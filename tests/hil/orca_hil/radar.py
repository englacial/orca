"""Build and run the radar binary (sdr/build/radar) against the fixture's UHD.

run.py is not usable as a library (exit(1), unconditional gpspipe/gpxlogger,
fixed log names, no timeouts), so this module reimplements the parts the
harness needs, keeping run.py's stdout log format: each line prefixed with
``[<time.time():.3f>] \\t`` (process_stdout_log depends on it).
"""
import contextlib
import hashlib
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import REPO_ROOT, yamlio

from generate_chirp import generate_from_yaml_filename
from save_data import save_data
import processing_dask as pdk

SIGINT_GRACE_S = 10


class RadarTimeout(RuntimeError):
    pass


# ---------------------------------------------------------------- UHD install

def uhd_prefix(fixture) -> Path:
    return Path(fixture["uhd"]["install_prefix"]).expanduser()


def uhd_env(fixture) -> dict:
    """os.environ plus the fixture's uhd.env (paths expanded)."""
    env = dict(os.environ)
    for key, val in (fixture["uhd"].get("env") or {}).items():
        env[key] = os.path.expanduser(str(val))
    return env


def uhd_tool(fixture, name) -> Path:
    """UHD utilities are always called by full path, never through PATH."""
    return uhd_prefix(fixture) / "bin" / name


def run_uhd_tool(fixture, name, args=(), timeout=120):
    return subprocess.run([str(uhd_tool(fixture, name)), *args], env=uhd_env(fixture),
                          capture_output=True, text=True, timeout=timeout)


def build_stamp(fixture) -> dict | None:
    path = uhd_prefix(fixture) / "ORCA_UHD_BUILD.yaml"
    return yamlio.load(path) if path.exists() else None


# ---------------------------------------------------------------- building

@dataclass
class RadarBuild:
    binary: Path
    sha256: str
    libuhd: str | None


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolved_libuhd(binary, env=None) -> str | None:
    """The libuhd.so path that ``ldd`` resolves for a binary."""
    out = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, env=env).stdout
    m = re.search(r"libuhd\.so[^\s]*\s+=>\s+(\S+)", out)
    return m.group(1) if m else None


def build_radar(fixture, repo_root=REPO_ROOT, log_path=None) -> RadarBuild:
    """Configure and build sdr/build/radar against the fixture's UHD install."""
    if not os.environ.get("CONDA_PREFIX"):
        raise RuntimeError("activate the ORCA conda env (sdr/CMakeLists.txt uses $CONDA_PREFIX)")
    prefix = uhd_prefix(fixture)
    build = Path(repo_root) / "sdr" / "build"
    cmds = [
        ["cmake", "-S", "sdr", "-B", "sdr/build", f"-DCMAKE_PREFIX_PATH={prefix}",
         f"-DUHD_DIR={prefix}/lib/cmake/uhd", f"-DCMAKE_BUILD_RPATH={prefix}/lib"],
        ["cmake", "--build", "sdr/build", "--target", "radar", "-j4"],
    ]
    out = []
    for cmd in cmds:
        p = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        out.append(f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}")
        if p.returncode != 0:
            if log_path:
                Path(log_path).write_text("\n".join(out))
            raise RuntimeError(f"radar build failed: {' '.join(cmd)}\n{p.stderr[-2000:]}")
    if log_path:
        Path(log_path).write_text("\n".join(out))
    binary = build / "radar"
    return RadarBuild(binary, sha256(binary), resolved_libuhd(binary))


# ---------------------------------------------------------------- running

@dataclass
class RunResult:
    config_rel: Path
    prefix: str | None = None          # absolute save_data prefix
    zarr: str | None = None
    log_path: Path | None = None       # case-dir copy of the stdout log
    returncode: int | None = None
    duration_s: float = 0.0
    timed_out: bool = False
    killed: bool = False
    sigint_sent_at: float | None = None
    exited_at: float | None = None
    summary: dict = field(default_factory=dict)


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_summary(log_text) -> dict:
    """Error count, pulses written/attempted, and markers from a radar stdout log."""
    def grab(pattern):
        m = re.search(pattern, log_text)
        return int(m.group(1)) if m else None
    return {
        "error_count": grab(r"\[RX\] Error count: (\d+)"),
        "pulses_written": grab(r"\[RX\] Total pulses written: (\d+)"),
        "pulses_attempted": grab(r"\[RX\] Total pulses attempted: (\d+)"),
        "start_present": "[START]" in log_text,
        "close_file_lines": re.findall(r"\[CLOSE FILE\] (\S+)", log_text),
    }


def expected_duration_s(cfg) -> float:
    n = cfg["CHIRP"]["num_pulses"]
    pri = float(cfg["CHIRP"]["pulse_rep_int"])
    return float(cfg["CHIRP"].get("time_offset", 1)) + max(n, 0) * pri + 10.0


class RadarRunner:
    def __init__(self, fixture, repo_root=REPO_ROOT, echo=False):
        self.fixture = fixture
        self.repo_root = Path(repo_root)
        self.echo = echo

    def run(self, config_rel, stop_after_s=None, timeout_s=None, convert=True) -> RunResult:
        """Run one capture with the config at repo-relative ``config_rel``.

        stop_after_s: send SIGINT after this many seconds (for num_pulses = -1)
        timeout_s: default expected duration x 2 + 60 s; raises RadarTimeout
        """
        config_rel = Path(config_rel)
        cfg_abs = self.repo_root / config_rel
        cfg = yamlio.load(cfg_abs)
        case_dir = cfg_abs.parent
        if timeout_s is None:
            base = stop_after_s + 10 if stop_after_s is not None else expected_duration_s(cfg)
            timeout_s = 2 * base + 60
        result = RunResult(config_rel)

        generate_from_yaml_filename(str(cfg_abs))
        log_path = case_dir / "uhd_stdout.log"
        t0 = time.time()
        with open(log_path, "w") as log, open(case_dir / "uhd_stderr.log", "w") as err:
            proc = subprocess.Popen(["./radar", str(config_rel)], cwd=self.repo_root / "sdr" / "build",
                                    env=uhd_env(self.fixture), stdout=subprocess.PIPE, stderr=err,
                                    text=True, bufsize=1)
            reader = threading.Thread(target=self._pump, args=(proc.stdout, log), daemon=True)
            reader.start()
            try:
                if stop_after_s is not None:
                    try:
                        proc.wait(timeout=stop_after_s)
                    except subprocess.TimeoutExpired:
                        result.sigint_sent_at = time.time()
                        proc.send_signal(signal.SIGINT)
                        try:
                            proc.wait(timeout=SIGINT_GRACE_S)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            result.killed = True
                proc.wait(timeout=max(timeout_s - (time.time() - t0), 1))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                result.timed_out = result.killed = True
            result.exited_at = time.time()
            reader.join(timeout=10)
        result.returncode = proc.returncode
        result.duration_s = result.exited_at - t0
        result.log_path = log_path
        result.summary = parse_summary(log_path.read_text())
        if result.timed_out:
            raise RadarTimeout(f"radar did not finish within {timeout_s:.0f} s ({config_rel})")

        with _cwd(self.repo_root):
            prefix = save_data(str(config_rel), extra_files={str(log_path.relative_to(self.repo_root)):
                                                             "uhd_stdout.log"})
        result.prefix = str(self.repo_root / prefix)
        if convert and result.returncode == 0:
            result.zarr = pdk.save_radar_data_to_zarr(result.prefix)
        return result

    def _pump(self, out, log):
        for line in iter(out.readline, ""):
            stamped = f"[{time.time():0.3f}] \t{line}"
            log.write(stamped)
            if self.echo:
                print(stamped, end="")
        out.close()
