"""Golden captures for replay (HIL plan Section 8).

    python -m tests.hil.orca_hil.golden make      # capture on the fixture, package, write golden.yaml
    python -m tests.hil.orca_hil.golden upload    # create the GitHub release from the packaged files
    python -m tests.hil.orca_hil.golden fetch     # download and verify into the cache

`make` packages each test's raw prefix files as one uncompressed tar, then computes the
expected metrics by replaying those tars exactly as the replay tests do. golden.yaml
freezes everything analyze() needs besides the captures: the fixture, and any
baseline-derived params.
"""
import argparse
import hashlib
import math
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from . import HIL_DIR, scratch, yamlio
from .baselines import Baselines

GOLDEN_YAML = HIL_DIR / "golden.yaml"
RELEASE_REPO = "englacial/orca-hil-testing-results"
CACHE = Path(os.environ.get("ORCA_HIL_GOLDEN_CACHE", "~/.cache/orca-hil-golden")).expanduser()
REL_TOL = 1e-6
ABS_TOL = 1e-6

T5_RX_DURATION = 40e-6
T10_PULSES = 20000          # 2000 records at presums 10


def capture_plan():
    """{test_id: {case: overrides}}, except T10 (its PRI comes from a duty search)."""
    from tests.hil.test_t04_spectral import CASES as T4_CASES
    from tests.hil.test_t05_dithering import CASES as T5_CASES
    return {
        "t01": {"a_profile": {"CHIRP": {"num_pulses": 3000}}},
        "t02": {"hamming": {"GENERATE": {"window": "hamming"}, "CHIRP": {"num_pulses": 3000}}},
        "t03": {"rx_only": {"RF0": {"transmit": False}, "CHIRP": {"num_pulses": 3000}}},
        "t04": {c: {"GENERATE": {"lo_offset_sw": lo}, "CHIRP": {"phase_dithering": d, "num_pulses": 2000}}
                for c, (lo, d) in T4_CASES.items() if c in ("a", "b")},
        "t05": {c: {"GENERATE": {"lo_offset_sw": lo},
                    "CHIRP": {"phase_dithering": d, "num_pulses": 2000, "num_presums": 1,
                              "rx_duration": T5_RX_DURATION}}
                for c, (lo, d) in T5_CASES.items()},
        "t11": {case: {"CHIRP": {"num_pulses": 10000, "num_presums": n}} for case, n in (("p1", 1), ("p100", 100))},
    }


def analysis_params(baselines):
    """Per-test analyze() params, frozen into golden.yaml."""
    from . import testkit
    return {
        "t03": {"baseline_lines": baselines.get_raw("t03", "narrowband_lines"),
                "loopback_index": testkit.loopback_index_from_baseline(baselines)},
        "t04": {"baseline_spurs": baselines.get_raw("t04", "spurs")},
        "t05": {"plateau_case": "default", "dither_pair": ["default", "dither"], "fit_min_n": 10},
    }


def analyze(test_id, zarrs, fixture, params):
    """Run one test's analyze() on {case: zarr}; returns (reduced, [Measurement])."""
    from .analysis.noise import analyze_rx_only
    from .analysis.presum import analyze_presum
    from .analysis.profile import analyze_profile
    from .analysis.recovery import analyze_recovery
    from .analysis.sidelobes import analyze_sidelobes
    from .analysis.spectral import analyze_spectral
    from .analysis.stacking import analyze_stacking
    p = dict(params or {})
    if "dither_pair" in p:
        p["dither_pair"] = tuple(p["dither_pair"])
    return {
        "t01": lambda: analyze_profile(zarrs["a_profile"], fixture),
        "t02": lambda: analyze_sidelobes(zarrs, fixture),
        "t03": lambda: analyze_rx_only(zarrs["rx_only"], fixture, p),
        "t04": lambda: analyze_spectral(zarrs, fixture, p),
        "t05": lambda: analyze_stacking(zarrs, fixture, p),
        "t10": lambda: analyze_recovery(zarrs["errors"], fixture),
        "t11": lambda: analyze_presum(zarrs, fixture),
    }[test_id]()


def metric_values(measurements):
    return [{"name": m.name, "case": m.case, "value": yamlio.to_builtin(m.value)} for m in measurements]


def _close(exp, got, tol):
    if isinstance(exp, bool) or isinstance(got, bool) or not isinstance(exp, (int, float)):
        return exp == got
    if not isinstance(got, (int, float)):
        return False
    if math.isnan(exp) or math.isnan(got):
        return math.isnan(exp) and math.isnan(got)
    return abs(got - exp) <= max(tol.get("abs", ABS_TOL), tol.get("rel", REL_TOL) * abs(exp))


def compare(test_id, expected, got, tolerances=None):
    """Lines describing every difference between expected and replayed metrics (empty if equal)."""
    key = lambda m: (m["name"], m["case"] or "")
    exp, new = {key(m): m["value"] for m in expected}, {key(m): m["value"] for m in got}
    diffs = []
    for k in sorted(exp.keys() | new.keys()):
        label = k[0] + (f" [{k[1]}]" if k[1] else "")
        if k not in new:
            diffs.append(f"{label}: missing (expected {exp[k]!r})")
        elif k not in exp:
            diffs.append(f"{label}: unexpected new metric = {new[k]!r}")
        elif not _close(exp[k], new[k], (tolerances or {}).get(f"{test_id}.{k[0]}", {})):
            diffs.append(f"{label}: expected {exp[k]!r}, got {new[k]!r}")
    return diffs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package(test_id, caps, out_dir):
    """One tar per test; members are <case>/<prefix basename>_*."""
    path = Path(out_dir) / f"{test_id}.tar"
    with tarfile.open(path, "w") as tar:
        for cap in caps:
            prefix = Path(cap.prefix)
            for f in sorted(prefix.parent.glob(prefix.name + "_*")):
                if f.is_file():
                    tar.add(f, arcname=f"{cap.case}/{f.name}")
    return path


def prepare(golden_dir, test_id):
    """Extract <test_id>.tar (once) and convert each case to zarr; returns {case: zarr path}."""
    import processing_dask as pdk
    dest = Path(golden_dir) / test_id
    if not dest.exists():
        with tarfile.open(Path(golden_dir) / f"{test_id}.tar") as tar:
            tar.extractall(dest.with_suffix(".tmp"), filter="data")
        dest.with_suffix(".tmp").rename(dest)
    zarrs = {}
    for case_dir in sorted(p for p in dest.iterdir() if p.is_dir()):
        prefix = str(next(case_dir.glob("*_rx_samps.bin"))).removesuffix("_rx_samps.bin")
        zarrs[case_dir.name] = pdk.save_radar_data_to_zarr(prefix)
    return zarrs


def load(path=GOLDEN_YAML):
    return yamlio.load(path) if Path(path).exists() else None


def fetch(golden=None, cache=CACHE):
    """Download the release assets into <cache>/<tag> (if missing) and verify their SHA-256s."""
    golden = golden or load()
    dest = Path(cache) / golden["tag"]
    dest.mkdir(parents=True, exist_ok=True)
    missing = [a for a in golden["assets"] if not (dest / a).exists()]
    if missing:
        subprocess.run(["gh", "release", "download", golden["tag"], "--repo", golden["repo"], "--dir", str(dest),
                        *sum((["--pattern", a] for a in missing), [])], check=True)
    for name, meta in golden["assets"].items():
        if sha256(dest / name) != meta["sha256"]:
            raise RuntimeError(f"golden asset {name} does not match its SHA-256 in golden.yaml")
    return dest


def replay_all(golden_dir, golden):
    return {tid: metric_values(analyze(tid, prepare(golden_dir, tid), golden["fixture"],
                                       golden["params"].get(tid))[1])
            for tid in sorted(golden["assets_by_test"])}


def make(fixture_path, out_root):
    from tests.hil.test_t10_error_recovery import OVERRIDES as T10_OVERRIDES, find_duty
    from .analysis.duty import duty_to_pri
    from .capture import CaptureManager
    from .configs import merged_config
    from .radar import build_radar
    fixture = yamlio.load(fixture_path)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    tag = f"golden-{fixture['fixture_id']}-{date}"
    out_dir = Path(out_root) / tag
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; remove it to make a new golden set")
    out_dir.mkdir(parents=True)
    build_radar(fixture, log_path=out_dir / "build.log")
    baselines = Baselines(fixture["fixture_id"], "fast")
    run_id = f"golden_{date}"
    mgr = CaptureManager(fixture, run_id, out_dir / "run", out_dir / "kept")
    plan = capture_plan()
    assets = {}
    try:
        for tid, cases in plan.items():
            caps = [mgr.acquire(tid, case, ov, convert=False) for case, ov in cases.items()]
            assets[tid] = package(tid, caps, out_dir)
            mgr.release_test(tid, passed=True)
        duty, probes = find_duty(mgr, fixture, baselines, max_probes=4)
        base = merged_config(fixture, T10_OVERRIDES)
        ov = {"CHIRP": {**T10_OVERRIDES["CHIRP"], "num_pulses": T10_PULSES, "pulse_rep_int": duty_to_pri(base, duty)}}
        plan["t10"] = {"errors": ov}
        assets["t10"] = package("t10", [mgr.acquire("t10", "errors", ov, convert=False)], out_dir)
        mgr.release_test("t10", passed=True)
    finally:
        mgr.release_all()
        shutil.rmtree(scratch.run_scratch_dir(run_id), ignore_errors=True)

    golden = {
        "tag": tag, "repo": RELEASE_REPO, "fixture_id": fixture["fixture_id"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assets": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in assets.values()},
        "assets_by_test": {tid: p.name for tid, p in assets.items()},
        "captures": plan, "t10_duty": duty, "t10_probes": probes,
        "fixture": fixture, "params": analysis_params(baselines), "tolerances": {},
    }
    golden["expected"] = replay_all(out_dir, golden)
    yamlio.dump(golden, GOLDEN_YAML)
    return out_dir, golden


def upload(golden=None, out_root=None):
    golden = golden or load()
    src = Path(out_root) / golden["tag"]
    subprocess.run(["gh", "release", "create", golden["tag"], "--repo", golden["repo"],
                    "--title", golden["tag"], "--notes", f"HIL golden captures for {golden['fixture_id']}",
                    *[str(src / a) for a in golden["assets"]]], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HIL golden captures.")
    ap.add_argument("action", choices=("make", "upload", "fetch"))
    ap.add_argument("--fixture", default=None, help="fixture id (default: the only file in fixtures/)")
    a = ap.parse_args(argv)
    files = sorted((HIL_DIR / "fixtures").glob("*.yaml"))
    fx_path = HIL_DIR / "fixtures" / f"{a.fixture}.yaml" if a.fixture else files[0]
    out_root = Path(yamlio.load(fx_path)["host"]["data_root"]).expanduser() / "golden"
    if a.action == "make":
        out_dir, golden = make(fx_path, out_root)
        total = sum(m["bytes"] for m in golden["assets"].values()) / 1e6
        print(f"golden set {golden['tag']}: {total:.0f} MB in {out_dir}; wrote {GOLDEN_YAML}")
    elif a.action == "upload":
        upload(out_root=out_root)
    else:
        print(fetch())


if __name__ == "__main__":
    main()
