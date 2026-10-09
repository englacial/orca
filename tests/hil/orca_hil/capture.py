"""acquire() helpers: capture directories, the session capture cache, retention."""
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import configs, scratch
from .radar import RadarRunner, RunResult


@dataclass
class Capture:
    test_id: str
    case: str
    case_dir: Path               # absolute, under <repo>/hil_scratch
    config_rel: Path             # repo-relative generated config
    result: RunResult | None = None
    shared: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def prefix(self):
        return self.result.prefix if self.result else None

    @property
    def zarr(self):
        return self.result.zarr if self.result else None


class CaptureManager:
    """Owns every capture in a session.

    Captures belong to the test that acquired them and are released (deleted, or
    raw files kept on failure) when that test finishes. ``shared_key`` captures
    can be reused by other tests (e.g. T6/T7) and are released at session end.
    """

    def __init__(self, fixture, run_id, run_dir, kept_root, keep_raw=False, runner=None):
        self.fixture = fixture
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.kept_root = Path(kept_root)
        self.keep_raw = keep_raw
        self.runner = runner or RadarRunner(fixture)
        self.quota_bytes = float((fixture.get("storage") or {}).get("kept_raw_quota_gb", 100)) * 1e9
        self._owned = {}     # test_id -> [Capture]
        self._shared = {}    # key -> Capture
        self._shared_ok = {}  # key -> all users passed so far

    def new_case(self, test_id, case, overrides) -> Capture:
        """Create the case directory and generated config (no radar run)."""
        cdir = scratch.case_dir(self.run_id, test_id, case)
        if cdir.exists():
            shutil.rmtree(cdir)
        cfg_rel = configs.build_config(self.fixture, overrides, cdir)
        cap = Capture(test_id, case, cdir, cfg_rel)
        self._owned.setdefault(test_id, []).append(cap)
        return cap

    def acquire(self, test_id, case, overrides, stop_after_s=None, shared_key=None, timeout_s=None):
        if shared_key and shared_key in self._shared:
            return self._shared[shared_key]
        cap = self.new_case(test_id, case, overrides)
        cap.result = self.runner.run(cap.config_rel, stop_after_s=stop_after_s, timeout_s=timeout_s)
        self.archive(cap)
        if shared_key:
            self._owned[test_id].remove(cap)
            cap.shared = True
            self._shared[shared_key] = cap
            self._shared_ok[shared_key] = True
        return cap

    def archive(self, cap: Capture):
        """Copy the generated config and stdout log into the run directory."""
        tdir = self.run_dir / cap.test_id
        if cap.prefix:
            src_cfg, src_log = Path(cap.prefix + "_config.yaml"), Path(cap.prefix + "_uhd_stdout.log")
        else:
            src_cfg, src_log = cap.case_dir / "config.yaml", cap.case_dir / "uhd_stdout.log"
        for src, dest in ((src_cfg, tdir / "configs" / f"{cap.case}_config.yaml"),
                          (src_log, tdir / "logs" / f"{cap.case}_uhd_stdout.log")):
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)

    def mark_shared_users(self, keys, passed):
        for k in keys:
            if k in self._shared_ok:
                self._shared_ok[k] &= passed

    def release_test(self, test_id, passed):
        kept = []
        for cap in self._owned.pop(test_id, []):
            k = scratch.release_case(cap.case_dir, passed, self.kept_root, self.run_id, test_id,
                                     keep_raw=self.keep_raw)
            if k:
                kept.append(k)
        if kept:
            scratch.enforce_quota(self.kept_root, self.quota_bytes, protect=self.run_id)
        return kept

    def release_all(self):
        for test_id in list(self._owned):
            self.release_test(test_id, passed=False)
        for key, cap in self._shared.items():
            scratch.release_case(cap.case_dir, self._shared_ok.get(key, False), self.kept_root,
                                 self.run_id, cap.test_id, keep_raw=self.keep_raw)
        self._shared.clear()
        scratch.enforce_quota(self.kept_root, self.quota_bytes, protect=self.run_id)
        shutil.rmtree(scratch.run_scratch_dir(self.run_id), ignore_errors=True)
