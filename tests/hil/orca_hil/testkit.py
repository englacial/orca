"""Small helpers shared by the hardware test files."""
import csv

from . import REPO_ROOT, yamlio
from .reduced import write_reduced


def case_configs(captures):
    """{case: generated config} for a list of Capture objects."""
    return {c.case: yamlio.load(c.case_dir / "config.yaml") for c in captures}


def finish(hil, recorder, tier, reduced, captures, plots):
    """Write reduced.zarr.zip and plots, then finalize the recorder.

    plots: {filename: callable(status, path)}; summary.png first by convention.
    Plots are drawn even if some metrics failed, so failures can be inspected.
    """
    tid = recorder.test_id
    out = hil.run_dir / tid
    status = "fail" if recorder.failures else recorder.status
    if reduced is not None:
        write_reduced(reduced, out / "reduced.zarr.zip", test_id=tid, run_id=hil.run_id,
                      git_sha=hil.git["sha"], fixture_id=hil.fixture_id, tier=tier,
                      configs=case_configs(captures))
    for name, draw in plots.items():
        draw(status, out / name)
    return recorder.finalize()


def baseline_mean(baselines, test_id, name, case=None):
    b = baselines.get(test_id, name, case)
    return None if b is None else b["mean"]


def loopback_index_from_baseline(baselines):
    """Expected loopback position (compressed samples) from the T1 baseline, or None."""
    t1 = baseline_mean(baselines, "t01", "leakage_peak_index")
    sep = baseline_mean(baselines, "t01", "peak_separation_samples")
    return None if t1 is None or sep is None else t1 + sep


def repo_path(rel):
    return REPO_ROOT / rel


# Captures shared by T6 and T7 (whichever runs first acquires them)
RAW_BURST = {"CHIRP": {"num_pulses": 40000, "num_presums": 1}}                             # 10 s
LONG_CAPTURE = {"CHIRP": {"num_pulses": 6_000_000, "num_presums": 40, "rx_duration": 25e-6}}   # 25 min


def coherence_captures(capture_manager, test_id):
    """(raw_burst, long) shared captures."""
    return (capture_manager.acquire(test_id, "raw_burst", RAW_BURST, shared_key="raw_burst"),
            capture_manager.acquire(test_id, "long", LONG_CAPTURE, shared_key="long"))


def telemetry_rows(hil, recorder, t0, t1):
    """Telemetry samples between wall times t0 and t1 from this run (all tests so far)."""
    rows = []
    for path in hil.run_dir.glob("*/telemetry.csv"):
        with open(path) as f:
            rows += [{k: float(v) if v not in ("", "None") else None for k, v in r.items()}
                     for r in csv.DictReader(f)]
    if hasattr(recorder, "telemetry"):
        rows += recorder.telemetry.rows
    return sorted((r for r in rows if r["time"] is not None and t0 <= r["time"] <= t1), key=lambda r: r["time"])
