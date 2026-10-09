"""Small helpers shared by the hardware test files."""
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
