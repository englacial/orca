"""T11 - Presum equivalence: on-device presumming vs offline stacking (Sec. III-B, VII)."""
import pytest

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.presum import analyze_presum

PARAMS = {"fast": {"n_pulses": 50000, "presums": 100}}
PARAMS["full"] = PARAMS["fast"]


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t11", f"offline stack vs {reduced.attrs['presums']} on-device presums",
                                        run_id, status)
    ax = axes[0, 0]
    n = reduced["sample"].values
    plotting.plot_baseline(ax, n, reduced["offline_profile_db"], label="p1 stacked offline")
    plotting.plot_current(ax, n, reduced["on_device_profile_db"], label="on-device presums", ls="--")
    ax.set(xlabel="compressed sample", ylabel="power (dBFS)")
    ax.legend()
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t11
@pytest.mark.timeout(900)
def test_t11_presum(hil, tier, fixture_cfg, capture_manager, recorder):
    p = PARAMS[tier]
    caps = [capture_manager.acquire("t11", case, {"CHIRP": {"num_pulses": p["n_pulses"], "num_presums": presums}})
            for case, presums in (("p1", 1), ("p100", p["presums"]))]
    reduced, ms = analyze_presum({c.case: c.zarr for c in caps}, fixture_cfg)
    recorder.record_all(ms)
    testkit.finish(hil, recorder, tier, reduced, caps, {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
    })
