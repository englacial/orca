"""T6 - Short-term coherence vs integration time (Fig. 3)."""
import numpy as np
import pytest

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.coherence import analyze_coherence


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t06", "Fig. 3 coherent power vs integration time", run_id, status)
    ax = axes[0, 0]
    T = reduced["integration_time"].values
    p = -reduced["power_loss_db"].values
    sd = np.nan_to_num(reduced["spread_1sigma_db"].values)
    ax.fill_between(T, p - sd, p + sd, color=plotting.BAND, label="+/-1 sigma across windows")
    plotting.plot_current(ax, T, p, marker="o", label="mean")
    ax.axvline(reduced.attrs["long_record_s"], color=plotting.BASELINE, ls=":", lw=0.8)
    ax.text(reduced.attrs["long_record_s"], 0.02, " burst | long capture", transform=ax.get_xaxis_transform(),
            fontsize=7, color=plotting.BASELINE)
    plotting.hband(ax, -0.2, 0.2, label="+/-0.2 dB")
    ax.set_xscale("log")
    ax.set(xlabel="integration time (s)", ylabel="coherent power rel. single pulse (dB)")
    ax.legend(loc="lower left")
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.full
@pytest.mark.t06
@pytest.mark.timeout(4800)
def test_t06_coherence(hil, tier, fixture_cfg, capture_manager, recorder):
    burst, long = testkit.coherence_captures(capture_manager, "t06")
    reduced, ms = analyze_coherence(burst.zarr, long.zarr, fixture_cfg)
    recorder.record_all(ms)
    capture_manager.mark_shared_users(["raw_burst", "long"], not recorder.failures)
    testkit.finish(hil, recorder, tier, reduced, [burst, long], {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
    })
