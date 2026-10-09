"""T12 - Frequency response sweep (not in the paper)."""
import pytest

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.freqresp import analyze_freq_response, case_name

FREQS_HZ = [150e6, 250e6, 350e6, 450e6, 600e6, 750e6, 900e6]
N_PULSES = 5000


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t12", "path gain vs frequency", run_id, status, 2, 1, sharex=True)
    f = reduced["freq"].values / 1e6
    ax = axes[0, 0]
    plotting.plot_current(ax, f, reduced["loopback_power_dbfs"], marker="o", label="loopback peak")
    ax.plot(f, reduced["loopback_fit_dbfs"], "k--", lw=0.8, label="2nd-order fit in sqrt(f)")
    ax.plot(f, reduced["leakage_power_dbfs"], "s:", color=plotting.BASELINE, label="internal leakage peak")
    ax.set(ylabel="dBFS")
    ax.legend()
    ax = axes[1, 0]
    plotting.plot_current(ax, f, reduced["loopback_power_dbfs"] - reduced["leakage_power_dbfs"], marker="o",
                          label="loopback / leakage")
    ax.set(xlabel="RF frequency (MHz)", ylabel="dB")
    ax.legend()
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.full
@pytest.mark.t12
@pytest.mark.timeout(1800)
def test_t12_freq_response(hil, tier, fixture_cfg, capture_manager, recorder):
    caps = [capture_manager.acquire("t12", case_name(f), {"RF0": {"freq": f}, "CHIRP": {"num_pulses": N_PULSES}})
            for f in FREQS_HZ]
    reduced, ms = analyze_freq_response({f: c.zarr for f, c in zip(FREQS_HZ, caps)}, fixture_cfg)
    recorder.record_all(ms)
    testkit.finish(hil, recorder, tier, reduced, caps, {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
    })
