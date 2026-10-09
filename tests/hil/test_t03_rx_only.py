"""T3 - RX-only noise and interference sentinel (adapts Fig. 6; passive mode).

A failure here means the fixture needs attention; T4-T7 still run but their
metrics are flagged fixture_suspect.
"""
import numpy as np
import pytest

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.noise import analyze_rx_only

PARAMS = {"fast": {"n_pulses": 10000}, "full": {"n_pulses": 100000}}


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t03", "Fig. 6 RX-only noise", run_id, status, 2, 2)
    ax = axes[0, 0]
    n = reduced["fast_sample"].values
    ax.plot(n, reduced["raw_example_i"], lw=0.5, label="I", color=plotting.ACCENT)
    ax.plot(n, reduced["raw_example_q"], lw=0.5, label="Q", color=plotting.BASELINE)
    ax.set(title="raw record (first)", xlabel="sample", ylabel="FS")
    ax.legend()

    ax = axes[0, 1]
    f = reduced["freq"].values / 1e6
    plotting.plot_current(ax, f, reduced["psd_db"], label="PSD")
    for ln in reduced.attrs["lines"]:
        new = any(abs(ln["freq"] - x["freq"]) < 1 for x in reduced.attrs["new_lines"])
        ax.axvline(ln["freq"] / 1e6, color=plotting.FAIL if new else plotting.BASELINE, lw=0.8, ls=":")
    ax.set(title=f"PSD ({len(reduced.attrs['lines'])} lines, {len(reduced.attrs['new_lines'])} new)",
           xlabel="frequency (MHz)", ylabel="dBFS/Hz")

    ax = axes[1, 0]
    a = reduced["amplitude"].values
    s = reduced.attrs["sigma_component"]
    ax.semilogy(a, reduced["amplitude_pdf"], ".", color=plotting.ACCENT, label="measured")
    ax.semilogy(a, a / s ** 2 * np.exp(-a ** 2 / (2 * s ** 2)), color=plotting.BASELINE, label="Rayleigh")
    ax.set(title="amplitude distribution", xlabel="|x| (FS)", ylabel="pdf")
    ax.legend()

    ax = axes[1, 1]
    ns = reduced["n_stack"].values
    curve = reduced["noise_vs_n_db"].values
    ax.semilogx(ns, curve, "o-", color=plotting.ACCENT, label="measured")
    ax.semilogx(ns, curve[0] - 10 * np.log10(ns), "--", color=plotting.BASELINE, label="-10 dB/decade")
    ax.set(title="Fig. 6b-style noise vs stacking", xlabel="pulses stacked", ylabel="noise (dBFS)")
    ax.legend()
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t03
@pytest.mark.timeout(1200)
def test_t03_rx_only(hil, tier, fixture_cfg, capture_manager, recorder, baselines):
    p = PARAMS[tier]
    cap = capture_manager.acquire("t03", "rx_only", {"RF0": {"transmit": False},
                                                     "CHIRP": {"num_pulses": p["n_pulses"]}})
    params = {"baseline_lines": baselines.get_raw("t03", "narrowband_lines"),
              "loopback_index": testkit.loopback_index_from_baseline(baselines)}
    reduced, ms = analyze_rx_only(cap.zarr, fixture_cfg, params)
    recorder.record_all(ms)
    testkit.finish(hil, recorder, tier, reduced, [cap], {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
    })
