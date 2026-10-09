"""T7 - Long-term phase stability (Fig. 4)."""
import numpy as np
import pytest

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.coherence import analyze_phase_stability


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t07", "Fig. 4 loopback phase vs slow time", run_id, status)
    ax = axes[0, 0]
    t = reduced["slow_time"].values
    a = reduced.attrs
    plotting.plot_current(ax, t, np.degrees(reduced["phase_rad"].values), lw=0.5, label="loopback peak phase")
    ax.plot(t, np.degrees(a["phase0_rad"] + a["drift_rad_s"] * t), "k--", lw=0.8,
            label=f"fit: {np.degrees(a['drift_rad_s']) * 1e3:.3f} mdeg/s")
    ax.text(0.02, 0.95, f"single-pulse Var(phi) = {a['var_phi_rad2']:.3g} rad^2\n"
            f"1/(2 SNR) = {a['theory_var_rad2']:.3g} rad^2 (SNR = |A|^2 / complex noise var, {a['snr_db']:.1f} dB)",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set(xlabel="slow time (s)", ylabel="phase (deg)")
    ax.legend(loc="lower right")
    return plotting.save(fig, path)


def plot_residual(reduced, temps, run_id, status, path):
    fig, axes = plotting.summary_figure("t07", "phase residual after linear fit", run_id, status, 1, 2,
                                        gridspec_kw={"width_ratios": [1, 2.5]})
    r = reduced["residual_deg"].values
    axes[0, 0].hist(r, bins=100, color=plotting.ACCENT)
    axes[0, 0].set(xlabel="residual (deg)", ylabel="records", yscale="log")
    ax = axes[0, 1]
    t = reduced["slow_time"].values
    ax.plot(t, r, lw=0.4, color=plotting.ACCENT)
    for k in reduced["step_record"].values:
        ax.axvline(t[k], color=plotting.FAIL, lw=0.6)
    ax.set(xlabel="slow time (s)", ylabel="residual (deg)")
    if temps:
        tw = ax.twinx()
        tw.plot([x for x, _ in temps], [y for _, y in temps], color=plotting.FAIL, lw=1, label="SoC temp")
        tw.set_ylabel("SoC temperature (degC)", color=plotting.FAIL)
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.full
@pytest.mark.t07
@pytest.mark.timeout(4800)
def test_t07_phase_stability(hil, tier, fixture_cfg, capture_manager, recorder):
    burst, long = testkit.coherence_captures(capture_manager, "t07")
    reduced, ms = analyze_phase_stability(long.zarr, burst.zarr, fixture_cfg)
    recorder.record_all(ms)
    capture_manager.mark_shared_users(["raw_burst", "long"], not recorder.failures)
    t1 = long.result.exited_at
    t0 = t1 - long.result.duration_s
    temps = [(r["time"] - t0, r["soc_temp_c"]) for r in testkit.telemetry_rows(hil, recorder, t0, t1)
             if r["soc_temp_c"] is not None]
    testkit.finish(hil, recorder, tier, reduced, [burst, long], {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
        "residual.png": lambda s, path: plot_residual(reduced, temps, hil.run_id, s, path),
    })
