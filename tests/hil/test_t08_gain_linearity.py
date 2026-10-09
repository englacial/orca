"""T8 - Gain linearity and received-power calibration (Fig. 8)."""
import numpy as np
import pytest
import xarray as xr

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.gain import analyze_gain

N_PULSES = 2000


def sweeps(tier, fixture):
    tx0, rx0 = fixture["rf_base"]["tx_gain"], fixture["rf_base"]["rx_gain"]
    cap = fixture["safety"]["tx_gain_max"]
    if tier == "fast":
        return {"tx": list(np.arange(tx0 - 12, tx0 + 12 + 1e-9, 6)), "rx": []}
    return {"tx": list(np.arange(tx0 - 30, cap + 1e-9, 3)),
            "rx": list(np.arange(rx0 - 20, rx0 + 15 + 1e-9, 5))}


def plot_summary(reduced, run_id, status, path):
    sweeps_present = [s for s in ("tx", "rx") if (reduced["sweep"] == s).any()]
    fig, axes = plotting.summary_figure("t08", "Fig. 8 received power vs gain", run_id, status,
                                        1, len(sweeps_present))
    for ax, s in zip(axes.flat, sweeps_present):
        sel = reduced.where(reduced["sweep"] == s, drop=True)
        g = sel["gain"].values
        plotting.plot_current(ax, g, sel["loopback_dbfs"], marker="o", label="loopback peak")
        ax.plot(g, sel["noise_dbfs"], "x--", color=plotting.BASELINE, label="noise floor (compressed)")
        fit = reduced.attrs["fits"].get(s, {})
        if np.isfinite(fit.get("slope", np.nan)):
            ax.text(0.02, 0.95, f"slope {fit['slope']:.3f} dB/dB", transform=ax.transAxes, va="top")
        if fit.get("p1db") is not None:
            ax.axvline(fit["p1db"], color=plotting.FAIL, ls=":", label="P1dB")
        ax.set(xlabel=f"{s.upper()} gain (dB)", ylabel="dBFS")
        ax.legend(loc="lower right")
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t08
@pytest.mark.timeout(2400)
def test_t08_gain_linearity(hil, tier, fixture_cfg, capture_manager, recorder):
    fx = fixture_cfg
    clip = fx["safety"]["rx_clip_fraction_max"]
    points, caps = [], []
    for sweep, gains in sweeps(tier, fx).items():
        for g in gains:
            key = "tx_gain" if sweep == "tx" else "rx_gain"
            cap = capture_manager.acquire("t08", f"{sweep}{g:g}", {"RF0": {key: float(g)},
                                                                  "CHIRP": {"num_pulses": N_PULSES}})
            caps.append(cap)
            points.append({"sweep": sweep, "gain": float(g), "capture": cap.zarr})
            raw_max = float(np.abs(xr.open_zarr(cap.zarr)["radar_data"]).max())
            if sweep == "tx" and raw_max > clip:
                break
    reduced, ms = analyze_gain(points, fx, {"base_tx_gain": fx["rf_base"]["tx_gain"],
                                            "base_rx_gain": fx["rf_base"]["rx_gain"], "full": tier == "full"})
    recorder.record_all(ms)
    testkit.finish(hil, recorder, tier, reduced, caps, {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
    })
