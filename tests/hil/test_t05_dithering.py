"""T5 - Coherent stacking and phase dithering noise floor (Fig. 7)."""
import numpy as np
import pytest
import xarray as xr

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.stacking import analyze_stacking

CASES = {"default": (0.0, False), "lo_offset": (12.5e6, False), "dither": (0.0, True),
         "lo_offset_dither": (12.5e6, True)}   # lo_offset_sw, phase_dithering
PARAMS = {
    "fast": {"groups": [{"suffix": "", "n_pulses": 10000, "presums": 1}]},
    "full": {"groups": [{"suffix": "", "n_pulses": 100000, "presums": 1},
                        {"suffix": "_p10", "n_pulses": 1000000, "presums": 10}]},
}
RX_DURATION = 40e-6   # 2240 samples: a longer noise window


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t05", "Fig. 7 noise floor vs coherent stacking", run_id, status)
    ax = axes[0, 0]
    n = reduced["n_stack"].values
    for c in reduced["case"].values:
        ax.semilogx(n, reduced["noise_mean_db"].sel(case=c), "o-", ms=3, label=c)
    ref = reduced["noise_mean_db"].values[:, 0].mean()
    ax.semilogx(n, ref - 10 * np.log10(n / n[0]), "k--", lw=0.8, label="-10 dB/decade")
    ax.set(xlabel="pulses stacked", ylabel="noise power (dBFS)")
    ax.legend()
    return plotting.save(fig, path)


def plot_variance(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t05", "variance of per-record noise power", run_id, status)
    ax = axes[0, 0]
    for c in reduced["case"].values:
        ax.loglog(reduced["n_stack"], reduced["noise_var"].sel(case=c), "o-", ms=3, label=c)
    ax.set(xlabel="pulses stacked", ylabel="variance (FS^4)")
    ax.legend()
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t05
@pytest.mark.timeout(5400)
def test_t05_dithering(hil, tier, fixture_cfg, capture_manager, recorder):
    caps, reduced_parts = [], []
    for g in PARAMS[tier]["groups"]:
        group = {}
        for case, (lo, dither) in CASES.items():
            cap = capture_manager.acquire("t05", case + g["suffix"], {
                "GENERATE": {"lo_offset_sw": lo},
                "CHIRP": {"phase_dithering": dither, "num_pulses": g["n_pulses"], "num_presums": g["presums"],
                          "rx_duration": RX_DURATION}})
            group[case + g["suffix"]] = cap.zarr
            caps.append(cap)
        params = {"plateau_case": "default" + g["suffix"],
                  "dither_pair": ("default" + g["suffix"], "dither" + g["suffix"]),
                  "fit_min_n": 10 * g["presums"]}
        reduced, ms = analyze_stacking(group, fixture_cfg, params)
        if g["suffix"]:
            # One dither-gain metric per run: keep the presums-1 group's
            ms = [m for m in ms if m.name != "dither_gain_at_nmax_db"]
            reduced = reduced.rename({"case": "case" + g["suffix"], "n_stack": "n_stack" + g["suffix"]})
            reduced = reduced.rename({v: v + g["suffix"] for v in reduced.data_vars})
        recorder.record_all(ms)
        reduced_parts.append(reduced)
    reduced = xr.merge(reduced_parts)
    testkit.finish(hil, recorder, tier, reduced, caps, {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
        "noise_variance.png": lambda s, path: plot_variance(reduced, hil.run_id, s, path),
    })
