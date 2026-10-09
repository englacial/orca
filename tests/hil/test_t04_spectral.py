"""T4 - Spectral artifacts: LO leakage, PLL spurs, chirp copies (Fig. 5)."""
import numpy as np
import pytest

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.spectral import analyze_spectral

CASES = {"a": (0.0, False), "b": (0.0, True), "c": (12.5e6, True)}   # lo_offset_sw, phase_dithering
PARAMS = {"fast": {"n_pulses": 10000}, "full": {"n_pulses": 10000}}
LABELS = {"a": "no offset, no dither", "b": "no offset, dither", "c": "12.5 MHz offset, dither"}


def plot_summary(reduced, run_id, status, path):
    cases = list(reduced["case"].values)
    fig, axes = plotting.summary_figure("t04", "Fig. 5 spectrograms of the stacked pulse", run_id, status,
                                        1, len(cases), sharey=True)
    t = reduced["stft_time"].values * 1e6
    f = reduced["stft_freq"].values / 1e6
    for ax, c in zip(axes.flat, cases):
        z = reduced["spectrogram_db"].sel(case=c).values
        im = ax.pcolormesh(t, f, z, vmin=np.nanmax(z) - 70, vmax=np.nanmax(z), shading="auto", cmap="viridis")
        for sp in reduced.attrs["spurs"][c]:
            ax.axhline(sp["freq"] / 1e6, color="w", lw=0.6, ls=":")
        ax.axhline(0, color="r", lw=0.6, ls="--")
        lo, hi = reduced.attrs["bands_hz"][c]
        ax.set(title=f"{c}: {LABELS.get(c, c)}\nLO leakage {float(reduced['lo_leakage_rel_chirp_db'].sel(case=c)):.1f} dB",
               xlabel="time (us)")
        ax.text(0.02, 0.02, f"-10 dB band {lo / 1e6:.1f}..{hi / 1e6:.1f} MHz", transform=ax.transAxes,
                color="w", fontsize=7)
    axes[0, 0].set_ylabel("frequency (MHz)")
    fig.subplots_adjust(top=0.82, wspace=0.12)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="dB rel. chirp")
    fig.savefig(path, dpi=plotting.DPI)
    plotting.plt.close(fig)
    return path


def plot_spectra(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t04", "signal-free tail spectra", run_id, status)
    ax = axes[0, 0]
    f = reduced["freq"].values / 1e6
    for c in reduced["case"].values:
        ax.plot(f, reduced["tail_psd_db"].sel(case=c), lw=0.8, label=f"{c}: {LABELS.get(c, c)}")
    ax.set(xlabel="frequency (MHz)", ylabel="dB rel. chirp power")
    ax.legend()
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t04
@pytest.mark.timeout(900)
def test_t04_spectral(hil, tier, fixture_cfg, capture_manager, recorder, baselines):
    n = PARAMS[tier]["n_pulses"]
    caps = [capture_manager.acquire("t04", c, {"GENERATE": {"lo_offset_sw": lo},
                                               "CHIRP": {"phase_dithering": dither, "num_pulses": n}})
            for c, (lo, dither) in CASES.items()]
    reduced, ms = analyze_spectral({c.case: c.zarr for c in caps}, fixture_cfg,
                                   {"baseline_spurs": baselines.get_raw("t04", "spurs")})
    recorder.record_all(ms)
    testkit.finish(hil, recorder, tier, reduced, caps, {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
        "spectra.png": lambda s, path: plot_spectra(reduced, hil.run_id, s, path),
    })
