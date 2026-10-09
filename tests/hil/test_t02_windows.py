"""T2 - Window / sidelobe sweep (Fig. 1c sidelobe discussion)."""
import numpy as np
import pytest

from tests.hil.orca_hil import plotting, testkit
from tests.hil.orca_hil.analysis.sidelobes import analyze_sidelobes

ALL_WINDOWS = ["rectangular", "hamming", "blackman", "kaiser10", "kaiser14", "kaiser18"]
PARAMS = {
    "fast": {"windows": ["rectangular", "hamming"], "n_pulses": 10000},
    "full": {"windows": ALL_WINDOWS, "n_pulses": 10000},
}


def plot_summary(reduced, run_id, status, path):
    windows = list(reduced["window"].values)
    ncols = min(len(windows), 3)
    nrows = int(np.ceil(len(windows) / ncols))
    fig, axes = plotting.summary_figure("t02", "measured vs ideal compressed response", run_id, status,
                                        nrows, ncols, sharex=True, sharey=True)
    for ax, w in zip(axes.flat, windows):
        off = reduced["offset"].values
        plotting.plot_baseline(ax, off, reduced["ideal_db"].sel(window=w), label="ideal")
        plotting.plot_current(ax, off, reduced["measured_db"].sel(window=w), label="measured")
        ax.set_title(f"{w}: PSLR {float(reduced['pslr_db'].sel(window=w)):.1f} dB "
                     f"(ideal {float(reduced['ideal_pslr_db'].sel(window=w)):.1f})")
        ax.set_ylim(-110, 5)
        ax.set_xlabel("offset from loopback peak (samples)")
        ax.set_ylabel("dB rel. peak")
    for ax in list(axes.flat)[len(windows):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper right")
    return plotting.save(fig, path)


def plot_table(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t02", "sidelobe metrics", run_id, status)
    ax = axes[0, 0]
    ax.axis("off")
    rows = [[w, f"{float(reduced['pslr_db'].sel(window=w)):.2f}", f"{float(reduced['ideal_pslr_db'].sel(window=w)):.2f}",
             f"{float(reduced['islr_db'].sel(window=w)):.2f}", f"{float(reduced['ideal_islr_db'].sel(window=w)):.2f}",
             f"{float(reduced['width_3db'].sel(window=w)):.3f}"] for w in reduced["window"].values]
    t = ax.table(cellText=rows, colLabels=["window", "PSLR dB", "ideal PSLR", "ISLR dB", "ideal ISLR",
                                           "-3 dB width (samples)"], loc="center")
    t.scale(1, 1.6)
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t02
@pytest.mark.timeout(900)
def test_t02_windows(hil, tier, fixture_cfg, capture_manager, recorder):
    p = PARAMS[tier]
    caps = [capture_manager.acquire("t02", w, {"GENERATE": {"window": w}, "CHIRP": {"num_pulses": p["n_pulses"]}})
            for w in p["windows"]]
    reduced, ms = analyze_sidelobes({c.case: c.zarr for c in caps}, fixture_cfg)
    recorder.record_all(ms)
    testkit.finish(hil, recorder, tier, reduced, caps, {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
        "pslr_table.png": lambda s, path: plot_table(reduced, hil.run_id, s, path),
    })
