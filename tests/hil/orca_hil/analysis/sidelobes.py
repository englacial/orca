"""T2 window / sidelobe analysis."""
import numpy as np
import xarray as xr

from ..metrics import Measurement
from .common import db10, noise_window_start, open_capture, reference_chirp, upsample
from .peaks import leakage_removed, mainlobe_metrics
from .profile import UP, locate_peaks

FLOOR_MARGIN_DB = 6.0


def window_metrics(capture, fixture):
    """Measured and ideal mainlobe/sidelobe metrics for one window's capture."""
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    chirp = reference_chirp(cfg)
    lc = len(chirp)
    prof, resp, fit, _ = locate_peaks(ds, fixture, chirp)
    loop = leakage_removed(prof, resp, fit)

    nstart = noise_window_start(fit.t2, lc)
    floor = np.mean(np.abs(prof[nstart:]) ** 2) * 10 ** (FLOOR_MARGIN_DB / 10) if nstart < len(prof) else None
    meas = mainlobe_metrics(loop, fit.t2, lc, UP, floor_power=floor)

    # Ideal response at the same fractional position and level, with the same floor
    n = np.arange(len(prof))
    ideal_x = fit.a2 * resp.at(n - fit.t2)
    ideal = mainlobe_metrics(ideal_x, fit.t2, lc, UP, floor_power=floor)

    lo, hi = max(int(fit.t2 - lc) - 8, 0), min(int(fit.t2 + lc) + 9, len(prof))
    curves = {
        "measured_db": db10(np.abs(upsample(loop[lo:hi], UP)) ** 2) - db10(meas.peak_power),
        "ideal_db": db10(np.abs(upsample(ideal_x[lo:hi], UP)) ** 2) - db10(ideal.peak_power),
        "offset": lo + np.arange((hi - lo) * UP) / UP - meas.peak_pos,
    }
    return meas, ideal, lc, curves


def analyze_sidelobes(captures, fixture, params=None):
    """captures: {window_name: capture}. Returns (reduced Dataset, [Measurement])."""
    ms, rows = [], {}
    for window, cap in captures.items():
        meas, ideal, lc, curves = window_metrics(cap, fixture)
        ms += [
            Measurement("pslr_minus_ideal_db", meas.pslr_db - ideal.pslr_db, case=window),
            Measurement("mainlobe_width_ratio", meas.width_3db / ideal.width_3db, case=window),
            Measurement("islr_minus_ideal_db", meas.islr_db - ideal.islr_db, case=window),
            Measurement("sidelobe_extent_samples", meas.extent, case=window,
                        params={"chirp_length": lc},
                        inconclusive=meas.extent is None, note=None if meas.extent is not None
                        else "no noise window to set the sidelobe floor"),
        ]
        rows[window] = (meas, ideal, curves)

    windows = list(rows)
    lc_max = max(int(np.ceil(np.max(np.abs(rows[w][2]["offset"])))) for w in windows)
    grid = np.arange(-lc_max, lc_max + 1 / UP / 2, 1 / UP)

    def on_grid(w, key):
        c = rows[w][2]
        return np.interp(grid, c["offset"], c[key], left=np.nan, right=np.nan)

    reduced = xr.Dataset(
        {
            "measured_db": (("window", "offset"), np.stack([on_grid(w, "measured_db") for w in windows]),
                            {"units": "dB rel. peak"}),
            "ideal_db": (("window", "offset"), np.stack([on_grid(w, "ideal_db") for w in windows]),
                         {"units": "dB rel. peak"}),
            "pslr_db": ("window", [rows[w][0].pslr_db for w in windows], {"units": "dB"}),
            "ideal_pslr_db": ("window", [rows[w][1].pslr_db for w in windows], {"units": "dB"}),
            "islr_db": ("window", [rows[w][0].islr_db for w in windows], {"units": "dB"}),
            "ideal_islr_db": ("window", [rows[w][1].islr_db for w in windows], {"units": "dB"}),
            "width_3db": ("window", [rows[w][0].width_3db for w in windows], {"units": "samples"}),
        },
        coords={"window": windows,
                "offset": ("offset", grid, {"units": "samples"})},
    )
    return reduced, ms
