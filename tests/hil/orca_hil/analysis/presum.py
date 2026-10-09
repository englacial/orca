"""T11 presum equivalence: on-device presumming vs offline stacking (Sec. III-B, VII)."""
import numpy as np
import xarray as xr

import processing_dask as pdk

from ..metrics import Measurement
from .common import db10, noise_window_start, open_capture, reference_chirp, stacked_profile
from .profile import locate_peaks, single_pulse_stats

CORR_HALF_WIDTH = 100


def record_powers(ds, chirp, t1, t2, nstart):
    """Mean per-record loopback peak, leakage peak (noise power removed) and noise powers."""
    peaks, noise = single_pulse_stats(ds, chirp, t2, nstart)
    leak, _ = single_pulse_stats(ds, chirp, t1, nstart)
    n = float(np.mean(np.abs(noise) ** 2))
    return float(np.mean(np.abs(peaks) ** 2)) - n, float(np.mean(np.abs(leak) ** 2)) - n, n


def analyze_presum(captures, fixture, params=None):
    """captures: {"p1": presums 1, "p100": presums P}; p1 is stacked offline by P."""
    d1, dp = open_capture(captures["p1"]), open_capture(captures["p100"])
    P = int(dp.attrs["config"]["CHIRP"]["num_presums"])
    chirp = reference_chirp(dp.attrs["config"])
    _, _, fit, _ = locate_peaks(dp, fixture, chirp)
    nstart = noise_window_start(fit.t2, len(chirp))

    d1_st = pdk.stack(d1[["radar_data"]], P)
    d1_st.attrs = d1.attrs
    # Each capture is normalized by its own internal-leakage peak: overall gain varies
    # by a few tenths of a dB from capture to capture on this hardware.
    peak_st, leak_st, noise_st = record_powers(d1_st, chirp, fit.t1, fit.t2, nstart)
    peak_p, leak_p, noise_p = record_powers(dp, chirp, fit.t1, fit.t2, nstart)
    _, leak_1, noise_1 = record_powers(d1, chirp, fit.t1, fit.t2, nstart)

    prof1 = np.abs(stacked_profile(d1, chirp))
    profp = np.abs(stacked_profile(dp, chirp))
    c = int(round(fit.t2))
    win = slice(max(c - CORR_HALF_WIDTH, 0), c + CORR_HALF_WIDTH + 1)
    corr = float(np.corrcoef(prof1[win], profp[win])[0, 1])

    rel = lambda p, leak: db10(p / leak)  # noqa: E731
    ms = [
        Measurement("peak_power_diff_db", abs(rel(peak_st, leak_st) - rel(peak_p, leak_p)),
                    note="loopback peak rel. leakage peak"),
        Measurement("noise_power_diff_db", abs(rel(noise_st, leak_st) - rel(noise_p, leak_p)),
                    note="noise rel. leakage peak"),
        Measurement("p100_noise_vs_p1_single_db", rel(noise_p, leak_p) - rel(noise_1, leak_1),
                    note=f"rel. leakage peak; theory {-10 * np.log10(P):.1f} dB for {P} presums"),
        Measurement("profile_magnitude_correlation", corr),
        Measurement("capture_gain_diff_db", db10(leak_p) - db10(leak_1),
                    note="leakage peak power, p100 capture minus p1 capture"),
    ]
    n = np.arange(len(prof1))
    reduced = xr.Dataset(
        {"offline_profile_db": ("sample", db10(prof1 ** 2), {"units": "dBFS"}),
         "on_device_profile_db": ("sample", db10(profp ** 2), {"units": "dBFS"})},
        coords={"sample": ("sample", n, {"units": "samples"})},
        attrs={"presums": P, "t2": fit.t2, "peak_power_offline_dbfs": db10(peak_st),
               "peak_power_on_device_dbfs": db10(peak_p), "noise_offline_dbfs": db10(noise_st),
               "noise_on_device_dbfs": db10(noise_p), "noise_single_pulse_dbfs": db10(noise_1)},
    )
    return reduced, ms
