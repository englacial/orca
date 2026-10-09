"""T11 presum equivalence: on-device presumming vs offline stacking (Sec. III-B, VII)."""
import numpy as np
import xarray as xr

import processing_dask as pdk

from ..metrics import Measurement
from .common import db10, noise_window_start, open_capture, reference_chirp, stacked_profile
from .profile import locate_peaks, single_pulse_stats

CORR_WINDOW = (-5, 100)   # samples around the loopback peak; excludes the internal-leakage peak


def record_powers(ds, chirp, t2, nstart):
    """Mean per-record loopback peak power (noise power removed) and noise power."""
    peaks, noise = single_pulse_stats(ds, chirp, t2, nstart)
    n = float(np.mean(np.abs(noise) ** 2))
    return float(np.mean(np.abs(peaks) ** 2)) - n, n


def analyze_presum(captures, fixture, params=None):
    """captures: {"p1": presums 1, "p100": presums P}; p1 is stacked offline by P.

    The peak is compared as peak-to-noise ratio (gain varies by a few tenths of a dB
    between captures on this hardware, and the internal-leakage peak by up to ~2 dB);
    noise powers are compared in dBFS.
    """
    d1, dp = open_capture(captures["p1"]), open_capture(captures["p100"])
    P = int(dp.attrs["config"]["CHIRP"]["num_presums"])
    chirp = reference_chirp(dp.attrs["config"])
    _, _, fit, _ = locate_peaks(dp, fixture, chirp)
    nstart = noise_window_start(fit.t2, len(chirp))

    d1_st = pdk.stack(d1[["radar_data"]], P)
    d1_st.attrs = d1.attrs
    peak_st, noise_st = record_powers(d1_st, chirp, fit.t2, nstart)
    peak_p, noise_p = record_powers(dp, chirp, fit.t2, nstart)
    peak_1, noise_1 = record_powers(d1, chirp, fit.t2, nstart)

    prof1 = np.abs(stacked_profile(d1, chirp))
    profp = np.abs(stacked_profile(dp, chirp))
    c = int(round(fit.t2))
    # The internal-leakage peak (one round trip before) varies by a few dB between
    # captures, so it is left out of the shape comparison.
    win = slice(max(c + CORR_WINDOW[0], 0), c + CORR_WINDOW[1] + 1)
    corr = float(np.corrcoef(prof1[win], profp[win])[0, 1])

    ms = [
        Measurement("peak_power_diff_db", abs(db10(peak_st / noise_st) - db10(peak_p / noise_p)),
                    note="loopback peak-to-noise ratio, offline stack vs on-device"),
        Measurement("noise_power_diff_db", abs(db10(noise_st) - db10(noise_p))),
        Measurement("p100_noise_vs_p1_single_db", db10(noise_p) - db10(noise_1),
                    note=f"theory {-10 * np.log10(P):.1f} dB for {P} presums"),
        Measurement("profile_magnitude_correlation", corr),
        Measurement("capture_gain_diff_db", db10(peak_p) - db10(peak_1),
                    note="loopback peak power, p100 capture minus p1 capture"),
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
