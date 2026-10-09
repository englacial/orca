"""T12 frequency response sweep (not in the paper)."""
import numpy as np
import xarray as xr

from ..metrics import Measurement
from .common import db10, noise_window_start, open_capture, reference_chirp
from .profile import locate_peaks, single_pulse_stats


def case_name(freq_hz):
    return f"{freq_hz / 1e6:g}MHz"


def point_metrics(capture, fixture):
    ds = open_capture(capture)
    chirp = reference_chirp(ds.attrs["config"])
    _, _, fit, _ = locate_peaks(ds, fixture, chirp)
    _, noise = single_pulse_stats(ds, chirp, fit.t2, noise_window_start(fit.t2, len(chirp)))
    return {"loopback": db10(abs(fit.a2) ** 2), "leakage": db10(abs(fit.a1) ** 2),
            "noise": db10(np.mean(np.abs(noise) ** 2)) if noise.size else np.nan}


def smooth_fit(freq_hz, power_db):
    """2nd-order polynomial in sqrt(f) (cable loss ~ sqrt(f)); returns (fitted, ripple_db)."""
    x = np.sqrt(np.asarray(freq_hz) / 1e6)
    fitted = np.polyval(np.polyfit(x, power_db, 2), x)
    return fitted, float(np.max(np.abs(power_db - fitted)))


def analyze_freq_response(captures, fixture, params=None):
    """captures: {freq_hz: capture}."""
    freqs = np.array(sorted(captures))
    pts = [point_metrics(captures[f], fixture) for f in freqs]
    loop = np.array([p["loopback"] for p in pts])
    leak = np.array([p["leakage"] for p in pts])
    noise = np.array([p["noise"] for p in pts])
    fitted, ripple = smooth_fit(freqs, loop)

    ms = []
    for f, lp, lk, nf in zip(freqs, loop, leak, noise):
        c = case_name(f)
        ms += [Measurement("loopback_power_dbfs", lp, case=c),
               Measurement("loopback_to_leakage_db", lp - lk, case=c),
               Measurement("noise_floor_dbfs", nf, case=c)]
    ms.append(Measurement("ripple_db", ripple, note="max |loopback - smooth fit in sqrt(f)|"))

    reduced = xr.Dataset(
        {"loopback_power_dbfs": ("freq", loop, {"units": "dBFS"}),
         "leakage_power_dbfs": ("freq", leak, {"units": "dBFS"}),
         "noise_floor_dbfs": ("freq", noise, {"units": "dBFS"}),
         "loopback_fit_dbfs": ("freq", fitted, {"units": "dBFS"})},
        coords={"freq": ("freq", freqs, {"units": "Hz"})},
    )
    return reduced, ms
