"""T1 range-profile analysis (Fig. 1c, Fig. 2)."""
import numpy as np
import xarray as xr

from ..metrics import Measurement
from .common import (compress, db10, expected_separation_samples, find_loopback_peaks,
                     interp_complex, noise_window_start, open_capture, reference_chirp,
                     sample_rate, stacked_profile, upsample)
from .peaks import ChirpResponse, fit_two_peaks, leakage_removed, mainlobe_metrics

UP = 8


def locate_peaks(ds, fixture, chirp=None):
    """Stacked profile, response model and two-peak fit for a loopback capture."""
    cfg = ds.attrs["config"]
    chirp = reference_chirp(cfg) if chirp is None else chirp
    prof = stacked_profile(ds, chirp)
    exp_sep = expected_separation_samples(fixture, sample_rate(ds))
    p1, p2 = find_loopback_peaks(np.abs(upsample(prof, UP)) ** 2, exp_sep, up=UP)
    resp = ChirpResponse(chirp)
    fit = fit_two_peaks(prof, resp, p1, p2)
    return prof, resp, fit, exp_sep


def single_pulse_stats(ds, chirp, peak_pos, noise_start, half_taps=16):
    """Per-record complex peak values (interpolated at peak_pos) and noise-window samples."""
    comp = compress(ds, chirp)
    n_tt = comp.sizes["travel_time"]
    lo = max(int(np.floor(peak_pos)) - half_taps, 0)
    hi = min(int(np.floor(peak_pos)) + half_taps + 2, n_tt)
    sel = comp.isel(travel_time=np.r_[lo:hi, noise_start:n_tt]).values
    peaks = interp_complex(sel[:, :hi - lo], peak_pos - lo, half_taps=half_taps)
    return peaks, sel[:, hi - lo:]


def analyze_profile(capture, fixture, params=None):
    """Return (reduced Dataset, [Measurement]) for a T1 loopback capture."""
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    chirp = reference_chirp(cfg)
    lc = len(chirp)
    prof, resp, fit, exp_sep = locate_peaks(ds, fixture, chirp)

    nstart = noise_window_start(fit.t2, lc)
    peaks, noise = single_pulse_stats(ds, chirp, fit.t2, nstart)
    noise_power = float(np.mean(np.abs(noise) ** 2)) if noise.size else np.nan
    snr = db10(np.median(np.abs(peaks) ** 2) / noise_power)
    raw_max = float(np.abs(ds["radar_data"]).max().compute())

    n = np.arange(len(prof))
    loop = mainlobe_metrics(leakage_removed(prof, resp, fit), fit.t2, lc, UP)
    ideal = mainlobe_metrics(resp.at(n - fit.t2), fit.t2, lc, UP)

    ms = [
        Measurement("leakage_peak_index", fit.t1),
        Measurement("peak_separation_samples", fit.t2 - fit.t1, params={"expected": exp_sep}),
        Measurement("loopback_peak_power_dbfs", db10(abs(fit.a2) ** 2)),
        Measurement("loopback_to_leakage_ratio_db", db10(abs(fit.a2) ** 2 / abs(fit.a1) ** 2)),
        Measurement("single_pulse_snr_db", snr, inconclusive=not noise.size,
                    note=None if noise.size else "no samples in noise window"),
        Measurement("mainlobe_3db_width_samples", loop.width_3db, params={"ideal": ideal.width_3db}),
        Measurement("model_residual_ratio_db", fit.residual_ratio_db),
        Measurement("noise_floor_dbfs", db10(noise_power)),
        Measurement("raw_max_abs_fs", raw_max),
    ]

    up_lo, up_hi = max(int(fit.t1) - 30, 0), min(int(fit.t2) + 40, len(prof))
    seg_up = upsample(prof[up_lo:up_hi], UP)
    model_full = np.zeros(len(prof), complex)
    model_full[fit.region] = fit.model
    reduced = xr.Dataset(
        {
            "profile_power_db": ("sample", db10(np.abs(prof) ** 2), {"units": "dBFS"}),
            "model_power_db": ("sample", db10(np.abs(model_full) ** 2), {"units": "dBFS"}),
            "peak_region_power_db": ("sample_up", db10(np.abs(seg_up) ** 2), {"units": "dBFS"}),
        },
        coords={
            "sample": ("sample", n, {"units": "samples"}),
            "sample_up": ("sample_up", up_lo + np.arange(len(seg_up)) / UP, {"units": "samples"}),
        },
        attrs={"t1": fit.t1, "t2": fit.t2, "expected_separation": exp_sep,
               "noise_window_start": nstart, "n_records": int(ds.sizes["pulse_idx"])},
    )
    return reduced, ms
