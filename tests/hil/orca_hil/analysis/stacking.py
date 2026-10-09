"""T5 coherent stacking and phase-dithering noise floor (Fig. 7)."""
import numpy as np
import xarray as xr

import processing_dask as pdk

from ..metrics import Measurement
from .common import compress, db10, interp_complex, noise_window_start, open_capture, reference_chirp
from .noise import stack_sizes
from .profile import locate_peaks

PLATEAU_SLOPE_DB_PER_DECADE = -5.0


def stacking_curves(capture, fixture, max_noise_samples=None):
    """Noise power (mean and variance across records) and signal peak power vs N.

    N counts radar pulses, so on-device presums are included (N = n_stack * num_presums).
    Returns a dict of numpy arrays and scalars.
    """
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    P = int(cfg["CHIRP"].get("num_presums", 1))
    chirp = reference_chirp(cfg)
    _, _, fit, _ = locate_peaks(ds, fixture, chirp)
    nstart = noise_window_start(fit.t2, len(chirp))

    comp = compress(ds, chirp)
    n_tt = comp.sizes["travel_time"]
    if nstart >= n_tt:
        raise ValueError("noise window is empty; increase rx_duration")
    nstop = n_tt if max_noise_samples is None else min(n_tt, nstart + max_noise_samples)
    lo = max(int(np.floor(fit.t2)) - 16, 0)
    hi = int(np.floor(fit.t2)) + 18
    sel = comp.isel(travel_time=np.r_[lo:hi, nstart:nstop]).values
    cds = xr.Dataset({"radar_data": (("pulse_idx", "travel_time"), sel)},
                     coords={"pulse_idx": np.arange(sel.shape[0])})

    ns = stack_sizes(sel.shape[0])
    noise_mean, noise_var, sig = [], [], []
    for n in ns:
        st = pdk.stack(cds, n)["radar_data"].values
        npow = np.abs(st[:, hi - lo:]) ** 2
        noise_mean.append(npow.mean())
        noise_var.append(npow.mean(axis=1).var() if npow.shape[0] > 1 else np.nan)
        pk = interp_complex(st[:, :hi - lo], fit.t2 - lo)
        sig.append(np.mean(np.abs(pk) ** 2) - npow.mean())
    return {"n": np.array(ns) * P, "noise": np.array(noise_mean), "noise_var": np.array(noise_var),
            "signal": np.array(sig), "presums": P,
            "dithered": bool(cfg["CHIRP"].get("phase_dithering", False))}


def slope_db_per_decade(n, noise, n_min):
    m = n >= n_min
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log10(n[m]), db10(noise[m]), 1)[0])


def plateau_onset(n, noise):
    """First N where the local slope to the next N exceeds -5 dB/decade (max N if never)."""
    db = db10(noise)
    for i in range(len(n) - 1):
        s = (db[i + 1] - db[i]) / (np.log10(n[i + 1]) - np.log10(n[i]))
        if s > PLATEAU_SLOPE_DB_PER_DECADE:
            return float(n[i])
    return float(n[-1])


def analyze_stacking(captures, fixture, params=None):
    """captures: {case: capture}; cases follow the T5 table (default, lo_offset, dither,
    lo_offset_dither).

    params:
      fit_min_n: lower end of the slope fit (default 10)
      plateau_case: case judged for the coherent plateau (default "default")
      dither_pair: (undithered case, dithered case) for the dither gain
      max_noise_samples: cap on noise-window samples per record (memory)
    """
    params = params or {}
    n_min = params.get("fit_min_n", 10)
    plateau_case = params.get("plateau_case", "default")
    pair = tuple(params.get("dither_pair", ("default", "dither")))
    curves = {c: stacking_curves(cap, fixture, params.get("max_noise_samples"))
              for c, cap in captures.items()}

    ms = []
    for case, cv in curves.items():
        n, noise = cv["n"], cv["noise"]
        if cv["dithered"]:
            theory = db10(noise[0]) - 10 * np.log10(n[-1] / n[0])
            ms += [
                Measurement("slope_db_per_decade", slope_db_per_decade(n, noise, n_min), case=case),
                Measurement("deviation_from_theory_at_nmax_db", db10(noise[-1]) - theory, case=case),
                Measurement("signal_peak_loss_at_nmax_db",
                            db10(cv["signal"][0]) - db10(cv["signal"][-1]), case=case),
            ]
        if case == plateau_case:
            ms += [
                Measurement("plateau_level_dbfs", db10(noise[-1]), case=case),
                Measurement("plateau_onset_n", plateau_onset(n, noise), case=case),
            ]
    if all(c in curves for c in pair):
        ms.append(Measurement("dither_gain_at_nmax_db",
                              db10(curves[pair[0]]["noise"][-1]) - db10(curves[pair[1]]["noise"][-1])))

    cases = list(curves)
    n_ref = curves[cases[0]]["n"]
    if any(len(curves[c]["n"]) != len(n_ref) or np.any(curves[c]["n"] != n_ref) for c in cases):
        raise ValueError("all T5 cases must have the same pulse count and presums")
    reduced = xr.Dataset(
        {
            "noise_mean_db": (("case", "n_stack"), db10(np.stack([curves[c]["noise"] for c in cases])),
                              {"units": "dBFS"}),
            "noise_var": (("case", "n_stack"), np.stack([curves[c]["noise_var"] for c in cases]),
                          {"units": "FS^4"}),
            "signal_peak_db": (("case", "n_stack"), db10(np.stack([curves[c]["signal"] for c in cases])),
                               {"units": "dBFS"}),
        },
        coords={"case": cases, "n_stack": ("n_stack", n_ref, {"units": "pulses"})},
    )
    return reduced, ms
