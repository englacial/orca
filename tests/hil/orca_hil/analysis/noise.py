"""T3 RX-only noise and interference analysis (adapts Fig. 6)."""
import numpy as np
import scipy.signal
import xarray as xr

import processing_dask as pdk

from ..metrics import Measurement
from .common import compress, db10, interp_complex, open_capture, reference_chirp, sample_rate

LINE_THRESHOLD_DB = 10.0
LINE_MATCH_HZ = 50e3
IMPULSE_SIGMA = 6.0
MAX_PSD_RECORDS = 2000
MAX_MAD_RECORDS = 2000


def stack_sizes(n):
    """1, 3, 10, 30, ... up to n (n itself included)."""
    out, v = [], 1
    while v <= n:
        out.append(v)
        v = int(round(v * (3 if str(v)[0] == "1" else 10 / 3)))
    if out[-1] != n:
        out.append(n)
    return out


def raw_statistics(raw):
    """Noise power, Pearson kurtosis of I and Q, and the 6-sigma impulse fraction.

    raw: dask/xarray DataArray (sample_idx, pulse_idx), complex.
    """
    x = raw.data
    power = float(np.mean(np.abs(x) ** 2).compute())
    stats = {}
    for name, comp in (("i", x.real), ("q", x.imag)):
        c = comp - comp.mean()
        m2, m4 = np.mean(c ** 2), np.mean(c ** 4)
        m2, m4 = float(m2.compute()), float(m4.compute())
        stats[name] = m4 / m2 ** 2
    sub = raw.isel(pulse_idx=slice(0, MAX_MAD_RECORDS)).values
    sig = np.mean([1.4826 * np.median(np.abs(v - np.median(v))) for v in (sub.real, sub.imag)])
    impulse_frac = float((np.abs(x) > IMPULSE_SIGMA * sig).mean().compute())
    return power, stats["i"], stats["q"], impulse_frac, sig


def mean_psd(raw, fs):
    """Welch PSD averaged over records (two-sided, fftshifted). Returns (freqs, psd)."""
    sub = raw.isel(pulse_idx=slice(0, MAX_PSD_RECORDS)).values  # (samples, records)
    nper = min(1024, sub.shape[0])
    f, p = scipy.signal.welch(sub, fs=fs, nperseg=nper, nfft=2 * nper, axis=0,
                              return_onesided=False, detrend=False)
    psd = p.mean(axis=1)
    return np.fft.fftshift(f), np.fft.fftshift(psd)


def find_lines(freqs, psd, threshold_db=LINE_THRESHOLD_DB):
    """Narrowband lines more than threshold_db above the median PSD.

    Returns [{"freq": Hz, "level_db": dB above median}], one per contiguous run of bins.
    """
    rel = db10(psd / np.median(psd))
    above = rel > threshold_db
    lines = []
    i = 0
    while i < len(rel):
        if above[i]:
            j = i
            while j + 1 < len(rel) and above[j + 1]:
                j += 1
            k = i + int(np.argmax(rel[i:j + 1]))
            lines.append({"freq": float(freqs[k]), "level_db": float(rel[k])})
            i = j + 1
        else:
            i += 1
    return lines


def match_lines(lines, reference, tol_hz=LINE_MATCH_HZ):
    """Split lines into (new, matched) against a reference list; matched holds (line, ref) pairs."""
    new, matched = [], []
    for ln in lines:
        refs = [r for r in reference if abs(r["freq"] - ln["freq"]) <= tol_hz]
        if refs:
            matched.append((ln, min(refs, key=lambda r: abs(r["freq"] - ln["freq"]))))
        else:
            new.append(ln)
    return new, matched


def analyze_rx_only(capture, fixture, params=None):
    """T3 analysis.

    params:
      baseline_lines: [{"freq", "level_db"}] narrowband lines from the baseline, or None
      loopback_index: expected loopback position in compressed samples (from the T1
        baseline), or None
    """
    params = params or {}
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    fs = sample_rate(ds)
    raw = ds["radar_data"]

    power, kurt_i, kurt_q, impulse_frac, sigma = raw_statistics(raw)
    freqs, psd = mean_psd(raw, fs)
    lines = find_lines(freqs, psd)
    ref = params.get("baseline_lines")
    if ref is not None:
        new, matched = match_lines(lines, ref)
    else:
        new, matched = lines, []
    max_line = max((ln["level_db"] for ln in lines), default=0.0)

    # TX isolation and the stacking curve, on compressed data
    chirp = reference_chirp(cfg)
    comp = compress(ds, chirp)
    n_tt = comp.sizes["travel_time"]
    loop_idx = params.get("loopback_index")
    nstart = 0 if loop_idx is None else min(int(np.ceil(loop_idx)) + len(chirp) + 20, n_tt - 1)
    comp_np = comp.values  # (records, travel_time)
    cds = xr.Dataset({"radar_data": (("pulse_idx", "travel_time"), comp_np)},
                     coords={"pulse_idx": np.arange(comp_np.shape[0])})
    n_rec = comp_np.shape[0]
    ns = stack_sizes(n_rec)
    noise_curve = []
    for n in ns:
        st = pdk.stack(cds, n)["radar_data"].values
        noise_curve.append(float(np.mean(np.abs(st[:, nstart:]) ** 2)))
    stacked = comp_np.mean(axis=0)
    floor = np.mean(np.abs(stacked[nstart:]) ** 2)
    if loop_idx is not None:
        at_loop = np.abs(interp_complex(stacked, float(loop_idx))) ** 2
        isolation = db10(at_loop / floor)
    else:
        isolation = np.nan

    ms = [
        Measurement("noise_power_dbfs", db10(power)),
        Measurement("kurtosis_i", kurt_i),
        Measurement("kurtosis_q", kurt_q),
        Measurement("impulse_fraction_6sigma", impulse_frac),
        Measurement("new_narrowband_lines", len(new), inconclusive=ref is None,
                    note=None if ref is not None else "no baseline line list",),
        Measurement("narrowband_max_line_db", max_line),
        Measurement("tx_isolation_margin_db", isolation, inconclusive=loop_idx is None,
                    note=None if loop_idx is not None else "no loopback index from the T1 baseline"),
        Measurement("noise_at_nmax_dbfs", db10(noise_curve[-1])),
    ]

    hist_edges = np.linspace(0, 6 * sigma, 61)
    amp = np.abs(raw.isel(pulse_idx=slice(0, MAX_MAD_RECORDS)).values).ravel()
    hist, _ = np.histogram(amp, bins=hist_edges, density=True)
    reduced = xr.Dataset(
        {
            "psd_db": ("freq", db10(psd), {"units": "dBFS/Hz"}),
            "noise_vs_n_db": ("n_stack", db10(np.array(noise_curve)), {"units": "dBFS"}),
            "amplitude_pdf": ("amplitude", hist, {"units": "1/FS"}),
            "raw_example_i": ("fast_sample", raw.isel(pulse_idx=0).values.real),
            "raw_example_q": ("fast_sample", raw.isel(pulse_idx=0).values.imag),
        },
        coords={
            "freq": ("freq", freqs, {"units": "Hz"}),
            "n_stack": ("n_stack", ns),
            "amplitude": ("amplitude", 0.5 * (hist_edges[1:] + hist_edges[:-1]), {"units": "FS"}),
            "fast_sample": ("fast_sample", np.arange(raw.sizes["sample_idx"])),
        },
        attrs={"lines": lines, "new_lines": new, "sigma_component": sigma},
    )
    return reduced, ms
