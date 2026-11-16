"""Shared analysis helpers: capture loading, pulse compression, peaks, interpolation."""
import numpy as np
import scipy.signal
import xarray as xr

from generate_chirp import generate_chirp
import processing_dask as pdk

from ..fixtures import chain_delay_s


def open_capture(capture) -> xr.Dataset:
    """Accept an xarray Dataset or a path to a zarr written by save_radar_data_to_zarr."""
    if isinstance(capture, xr.Dataset):
        return capture
    return xr.open_zarr(str(capture))


def capture_config(ds):
    return ds.attrs["config"]


def sample_rate(ds):
    return float(capture_config(ds)["GENERATE"]["sample_rate"])


def reference_chirp(config):
    """The chirp the radar transmitted, from the capture's own config."""
    _, chirp = generate_chirp(config)
    return chirp


def chirp_length_samples(config):
    return len(reference_chirp(config))


def expected_separation_samples(fixture, fs):
    """RF-chain delay between the internal-leakage and loopback peaks, in samples."""
    return chain_delay_s(fixture) * fs


def compress(ds, chirp=None) -> xr.DataArray:
    """Pulse-compress every record (lazy). Returns radar_data with dims (pulse_idx, travel_time)."""
    cfg = capture_config(ds)
    if chirp is None:
        chirp = reference_chirp(cfg)
    out = pdk.pulse_compress(ds[["radar_data"]], chirp, sample_rate(ds))
    return out["radar_data"].transpose("pulse_idx", "travel_time")


def stacked_profile(ds, chirp=None) -> np.ndarray:
    """Coherent mean of all records, pulse-compressed (complex, length n_samples - len(chirp) + 1)."""
    n = ds.sizes["pulse_idx"]
    stacked = pdk.stack(ds[["radar_data"]], n)
    return compress(stacked, chirp).isel(pulse_idx=0).values


def upsample(x, factor):
    """Band-limited upsampling along the last axis."""
    return scipy.signal.resample(x, x.shape[-1] * factor, axis=-1)


def db10(x):
    with np.errstate(divide="ignore"):
        return 10 * np.log10(x)


def interp_complex(x, pos, half_taps=16, beta=8.0):
    """Kaiser-windowed sinc interpolation of x (..., N) at fractional index ``pos``."""
    i0 = int(np.floor(pos))
    idx = np.arange(i0 - half_taps + 1, i0 + half_taps + 1)
    idx = idx[(idx >= 0) & (idx < x.shape[-1])]
    d = pos - idx
    w = np.sinc(d) * np.i0(beta * np.sqrt(np.clip(1 - (d / half_taps) ** 2, 0, None))) / np.i0(beta)
    return np.tensordot(x[..., idx], w, axes=([-1], [0]))


def find_loopback_peaks(power, expected_sep=None, min_sep=4.0, up=1, sep_tol=3.0, expected_leak=None):
    """Locate the internal-leakage and loopback peaks in a power profile.

    ``power`` may be upsampled by ``up``. The global maximum is one of the two
    peaks. The other is the strongest local maximum within sep_tol samples of
    expected_sep away from it (either side), or, without expected_sep, at least
    min_sep away. The earlier peak is the internal leakage (shorter path).
    With ``expected_leak`` (samples), only the side that puts the leakage nearer
    to it is searched, so a leakage peak buried in noise is not replaced by a
    noise peak on the other side.
    Returns (leakage_pos, loopback_pos) in (non-upsampled) samples.
    """
    p = np.asarray(power)
    peaks, _ = scipy.signal.find_peaks(p)
    if len(peaks) < 2:
        raise ValueError("fewer than two peaks in profile")
    top = peaks[np.argmax(p[peaks])]
    dist = np.abs(peaks - top) / up
    if expected_sep is not None:
        ok = np.abs(dist - expected_sep) <= sep_tol
        if expected_leak is not None:
            # top is the loopback if it is nearer expected_leak + expected_sep than expected_leak
            top_is_loopback = abs(top / up - (expected_leak + expected_sep)) < abs(top / up - expected_leak)
            ok &= (peaks < top) if top_is_loopback else (peaks > top)
            if not ok.any():   # partner below the noise: put it at the expected separation
                other = top - expected_sep * up if top_is_loopback else top + expected_sep * up
                a, b = sorted((top, other))
                return a / up, b / up
    else:
        ok = dist >= min_sep
    if not ok.any():
        raise ValueError("no second peak found")
    cand = peaks[ok]
    other = cand[np.argmax(p[cand])]
    a, b = sorted((top, other))
    return a / up, b / up


def noise_window_start(loopback_pos, chirp_len, margin=20):
    """First compressed sample of the noise window: loopback + chirp length + margin."""
    return int(np.ceil(loopback_pos)) + chirp_len + margin
