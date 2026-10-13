"""T4 spectral artifacts: LO leakage, PLL spurs, chirp occupancy (Fig. 5)."""
import numpy as np
import scipy.signal
import xarray as xr

from ..metrics import Measurement
from .common import db10, noise_window_start, open_capture, reference_chirp, sample_rate
from .noise import find_lines, match_lines
from .profile import locate_peaks
from ..sim import fractional_delay

NFFT = 4096
STFT_WIN = 64
STFT_OVERLAP = 48
SPUR_THRESHOLD_DB = 10.0
SPUR_JUDGE_MARGIN_DB = 6.0  # spurs within this of the threshold come and go between runs; not judged
PRE_CHIRP_MARGIN = 8        # samples kept clear before the first received chirp
FLOOR_MARGIN_DB = 6.0       # a DC estimate within this of its noise floor is a bound only
OCC_BW_DB = 10.0


def stacked_raw_pulse(ds):
    """Coherent mean of all records, before pulse compression."""
    return ds["radar_data"].mean("pulse_idx").values


def line_psd(x, fs, nfft=NFFT, window="blackmanharris"):
    """Windowed periodogram scaled so a tone of amplitude A peaks at |A|^2.

    The default Blackman-Harris window (-92 dB sidelobes) keeps a strong coherent
    tone, such as LO leakage, from producing false lines.
    """
    w = scipy.signal.get_window(window, len(x), fftbins=False)
    X = np.fft.fftshift(np.fft.fft(x * w, nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    return f, np.abs(X) ** 2 / np.sum(w) ** 2


def occupied_band(f, psd, rel_db=OCC_BW_DB):
    """Outermost frequencies where psd is within rel_db of its maximum: (lo, hi)."""
    idx = np.nonzero(db10(psd / psd.max()) >= -rel_db)[0]
    return float(f[idx[0]]), float(f[idx[-1]])


def case_spectra(capture, fixture):
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    fs = sample_rate(ds)
    chirp = reference_chirp(cfg)
    lc = len(chirp)
    _, _, fit, _ = locate_peaks(ds, fixture, chirp)
    x = stacked_raw_pulse(ds)
    t_start = noise_window_start(fit.t2, lc)
    # Both received chirps (leakage and loopback) in full
    chirp_seg = x[int(np.floor(fit.t1)):int(np.ceil(fit.t2)) + lc]
    tail = x[t_start:]
    p_chirp = float(np.mean(np.abs(chirp_seg) ** 2))
    # LO leakage: DC of the stacked pulse before the first received chirp. (After the
    # chirp, a signal-derived DC tail appears when the chirp contains DC; it is
    # coherent with the dithered TX signal and reported separately.)
    pre = x[:max(int(np.floor(fit.t1)) - PRE_CHIRP_MARGIN, 2)]
    dc_pre = complex(np.mean(pre))
    dc_floor = float(np.var(pre) / len(pre))      # power of the DC estimate's noise
    dc_tail = complex(np.mean(tail)) if len(tail) else np.nan

    f_tail, psd_tail = line_psd(tail, fs)
    resolution = 2 * fs / max(len(tail), 1)
    lines = [ln for ln in find_lines(f_tail, psd_tail, SPUR_THRESHOLD_DB) if abs(ln["freq"]) > resolution]

    # Ideal: the same two chirps (fitted delays and amplitudes), without hardware effects
    ideal = (fit.a1 * fractional_delay(chirp, fit.t1, len(x))
             + fit.a2 * fractional_delay(chirp, fit.t2, len(x)))
    ideal_seg = ideal[int(np.floor(fit.t1)):int(np.ceil(fit.t2)) + lc]
    f_c, psd_c = line_psd(chirp_seg, fs, window="boxcar")
    f_i, psd_i = line_psd(ideal_seg, fs, window="boxcar")
    band = occupied_band(f_c, psd_c)
    band_ideal = occupied_band(f_i, psd_i)

    f_s, t_s, Z = scipy.signal.stft(x, fs=fs, window="hann", nperseg=STFT_WIN, noverlap=STFT_OVERLAP,
                                    return_onesided=False, boundary=None, padded=False)
    return {
        "config": cfg, "lo_leakage_rel_chirp_db": db10(abs(dc_pre) ** 2 / p_chirp),
        "lo_leakage_floor_rel_chirp_db": db10(dc_floor / p_chirp),
        "post_chirp_dc_rel_chirp_db": db10(abs(dc_tail) ** 2 / p_chirp), "spurs": lines,
        "band": band, "band_ideal": band_ideal, "f_tail": f_tail,
        "tail_psd_db": db10(psd_tail / p_chirp),
        "stft_f": np.fft.fftshift(f_s), "stft_t": t_s,
        "stft_db": db10(np.abs(np.fft.fftshift(Z, axes=0)) ** 2 / p_chirp),
    }


def analyze_spectral(captures, fixture, params=None):
    """captures: {"a": no offset/no dither, "b": no offset/dither, "c": offset/dither}.

    params:
      baseline_spurs: [{"freq", "level_db"}] from the baseline (case a), or None
    """
    params = params or {}
    spec = {case: case_spectra(cap, fixture) for case, cap in captures.items()}
    ms = []
    if "a" in spec:
        ms.append(Measurement("lo_leakage_rel_chirp_db", spec["a"]["lo_leakage_rel_chirp_db"], case="a"))
        ref = params.get("baseline_spurs")
        found = spec["a"]["spurs"]
        if ref is None:
            ms += [Measurement("spur_freqs_match", False, case="a", inconclusive=True,
                               note="no baseline spur list"),
                   Measurement("new_spurs_above_baseline", len(found), case="a", inconclusive=True,
                               note="no baseline spur list")]
        else:
            judged = SPUR_THRESHOLD_DB + SPUR_JUDGE_MARGIN_DB
            missing = [r for r in ref if r["level_db"] >= judged and not match_lines([r], found)[1]]
            new, matched = match_lines(found, ref)
            new = [ln for ln in new if ln["level_db"] >= judged]
            raised = [ln for ln, r in matched if ln["level_db"] > r["level_db"] + 6]
            ms += [Measurement("spur_freqs_match", not missing, case="a",
                               note=f"{len(missing)} baseline spurs not found"),
                   Measurement("new_spurs_above_baseline", len(new) + len(raised), case="a",
                               note=f"{len(new)} new, {len(raised)} raised by more than 6 dB")]
    for case, sp in spec.items():
        ms.append(Measurement("post_chirp_dc_rel_chirp_db", sp["post_chirp_dc_rel_chirp_db"], case=case))
    if "a" in spec and "b" in spec:
        a, b = spec["a"], spec["b"]
        at_floor = b["lo_leakage_rel_chirp_db"] < b["lo_leakage_floor_rel_chirp_db"] + FLOOR_MARGIN_DB
        if at_floor:
            bound = a["lo_leakage_rel_chirp_db"] - b["lo_leakage_floor_rel_chirp_db"] - FLOOR_MARGIN_DB
            ms.append(Measurement("lo_leakage_suppression_by_dither_db", bound,
                                  inconclusive=bound < 20,
                                  note=f"lower bound: dithered case at the estimate's noise floor "
                                       f"({b['lo_leakage_floor_rel_chirp_db']:.1f} dB rel. chirp)"))
        else:
            ms.append(Measurement("lo_leakage_suppression_by_dither_db",
                                  a["lo_leakage_rel_chirp_db"] - b["lo_leakage_rel_chirp_db"]))
    if "c" in spec:
        lo, hi = spec["c"]["band"]
        ilo, ihi = spec["c"]["band_ideal"]
        ms += [Measurement("occupied_bw_ratio", (hi - lo) / (ihi - ilo), case="c",
                           note="ratio to the ideal received chirps' -10 dB bandwidth"),
               Measurement("center_offset_error_mhz", abs((hi + lo) / 2 - (ihi + ilo) / 2) / 1e6, case="c")]

    cases = list(spec)
    n_t = min(len(spec[c]["stft_t"]) for c in cases)
    n_f = min(len(spec[c]["f_tail"]) for c in cases)
    reduced = xr.Dataset(
        {
            "spectrogram_db": (("case", "stft_freq", "stft_time"),
                               np.stack([spec[c]["stft_db"][:, :n_t] for c in cases]), {"units": "dB rel. chirp"}),
            "tail_psd_db": (("case", "freq"), np.stack([spec[c]["tail_psd_db"][:n_f] for c in cases]),
                            {"units": "dB rel. chirp"}),
            "lo_leakage_rel_chirp_db": ("case", [spec[c]["lo_leakage_rel_chirp_db"] for c in cases], {"units": "dB"}),
        },
        coords={"case": cases,
                "stft_freq": ("stft_freq", spec[cases[0]]["stft_f"], {"units": "Hz"}),
                "stft_time": ("stft_time", spec[cases[0]]["stft_t"][:n_t], {"units": "s"}),
                "freq": ("freq", spec[cases[0]]["f_tail"][:n_f], {"units": "Hz"})},
        attrs={"spurs": {c: spec[c]["spurs"] for c in cases},
               "bands_hz": {c: list(spec[c]["band"]) for c in cases}},
    )
    return reduced, ms
