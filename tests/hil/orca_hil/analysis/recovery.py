"""T10 error recovery and phase continuity (Sec. III-A, VIII-A)."""
import numpy as np
import scipy.stats
import xarray as xr

import processing_dask as pdk

from ..metrics import Measurement
from .common import db10, noise_window_start, open_capture, reference_chirp
from .profile import locate_peaks, single_pulse_stats
from .timing import time_offset_increases

WINDOW_RECORDS = 100
MIN_WINDOW_RECORDS = 10
FALSE_ALARM = 1e-3      # chance that healthy (Gaussian) gap jumps exceed the limit
MIN_JUMP_LIMIT_SIGMA = 3.0


def jump_limit_sigma(n_gaps):
    """Max-|jump| limit for n Gaussian gap jumps at FALSE_ALARM (at least 3 sigma)."""
    if n_gaps < 1:
        return MIN_JUMP_LIMIT_SIGMA
    return max(MIN_JUMP_LIMIT_SIGMA, float(scipy.stats.norm.isf(FALSE_ALARM / (2 * n_gaps))))


def wrap(phi):
    return (np.asarray(phi) + np.pi) % (2 * np.pi) - np.pi


def error_records(errors, presums):
    """Record index of the first record after each error gap (deduplicated, sorted).

    An error at scheduled pulse s falls after g = s - (errors before s) error-free
    pulses, so the gap lies at the start of record g // P, or inside it when
    g % P != 0 (that record then mixes pulses from both sides of the gap).
    """
    errs = sorted(errors)
    out = {}
    for i, s in enumerate(errs):
        g = s - i
        out[g // presums] = out.get(g // presums, False) or (g % presums != 0)
    return sorted(out.items())  # [(record, straddles), ...]


def gap_phase_jumps(phase, gaps):
    """Phase jumps across gaps vs jumps between normal neighbouring records.

    Returns (jumps_sigma, normal_dphi, gap_dphi): each gap's jump in units of the
    robust std of normal jumps at the same record spacing.
    """
    n = len(phase)
    near_gap = np.zeros(n, bool)
    pairs = []
    for k, straddles in gaps:
        a, b = (k - 1, k + 1) if straddles else (k - 1, k)
        if a >= 0 and b < n:
            pairs.append((a, b))
        near_gap[max(a, 0):min(b, n - 1) + 1] = True
    out, gap_all, normal_all = [], [], []
    for spacing in sorted({b - a for a, b in pairs}):
        idx = np.arange(n - spacing)
        clean = ~(near_gap[idx] | near_gap[idx + spacing])
        normal = wrap(phase[idx + spacing] - phase[idx])[clean]
        if normal.size < 10:
            continue
        med = np.median(normal)
        sigma = 1.4826 * np.median(np.abs(normal - med))
        gd = np.array([wrap(phase[b] - phase[a]) for a, b in pairs if b - a == spacing])
        out += list(np.abs(wrap(gd - med)) / sigma)
        gap_all += list(gd)
        normal_all += list(normal)
    return np.array(out), np.array(normal_all), np.array(gap_all)


def analyze_recovery(capture, fixture, params=None):
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    P = int(cfg["CHIRP"].get("num_presums", 1))
    log = ds.attrs["stdout_log"]
    _, errors, attempted = pdk.process_stdout_log(log)
    chirp = reference_chirp(cfg)
    _, _, fit, _ = locate_peaks(ds, fixture, chirp)
    peaks, _ = single_pulse_stats(ds, chirp, fit.t2, noise_window_start(fit.t2, len(chirp)))
    phase = np.unwrap(np.angle(peaks))
    power = np.abs(peaks) ** 2
    n = len(peaks)
    gaps = error_records(errors, P)

    ms = [
        Measurement("error_rate", len(errors) / attempted if attempted else np.nan,
                    note=f"{len(errors)} errors in {attempted} pulses"),
        Measurement("records_times_presums_equals_requested", n * P == int(cfg["CHIRP"]["num_pulses"])),
    ]
    if errors:
        ms.append(Measurement("error_file_type_is_not_included",
                              pdk.check_if_error_data_exists(ds) == "error_data_not_included"))
        ms.append(Measurement("time_offset_increases_logged", bool(time_offset_increases(log))))

    jumps, normal, gap_d = gap_phase_jumps(phase, gaps)
    no_gaps = jumps.size == 0
    limit = jump_limit_sigma(jumps.size)
    ms.append(Measurement("max_gap_phase_jump_sigma", float(jumps.max()) if jumps.size else np.nan,
                          params={"limit": limit}, inconclusive=no_gaps,
                          note=f"{jumps.size} gaps evaluated; limit {limit:.2f} sigma"))

    if gaps:
        # Reference gap: the first one with a full window of clean records before it
        # (runs usually start with an error burst), else the one with the most.
        starts = [k for k, _ in gaps]
        clean_before = [k - (starts[i - 1] + 1 if i else 0) for i, k in enumerate(starts)]
        ok = [i for i, c in enumerate(clean_before) if c >= WINDOW_RECORDS]
        i_ref = ok[0] if ok else int(np.argmax(clean_before))
        # Last gap with enough clean records after it (the capture may end mid-burst)
        clean_after = [(starts[i + 1] if i + 1 < len(starts) else n) - k - 1 for i, k in enumerate(starts)]
        late = [i for i, c in enumerate(clean_after) if c >= MIN_WINDOW_RECORDS and i >= i_ref]
        k_first = starts[i_ref]
        k_last = starts[late[-1]] if late else starts[-1]
        prev = starts[i_ref - 1] + 1 if i_ref else 0
        before = slice(max(k_first - WINDOW_RECORDS, prev), k_first)
        after_first = slice(k_first + 1, min(k_first + 1 + WINDOW_RECORDS, n))
        nxt = [k for k in starts if k > k_last]
        after_last = slice(k_last + 1, min(k_last + 1 + WINDOW_RECORDS, nxt[0] if nxt else n))
        nb = before.stop - before.start
        short = nb < MIN_WINDOW_RECORDS or after_last.stop - after_last.start < MIN_WINDOW_RECORDS
        drop = db10(power[before].mean() / power[after_first].mean()) if nb else np.nan
        change = (db10(abs(peaks[after_last].mean()) ** 2 / abs(peaks[before].mean()) ** 2)
                  if not short else np.nan)
        note = (f"gaps at records {k_first} (first with clean records before) and {k_last} "
                f"(last with clean records after); {nb} records before")
        ms += [Measurement("record_power_drop_after_first_error_db", drop,
                           inconclusive=nb < MIN_WINDOW_RECORDS, note=note),
               Measurement("stacked_power_change_after_errors_db", abs(change), inconclusive=short, note=note)]
    else:
        ms += [Measurement("record_power_drop_after_first_error_db", np.nan, inconclusive=True, note="no errors"),
               Measurement("stacked_power_change_after_errors_db", np.nan, inconclusive=True, note="no errors")]

    reduced = xr.Dataset(
        {"peak_phase_rad": ("record", phase.astype(np.float32), {"units": "rad"}),
         "peak_power_db": ("record", db10(power).astype(np.float32), {"units": "dBFS"}),
         "gap_record": ("gap", np.array([k for k, _ in gaps], dtype=np.int64)),
         "normal_jump_rad": ("normal_jump", normal.astype(np.float32), {"units": "rad"}),
         "gap_jump_rad": ("gap_jump", gap_d.astype(np.float32), {"units": "rad"})},
        coords={"record": np.arange(n)},
        attrs={"presums": P, "n_errors": len(errors)},
    )
    return reduced, ms
