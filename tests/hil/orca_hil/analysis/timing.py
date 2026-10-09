"""Pulse schedule reconstruction from the radar stdout log."""
import re

import numpy as np

import processing_dask as pdk

_INCREASE_RE = re.compile(r"\[TX\] \(Chirp (\d+)\) time_offset increased by ([0-9.eE+-]+)")


def time_offset_increases(log):
    """[(chirp_index, delay_s), ...] from '[TX] (Chirp N) time_offset increased by D' lines."""
    return [(int(n), float(d)) for n, d in _INCREASE_RE.findall(log)]


def expected_slow_time(config, log, n_records):
    """Correct slow_time per record (HIL plan, Section 8.3).

    slow_time[k] is the scheduled time of the first error-free pulse in record k,
    relative to the scheduled time of pulse 0:
        s = scheduled index of error-free pulse g = k * num_presums
        slow_time[k] = PRI * s + sum(D_j for N_j <= s)
    """
    pri = float(config["CHIRP"]["pulse_rep_int"])
    P = int(config["CHIRP"].get("num_presums", 1))
    _, errors, _ = pdk.process_stdout_log(log)
    err = set(errors)
    incs = sorted(time_offset_increases(log))
    inc_n = np.array([n for n, _ in incs], dtype=np.int64)
    inc_cum = np.cumsum([d for _, d in incs]) if incs else np.array([])

    n_good_needed = n_records * P
    good = []
    s = 0
    while len(good) < n_good_needed:
        if s not in err:
            good.append(s)
        s += 1
    good = np.array(good, dtype=np.int64)
    sched = good[::P][:n_records]
    n_applied = np.searchsorted(inc_n, sched, side="right")
    offset = np.where(n_applied > 0, inc_cum[np.maximum(n_applied - 1, 0)] if incs else 0.0, 0.0)
    return pri * sched + offset
