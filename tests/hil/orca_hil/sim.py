"""Synthetic loopback captures in the same on-disk format as the radar pipeline.

``simulate_capture`` writes ``<YYYYMMDD_HHMMSS>_rx_samps.bin`` (interleaved fc32),
``_config.yaml`` and ``_uhd_stdout.log`` so the output can go through
``processing_dask.save_radar_data_to_zarr`` exactly like hardware data.

Behavior mirrored from sdr/main.cpp:
- Nothing is written for error pulses; the radar keeps scheduling until
  ``num_pulses`` error-free pulses were received, so attempted = num_pulses + errors.
- Presumming averages ``num_presums`` consecutive error-free pulses per record.
- After errors are noticed, the scheduler adds ``new_errors * 2 * PRI`` to
  time_offset before scheduling chirp N and logs
  ``[TX] (Chirp N) time_offset increased by D``. Pulse s is scheduled at
  ``time_offset + PRI * s``.
- Phase dithering is undone on RX, so signal terms are unaffected; coherent
  artifacts that are not dithered at TX (LO leakage, spurs) get the inverse
  random phase and decorrelate across pulses.
"""
from datetime import datetime
from pathlib import Path

import numpy as np

from . import yamlio
from .configs import deep_merge

from generate_chirp import generate_chirp  # noqa: E402 (path set up by orca_hil)


def fractional_delay(x, delay, length, nfft=None):
    """Place ``x`` starting at fractional sample ``delay`` in a record of ``length``.

    Uses an FFT phase ramp (band-limited interpolation).
    """
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(length + len(x) + 64)))
    k = np.fft.fftfreq(nfft) * nfft
    X = np.fft.fft(x, nfft)
    return np.fft.ifft(X * np.exp(-2j * np.pi * k * delay / nfft))[:length]


def _time_offset_events(errors, total, lag):
    """Return {chirp_index: D_multiplier} for time_offset increases (D = mult * 2 * PRI)."""
    events = {}
    for e in sorted(errors):
        n = e + lag
        while n in events:  # one scheduler iteration per chirp; later errors land on later chirps
            n += 1
        if n < total:
            events[n] = 1
    return events


def scheduled_times(n_pulses, errors, pri, lag=2):
    """Scheduled time (relative to pulse 0) of every attempted pulse, and the event dict."""
    total = n_pulses + len(errors)
    events = _time_offset_events(errors, total, lag)
    extra = np.zeros(total)
    for n, mult in events.items():
        extra[n:] += mult * 2 * pri
    return pri * np.arange(total) + extra, events


def simulate_capture(config, *, out_dir, n_pulses, leakage=None, loopback=None,
                     noise_power=0.0, spurs=(), lo_leakage=0j, phase_noise_std=0.0,
                     drift_rad_s=0.0, dithering=None, errors=(), presums=None,
                     desync_after_error=False, error_notice_lag=2, impulses=None,
                     seed=0, timestamp=None, chunk_pulses=500):
    """Write a synthetic capture and return its prefix (str).

    config: radar config dict (e.g. from configs.merged_config)
    leakage, loopback: {"delay": samples (float), "amp": complex} or None
    noise_power: E|n|^2 per raw complex sample
    spurs: [{"freq": Hz, "amp": complex, "chirp_copy": bool}]; tones are coherent in
        absolute time. With chirp_copy, the tone instead modulates a copy of the
        leakage chirp (coherent with the signal, unaffected by dithering).
    lo_leakage: complex DC term (not dithered at TX)
    phase_noise_std: random-walk phase step per pulse [rad], applied to signal terms
    drift_rad_s: linear phase drift of signal terms vs scheduled time
    dithering: override CHIRP.phase_dithering (default: config value)
    errors: scheduled pulse indices that fail (ERROR_CODE_LATE_COMMAND)
    presums: override CHIRP.num_presums (default: config value)
    desync_after_error: after the first error, signal terms get a random phase per
        pulse (emulates an RX dithering sequence that slipped out of step)
    impulses: {"rate_hz": r, "amp": a} impulsive noise; each raw sample is hit with
        probability r / fs by a spike of magnitude a and random phase
    """
    rng = np.random.default_rng(seed)
    cfg = deep_merge(config, {"CHIRP": {"num_pulses": int(n_pulses)}})
    if presums is not None:
        cfg["CHIRP"]["num_presums"] = int(presums)
    if dithering is not None:
        cfg["CHIRP"]["phase_dithering"] = bool(dithering)
    P = int(cfg["CHIRP"].get("num_presums", 1))
    dither = bool(cfg["CHIRP"].get("phase_dithering", False))
    if n_pulses % P:
        raise ValueError("n_pulses must be a multiple of num_presums")

    fs = float(cfg["GENERATE"]["sample_rate"])
    pri = float(cfg["CHIRP"]["pulse_rep_int"])
    L = int(round(float(cfg["CHIRP"]["rx_duration"]) * fs))
    _, chirp = generate_chirp(cfg)
    n_idx = np.arange(L)

    errors = sorted(set(int(e) for e in errors))
    total = n_pulses + len(errors)
    if errors and errors[-1] >= total:
        raise ValueError("error indices must be < n_pulses + len(errors)")
    t_sched, events = scheduled_times(n_pulses, errors, pri, error_notice_lag)
    err_set = set(errors)
    good = np.array([s for s in range(total) if s not in err_set])

    # Per-attempted-pulse random sequences (drawn for all pulses so that the
    # signal does not depend on which pulses error)
    phase = np.cumsum(rng.normal(0, phase_noise_std, total)) if phase_noise_std else np.zeros(total)
    phase = phase + drift_rad_s * t_sched
    if desync_after_error and errors:
        bad = np.arange(total) > errors[0]
        phase[bad] += rng.uniform(0, 2 * np.pi, bad.sum())
    dither_phase = rng.uniform(0, 2 * np.pi, total) if dither else np.zeros(total)

    templates = []
    for term in (leakage, loopback):
        if term:
            templates.append((complex(term["amp"]), fractional_delay(chirp, term["delay"], L)))
    copy_template = fractional_delay(chirp, leakage["delay"], L) if leakage else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    prefix = out_dir / stamp

    chunk = max(P, (chunk_pulses // P) * P)
    with open(f"{prefix}_rx_samps.bin", "wb") as fout:
        for start in range(0, n_pulses, chunk):
            s = good[start:start + chunk]
            m = len(s)
            sig = np.zeros((m, L), complex)
            for amp, tmpl in templates:
                sig += amp * tmpl[None, :]
            sig *= np.exp(1j * phase[s])[:, None]

            art = np.full((m, L), complex(lo_leakage))
            for sp in spurs:
                tone_t0 = np.exp(2j * np.pi * sp["freq"] * t_sched[s])[:, None]
                tone_n = np.exp(2j * np.pi * sp["freq"] * n_idx / fs)[None, :]
                if sp.get("chirp_copy"):
                    sig += complex(sp["amp"]) * copy_template[None, :] * tone_n * tone_t0
                else:
                    art += complex(sp["amp"]) * tone_n * tone_t0
            if dither:
                art *= np.exp(-1j * dither_phase[s])[:, None]

            x = sig + art
            if noise_power:
                x += np.sqrt(noise_power / 2) * (rng.standard_normal((m, L))
                                                 + 1j * rng.standard_normal((m, L)))
            if impulses:
                hit = rng.random((m, L)) < impulses["rate_hz"] / fs
                x[hit] += impulses["amp"] * np.exp(2j * np.pi * rng.random(hit.sum()))
            rec = x.reshape(m // P, P, L).mean(axis=1)
            fout.write(rec.astype(np.complex64).tobytes())

    yamlio.dump(cfg, f"{prefix}_config.yaml")
    _write_log(f"{prefix}_uhd_stdout.log", cfg, errors, events, pri, total, n_pulses)
    return str(prefix)


def _write_log(path, cfg, errors, events, pri, total, n_pulses):
    t0 = 1.7e9
    lines = [(t0, f"Reading from config file: ../../{cfg['FILES'].get('output_dir', '')}/config.yaml"),
             (t0, "[VERSION] 0.0.1"),
             (t0, "Note: Nothing is written to the file for error pulses."),
             (t0 + 0.5, "[START] Beginning main loop")]
    t_start = t0 + 0.5 + float(cfg["CHIRP"].get("time_offset", 1))
    timed = []
    for e in errors:
        timed.append((e, 0, f"[ERROR] (Chirp {e}) Receiver error: ERROR_CODE_LATE_COMMAND"))
    for n, mult in events.items():
        timed.append((n, 1, f"[TX] (Chirp {n}) time_offset increased by {mult * 2 * pri:g}"))
    for idx, _, text in sorted(timed):
        lines.append((t_start + idx * pri, text))
    t_end = t_start + total * pri
    save_loc = f"../../{cfg['FILES'].get('output_dir', '')}/{cfg['FILES']['save_loc']}"
    for text in ("[RX] Closing output file.", f"[CLOSE FILE] {save_loc}",
                 f"[RX] Error count: {len(errors)}", f"[RX] Total pulses written: {n_pulses}",
                 f"[RX] Total pulses attempted: {total}",
                 "[RX] Done. Calling join_all() on transmit thread group.",
                 "[RX] transmit_thread.join_all() complete."):
        lines.append((t_end, text))
    with open(path, "w") as f:
        for t, text in lines:
            f.write(f"[{t:0.3f}] \t{text}\n")
