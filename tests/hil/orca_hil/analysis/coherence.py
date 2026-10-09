"""T6 short-term coherence (Fig. 3) and T7 long-term phase stability (Fig. 4).

Record times come from the pulse schedule reconstructed from the stdout log
(timing.expected_slow_time), not the stored slow_time (Section 8.3). Errors
therefore only shift records by a few PRIs, which is negligible next to the
integration times and drift rates measured here, so captures with errors are
still analyzed; the error count is recorded.
"""
import numpy as np
import xarray as xr

import processing_dask as pdk

from ..metrics import Measurement
from .common import noise_window_start, open_capture, reference_chirp
from .profile import locate_peaks, single_pulse_stats
from .timing import expected_slow_time

INTEGRATION_TIMES_S = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)
JUDGED_TIMES_S = (1.0, 10.0, 100.0)
SPREAD_TIME_S = 100.0
MIN_FILL = 0.9      # a window counts if it holds at least this fraction of its nominal records
STEP_SIGMA = 5.0


def peak_series(capture, fixture):
    """Complex loopback peak per record, record noise power at the peak and record times."""
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    chirp = reference_chirp(cfg)
    _, _, fit, _ = locate_peaks(ds, fixture, chirp)
    peaks, noise = single_pulse_stats(ds, chirp, fit.t2, noise_window_start(fit.t2, len(chirp)))
    log = ds.attrs["stdout_log"]
    _, errors, attempted = pdk.process_stdout_log(log)
    P = int(cfg["CHIRP"].get("num_presums", 1))
    pri = float(cfg["CHIRP"]["pulse_rep_int"])
    return {"peaks": peaks, "noise_power": float(np.mean(np.abs(noise) ** 2)),
            "t": expected_slow_time(cfg, log, len(peaks)), "record_s": pri * P, "pri": pri,
            "errors": len(errors), "attempted": attempted}


def coherence_ratios(peaks, t, noise_power, T, record_s):
    """Coherent-to-incoherent power ratio of each window of length T (noise bias removed).

    ratio = (|mean x|^2 - s2 / n) / (mean |x|^2 - s2): 1 for a perfectly coherent
    signal, independent of slow gain changes between windows.
    """
    w = np.floor((t - t[0]) / T).astype(np.int64)
    nominal = max(int(round(T / record_s)), 1)
    out = []
    for k in np.unique(w):
        x = peaks[w == k]
        if len(x) < MIN_FILL * nominal:
            continue
        coh = abs(x.mean()) ** 2 - noise_power / len(x)
        inc = np.mean(np.abs(x) ** 2) - noise_power
        out.append(coh / inc)
    return np.array(out)


def coherence_curve(series, times):
    rows = []
    for T in times:
        if T < series["record_s"] * (1 - 1e-6):
            continue
        r = coherence_ratios(series["peaks"], series["t"], series["noise_power"], T, series["record_s"])
        if r.size == 0:
            continue
        db = 10 * np.log10(np.clip(r, 1e-12, None))
        rows.append((T, -10 * np.log10(r.mean()), db.std() if r.size > 1 else np.nan, r.size))
    return rows


def analyze_coherence(burst, long, fixture, params=None):
    """burst: presums-1 capture (T <= one long-capture record); long: presummed capture.

    Power loss at T is relative to single pulses: loss within one long-capture record
    (from the burst) plus loss from one record to T (from the long capture).
    """
    b, lg = peak_series(burst, fixture), peak_series(long, fixture)
    rec = lg["record_s"]
    short = coherence_curve(b, [b["pri"]] + [T for T in INTEGRATION_TIMES_S if T <= rec * (1 + 1e-6)])
    rec_loss = coherence_curve(b, [rec])
    rec_loss = rec_loss[0][1] if rec_loss else 0.0
    span = lg["t"][-1] - lg["t"][0] + rec
    long_times = [T for T in INTEGRATION_TIMES_S if rec * (1 + 1e-6) < T < span] + [span]
    longc = [(T, loss + rec_loss, sd, n) for T, loss, sd, n in coherence_curve(lg, long_times)]
    rows = short + longc
    T = np.array([r[0] for r in rows])

    ms = [Measurement("errors_in_captures", b["errors"] + lg["errors"],
                      note=f"burst {b['errors']}/{b['attempted']}, long {lg['errors']}/{lg['attempted']}")]
    for Tj in JUDGED_TIMES_S:
        hit = [r for r in longc if abs(r[0] - Tj) < 1e-9]
        ms.append(Measurement("power_loss_db", hit[0][1] if hit else np.nan, case=f"{Tj:g}s",
                              inconclusive=not hit, note=None if hit else "capture shorter than T"))
    hit = [r for r in longc if abs(r[0] - SPREAD_TIME_S) < 1e-9 and r[3] > 1]
    ms.append(Measurement("spread_1sigma_db", hit[0][2] if hit else np.nan, case=f"{SPREAD_TIME_S:g}s",
                          inconclusive=not hit, note=f"{hit[0][3]} windows" if hit else "fewer than 2 windows"))

    reduced = xr.Dataset(
        {"power_loss_db": ("integration_time", np.array([r[1] for r in rows]), {"units": "dB"}),
         "spread_1sigma_db": ("integration_time", np.array([r[2] for r in rows]), {"units": "dB"}),
         "n_windows": ("integration_time", np.array([r[3] for r in rows], dtype=np.int64))},
        coords={"integration_time": ("integration_time", T, {"units": "s"})},
        attrs={"long_record_s": rec},
    )
    return reduced, ms


def detect_steps(resid, n_sigma=STEP_SIGMA):
    """Indices of first differences beyond n_sigma robust sigma (MAD x 1.4826)."""
    d = np.diff(resid)
    med = np.median(d)
    sigma = 1.4826 * np.median(np.abs(d - med))
    return np.flatnonzero(np.abs(d - med) > n_sigma * sigma), sigma


def analyze_phase_stability(long, burst, fixture, params=None):
    """long: presummed capture for the phase timeline; burst: presums-1 capture for
    single-pulse phase variance. Var(phi) is compared with 1/(2 SNR), SNR = |A|^2 / s2
    with s2 the complex noise variance at the peak."""
    lg, b = peak_series(long, fixture), peak_series(burst, fixture)
    phi = np.unwrap(np.angle(lg["peaks"]))
    t = lg["t"]
    coef = np.polyfit(t, phi, 1)
    resid = phi - np.polyval(coef, t)
    steps, step_sigma = detect_steps(resid)

    bphi = np.unwrap(np.angle(b["peaks"]))
    var_phi = np.var(np.diff(bphi)) / 2           # first differences remove slow wander
    a2 = np.mean(np.abs(b["peaks"]) ** 2) - b["noise_power"]
    snr = a2 / b["noise_power"]
    theory = 1 / (2 * snr)

    ms = [
        Measurement("errors_in_captures", lg["errors"] + b["errors"],
                    note=f"long {lg['errors']}/{lg['attempted']}, burst {b['errors']}/{b['attempted']}"),
        Measurement("phase_drift_mdeg_per_s", float(np.degrees(coef[0]) * 1e3),
                    note=f"over {t[-1] - t[0]:.0f} s"),
        Measurement("phase_var_to_theory_ratio", float(var_phi / theory),
                    note=f"Var(phi) {var_phi:.3g} rad^2, 1/(2 SNR) {theory:.3g}, SNR {10 * np.log10(snr):.1f} dB"),
        Measurement("phase_steps_count", int(len(steps)),
                    note=f"first differences > {STEP_SIGMA:g} x {np.degrees(step_sigma):.3g} deg"),
        Measurement("residual_std_deg", float(np.degrees(resid.std()))),
    ]
    reduced = xr.Dataset(
        {"phase_rad": ("record", phi.astype(np.float32), {"units": "rad"}),
         "residual_deg": ("record", np.degrees(resid).astype(np.float32), {"units": "deg"}),
         "step_record": ("step", steps.astype(np.int64))},
        coords={"slow_time": ("record", t, {"units": "s"})},
        attrs={"drift_rad_s": float(coef[0]), "phase0_rad": float(coef[1]), "var_phi_rad2": float(var_phi),
               "theory_var_rad2": float(theory), "snr_db": float(10 * np.log10(snr)),
               "step_sigma_deg": float(np.degrees(step_sigma))},
    )
    return reduced, ms
