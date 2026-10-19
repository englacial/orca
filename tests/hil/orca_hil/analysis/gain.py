"""T8 gain linearity and received-power calibration (Fig. 8)."""
import numpy as np
import xarray as xr

from ..metrics import Measurement
from .common import db10, noise_window_start, open_capture, reference_chirp
from .profile import locate_peaks, single_pulse_stats

MIN_SNR_DB = 10.0
COMPRESSION_DB = 1.0
LINEAR_MARGIN_DB = 10.0
RX_COMPRESSION_MIN_RAW = 0.5   # RX-sweep compression needs a large input; below this it is gain-table drift


def point_metrics(capture, fixture):
    """Loopback peak power, raw chirp RMS, raw max and noise floor for one sweep point."""
    ds = open_capture(capture)
    cfg = ds.attrs["config"]
    chirp = reference_chirp(cfg)
    lc = len(chirp)
    _, _, fit, _ = locate_peaks(ds, fixture, chirp)
    nstart = noise_window_start(fit.t2, lc)
    _, noise = single_pulse_stats(ds, chirp, fit.t2, nstart)
    c0 = int(np.floor(fit.t2))
    raw = ds["radar_data"]
    rms = float((np.abs(raw.isel(sample_idx=slice(c0, c0 + lc))) ** 2).mean().compute())
    return {"loopback_dbfs": float(db10(abs(fit.a2) ** 2)), "chirp_rms_dbfs": float(db10(rms)),
            "raw_max": float(np.abs(raw).max().compute()),
            "noise_dbfs": float(db10(np.mean(np.abs(noise) ** 2)))}


def sweep_fit(gains, levels, noise, raw_max=None, min_raw=0.0):
    """Slope over the linear region and the 1 dB compression point.

    The linear reference has unit slope through the low-gain points with SNR >= 10 dB.
    P1dB is the first gain where the level falls 1 dB below it, counting only points whose
    raw max is at least min_raw. The slope is fitted over points with SNR >= 10 dB and at
    least 10 dB below P1dB (all such points if there is no compression in the sweep).
    """
    g, lv, nz = map(np.asarray, (gains, levels, noise))
    rm = np.full(len(g), np.inf) if raw_max is None else np.asarray(raw_max)
    order = np.argsort(g)
    g, lv, nz, rm = g[order], lv[order], nz[order], rm[order]
    ok = lv - nz >= MIN_SNR_DB
    if ok.sum() < 2:
        return np.nan, None, ok
    ref_pts = np.nonzero(ok)[0][:3]
    offset = np.mean(lv[ref_pts] - g[ref_pts])
    dev = lv - (g + offset)
    p1db = None
    for i in np.nonzero(ok)[0]:
        if dev[i] < -COMPRESSION_DB and rm[i] >= min_raw:
            j = i - 1
            p1db = float(g[i]) if j < 0 else float(np.interp(-COMPRESSION_DB, [dev[i], dev[j]], [g[i], g[j]]))
            break
    lin = ok & ((g <= p1db - LINEAR_MARGIN_DB) if p1db is not None else True)
    slope = float(np.polyfit(g[lin], lv[lin], 1)[0]) if lin.sum() >= 2 else np.nan
    return slope, p1db, lin


def analyze_gain(points, fixture, params):
    """points: [{"sweep": "tx"|"rx", "gain": dB, "capture": capture}].

    params: base_tx_gain, base_rx_gain
    """
    rows = []
    for p in points:
        m = point_metrics(p["capture"], fixture)
        rows.append({"sweep": p["sweep"], "gain": float(p["gain"]), **m})

    ms = []
    fits = {}
    for sweep in ("tx", "rx"):
        r = [x for x in rows if x["sweep"] == sweep]
        if not r:
            continue
        gains = [x["gain"] for x in r]
        slope, p1db, lin = sweep_fit(gains, [x["loopback_dbfs"] for x in r], [x["noise_dbfs"] for x in r],
                                     [x["raw_max"] for x in r], RX_COMPRESSION_MIN_RAW if sweep == "rx" else 0.0)
        fits[sweep] = {"slope": slope, "p1db": p1db}
        ms.append(Measurement(f"{sweep}_sweep_slope_db_per_db", slope, note=f"{int(np.sum(lin))} points in fit"))
        if sweep == "tx":
            levels = np.array([x["loopback_dbfs"] for x in r])
            order = np.argsort(gains)
            at_base = float(np.interp(params["base_tx_gain"], np.array(gains)[order], levels[order]))
            ms.append(Measurement("dbfs_at_base_gain", at_base))
            if params.get("full"):
                noise = float(np.median([x["noise_dbfs"] for x in r]))
                if p1db is not None:
                    ms.append(Measurement("p1db_tx_gain", p1db))
                    top = float(np.interp(p1db, np.array(gains)[order], levels[order]))
                    ms.append(Measurement("dynamic_range_db", top - noise))
                else:
                    ms.append(Measurement("linear_to_tx_gain", max(gains), note="no compression within the sweep"))
                    ms.append(Measurement("linear_range_db", float(levels.max()) - noise,
                                          note="top of sweep minus noise floor"))

    dims = {"point": np.arange(len(rows))}
    reduced = xr.Dataset(
        {k: ("point", [x[k] for x in rows], {"units": u}) for k, u in (
            ("gain", "dB"), ("loopback_dbfs", "dBFS"), ("chirp_rms_dbfs", "dBFS"), ("raw_max", "FS"),
            ("noise_dbfs", "dBFS"))} | {"sweep": ("point", [x["sweep"] for x in rows])},
        coords=dims, attrs={"fits": fits},
    )
    return reduced, ms
