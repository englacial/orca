"""T9 duty cycle vs error rate (Fig. 9)."""
import re
from collections import Counter

import numpy as np
import xarray as xr

import processing_dask as pdk

from ..metrics import Measurement

KNEE_ERROR_RATE = 0.02


def active_time(cfg):
    """Full-duplex active time, as in tests/error_code_late_command_sweep.py."""
    return max(float(cfg["CHIRP"]["tx_duration"]), float(cfg["CHIRP"]["rx_duration"]))


def duty_to_pri(cfg, duty):
    return active_time(cfg) / duty


def point_stats(log_text, cfg, killed=False):
    _, errors, attempted = pdk.process_stdout_log(log_text)
    codes = Counter(errors.values())
    m = re.search(r"\[RX\] Total pulses written: (\d+)", log_text)
    written = int(m.group(1)) if m else None
    requested = int(cfg["CHIRP"]["num_pulses"])
    return {"errors": len(errors), "attempted": attempted,
            "late_command": codes.get("ERROR_CODE_LATE_COMMAND", 0),
            "other_errors": sum(v for k, v in codes.items() if k != "ERROR_CODE_LATE_COMMAND"),
            "error_rate": len(errors) / attempted if attempted else np.nan,
            "written_ok": written == requested, "killed": bool(killed)}


def knee(duties, rates, limit=KNEE_ERROR_RATE):
    """Largest duty such that every point at or below it has an error rate below ``limit``."""
    best = np.nan
    for d, r in sorted(zip(duties, rates)):
        if not r < limit:
            break
        best = d
    return float(best)


def analyze_duty(points, params=None):
    """points: [{"duty": fraction, "log": stdout text, "config": cfg, "killed": bool}].

    params: report_at (duties with a dedicated error-rate metric, e.g. {0.5: "error_rate_at_50pct"}),
            full (bool: also record the knee)
    """
    params = params or {}
    rows = [{"duty": p["duty"], "pri": float(p["config"]["CHIRP"]["pulse_rep_int"]),
             **point_stats(p["log"], p["config"], p.get("killed", False))} for p in points]
    ms = []
    for r in rows:
        case = f"duty_{round(r['duty'] * 100):d}"
        ms.append(Measurement("error_rate", r["error_rate"], case=case,
                              note=f"{r['errors']} errors ({r['other_errors']} not late-command) "
                                   f"in {r['attempted']} pulses"))
        ms.append(Measurement("records_written_equals_requested", r["written_ok"], case=case))
    ms.append(Measurement("process_killed", any(r["killed"] for r in rows)))
    for duty, name in (params.get("report_at") or {}).items():
        near = min(rows, key=lambda r: abs(r["duty"] - duty))
        ms.append(Measurement(name, near["error_rate"], note=f"measured at duty {near['duty']:.3f}"))
    if params.get("full"):
        ms.append(Measurement("max_duty_with_error_rate_below_2pct",
                              knee([r["duty"] for r in rows], [r["error_rate"] for r in rows])))

    reduced = xr.Dataset(
        {k: ("duty", [r[k] for r in rows], {"units": u}) for k, u in (
            ("pri", "s"), ("error_rate", "fraction"), ("errors", "count"), ("attempted", "count"),
            ("other_errors", "count"))},
        coords={"duty": ("duty", [r["duty"] for r in rows], {"units": "fraction"})},
    )
    return reduced, ms
