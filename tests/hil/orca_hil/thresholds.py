"""Threshold definitions (thresholds.yaml) and bound evaluation.

Each metric in thresholds.yaml maps to one check, or a list of checks:

    t01:
      noise_floor_dbfs: {kind: base, bound: {type: upper, tol: 2.0}, units: dBFS}
      peak_separation_samples:
        - {kind: abs, bound: {type: range_about, center: expected, tol: 1.5}, units: samples}
        - {kind: base, bound: {type: two_sided, tol: 0.1}, units: samples}

String values inside a bound (other than ``type``) are placeholders that the
test fills in at record time, e.g. ``center: expected``.
"""
from dataclasses import dataclass, field
import math

from . import HIL_DIR, yamlio

THRESHOLDS_PATH = HIL_DIR / "thresholds.yaml"
DEFAULT_K = 4.0

ABS_TYPES = {"range", "range_about", "max", "min", "abs_max", "equals"}
BASE_TYPES = {"two_sided", "lower", "upper", "ratio_upper", "factor_two_sided"}
KINDS = {"abs", "base", "info"}


@dataclass
class Check:
    kind: str
    units: str | None
    bound: dict = field(default_factory=dict)


@dataclass
class Evaluation:
    status: str                # pass / fail / inconclusive / info
    bound: dict                # bound with placeholders resolved
    lo: float | None = None    # effective limits, for the history CSV
    hi: float | None = None


def load_thresholds(path=THRESHOLDS_PATH) -> dict:
    """Return {test_id: {metric: [Check, ...]}, '_k': k}."""
    raw = yamlio.load(path) or {}
    k = float((raw.pop("defaults", None) or {}).get("k", DEFAULT_K))
    out = {"_k": k}
    for test_id, metrics in raw.items():
        out[test_id] = {}
        for name, spec in (metrics or {}).items():
            specs = spec if isinstance(spec, list) else [spec]
            checks = []
            for s in specs:
                check = Check(kind=s["kind"], units=s.get("units"), bound=dict(s.get("bound") or {}))
                validate_check(f"{test_id}.{name}", check)
                checks.append(check)
            out[test_id][name] = checks
    return out


def validate_check(label, check: Check):
    if check.kind not in KINDS:
        raise ValueError(f"{label}: unknown kind {check.kind!r}")
    if check.kind == "info":
        return
    btype = check.bound.get("type")
    allowed = ABS_TYPES if check.kind == "abs" else BASE_TYPES
    if btype not in allowed:
        raise ValueError(f"{label}: bound type {btype!r} is not valid for kind {check.kind!r}")


def resolve(bound: dict, params: dict) -> dict:
    """Substitute string placeholders in a bound from ``params``."""
    out = {}
    for key, val in bound.items():
        if key != "type" and isinstance(val, str):
            if val not in params:
                raise KeyError(f"bound placeholder {val!r} not supplied by the test")
            val = params[val]
        out[key] = val
    return out


def evaluate(value, check: Check, baseline: dict | None = None, params: dict | None = None,
             k: float = DEFAULT_K) -> Evaluation:
    if check.kind == "info":
        return Evaluation("info", {})
    b = resolve(check.bound, params or {})
    btype = b["type"]

    if btype == "equals":
        return Evaluation("pass" if value == b["value"] else "fail", b)

    if value is None or not math.isfinite(float(value)):
        return Evaluation("fail", b)
    v = float(value)

    if check.kind == "base":
        if not baseline or baseline.get("mean") is None:
            return Evaluation("inconclusive", b)
        m = float(baseline["mean"])
        s = float(baseline.get("std") or 0.0)
        if btype in ("two_sided", "lower", "upper"):
            margin = max(float(b["tol"]), k * s)
            lo = m - margin if btype in ("two_sided", "lower") else None
            hi = m + margin if btype in ("two_sided", "upper") else None
        elif btype == "ratio_upper":
            lo, hi = None, max(float(b["factor"]) * m, float(b.get("floor", -math.inf)))
        else:  # factor_two_sided
            f = float(b["factor"])
            lo, hi = sorted((m / f, m * f))
    else:
        if btype == "range":
            lo, hi = float(b["lo"]), float(b["hi"])
        elif btype == "range_about":
            c = float(b["center"])
            tol = float(b["tol"]) if "tol" in b else abs(c) * float(b["rel_tol"])
            lo, hi = c - tol, c + tol
        elif btype == "max":
            lo, hi = None, float(b["value"])
        elif btype == "min":
            lo, hi = float(b["value"]), None
        else:  # abs_max
            lo, hi = -float(b["value"]), float(b["value"])

    ok = (lo is None or v >= lo) and (hi is None or v <= hi)
    return Evaluation("pass" if ok else "fail", b, lo, hi)
