"""Fixture files: lookup, ``extends`` inheritance and RF-chain validation.

A fixture describes one bench setup. ``hardware.chain`` lists the RF parts
between SDR TX and SDR RX, in order. A fixture can ``extends: <parent>`` another
one (by id from tests/hil/fixtures/, or by a path relative to the child). The
child's mappings are merged into the parent's; lists and scalars replace the
parent's, so a child that changes the RF chain restates all of it.

``status: commissioned`` marks a fixture with its own baselines; anything else is
``exploratory`` and its runs are logged as non-standard. These keys are never
inherited: status, judge, expect, skip_tests, description.
"""
import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from . import HIL_DIR, yamlio

FIXTURE_DIR = HIL_DIR / "fixtures"
C0 = 299792458.0
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
STATUSES = ("commissioned", "exploratory")
JUDGE_MODES = ("info", "expect")
NOT_INHERITED = ("status", "judge", "expect", "skip_tests", "description")

# Per type: required fields, optional fields. Every element may also have the COMMON fields.
COMMON_FIELDS = {"id", "type", "model", "serial", "source", "delay_ns", "max_input_dbm", "notes"}
ELEMENT_FIELDS = {
    "cable": ({"length_m", "velocity_factor"}, {"loss_db", "loss_db_per_100m"}),
    "adapter": (set(), {"loss_db"}),
    "attenuator": ({"loss_db"}, {"loss_tol_db", "power_rating_w"}),
    "limiter": ({"loss_db", "threshold_dbm"}, {"loss_tol_db", "leakage_dbm"}),
    "filter": ({"loss_db"}, {"loss_tol_db", "passband_mhz", "stopband_rejection_db"}),
    "amplifier": ({"role", "gain_db"}, {"gain_tol_db", "nf_db", "p1db_out_dbm", "powered", "supply",
                                        "watchdog"}),
}
SOURCES = ("datasheet", "measured")
AMP_ROLES = ("lna", "pa")
# Fields that may be a number or a frequency table {freq_mhz: [...], value: [...]}
FREQ_FIELDS = {"loss_db", "loss_db_per_100m", "loss_tol_db", "gain_db", "gain_tol_db", "nf_db",
               "p1db_out_dbm", "max_input_dbm", "threshold_dbm", "leakage_dbm", "delay_ns",
               "stopband_rejection_db"}


class FixtureError(ValueError):
    pass


def resolve_path(arg=None, directory=FIXTURE_DIR) -> Path:
    """--fixture value -> file. A path (contains '/' or ends in .yaml) or an id in ``directory``.

    Without an argument, the only fixture file in ``directory``.
    """
    if arg:
        if "/" in arg or arg.endswith((".yaml", ".yml")):
            p = Path(arg).expanduser().resolve()
        else:
            p = directory / f"{arg}.yaml"
        if not p.is_file():
            raise FixtureError(f"fixture file not found: {p}")
        return p
    files = sorted(directory.glob("*.yaml"))
    if len(files) != 1:
        raise FixtureError(f"--fixture is required ({len(files)} fixture files in {directory})")
    return files[0]


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load_raw(path, seen):
    path = Path(path).resolve()
    if path in seen:
        raise FixtureError(f"extends cycle at {path}")
    raw = yamlio.load(path) or {}
    parent_ref = raw.get("extends")
    if not parent_ref:
        return raw, [path]
    if "/" in parent_ref or parent_ref.endswith((".yaml", ".yml")):
        ppath = (path.parent / parent_ref).resolve()
    else:
        ppath = FIXTURE_DIR / f"{parent_ref}.yaml"
    if not ppath.is_file():
        raise FixtureError(f"{path.name}: parent fixture not found: {ppath}")
    parent, chain = _load_raw(ppath, seen | {path})
    parent = {k: v for k, v in parent.items() if k not in NOT_INHERITED}
    merged = _merge(parent, raw)
    merged["extends"] = parent.get("fixture_id", parent_ref)
    return merged, [path] + chain


def load(path):
    """Load, merge and validate a fixture. Returns the merged fixture dict.

    The merged fixture has ``status`` (default exploratory), ``judge`` (default info
    for exploratory fixtures) and, if it extends another, ``extends`` = the parent id.
    """
    fx, files = _load_raw(path, frozenset())
    fx.setdefault("status", "exploratory")
    if fx["status"] == "exploratory":
        fx.setdefault("judge", "info")
    validate(fx)
    fx["source_files"] = [str(p) for p in files]
    return fx


def parent(fixture):
    """The merged parent fixture, or None."""
    files = fixture.get("source_files") or []
    return load(files[1]) if fixture.get("extends") and len(files) > 1 else None


def digest(fixture) -> str:
    """sha256 of the merged fixture content (source file paths excluded)."""
    content = {k: v for k, v in fixture.items() if k != "source_files"}
    text = json.dumps(yamlio.to_builtin(content), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def is_standard(fixture) -> bool:
    return fixture.get("status") == "commissioned"


# ---------------------------------------------------------------- validation

def validate(fx):
    fid = fx.get("fixture_id")
    if not fid or not ID_RE.match(str(fid)):
        raise FixtureError(f"fixture_id {fid!r} must match {ID_RE.pattern}")
    if fx["status"] not in STATUSES:
        raise FixtureError(f"{fid}: status must be one of {STATUSES}")
    if fx["status"] == "commissioned" and ("judge" in fx or "expect" in fx):
        raise FixtureError(f"{fid}: judge/expect apply only to exploratory fixtures")
    if "judge" in fx and fx["judge"] not in JUDGE_MODES:
        raise FixtureError(f"{fid}: judge must be one of {JUDGE_MODES}")
    hw = fx.get("hardware") or {}
    if "chain" not in hw:
        raise FixtureError(f"{fid}: hardware.chain is required")
    ids = set()
    for i, el in enumerate(hw["chain"] or []):
        validate_element(f"{fid}: chain[{i}]", el)
        if el["id"] in ids:
            raise FixtureError(f"{fid}: duplicate chain id {el['id']!r}")
        ids.add(el["id"])


def validate_element(label, el):
    if not isinstance(el, dict) or "id" not in el or "type" not in el:
        raise FixtureError(f"{label}: every element needs id and type")
    label = f"{label} ({el['id']})"
    if el["type"] not in ELEMENT_FIELDS:
        raise FixtureError(f"{label}: unknown type {el['type']!r} (one of {sorted(ELEMENT_FIELDS)})")
    required, optional = ELEMENT_FIELDS[el["type"]]
    missing = required - el.keys()
    if missing:
        raise FixtureError(f"{label}: missing {sorted(missing)}")
    unknown = el.keys() - required - optional - COMMON_FIELDS
    if unknown:
        raise FixtureError(f"{label}: unknown fields {sorted(unknown)}")
    if el.get("source", "datasheet") not in SOURCES:
        raise FixtureError(f"{label}: source must be one of {SOURCES}")
    if el["type"] == "amplifier" and el["role"] not in AMP_ROLES:
        raise FixtureError(f"{label}: role must be one of {AMP_ROLES}")
    for k in FREQ_FIELDS & el.keys():
        _check_value(f"{label}.{k}", el[k])


def _check_value(label, v):
    if isinstance(v, dict):
        if set(v) != {"freq_mhz", "value"}:
            raise FixtureError(f"{label}: a table needs exactly freq_mhz and value")
        f, y = v["freq_mhz"], v["value"]
        if len(f) != len(y) or len(f) < 1:
            raise FixtureError(f"{label}: freq_mhz and value must be equal, non-empty lists")
        if any(b <= a for a, b in zip(f, f[1:])):
            raise FixtureError(f"{label}: freq_mhz must increase")
        vals = list(f) + list(y)
    else:
        vals = [v]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vals):
        raise FixtureError(f"{label}: values must be finite numbers")


def from_legacy(fixture):
    """Convert a pre-chain fixture snapshot (hardware.attenuator_db/cable) to the chain format."""
    hw = fixture.get("hardware") or {}
    if "chain" in hw:
        return fixture
    out = copy.deepcopy(fixture)
    hw = out["hardware"]
    chain = [{"id": "pad", "type": "attenuator", "loss_db": hw.pop("attenuator_db")}] if "attenuator_db" in hw else []
    cable = hw.pop("cable", None)
    if cable:
        chain.append({"id": "cable", "type": "cable", "model": cable.get("type"),
                      "length_m": cable["length_m"], "velocity_factor": cable["velocity_factor"]})
    hw.pop("adapters", None)
    hw["chain"] = chain
    return out


# ---------------------------------------------------------------- chain values

def value_at(v, freq_hz, default=0.0):
    """A scalar or frequency-table field at freq_hz (linear interpolation, clamped at the ends)."""
    if v is None:
        return default
    if isinstance(v, dict):
        return float(np.interp(freq_hz / 1e6, v["freq_mhz"], v["value"]))
    return float(v)


def element_delay_s(el, freq_hz):
    d = value_at(el.get("delay_ns"), freq_hz) * 1e-9
    if el["type"] == "cable":
        d += float(el["length_m"]) / (float(el["velocity_factor"]) * C0)
    return d


def chain_delay_s(fixture, freq_hz=None):
    """Total propagation delay of hardware.chain (cables plus each element's delay_ns)."""
    f = fixture["rf_base"]["freq"] if freq_hz is None else freq_hz
    return sum(element_delay_s(el, f) for el in fixture["hardware"]["chain"])
