"""Fixture files: lookup, ``extends`` inheritance and RF-chain validation.

A fixture describes one bench setup. ``hardware.chain`` lists the RF parts
between SDR TX and SDR RX, in order. A fixture can ``extends: <parent>`` another
one (by id from tests/hil/fixtures/, or by a path relative to the child). The
child's mappings are merged into the parent's; lists and scalars replace the
parent's, so a child that changes the RF chain restates all of it.

``status: commissioned`` marks a fixture with its own baselines; anything else is
``exploratory`` and its runs are logged as non-standard. These keys are never
inherited: status, judge, expect, skip_tests, description. In a child's chain,
a bare id (``- cable50``) stands for the parent's element with that id.
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
COMMON_FIELDS = {"id", "type", "model", "serial", "source", "delay_ns", "delay_tol_ns", "max_input_dbm",
                 "notes"}
ELEMENT_FIELDS = {
    "cable": ({"length_m", "velocity_factor"}, {"loss_db", "loss_db_per_100m"}),
    "adapter": (set(), {"loss_db"}),
    "attenuator": ({"loss_db"}, {"loss_tol_db", "power_rating_w"}),
    "limiter": ({"loss_db", "threshold_dbm"}, {"loss_tol_db", "leakage_dbm"}),
    "filter": ({"loss_db"}, {"loss_tol_db", "passband_mhz", "stopband_rejection_db"}),
    "amplifier": ({"role", "gain_db"}, {"gain_tol_db", "nf_db", "p1db_out_dbm", "powered", "supply",
                                        "watchdog", "max_gain_db", "max_output_dbm", "max_enable_duty",
                                        "startup_burst_dbm", "radar_config", "radar_config_min",
                                        "disable_command"}),
}
# Amplifier fields for a power amplifier:
#   max_gain_db        gain used by the power budget (e.g. the gain at maximum VVA setting until
#                      the actual setting is tested); gain_db stays the expected gain
#   max_output_dbm     output limit approved for this setup (budget error above it, less margin)
#   max_enable_duty    limit on the enable duty, tx_duration / pulse_rep_int, of every capture
#   startup_burst_dbm  output of bursts the amp emits independently of the drive (e.g. at radar
#                      start); followed through the rest of the chain by the budget
#   radar_config       radar config merged into every capture last (e.g. the GPIO enable pin)
#   radar_config_min   numeric radar config minimums applied after that, so a test's larger value
#                      is kept (e.g. CHIRP.rx_duration long enough for a padded pulse)
#   watchdog           {status_file, max_age_s}: captures run only while an external watchdog
#                      reports itself armed in status_file (orca_hil/watchdog.py)
#   disable_command    command (list of strings) that puts the amp in shutdown through the SDR,
#                      e.g. drives its enable GPIO inactive. The harness runs it before its first
#                      contact with the SDR (an FPGA load leaves the GPIO at its default), after
#                      each UHD tool step in T0, and at session end (orca_hil/watchdog.py)
SOURCES = ("datasheet", "measured", "estimate")
UNRATED_SAFE_DBM = 20.0     # elements that can see more than this from a power amplifier need a rating
AMP_ROLES = ("lna", "pa")
# Fields that may be a number or a frequency table {freq_mhz: [...], value: [...]}
FREQ_FIELDS = {"loss_db", "loss_db_per_100m", "loss_tol_db", "gain_db", "gain_tol_db", "nf_db",
               "max_gain_db", "max_output_dbm", "startup_burst_dbm",
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
    child_chain = (raw.get("hardware") or {}).get("chain")
    if child_chain is not None:
        merged["hardware"]["chain"] = _resolve_chain(path.name, child_chain, parent)
    return merged, [path] + chain


def _resolve_chain(label, chain, parent):
    """A bare id in a child's chain stands for the parent's element with that id."""
    by_id = {el["id"]: el for el in (parent.get("hardware") or {}).get("chain") or []}
    out = []
    for el in chain:
        if isinstance(el, str):
            if el not in by_id:
                raise FixtureError(f"{label}: chain entry {el!r} is not an element of the parent")
            el = copy.deepcopy(by_id[el])
        out.append(el)
    return out


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
    table = (fx.get("calibration") or {}).get("tx_power_table")
    if table is not None:
        try:
            gains = [float(r["tx_gain"]) for r in table] + [float(r["dbm"]) for r in table]
        except (TypeError, KeyError, ValueError):
            raise FixtureError(f"{fid}: calibration.tx_power_table must be a list of {{tx_gain, dbm}}")
        g = gains[:len(table)]
        if not table or any(b <= a for a, b in zip(g, g[1:])):
            raise FixtureError(f"{fid}: calibration.tx_power_table tx_gain must increase")
    ids = set()
    pa, peak = None, None        # worst-case power from a power amplifier, followed down the chain
    f0 = float((fx.get("rf_base") or {}).get("freq") or 0.0)
    for i, el in enumerate(hw["chain"] or []):
        validate_element(f"{fid}: chain[{i}]", el)
        if el["id"] in ids:
            raise FixtureError(f"{fid}: duplicate chain id {el['id']!r}")
        ids.add(el["id"])
        if peak is not None:
            if peak > UNRATED_SAFE_DBM and el["type"] != "cable" and input_limit_dbm(el, f0) is None:
                raise FixtureError(f"{fid}: {el['id']} can see {peak:.0f} dBm from power amplifier {pa} "
                                   "and needs max_input_dbm or power_rating_w")
            peak += budget_gain_db(el, f0)
        if el["type"] == "amplifier" and el["role"] == "pa":
            if "watchdog" not in el:
                raise FixtureError(f"{fid}: power amplifier {el['id']} needs a watchdog")
            if "disable_command" not in el:
                raise FixtureError(f"{fid}: power amplifier {el['id']} needs a disable_command")
            pa = el["id"]
            peak = max(value_at(el.get(k), f0, -math.inf)
                       for k in ("p1db_out_dbm", "max_output_dbm", "startup_burst_dbm"))
            if peak == -math.inf:
                raise FixtureError(f"{fid}: power amplifier {el['id']} needs p1db_out_dbm")


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
    if "watchdog" in el:
        w = el["watchdog"]
        if not isinstance(w, dict) or "status_file" not in w or set(w) - {"status_file", "max_age_s"}:
            raise FixtureError(f"{label}: watchdog must be {{status_file[, max_age_s]}}")
    if "disable_command" in el and not (isinstance(el["disable_command"], list) and el["disable_command"]
                                        and all(isinstance(a, str) for a in el["disable_command"])):
        raise FixtureError(f"{label}: disable_command must be a non-empty list of strings")
    if "radar_config_min" in el and not (isinstance(el["radar_config_min"], dict) and all(
            isinstance(v, dict) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v.values())
            for v in el["radar_config_min"].values())):
        raise FixtureError(f"{label}: radar_config_min must map config sections to numeric minimums")
    if "radar_config" in el and not isinstance(el["radar_config"], dict):
        raise FixtureError(f"{label}: radar_config must be a mapping of config sections")
    if "max_enable_duty" in el and not 0 < float(el["max_enable_duty"]) <= 1:
        raise FixtureError(f"{label}: max_enable_duty must be in (0, 1]")
    if "power_rating_w" in el and not float(el["power_rating_w"]) > 0:
        raise FixtureError(f"{label}: power_rating_w must be positive")


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


def element_gain_db(el, freq_hz):
    """Gain of one element in dB (negative for a loss), or None where it is not known.

    A filter's scalar loss_db applies only inside passband_mhz; outside it the
    loss is unknown unless loss_db is a table that covers freq_hz.
    """
    t = el["type"]
    if t == "amplifier":
        return value_at(el["gain_db"], freq_hz)
    if t == "filter" and el.get("passband_mhz") and not _table_covers(el["loss_db"], freq_hz):
        lo, hi = el["passband_mhz"]
        if not lo <= freq_hz / 1e6 <= hi:
            return None
    if t == "cable" and "loss_db" not in el:
        return -value_at(el.get("loss_db_per_100m"), freq_hz) * float(el["length_m"]) / 100.0
    return -value_at(el.get("loss_db"), freq_hz)


def _table_covers(v, freq_hz):
    return isinstance(v, dict) and v["freq_mhz"][0] <= freq_hz / 1e6 <= v["freq_mhz"][-1]


def element_tol_db(el, freq_hz):
    return abs(value_at(el.get("gain_tol_db", el.get("loss_tol_db")), freq_hz))


def chain_gain_db(fixture, freq_hz=None):
    """Net gain of hardware.chain in dB, or None if any element's gain is unknown at freq_hz."""
    f = fixture["rf_base"]["freq"] if freq_hz is None else freq_hz
    gains = [element_gain_db(el, f) for el in fixture["hardware"]["chain"]]
    return None if any(g is None for g in gains) else sum(gains)


# ---------------------------------------------------------------- power budget
#
# Power is followed from the SDR TX port through the chain at the chirp's peak
# envelope, assuming every element is linear (amplifiers do not saturate and
# limiters do not limit), which overestimates the power after them. Unknown
# losses (a filter outside its passband) count as 0 dB. Each limit is reduced
# by safety.power_margin_db.

DEFAULT_POWER_MARGIN_DB = 3.0
DEFAULT_LINEAR_MARGIN_DB = 3.0


def tx_power_dbm(fixture, tx_gain):
    """SDR TX output (dBm) from calibration.tx_power_table, extrapolated at 1 dB/dB."""
    table = (fixture.get("calibration") or {}).get("tx_power_table")
    if not table:
        raise FixtureError(f"{fixture['fixture_id']}: calibration.tx_power_table is needed for the power budget")
    g, p = [float(r["tx_gain"]) for r in table], [float(r["dbm"]) for r in table]
    if tx_gain < g[0]:
        return p[0] - (g[0] - tx_gain)
    if tx_gain > g[-1]:
        return p[-1] + (tx_gain - g[-1])
    return float(np.interp(tx_gain, g, p))


def input_limit_dbm(el, freq_hz):
    """An element's input limit: the lower of max_input_dbm and power_rating_w, or None."""
    lims = []
    if el.get("max_input_dbm") is not None:
        lims.append(value_at(el["max_input_dbm"], freq_hz))
    if el.get("power_rating_w") is not None:
        lims.append(10 * math.log10(float(el["power_rating_w"]) * 1000))
    return min(lims) if lims else None


def budget_gain_db(el, freq_hz):
    """Gain for the power budget: max_gain_db where given, unknown losses as 0 dB."""
    if el["type"] == "amplifier" and el.get("max_gain_db") is not None:
        return value_at(el["max_gain_db"], freq_hz)
    g = element_gain_db(el, freq_hz)
    return 0.0 if g is None else g


def power_budget(fixture, tx_gain, freq_hz):
    """[{id, type, input_dbm, output_dbm, burst_dbm}] for each element, then the SDR RX input.

    burst_dbm is the element's input from an upstream amplifier's startup_burst_dbm (or None).
    """
    p = tx_power_dbm(fixture, tx_gain)
    burst = None
    stages = []
    for el in fixture["hardware"]["chain"]:
        g = budget_gain_db(el, freq_hz)
        out = p + g
        stages.append({"id": el["id"], "type": el["type"], "input_dbm": p, "output_dbm": out, "burst_dbm": burst})
        burst = None if burst is None else burst + g
        if el.get("startup_burst_dbm") is not None:
            burst = max(burst if burst is not None else -math.inf, value_at(el["startup_burst_dbm"], freq_hz))
        p = out
    stages.append({"id": "sdr_rx", "type": "sdr", "input_dbm": p, "output_dbm": None, "burst_dbm": burst})
    return stages


def budget_problems(fixture, tx_gain, freq_hz):
    """(errors, nonlinear) at tx_gain and freq_hz.

    errors: limits exceeded (an element's max_input_dbm, the SDR's
    safety.rx_max_input_dbm, a limiter driven into limiting unless
    safety.allow_limiting). nonlinear: elements within safety.linear_margin_db
    of compression (amplifier p1db_out_dbm) or of a limiter's threshold.
    """
    safety = fixture.get("safety") or {}
    margin = float(safety.get("power_margin_db", DEFAULT_POWER_MARGIN_DB))
    lin_margin = float(safety.get("linear_margin_db", DEFAULT_LINEAR_MARGIN_DB))
    errors, nonlinear = [], []
    els = {el["id"]: el for el in fixture["hardware"]["chain"]}
    for st in power_budget(fixture, tx_gain, freq_hz):
        pin = st["input_dbm"]
        peak = max(pin, st["burst_dbm"]) if st["burst_dbm"] is not None else pin
        what = "input" if peak == pin else "start-up burst input"
        if st["id"] == "sdr_rx":
            lim = safety.get("rx_max_input_dbm")
            if lim is None:
                errors.append("safety.rx_max_input_dbm is not set")
            elif peak > lim - margin:
                errors.append(f"SDR RX {what} {peak:.1f} dBm > {lim} dBm - {margin:g} dB margin")
            continue
        el = els[st["id"]]
        lim = input_limit_dbm(el, freq_hz)
        if lim is not None and peak > lim - margin:
            errors.append(f"{el['id']} {what} {peak:.1f} dBm > limit {lim:.1f} dBm - {margin:g} dB margin")
        if el.get("max_output_dbm") is not None:
            mx = value_at(el["max_output_dbm"], freq_hz)
            if st["output_dbm"] > mx - margin:
                errors.append(f"{el['id']} output {st['output_dbm']:.1f} dBm > approved {mx:g} dBm - "
                              f"{margin:g} dB margin")
        if el["type"] == "limiter":
            thr = value_at(el["threshold_dbm"], freq_hz)
            if pin > thr - lin_margin:
                if not safety.get("allow_limiting"):
                    errors.append(f"limiter {el['id']} input {pin:.1f} dBm is within {lin_margin:g} dB of its "
                                  f"{thr:g} dBm threshold (set safety.allow_limiting to permit)")
                nonlinear.append(el["id"])
        if el["type"] == "amplifier" and el.get("p1db_out_dbm") is not None:
            if st["output_dbm"] > value_at(el["p1db_out_dbm"], freq_hz) - lin_margin:
                nonlinear.append(el["id"])
    return errors, nonlinear


def max_safe_tx_gain(fixture, freq_hz, lo=0.0, hi=None, step=0.5):
    """Highest tx_gain on a `step` grid in [lo, hi] with no budget errors, or None."""
    hi = float((fixture.get("safety") or {}).get("tx_gain_max", 89.8)) if hi is None else hi
    best = None
    grid = [float(g) for g in np.arange(lo, hi, step)] + [hi]
    for g in grid:
        if budget_problems(fixture, g, freq_hz)[0]:
            break
        best = g
    return best


def powered_elements(fixture):
    return [el["id"] for el in fixture["hardware"]["chain"] if el.get("powered")]


def watchdog_elements(fixture):
    return [el["id"] for el in fixture["hardware"]["chain"] if el.get("watchdog")]


def watchdogs(fixture):
    """[(element id, {status_file, max_age_s})] for chain elements that need a watchdog."""
    return [(el["id"], el["watchdog"]) for el in fixture["hardware"]["chain"] if el.get("watchdog")]


def chain_radar_config(fixture):
    """The chain elements' radar_config, merged in chain order."""
    out = {}
    for el in fixture["hardware"]["chain"]:
        for section, vals in (el.get("radar_config") or {}).items():
            out.setdefault(section, {}).update(vals)
    return out


def chain_radar_config_min(fixture):
    """The chain elements' radar_config_min, combined (largest minimum wins)."""
    out = {}
    for el in fixture["hardware"]["chain"]:
        for section, vals in (el.get("radar_config_min") or {}).items():
            sec = out.setdefault(section, {})
            for k, v in vals.items():
                sec[k] = max(sec.get(k, v), v)
    return out


def enable_duty(cfg):
    """Upper bound on the amp-enable duty of a radar config: tx_duration / pulse_rep_int."""
    return float(cfg["CHIRP"]["tx_duration"]) / float(cfg["CHIRP"]["pulse_rep_int"])


def max_test_duty(fixture, cfg):
    """Highest duty max(tx_duration, rx_duration) / PRI (as T9/T10 define it) that keeps every
    element within its max_enable_duty, for the durations in cfg; 1.0 without a limit."""
    caps = [float(el["max_enable_duty"]) for el in fixture["hardware"]["chain"] if el.get("max_enable_duty")]
    if not caps:
        return 1.0
    tx, rx = float(cfg["CHIRP"]["tx_duration"]), float(cfg["CHIRP"]["rx_duration"])
    return min(1.0, min(caps) * max(tx, rx) / tx)


def duty_problems(fixture, cfg):
    """Errors for chain elements whose max_enable_duty the config exceeds."""
    if not cfg["RF0"].get("transmit", True):
        return []
    d = enable_duty(cfg)
    return [f"{el['id']} enable duty {d:.3f} > max_enable_duty {el['max_enable_duty']}"
            for el in fixture["hardware"]["chain"]
            if el.get("max_enable_duty") is not None and d > float(el["max_enable_duty"])]
