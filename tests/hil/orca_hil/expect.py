"""Expected metric changes for a non-standard setup.

An exploratory fixture is compared with its reference (the nearest
commissioned ancestor, whose baselines are used). Each changed chain element,
and any change in rf_base gains, affects some of the ``affected_by`` tags in
thresholds.yaml (TYPE_EFFECTS). For each check of an affected metric:

  judge: info    -> recorded as info
  judge: expect  -> shift tags (gain, -gain, gain_atten_check, delay, snr_db) move
                    the limits by the predicted change and widen them by the changed
                    elements' tolerances; other tags are recorded as info

Predicted gain changes assume a linear chain: where any element is within
safety.linear_margin_db of compression or limiting, gain shifts become info.
The fixture's ``expect:`` entries override this, keyed ``<test>.<metric>``,
``<test>.<metric>/<case>`` or ``<test>.*``, with ``info``, ``judge`` (no
change) or ``{shift: <value>, tol: <value>}``.

With --setup-note on a commissioned fixture, the change is undeclared: every
[base] check is info and [abs] checks are judged as usual.
"""
import re
from dataclasses import dataclass

from . import configs, fixtures
from .thresholds import INFO_TAGS

TYPE_EFFECTS = {
    "cable": {"gain", "delay"},
    "adapter": {"gain", "delay"},
    "attenuator": {"gain", "delay"},
    "limiter": {"gain", "delay", "linearity"},
    "filter": {"gain", "delay", "bandwidth", "spectral", "phase"},
    "amplifier": {"gain", "delay", "noise", "linearity", "spectral", "bandwidth", "phase"},
}
DEFAULT_DELAY_TOL = 0.2          # fraction of delay_ns when delay_tol_ns is not given
SNR_GAIN_THRESHOLD_DB = 0.5      # smaller gain changes (with tolerance) leave snr metrics judged
CASE_FREQ_RE = re.compile(r"^(\d+(?:\.\d+)?)MHz$")
OVERRIDE_KEY_RE = re.compile(r"^t\d\d\.(\*|\w+(/[\w.-]+)?)$")


@dataclass
class Adjustment:
    info: bool = False
    shift: float = 0.0
    extra_tol: float = 0.0
    reason: str | None = None


NONE = Adjustment()


def reference(fixture):
    """Nearest commissioned ancestor (the fixture itself if commissioned), or None."""
    fx = fixture
    while fx is not None and not fixtures.is_standard(fx):
        fx = fixtures.parent(fx)
    return fx


def build(fixture, setup_note=None):
    """Expectation for a session, or None for a standard run."""
    if fixtures.is_standard(fixture) and not setup_note:
        return None
    return Expectation(fixture, setup_note)


def _changed(child, ref):
    """Chain elements of child that differ from ref, plus ref elements missing from child."""
    ref_by_id = {el["id"]: el for el in ref["hardware"]["chain"]}
    child_ids = {el["id"] for el in child["hardware"]["chain"]}
    changed = [el for el in child["hardware"]["chain"] if ref_by_id.get(el["id"]) != el]
    changed += [el for el in ref["hardware"]["chain"] if el["id"] not in child_ids]
    return changed


def _delay_tol_s(el):
    if "delay_tol_ns" in el:
        return abs(float(el["delay_tol_ns"])) * 1e-9
    return DEFAULT_DELAY_TOL * abs(fixtures.value_at(el.get("delay_ns"), 0.0)) * 1e-9


class Expectation:
    def __init__(self, fixture, setup_note=None):
        self.fixture = fixture
        self.setup_note = setup_note
        self.mode = "note" if fixtures.is_standard(fixture) else fixture.get("judge", "info")
        self.ref = None if self.mode == "note" else reference(fixture)
        self.overrides = dict(fixture.get("expect") or {})
        for key, val in self.overrides.items():
            if not OVERRIDE_KEY_RE.match(key):
                raise fixtures.FixtureError(f"expect key {key!r} must be <test>.<metric>[/<case>] or <test>.*")
            if val not in ("info", "judge") and not (isinstance(val, dict) and set(val) <= {"shift", "tol"}):
                raise fixtures.FixtureError(f"expect.{key}: use info, judge or {{shift, tol}}")
        self.skip_tests = list(fixture.get("skip_tests") or [])
        self._per_freq = {}
        if self.mode == "note":
            return
        if self.ref is None:
            self.affected, self.reasons = set(INFO_TAGS) | {"gain", "delay", "gain_atten_check"}, \
                {"all": "no commissioned reference fixture"}
            self.changed = []
            return
        self.changed = _changed(fixture, self.ref)
        self.affected = set()
        self.reasons = {}
        for el in self.changed:
            for tag in TYPE_EFFECTS[el["type"]]:
                self.affected.add(tag)
                self.reasons.setdefault(tag, f"chain element {el['id']} ({el['type']}) changed")
        if self.changed and any(el["type"] == "amplifier" for el in fixture["hardware"]["chain"]):
            self.affected.add("noise")
            self.reasons.setdefault("noise", "chain with an amplifier changed")
        rf, rf0 = fixture["rf_base"], self.ref["rf_base"]
        if rf.get("freq") != rf0.get("freq"):
            self.affected |= set(INFO_TAGS) | {"gain", "delay", "gain_atten_check"}
            self.reasons.setdefault("all", "rf_base.freq changed")
        if rf.get("tx_gain") != rf0.get("tx_gain") or rf.get("rx_gain") != rf0.get("rx_gain"):
            self.affected.add("gain")
            self.reasons.setdefault("gain", "rf_base gains changed")
        if rf.get("rx_gain") != rf0.get("rx_gain"):
            self.affected.add("noise")
            self.reasons.setdefault("noise", "rf_base.rx_gain changed")
        if "gain" in self.affected or self._att_gain(fixture) != self._att_gain(self.ref):
            self.affected.add("gain_atten_check")
        fs = float(configs.merged_config(fixture, {})["GENERATE"]["sample_rate"])
        self.delay_samples = (fixtures.chain_delay_s(fixture) - fixtures.chain_delay_s(self.ref)) * fs
        self.delay_tol_samples = sum(_delay_tol_s(el) for el in self.changed) * fs
        if "noise" in self.affected:
            self.affected.add("snr")
        elif "gain" in self.affected:
            shift, tol, why = self.gain_shift()
            if why or abs(shift) + tol >= SNR_GAIN_THRESHOLD_DB:
                self.affected.add("snr")
                self.reasons.setdefault("snr", why or f"gain change {shift:+.2g} dB")

    @staticmethod
    def _att_gain(fx):
        return ((fx.get("safety") or {}).get("attenuator_check") or {}).get("tx_gain")

    # ------------------------------------------------------------ gain predictions

    def gain_shift(self, freq_hz=None, at="base"):
        """(shift_db, tol_db, reason_if_unpredictable) for path gain at rf_base or the T0 check gain."""
        fx, ref = self.fixture, self.ref
        f = fx["rf_base"]["freq"] if freq_hz is None else freq_hz
        key = (f, at)
        if key in self._per_freq:
            return self._per_freq[key]
        g, g0 = fixtures.chain_gain_db(fx, f), fixtures.chain_gain_db(ref, f)
        tol = sum(fixtures.element_tol_db(el, f) for el in self.changed)
        if at == "base":
            tx, tx0 = fx["rf_base"]["tx_gain"], ref["rf_base"]["tx_gain"]
        else:
            tx, tx0 = self._att_gain(fx), self._att_gain(ref)
        reason = None
        if g is None or g0 is None:
            reason = f"chain gain unknown at {f / 1e6:g} MHz"
            shift = None
        else:
            shift = (g - g0) + (tx - tx0) + (fx["rf_base"]["rx_gain"] - ref["rf_base"]["rx_gain"])
            errors, nonlinear = fixtures.budget_problems(fx, tx, f)
            if nonlinear:
                reason = f"not linear at tx_gain {tx:g}: {', '.join(nonlinear)}"
        self._per_freq[key] = (shift, tol, reason)
        return self._per_freq[key]

    def atten_check_shift(self):
        """Predicted change of the T0 attenuator-check peak (linear), and its tolerance."""
        if self.mode == "note" or self.ref is None:
            return 0.0, 0.0
        shift, tol, _ = self.gain_shift(at="atten_check")
        return (shift or 0.0), tol

    def delay_shift_samples(self):
        return 0.0 if self.mode == "note" or self.ref is None else self.delay_samples

    # ------------------------------------------------------------ per-check decision

    def _override(self, test_id, name, case):
        for key in ([f"{test_id}.{name}/{case}"] if case else []) + [f"{test_id}.{name}", f"{test_id}.*"]:
            if key in self.overrides:
                return key, self.overrides[key]
        return None, None

    def adjust(self, test_id, name, case, check):
        key, ov = self._override(test_id, name, case)
        if ov == "judge":
            return NONE
        if ov == "info":
            return Adjustment(info=True, reason=f"fixture expect: {key}")
        if isinstance(ov, dict):
            return Adjustment(shift=float(ov.get("shift", 0.0)), extra_tol=float(ov.get("tol", 0.0)),
                              reason=f"fixture expect: {key}")
        if check.kind == "info":
            return NONE
        if self.mode == "note":
            return Adjustment(info=True, reason=f"setup note: {self.setup_note}") if check.kind == "base" else NONE
        tags = [t for t in check.affected_by if self._is_affected(t)]
        if not tags:
            return NONE
        info_tags = [t for t in tags if t in INFO_TAGS]
        if self.mode == "info" or info_tags:
            t = (info_tags or tags)[0]
            return Adjustment(info=True, reason=f"{t}: {self._reason(t)}")
        shift, tol = 0.0, 0.0
        for t in tags:
            s, e, why = self._shift(t, case)
            if s is None:
                return Adjustment(info=True, reason=f"{t}: {why}")
            shift, tol = shift + s, tol + e
        return Adjustment(shift=shift, extra_tol=tol,
                          reason=f"expected {'/'.join(tags)} change {shift:+.3g} (+/-{tol:.2g})")

    def _is_affected(self, tag):
        if tag == "-gain":
            return "gain" in self.affected
        if tag == "snr_db":
            return bool(self.affected & {"gain", "noise"})
        return tag in self.affected

    def _reason(self, tag):
        base = "gain" if tag in ("-gain", "snr_db") else tag
        return self.reasons.get(base) or self.reasons.get("all") or "setup changed"

    def _shift(self, tag, case):
        if tag == "delay":
            return self.delay_samples, self.delay_tol_samples, None
        if tag == "snr_db" and "noise" in self.affected:
            return None, 0.0, f"noise: {self._reason('noise')}"
        m = CASE_FREQ_RE.match(case or "")
        f = float(m.group(1)) * 1e6 if m else None
        shift, tol, why = self.gain_shift(f, at="atten_check" if tag == "gain_atten_check" else "base")
        if why:
            return None, 0.0, why
        return (-shift if tag == "-gain" else shift), tol, None

    def summary(self):
        out = {"mode": self.mode, "setup_note": self.setup_note,
               "reference_fixture": self.ref["fixture_id"] if self.ref else None,
               "skip_tests": self.skip_tests, "overrides": self.overrides}
        if self.mode != "note" and self.ref is not None:
            shift, tol, why = self.gain_shift()
            out.update(changed_elements=[el["id"] for el in self.changed],
                       affected_tags=sorted(self.affected),
                       gain_shift_db=shift, gain_tol_db=tol, gain_unpredictable=why,
                       delay_shift_samples=self.delay_samples, delay_tol_samples=self.delay_tol_samples,
                       atten_check_shift_db=self.atten_check_shift()[0])
        return out
