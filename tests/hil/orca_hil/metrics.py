"""MetricRecorder: applies thresholds.yaml and baselines, writes metrics.yaml."""
from dataclasses import dataclass, field
import time

from . import yamlio
from .thresholds import evaluate


@dataclass
class Measurement:
    """A value produced by an ``analyze()`` function, before thresholds are applied.

    ``params`` fills placeholders in the metric's bound (e.g. ``{"expected": 14.1}``).
    ``inconclusive`` marks a value that cannot be judged (missing reference data).
    """
    name: str
    value: object
    case: str | None = None
    params: dict = field(default_factory=dict)
    note: str | None = None
    inconclusive: bool = False


class MetricRecorder:
    def __init__(self, test_id, thresholds, baselines=None, out_dir=None,
                 fixture_suspect=False, ignore_base=False):
        """
        thresholds: output of ``thresholds.load_thresholds()``
        baselines: a ``baselines.Baselines`` (or None: every [base] metric is inconclusive)
        ignore_base: baseline-update mode; [base] metrics are recorded but not judged
        """
        self.test_id = test_id
        self.checks = {**thresholds.get("common", {}), **thresholds.get(test_id, {})}
        self.k = thresholds.get("_k", 4.0)
        self.baselines = baselines
        self.out_dir = out_dir
        self.fixture_suspect = fixture_suspect
        self.ignore_base = ignore_base
        self.records = []
        self.pre_finalize = []   # callables(recorder) run at the start of finalize()
        self.finalized = False
        self._t0 = time.time()

    def record(self, name, value, case=None, params=None, note=None, inconclusive=False):
        if name not in self.checks:
            raise KeyError(f"metric {self.test_id}.{name} is not defined in thresholds.yaml")
        value = yamlio.to_builtin(value)
        new = []
        for check in self.checks[name]:
            baseline = None
            if check.kind == "base" and self.baselines is not None:
                baseline = self.baselines.get(self.test_id, name, case)
            ev = evaluate(value, check, baseline=baseline, params=params, k=self.k)
            status = ev.status
            if inconclusive and status != "info":
                status = "inconclusive"
            if check.kind == "base" and self.ignore_base and status != "inconclusive":
                status = "info"
            rec = {"name": name}
            if case is not None:
                rec["case"] = case
            rec.update(value=value, units=check.units, kind=check.kind, bound=ev.bound,
                       bound_lo=ev.lo, bound_hi=ev.hi, status=status)
            if baseline is not None:
                rec["baseline"] = baseline
            if note:
                rec["note"] = note
            if self.fixture_suspect:
                rec["fixture_suspect"] = True
            new.append(rec)
        self.records.extend(new)
        return new

    def record_all(self, measurements):
        for m in measurements:
            self.record(m.name, m.value, case=m.case, params=m.params, note=m.note,
                        inconclusive=m.inconclusive)

    @property
    def failures(self):
        return [r for r in self.records if r["status"] == "fail"]

    @property
    def status(self):
        if self.failures:
            return "fail"
        if any(r["status"] == "inconclusive" for r in self.records):
            return "inconclusive"
        return "pass"

    def summary(self):
        return {"test_id": self.test_id, "status": self.status,
                "fixture_suspect": self.fixture_suspect,
                "duration_s": round(time.time() - self._t0, 2), "metrics": self.records}

    def finalize(self, fail=True):
        """Write metrics.yaml and, if any metric failed, fail the test listing all failures."""
        if not self.finalized:
            for hook in self.pre_finalize:
                hook(self)
        self.finalized = True
        if self.out_dir is not None:
            yamlio.dump(self.summary(), self.out_dir / "metrics.yaml")
        if fail and self.failures:
            import pytest
            pytest.fail(f"{self.test_id}: {len(self.failures)} metric(s) failed\n"
                        + format_table(self.failures), pytrace=False)
        return self.summary()


def format_table(records):
    rows = [("metric", "case", "value", "units", "kind", "limits")]
    for r in records:
        lim = f"[{_fmt(r.get('bound_lo'))}, {_fmt(r.get('bound_hi'))}]"
        if r["bound"].get("type") == "equals":
            lim = f"== {r['bound']['value']}"
        rows.append((r["name"], r.get("case") or "", _fmt(r["value"]), r.get("units") or "",
                     r["kind"], lim))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows)


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
