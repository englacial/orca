import math

import numpy as np
import pytest

from tests.hil.orca_hil import yamlio
from tests.hil.orca_hil.baselines import Baselines
from tests.hil.orca_hil.metrics import Measurement, MetricRecorder
from tests.hil.orca_hil.thresholds import load_thresholds

THRESHOLDS = """
t99:
  power_db: {kind: base, bound: {type: two_sided, tol: 1.0}, units: dB}
  clip: {kind: abs, bound: {type: max, value: 0.5}, units: FS}
  sep:
    - {kind: abs, bound: {type: range_about, center: expected, tol: 1.5}, units: samples}
    - {kind: base, bound: {type: two_sided, tol: 0.1}, units: samples}
  count: {kind: info, units: count}
"""


@pytest.fixture
def thresholds(tmp_path):
    p = tmp_path / "thresholds.yaml"
    p.write_text(THRESHOLDS)
    return load_thresholds(p)


@pytest.fixture
def baselines(tmp_path):
    b = Baselines("fx", directory=tmp_path)
    b.update_scalar("t99", "power_db", [-30.0, -30.1, -29.9], runs=["r1", "r2", "r3"])
    b.update_scalar("t99", "sep", [14.2, 14.2], runs=["r1", "r2"])
    return b


def test_unknown_metric_is_an_error(thresholds):
    rec = MetricRecorder("t99", thresholds)
    with pytest.raises(KeyError):
        rec.record("not_listed", 1.0)


def test_records_and_writes_yaml(thresholds, baselines, tmp_path):
    rec = MetricRecorder("t99", thresholds, baselines, out_dir=tmp_path)
    rec.record_all([
        Measurement("power_db", np.float32(-30.2)),
        Measurement("clip", np.float64(0.1)),
        Measurement("sep", 14.25, params={"expected": 14.1}),
        Measurement("count", np.int64(3)),
    ])
    summary = rec.finalize()
    assert summary["status"] == "pass"
    data = yamlio.load(tmp_path / "metrics.yaml")
    assert data["test_id"] == "t99"
    kinds = [(m["name"], m["kind"]) for m in data["metrics"]]
    assert ("sep", "abs") in kinds and ("sep", "base") in kinds
    assert all(isinstance(m["value"], (int, float)) for m in data["metrics"])
    power = next(m for m in data["metrics"] if m["name"] == "power_db")
    assert power["baseline"]["n"] == 3


def test_finalize_fails_listing_all_failures(thresholds, baselines, tmp_path):
    rec = MetricRecorder("t99", thresholds, baselines, out_dir=tmp_path)
    rec.record("power_db", -35.0)
    rec.record("clip", 0.9)
    rec.record("sep", 14.2, params={"expected": 14.1})
    with pytest.raises(pytest.fail.Exception) as exc:
        rec.finalize()
    msg = str(exc.value)
    assert "2 metric(s) failed" in msg and "power_db" in msg and "clip" in msg
    assert (tmp_path / "metrics.yaml").exists()


def test_nan_written_as_yaml_nan(thresholds, tmp_path):
    rec = MetricRecorder("t99", thresholds, out_dir=tmp_path)
    rec.record("count", float("nan"))
    rec.finalize()
    assert ".nan" in (tmp_path / "metrics.yaml").read_text()
    assert math.isnan(yamlio.load(tmp_path / "metrics.yaml")["metrics"][0]["value"])


def test_no_baseline_is_inconclusive_but_passes(thresholds, tmp_path):
    rec = MetricRecorder("t99", thresholds, Baselines("none", directory=tmp_path))
    rec.record("power_db", -10.0)
    assert rec.records[0]["status"] == "inconclusive"
    assert rec.finalize()["status"] == "inconclusive"


def test_forced_inconclusive_and_fixture_suspect(thresholds):
    rec = MetricRecorder("t99", thresholds, fixture_suspect=True)
    rec.record("clip", 0.9, inconclusive=True, note="no reference")
    r = rec.records[0]
    assert r["status"] == "inconclusive" and r["fixture_suspect"] and r["note"] == "no reference"


def test_ignore_base_mode(thresholds, baselines):
    rec = MetricRecorder("t99", thresholds, baselines, ignore_base=True)
    rec.record("power_db", -50.0)
    rec.record("clip", 0.9)
    statuses = {r["name"]: r["status"] for r in rec.records}
    assert statuses == {"power_db": "info", "clip": "fail"}


def test_ignore_base_mode_without_baseline(thresholds, tmp_path):
    rec = MetricRecorder("t99", thresholds, Baselines("none", directory=tmp_path), ignore_base=True)
    rec.record("power_db", -50.0)
    rec.record("power_db", float("nan"), case="x", inconclusive=True)
    assert [r["status"] for r in rec.records] == ["info", "inconclusive"]
