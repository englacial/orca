import math

import pytest

from tests.hil.orca_hil.thresholds import Check, evaluate, load_thresholds, resolve


def ev(kind, bound, value, baseline=None, params=None, k=4.0):
    return evaluate(value, Check(kind, "u", bound), baseline=baseline, params=params, k=k)


@pytest.mark.parametrize("bound,value,status", [
    ({"type": "range", "lo": 1, "hi": 2}, 1.5, "pass"),
    ({"type": "range", "lo": 1, "hi": 2}, 2.1, "fail"),
    ({"type": "max", "value": 0.5}, 0.5, "pass"),
    ({"type": "max", "value": 0.5}, 0.51, "fail"),
    ({"type": "min", "value": 20}, 19, "fail"),
    ({"type": "abs_max", "value": 0.5}, -0.4, "pass"),
    ({"type": "abs_max", "value": 0.5}, -0.6, "fail"),
    ({"type": "equals", "value": True}, True, "pass"),
    ({"type": "equals", "value": 0}, 2, "fail"),
])
def test_abs_bounds(bound, value, status):
    assert ev("abs", bound, value).status == status


def test_range_about_placeholder_and_rel_tol():
    b = {"type": "range_about", "center": "expected", "tol": 1.5}
    assert ev("abs", b, 15.0, params={"expected": 14.1}).status == "pass"
    assert ev("abs", b, 16.0, params={"expected": 14.1}).status == "fail"
    r = ev("abs", {"type": "range_about", "center": "ideal", "rel_tol": 0.1}, 5.4, params={"ideal": 5.0})
    assert r.status == "pass" and r.lo == pytest.approx(4.5) and r.hi == pytest.approx(5.5)


def test_missing_placeholder_raises():
    with pytest.raises(KeyError):
        resolve({"type": "range_about", "center": "expected", "tol": 1}, {})


def test_base_bounds_use_max_of_tol_and_k_sigma():
    base = {"mean": -30.0, "std": 0.5, "n": 5}
    # k*std = 2.0 > tol = 1.0
    assert ev("base", {"type": "two_sided", "tol": 1.0}, -31.9, base).status == "pass"
    assert ev("base", {"type": "two_sided", "tol": 1.0}, -32.1, base).status == "fail"
    tight = {"mean": -30.0, "std": 0.01, "n": 5}
    assert ev("base", {"type": "two_sided", "tol": 1.0}, -30.9, tight).status == "pass"
    assert ev("base", {"type": "lower", "tol": 1.0}, -25.0, tight).status == "pass"
    assert ev("base", {"type": "lower", "tol": 1.0}, -31.5, tight).status == "fail"
    assert ev("base", {"type": "upper", "tol": 1.0}, -35.0, tight).status == "pass"
    assert ev("base", {"type": "upper", "tol": 1.0}, -28.5, tight).status == "fail"


def test_ratio_and_factor_bounds():
    base = {"mean": 2e-7, "std": 0.0, "n": 3}
    r = ev("base", {"type": "ratio_upper", "factor": 3, "floor": 1e-6}, 9e-7, base)
    assert r.status == "pass" and r.hi == 1e-6
    assert ev("base", {"type": "ratio_upper", "factor": 3}, 9e-7, base).status == "fail"
    base = {"mean": 100.0, "std": 0.0, "n": 3}
    assert ev("base", {"type": "factor_two_sided", "factor": 3}, 34, base).status == "pass"
    assert ev("base", {"type": "factor_two_sided", "factor": 3}, 301, base).status == "fail"


def test_base_without_baseline_is_inconclusive():
    assert ev("base", {"type": "upper", "tol": 1}, 3.0, None).status == "inconclusive"


def test_nonfinite_fails_and_info_never_judged():
    assert ev("abs", {"type": "max", "value": 1}, math.nan).status == "fail"
    assert ev("abs", {"type": "max", "value": 1}, math.inf).status == "fail"
    assert ev("info", {}, math.nan).status == "info"


def test_repo_thresholds_file_loads():
    th = load_thresholds()
    assert th["_k"] == 4
    assert len(th["t01"]["peak_separation_samples"]) == 2


def test_invalid_bound_type_for_kind(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("t99:\n  x: {kind: abs, bound: {type: two_sided, tol: 1}, units: dB}\n")
    with pytest.raises(ValueError):
        load_thresholds(p)
