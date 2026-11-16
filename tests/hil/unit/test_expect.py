"""Power budget, expected changes for non-standard chains, and adjusted judging."""
import copy

import pytest

from tests.hil.orca_hil import configs, expect, fixtures, yamlio
from tests.hil.orca_hil.baselines import Baselines
from tests.hil.orca_hil.metrics import MetricRecorder
from tests.hil.orca_hil.thresholds import Check, evaluate, load_thresholds

BASE_ID = "b206mini_pi5_loopback_50m"
LIM1 = {"id": "lim1", "type": "limiter", "loss_db": 0.5, "loss_tol_db": 0.2, "threshold_dbm": 10,
        "max_input_dbm": 30}
LNA = {"id": "lna", "type": "amplifier", "role": "lna", "gain_db": 20, "gain_tol_db": 1.0, "nf_db": 1.0,
       "p1db_out_dbm": 18, "max_input_dbm": 15, "powered": True, "delay_ns": 1.0}
LIM2 = {**LIM1, "id": "lim2"}
PAD10 = {"id": "pad10", "type": "attenuator", "loss_db": 10, "loss_tol_db": 0.3}


@pytest.fixture
def base():
    return fixtures.load(fixtures.resolve_path(BASE_ID))


def child(tmp_path, base, extra, **top):
    doc = {"fixture_id": "rx_chain", "extends": BASE_ID,
           "hardware": {"chain": copy.deepcopy(base["hardware"]["chain"]) + extra}, **top}
    path = tmp_path / f"{doc['fixture_id']}.yaml"
    yamlio.dump(doc, path)
    return fixtures.load(path)


@pytest.fixture
def thresholds():
    return load_thresholds()


def checks(thresholds, test_id, name):
    return thresholds[test_id][name]


# ---------------------------------------------------------------- power budget

def test_base_fixture_budget_is_clear_to_device_max(base):
    for g in (0, 35, 89.8):
        assert fixtures.budget_problems(base, g, 330e6) == ([], [])
    assert fixtures.max_safe_tx_gain(base, 330e6) == pytest.approx(89.8)


def test_tx_power_table_interpolates_and_extrapolates(base):
    assert fixtures.tx_power_dbm(base, 62) == pytest.approx(-3.0)
    assert fixtures.tx_power_dbm(base, 67) == pytest.approx(2.0)
    assert fixtures.tx_power_dbm(base, 35) == pytest.approx(-30.0)
    assert fixtures.tx_power_dbm(base, 86) == pytest.approx(21.0)


def test_budget_follows_the_chain(tmp_path, base):
    fx = child(tmp_path, base, [LIM1, LNA, LIM2, PAD10])
    st = {s["id"]: s for s in fixtures.power_budget(fx, 35, 330e6)}
    # -30 dBm TX, 30 dB pad, 21.2 dB cable (measured table), 0.5 dB limiter
    assert st["lna"]["input_dbm"] == pytest.approx(-30 - 30 - 21.2 - 0.5)
    assert st["sdr_rx"]["input_dbm"] == pytest.approx(-30 - 30 - 21.2 - 0.5 + 20 - 0.5 - 10)
    assert fixtures.chain_gain_db(fx) - fixtures.chain_gain_db(base) == pytest.approx(20 - 0.5 - 0.5 - 10)


def test_budget_limits_rx_input_and_caps_tx_gain(tmp_path, base):
    fx = child(tmp_path, base, [LIM1, LNA, LIM2])        # no pad after the LNA
    assert fixtures.budget_problems(fx, 35, 330e6)[0] == []
    errors, _ = fixtures.budget_problems(fx, 89.8, 330e6)
    assert any("SDR RX input" in e for e in errors)
    cap = fixtures.max_safe_tx_gain(fx, 330e6)
    # RX limit -15 dBm - 3 dB margin; RX input = tx_gain - 65 - 51.2 + 19
    assert cap == pytest.approx(79.0)
    assert fixtures.budget_problems(fx, cap + 0.5, 330e6)[0]


def test_limiter_in_limiting_is_refused_unless_allowed(tmp_path, base):
    lim = {**LIM1, "threshold_dbm": -85}         # input is -81.2 dBm at tx_gain 35
    fx = child(tmp_path, base, [lim])
    errors, nonlinear = fixtures.budget_problems(fx, 35, 330e6)
    assert any("limiter lim1" in e for e in errors) and nonlinear == ["lim1"]
    fx2 = child(tmp_path, base, [lim], safety={"allow_limiting": True})
    errors, nonlinear = fixtures.budget_problems(fx2, 35, 330e6)
    assert errors == [] and nonlinear == ["lim1"]


def test_element_max_input_and_compression(tmp_path, base):
    lna = {**LNA, "max_input_dbm": -80, "p1db_out_dbm": -63}   # input -81.7, output -61.7 dBm at tx_gain 35
    fx = child(tmp_path, base, [lna], safety={"rx_max_input_dbm": 10})
    errors, nonlinear = fixtures.budget_problems(fx, 35, 330e6)
    assert any(e.startswith("lna input") for e in errors) and nonlinear == ["lna"]


def test_filter_outside_passband_has_unknown_gain(tmp_path, base):
    bpf = {"id": "bpf", "type": "filter", "loss_db": 1.5, "passband_mhz": [280, 380]}
    fx = child(tmp_path, base, [bpf])
    assert fixtures.chain_gain_db(fx, 330e6) == pytest.approx(fixtures.chain_gain_db(base, 330e6) - 1.5)
    assert fixtures.chain_gain_db(fx, 600e6) is None
    # unknown loss counts as 0 dB in the budget
    assert fixtures.power_budget(fx, 35, 600e6)[-1]["input_dbm"] == pytest.approx(
        fixtures.power_budget(base, 35, 600e6)[-1]["input_dbm"])


def test_config_validation_rejects_unsafe_capture(tmp_path, base):
    fx = child(tmp_path, base, [LIM1, LNA, LIM2])
    cfg = configs.merged_config(fx, {"RF0": {"tx_gain": 89.8}, "CHIRP": {"num_pulses": 10},
                                     "FILES": {"output_dir": "hil_scratch/u"}})
    with pytest.raises(configs.ConfigError, match="power budget"):
        configs.validate(cfg, fx)
    cfg["RF0"]["transmit"] = False         # RX only: no TX power
    configs.validate(cfg, fx)


# ---------------------------------------------------------------- expectations

def test_standard_run_has_no_expectation(base):
    assert expect.build(base) is None


def test_expect_mode_shifts_gain_and_delay(tmp_path, base, thresholds):
    fx = child(tmp_path, base, [LIM1, LNA, LIM2, PAD10], judge="expect")
    e = expect.build(fx)
    assert e.ref["fixture_id"] == BASE_ID
    net, tol = 20 - 0.5 - 0.5 - 10, 0.2 + 1.0 + 0.2 + 0.3
    a = e.adjust("t01", "loopback_peak_power_dbfs", "a_profile", checks(thresholds, "t01", "loopback_peak_power_dbfs")[0])
    assert not a.info and a.shift == pytest.approx(net) and a.extra_tol == pytest.approx(tol)
    abs_sep, base_sep = checks(thresholds, "t01", "peak_separation_samples")
    assert e.adjust("t01", "peak_separation_samples", None, abs_sep) == expect.NONE   # 'expected' uses the chain
    a = e.adjust("t01", "peak_separation_samples", None, base_sep)
    assert a.shift == pytest.approx(1e-9 * 56e6) and a.extra_tol == pytest.approx(0.2e-9 * 56e6)
    a = e.adjust("t04", "lo_leakage_dithered_rel_chirp_db", None,
                 checks(thresholds, "t04", "lo_leakage_dithered_rel_chirp_db")[0])
    assert a.shift == pytest.approx(-net)
    assert e.adjust("t00", "attenuator_check_peak_dbfs", None,
                    checks(thresholds, "t00", "attenuator_check_peak_dbfs")[0]).shift == pytest.approx(net)


def test_expect_mode_unpredictable_tags_are_info(tmp_path, base, thresholds):
    e = expect.build(child(tmp_path, base, [LIM1, LNA], judge="expect"))
    for test_id, name in (("t01", "noise_floor_dbfs"), ("t01", "single_pulse_snr_db"), ("t08", "p1db_tx_gain"),
                          ("t02", "pslr_minus_ideal_db"), ("t05", "plateau_onset_n")):
        a = e.adjust(test_id, name, None, checks(thresholds, test_id, name)[0])
        assert a.info, (test_id, name)
    for test_id, name in (("t01", "exit_status_ok"), ("t03", "kurtosis_i"), ("t01", "raw_max_abs_fs"),
                          ("t01", "leakage_peak_index")):
        assert e.adjust(test_id, name, None, checks(thresholds, test_id, name)[0]) == expect.NONE


def test_passive_change_shifts_snr_and_keeps_noise_judged(tmp_path, base, thresholds):
    e = expect.build(child(tmp_path, base, [PAD10], judge="expect"))
    assert e.adjust("t01", "single_pulse_snr_db", None,
                    checks(thresholds, "t01", "single_pulse_snr_db")[0]).shift == pytest.approx(-10)
    assert e.adjust("t01", "noise_floor_dbfs", None, checks(thresholds, "t01", "noise_floor_dbfs")[0]) == expect.NONE
    assert e.adjust("t02", "mainlobe_width_ratio", None,
                    checks(thresholds, "t02", "mainlobe_width_ratio")[0]) == expect.NONE


def test_info_mode_demotes_every_affected_metric(tmp_path, base, thresholds):
    e = expect.build(child(tmp_path, base, [PAD10]))
    assert e.mode == "info"
    assert e.adjust("t01", "loopback_peak_power_dbfs", None,
                    checks(thresholds, "t01", "loopback_peak_power_dbfs")[0]).info
    assert e.adjust("t01", "exit_status_ok", None, checks(thresholds, "t01", "exit_status_ok")[0]) == expect.NONE


def test_rf_base_gain_change_is_part_of_the_shift(tmp_path, base, thresholds):
    fx = child(tmp_path, base, [LIM1, LNA, LIM2, PAD10], judge="expect", rf_base={"tx_gain": 26.0})
    e = expect.build(fx)
    a = e.adjust("t01", "loopback_peak_power_dbfs", None, checks(thresholds, "t01", "loopback_peak_power_dbfs")[0])
    assert a.shift == pytest.approx(9 - 9)
    # the attenuator check keeps its own tx_gain, so it sees the full chain change
    assert e.atten_check_shift()[0] == pytest.approx(9)


def test_nonlinear_chain_makes_gain_shifts_info(tmp_path, base, thresholds):
    lna = {**LNA, "p1db_out_dbm": -63}       # output -61.7 dBm at tx_gain 35
    e = expect.build(child(tmp_path, base, [lna], judge="expect", safety={"rx_max_input_dbm": 10}))
    a = e.adjust("t01", "loopback_peak_power_dbfs", None, checks(thresholds, "t01", "loopback_peak_power_dbfs")[0])
    assert a.info and "not linear" in a.reason


def test_per_frequency_case_uses_its_frequency(tmp_path, base, thresholds):
    bpf = {"id": "bpf", "type": "filter", "loss_db": {"freq_mhz": [300, 360], "value": [1.0, 3.0]},
           "passband_mhz": [280, 380]}
    e = expect.build(child(tmp_path, base, [bpf], judge="expect"))
    chk = checks(thresholds, "t12", "loopback_power_dbfs")[0]
    assert e.adjust("t12", "loopback_power_dbfs", "350MHz", chk).shift == pytest.approx(-(1 + 2 * 50 / 60))
    assert e.adjust("t12", "loopback_power_dbfs", "600MHz", chk).info


def test_overrides(tmp_path, base, thresholds):
    fx = child(tmp_path, base, [LIM1, LNA], judge="expect",
               expect={"t01.noise_floor_dbfs": {"shift": 12, "tol": 3}, "t01.loopback_peak_power_dbfs": "info",
                       "t04.*": "judge", "t12.loopback_power_dbfs/350MHz": "info"})
    e = expect.build(fx)
    a = e.adjust("t01", "noise_floor_dbfs", None, checks(thresholds, "t01", "noise_floor_dbfs")[0])
    assert (a.info, a.shift, a.extra_tol) == (False, 12, 3)
    assert e.adjust("t01", "loopback_peak_power_dbfs", None,
                    checks(thresholds, "t01", "loopback_peak_power_dbfs")[0]).info
    assert e.adjust("t04", "spur_freqs_match", None, checks(thresholds, "t04", "spur_freqs_match")[0]) == expect.NONE
    chk = checks(thresholds, "t12", "loopback_power_dbfs")[0]
    assert e.adjust("t12", "loopback_power_dbfs", "350MHz", chk).info
    assert not e.adjust("t12", "loopback_power_dbfs", "330MHz", chk).info


def test_bad_override_key(tmp_path, base):
    with pytest.raises(fixtures.FixtureError, match="expect key"):
        expect.build(child(tmp_path, base, [PAD10], expect={"noise_floor_dbfs": "info"}))


def test_setup_note_demotes_base_checks_only(base, thresholds):
    e = expect.build(base, setup_note="10 m cable")
    assert e.adjust("t01", "loopback_peak_power_dbfs", None,
                    checks(thresholds, "t01", "loopback_peak_power_dbfs")[0]).info
    assert e.adjust("t01", "raw_max_abs_fs", None, checks(thresholds, "t01", "raw_max_abs_fs")[0]) == expect.NONE
    assert e.atten_check_shift() == (0.0, 0.0)


def test_summary_is_serialisable(tmp_path, base):
    s = expect.build(child(tmp_path, base, [LIM1, LNA, LIM2, PAD10], judge="expect")).summary()
    yamlio.dump(s, tmp_path / "s.yaml")
    assert s["changed_elements"] == ["lim1", "lna", "lim2", "pad10"] and s["gain_shift_db"] == pytest.approx(9)


# ---------------------------------------------------------------- judging

def test_evaluate_shift_and_extra_tol():
    chk = Check(kind="base", units="dB", bound={"type": "two_sided", "tol": 1.0})
    bl = {"mean": -20.0, "std": 0.0}
    assert evaluate(-11.0, chk, bl).status == "fail"
    ev = evaluate(-11.0, chk, bl, shift=9.0, extra_tol=0.5)
    assert ev.status == "pass" and (ev.lo, ev.hi) == (pytest.approx(-12.5), pytest.approx(-9.5))
    with pytest.raises(ValueError):
        evaluate(True, Check(kind="abs", units=None, bound={"type": "equals", "value": True}), shift=1.0)


def test_recorder_applies_and_logs_adjustments(tmp_path, base, thresholds):
    fx = child(tmp_path, base, [LIM1, LNA, LIM2, PAD10], judge="expect")
    bl = Baselines(BASE_ID, "fast")
    mean = bl.get("t01", "loopback_peak_power_dbfs")["mean"]
    rec = MetricRecorder("t01", thresholds, bl, expectation=expect.build(fx))
    [r] = rec.record("loopback_peak_power_dbfs", mean + 9.2)
    assert r["status"] == "pass" and r["expect"]["unadjusted_status"] == "fail"
    assert r["expect"]["shift"] == pytest.approx(9)
    [r] = rec.record("noise_floor_dbfs", -10.0)
    assert r["status"] == "info" and r["expect"]["demoted_to_info"]
    [r] = rec.record("exit_status_ok", False)
    assert r["status"] == "fail" and "expect" not in r
