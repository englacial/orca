"""Watchdog status checks and power-amplifier fixture rules."""
import copy
import json
import subprocess
import time

import pytest

from tests.hil.orca_hil import configs, expect, fixtures, watchdog, yamlio

BASE_ID = "b206mini_pi5_loopback_50m"


def write_status(path, **kw):
    doc = {"state": "armed", "updated_unix": time.time(), "expires_unix": time.time() + 600,
           "powered": True, "reason": None}
    doc.update(kw)
    path.write_text(json.dumps(doc))


@pytest.fixture
def spec(tmp_path):
    return {"status_file": str(tmp_path / "guard.json"), "max_age_s": 1.0}


def test_armed_status_passes(spec, tmp_path):
    write_status(tmp_path / "guard.json")
    assert watchdog.problem(spec) is None
    watchdog.check([("pa", spec)], min_remaining_s=300)


@pytest.mark.parametrize("kw, match", [
    ({"state": "tripped", "reason": "TMP117 60 C"}, "tripped"),
    ({"updated_unix": time.time() - 10}, "old"),
    ({"powered": False}, "unpowered"),
    ({"expires_unix": time.time() + 100}, "expires"),
])
def test_status_problems(spec, tmp_path, kw, match):
    write_status(tmp_path / "guard.json", **kw)
    with pytest.raises(watchdog.WatchdogError, match=match):
        watchdog.check([("pa", spec)], min_remaining_s=300)


def test_missing_status_file(spec):
    assert "guard.json" in watchdog.problem(spec)


def test_monitor_stops_process_when_status_goes_stale(spec, tmp_path):
    write_status(tmp_path / "guard.json")
    proc = subprocess.Popen(["sleep", "30"])
    mon = watchdog.Monitor([("pa", spec)], proc)
    try:
        proc.wait(timeout=10)       # status is never refreshed, so it goes stale after 1 s
    finally:
        mon.close()
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode is not None and "old" in mon.failure


def test_monitor_leaves_healthy_run_alone(spec, tmp_path):
    proc = subprocess.Popen(["sleep", "1.5"])
    mon = watchdog.Monitor([("pa", spec)], proc)
    t_end = time.time() + 3
    while proc.poll() is None and time.time() < t_end:
        write_status(tmp_path / "guard.json")
        time.sleep(0.2)
    mon.close()
    assert proc.returncode == 0 and mon.failure is None


# ---------------------------------------------------------------- power-amplifier fixtures

def pa_element(spec, **kw):
    el = {"id": "pa", "type": "amplifier", "role": "pa", "gain_db": 26.5, "max_gain_db": 57.3,
          "max_input_dbm": 10, "max_output_dbm": 10, "startup_burst_dbm": 43.4, "max_enable_duty": 0.15,
          "powered": True, "watchdog": spec,
          "radar_config": {"GPIO": {"pwr_amp_pin": "2", "pwr_amp_active_low": True},
                           "GENERATE": {"pulse_length": 18e-6}, "CHIRP": {"tx_duration": 18e-6, "tx_lead": 1.5e-6}}}
    el.update(kw)
    return el


def pa_fixture(tmp_path, spec, pad_rating=100, **pa_kw):
    base = fixtures.load(fixtures.resolve_path(BASE_ID))
    chain = [pa_element(spec, **pa_kw), {"id": "pad40", "type": "attenuator", "loss_db": 40,
                                         **({"power_rating_w": pad_rating} if pad_rating else {})}]
    chain += copy.deepcopy(base["hardware"]["chain"])
    chain[2]["power_rating_w"] = 2
    doc = {"fixture_id": "pa", "extends": BASE_ID, "judge": "expect", "rf_base": {"tx_gain": 10.0},
           "safety": {"attenuator_check": {"tx_gain": 0.0}}, "hardware": {"chain": chain}}
    yamlio.dump(doc, tmp_path / "pa.yaml")
    return fixtures.load(tmp_path / "pa.yaml")


def test_pa_needs_watchdog_and_rated_parts_after_it(tmp_path, spec):
    with pytest.raises(fixtures.FixtureError, match="needs max_input_dbm or power_rating_w"):
        pa_fixture(tmp_path, spec, pad_rating=None)
    doc = yamlio.load(tmp_path / "pa.yaml")
    del doc["hardware"]["chain"][0]["watchdog"]
    doc["hardware"]["chain"][1]["power_rating_w"] = 100
    yamlio.dump(doc, tmp_path / "pa2.yaml")
    with pytest.raises(fixtures.FixtureError, match="needs a watchdog"):
        fixtures.load(tmp_path / "pa2.yaml")


def test_pa_budget_uses_max_gain_and_output_limit(tmp_path, spec):
    fx = pa_fixture(tmp_path, spec)
    st = {s["id"]: s for s in fixtures.power_budget(fx, 10, 330e6)}
    assert st["pa"]["output_dbm"] == pytest.approx(-55 + 57.3)      # max_gain_db, not gain_db
    # output limit 10 dBm - 3 dB margin: tx_gain <= 65 + 7 - 57.3 = 14.7
    assert fixtures.max_safe_tx_gain(fx, 330e6) == pytest.approx(14.5)
    assert any("approved 10 dBm" in e for e in fixtures.budget_problems(fx, 15, 330e6)[0])


def test_startup_burst_reaches_the_pads(tmp_path, spec):
    fx = pa_fixture(tmp_path, spec, pad_rating=10)   # 40 dBm rating < 43.4 dBm burst + margin
    errors, _ = fixtures.budget_problems(fx, 0, 330e6)
    assert any(e.startswith("pad40 start-up burst input 43.4") for e in errors)
    fx = pa_fixture(tmp_path, spec)                 # 100 W pad first; the 2 W pad30 sees 3.4 dBm
    assert fixtures.budget_problems(fx, 0, 330e6)[0] == []


def test_radar_config_is_applied_last_and_duty_is_limited(tmp_path, spec):
    fx = pa_fixture(tmp_path, spec)
    cfg = configs.merged_config(fx, {"CHIRP": {"tx_duration": 10e-6, "num_pulses": 10},
                                     "FILES": {"output_dir": "hil_scratch/u"}})
    assert cfg["GPIO"]["pwr_amp_pin"] == "2" and cfg["CHIRP"]["tx_duration"] == 18e-6
    configs.validate(cfg, fx)                                     # 18 us / 250 us = 7.2 %
    cfg["CHIRP"]["pulse_rep_int"] = 100e-6
    with pytest.raises(configs.ConfigError, match="enable duty 0.180"):
        configs.validate(cfg, fx)
    t9 = configs.merged_config(fx, {"CHIRP": {"rx_duration": 100e-6}})
    assert fixtures.max_test_duty(fx, t9) == pytest.approx(0.15 * 100 / 18)


def test_tx_lead_shifts_absolute_indices(tmp_path, spec):
    fx = pa_fixture(tmp_path, spec)
    e = expect.build(fx)
    assert e.timing_shift_samples() == pytest.approx(-1.5e-6 * 56e6)
    from tests.hil.orca_hil.thresholds import load_thresholds
    chk = load_thresholds()["t01"]["leakage_peak_index"][0]
    assert e.adjust("t01", "leakage_peak_index", None, chk).shift == pytest.approx(-84)
