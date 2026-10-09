import numpy as np
import pytest

from tests.hil.orca_hil.analysis.presum import analyze_presum
from tests.hil.orca_hil.analysis.recovery import analyze_recovery, error_records, jump_limit_sigma
from tests.hil.orca_hil.configs import deep_merge

SIG = {"leakage": {"delay": 159.0, "amp": 2e-3}, "loopback": {"delay": 173.25, "amp": 1e-2}}
ERRORS = [5000, 5001, 5002, 12000, 15000, 15001]


def by_name(ms):
    return {(m.name, m.case): m for m in ms}


@pytest.fixture
def t10_cfg(radar_cfg):
    return deep_merge(radar_cfg, {"CHIRP": {"rx_duration": 25e-6, "phase_dithering": True}})


def test_error_records():
    # P = 10: errors at scheduled 5000-5002 follow 5000 good pulses -> record 500 boundary
    assert error_records([5000, 5001, 5002], 10) == [(500, False)]
    assert error_records([12005], 10) == [(1200, True)]
    assert error_records([], 10) == []


def test_jump_limit_sigma():
    assert jump_limit_sigma(0) == 3.0
    assert jump_limit_sigma(10) == pytest.approx(3.89, abs=0.01)
    assert jump_limit_sigma(6479) == pytest.approx(5.25, abs=0.01)


def test_t10_continuous_through_errors(t10_cfg, fixture_cfg, make_capture):
    _, z = make_capture(t10_cfg, n_pulses=20000, presums=10, errors=ERRORS, noise_power=1e-4,
                        phase_noise_std=1e-3, seed=41, **SIG)
    reduced, ms = analyze_recovery(z, fixture_cfg)
    m = by_name(ms)
    assert m[("records_times_presums_equals_requested", None)].value is True
    assert m[("error_file_type_is_not_included", None)].value is True
    assert m[("time_offset_increases_logged", None)].value is True
    assert m[("max_gap_phase_jump_sigma", None)].value < 3.5
    assert abs(m[("record_power_drop_after_first_error_db", None)].value) < 0.5
    assert m[("stacked_power_change_after_errors_db", None)].value < 0.2
    assert list(reduced["gap_record"].values) == [500, 1199, 1499]


def test_t10_detects_desync(t10_cfg, fixture_cfg, make_capture):
    _, z = make_capture(t10_cfg, n_pulses=20000, presums=10, errors=ERRORS, noise_power=1e-4,
                        desync_after_error=True, seed=42, **SIG)
    _, ms = analyze_recovery(z, fixture_cfg)
    m = by_name(ms)
    assert m[("record_power_drop_after_first_error_db", None)].value > 5
    assert m[("stacked_power_change_after_errors_db", None)].value > 5


def test_t11_presum_equivalence(radar_cfg, fixture_cfg, make_capture):
    kw = dict(n_pulses=20000, noise_power=1e-4, seed=51, **SIG)
    _, z1 = make_capture(radar_cfg, presums=1, **kw)
    _, z100 = make_capture(radar_cfg, presums=100, **kw)
    reduced, ms = analyze_presum({"p1": z1, "p100": z100}, fixture_cfg)
    m = by_name(ms)
    assert m[("peak_power_diff_db", None)].value < 0.2
    assert m[("noise_power_diff_db", None)].value < 0.5
    assert m[("p100_noise_vs_p1_single_db", None)].value == pytest.approx(-20.0, abs=0.5)
    assert m[("profile_magnitude_correlation", None)].value > 0.999
    assert reduced.attrs["presums"] == 100
    assert m[("capture_gain_diff_db", None)].value == pytest.approx(0.0, abs=0.1)
