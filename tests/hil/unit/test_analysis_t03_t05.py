import numpy as np
import pytest

from tests.hil.orca_hil.analysis.noise import analyze_rx_only, find_lines, stack_sizes
from tests.hil.orca_hil.analysis.stacking import analyze_stacking
from tests.hil.orca_hil.configs import deep_merge


def by_name(ms):
    return {(m.name, m.case): m for m in ms}


@pytest.fixture
def rx_only_cfg(radar_cfg):
    return deep_merge(radar_cfg, {"RF0": {"transmit": False}})


def test_stack_sizes():
    assert stack_sizes(10000) == [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    assert stack_sizes(2000) == [1, 3, 10, 30, 100, 300, 1000, 2000]


def test_t03_gaussian_noise(rx_only_cfg, fixture_cfg, make_capture):
    _, z = make_capture(rx_only_cfg, n_pulses=2000, noise_power=1e-4, seed=11)
    reduced, ms = analyze_rx_only(z, fixture_cfg, {"baseline_lines": [], "loopback_index": 173.25})
    m = by_name(ms)
    assert m[("noise_power_dbfs", None)].value == pytest.approx(-40.0, abs=0.05)
    assert m[("kurtosis_i", None)].value == pytest.approx(3.0, abs=0.05)
    assert m[("kurtosis_q", None)].value == pytest.approx(3.0, abs=0.05)
    assert m[("impulse_fraction_6sigma", None)].value < 1e-6
    assert m[("new_narrowband_lines", None)].value == 0
    assert m[("tx_isolation_margin_db", None)].value < 3.0
    curve = reduced["noise_vs_n_db"].values
    n = reduced["n_stack"].values
    slope = np.polyfit(np.log10(n[2:]), curve[2:], 1)[0]
    assert slope == pytest.approx(-10.0, abs=0.5)


def test_t03_detects_impulses(rx_only_cfg, fixture_cfg, make_capture):
    noise = 1e-4
    amp = 20 * np.sqrt(noise / 2)
    _, z = make_capture(rx_only_cfg, n_pulses=1000, noise_power=noise, seed=12,
                        impulses={"rate_hz": 2e6, "amp": amp})
    _, ms = analyze_rx_only(z, fixture_cfg, {})
    m = by_name(ms)
    assert m[("impulse_fraction_6sigma", None)].value == pytest.approx(2e6 / 56e6, rel=0.1)
    assert m[("kurtosis_i", None)].value > 3.5
    assert m[("new_narrowband_lines", None)].inconclusive        # no baseline list
    assert m[("tx_isolation_margin_db", None)].inconclusive      # no loopback index


def test_t03_spur_lines(rx_only_cfg, fixture_cfg, make_capture):
    spur = {"freq": 3.1e6, "amp": 3e-3}
    _, z = make_capture(rx_only_cfg, n_pulses=500, noise_power=1e-4, spurs=[spur], seed=13)
    reduced, ms = analyze_rx_only(z, fixture_cfg, {"baseline_lines": []})
    m = by_name(ms)
    assert m[("new_narrowband_lines", None)].value == 1
    line = reduced.attrs["lines"][0]
    assert line["freq"] == pytest.approx(3.1e6, abs=50e3)
    _, ms2 = analyze_rx_only(z, fixture_cfg, {"baseline_lines": [{"freq": 3.12e6, "level_db": 20.0}]})
    assert by_name(ms2)[("new_narrowband_lines", None)].value == 0


def test_find_lines_groups_contiguous_bins():
    f = np.arange(100.0)
    psd = np.ones(100)
    psd[[10, 11, 12, 50]] = [50, 200, 60, 30]
    lines = find_lines(f, psd)
    assert [ln["freq"] for ln in lines] == [11.0, 50.0]


def _t05_cases(radar_cfg, make_capture, n=3000, dc=0.03):
    base = deep_merge(radar_cfg, {"CHIRP": {"rx_duration": 40e-6}})
    sig = {"leakage": {"delay": 159.0, "amp": 0.02}, "loopback": {"delay": 173.25, "amp": 0.1}}
    caps = {}
    for case, lo_sw, dither in (("default", 0.0, False), ("dither", 0.0, True),
                                ("lo_offset_dither", 12.5e6, True)):
        cfg = deep_merge(base, {"GENERATE": {"lo_offset_sw": lo_sw}})
        caps[case] = make_capture(cfg, n_pulses=n, noise_power=1.0, lo_leakage=dc, dithering=dither,
                                  seed=21, **sig)[1]
    return caps


def test_t05_slope_plateau_and_dither_gain(radar_cfg, fixture_cfg, make_capture):
    caps = _t05_cases(radar_cfg, make_capture)
    reduced, ms = analyze_stacking(caps, fixture_cfg)
    m = by_name(ms)
    for case in ("dither", "lo_offset_dither"):
        assert m[("slope_db_per_decade", case)].value == pytest.approx(-10.0, abs=0.3)
        assert abs(m[("deviation_from_theory_at_nmax_db", case)].value) < 1.0
        assert abs(m[("signal_peak_loss_at_nmax_db", case)].value) < 0.2
    assert ("slope_db_per_decade", "default") not in m
    assert m[("plateau_onset_n", "default")].value < 3000
    assert m[("dither_gain_at_nmax_db", None)].value > 10
    assert reduced["noise_mean_db"].dims == ("case", "n_stack")


def test_t05_pure_noise_has_no_plateau(radar_cfg, fixture_cfg, make_capture):
    caps = _t05_cases(radar_cfg, make_capture, dc=0.0)
    _, ms = analyze_stacking({"default": caps["default"]}, fixture_cfg)
    assert by_name(ms)[("plateau_onset_n", "default")].value == 3000
