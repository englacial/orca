import numpy as np
import pytest

from tests.hil.orca_hil.analysis.duty import analyze_duty, duty_to_pri, knee
from tests.hil.orca_hil.analysis.gain import analyze_gain, sweep_fit
from tests.hil.orca_hil.analysis.spectral import analyze_spectral
from tests.hil.orca_hil.configs import deep_merge

SIG = {"leakage": {"delay": 159.0, "amp": 2e-3}, "loopback": {"delay": 173.25, "amp": 1e-2}}


def by_name(ms):
    return {(m.name, m.case): m for m in ms}


@pytest.fixture
def t04_captures(radar_cfg, make_capture):
    dc, spur = 1e-3, {"freq": 5.0e6, "amp": 1e-3}
    caps = {}
    for case, lo_sw, dither in (("a", 0.0, False), ("b", 0.0, True), ("c", 12.5e6, True)):
        cfg = deep_merge(radar_cfg, {"GENERATE": {"lo_offset_sw": lo_sw}})
        caps[case] = make_capture(cfg, n_pulses=2000, noise_power=1e-6, lo_leakage=dc, spurs=[spur],
                                  dithering=dither, seed=31, **SIG)[1]
    return caps


def test_t04_leakage_spurs_and_occupancy(t04_captures, fixture_cfg):
    first, _ = analyze_spectral({"a": t04_captures["a"]}, fixture_cfg)
    baseline = [dict(s, freq=s["freq"] + 20e3) for s in first.attrs["spurs"]["a"]]
    reduced, ms = analyze_spectral(t04_captures, fixture_cfg, {"baseline_spurs": baseline})
    m = by_name(ms)
    assert m[("lo_leakage_suppression_by_dither_db", None)].value > 20
    assert m[("lo_leakage_dithered_rel_chirp_db", "b")].value < m[("lo_leakage_rel_chirp_db", "a")].value - 20
    assert m[("spur_freqs_match", "a")].value is True
    assert m[("new_spurs_above_baseline", "a")].value == 0
    # Spurs near the detection threshold may come and go without failing either check
    weak = {"freq": 17e6, "level_db": 12.0}
    _, ms = analyze_spectral(t04_captures, fixture_cfg, {"baseline_spurs": baseline + [weak]})
    assert by_name(ms)[("spur_freqs_match", "a")].value is True
    _, ms = analyze_spectral(t04_captures, fixture_cfg, {"baseline_spurs": baseline[1:]})
    assert by_name(ms)[("new_spurs_above_baseline", "a")].value == 1
    assert m[("occupied_bw_ratio", "c")].value == pytest.approx(1.0, abs=0.05)
    assert m[("center_offset_error_mhz", "c")].value < 0.5
    spur_freqs = [s["freq"] for s in reduced.attrs["spurs"]["a"]]
    assert len(spur_freqs) == 1 and abs(spur_freqs[0] - 5e6) < 50e3
    assert reduced["spectrogram_db"].dims == ("case", "stft_freq", "stft_time")


def test_t04_without_baseline_is_inconclusive(t04_captures, fixture_cfg):
    _, ms = analyze_spectral({"a": t04_captures["a"]}, fixture_cfg)
    m = by_name(ms)
    assert m[("spur_freqs_match", "a")].inconclusive
    assert m[("new_spurs_above_baseline", "a")].inconclusive


def test_sweep_fit_linear_and_compressed():
    g = np.arange(5, 76, 3.0)
    lin = g - 90.0
    level = lin - np.clip(g - 60, 0, None) ** 2 / 20   # compresses above 60 dB
    noise = np.full_like(g, -100.0)
    slope, p1db, used = sweep_fit(g, level, noise)
    assert slope == pytest.approx(1.0, abs=0.01)
    assert p1db == pytest.approx(64.5, abs=0.5)
    assert used[g > p1db - 10].sum() == 0
    slope2, p1db2, _ = sweep_fit(g[:10], lin[:10], noise[:10])
    assert slope2 == pytest.approx(1.0) and p1db2 is None
    # Drops below the line with a small raw signal are not compression when min_raw is set
    _, p1db3, _ = sweep_fit(g, level, noise, raw_max=np.full_like(g, 0.05), min_raw=0.5)
    assert p1db3 is None


def test_t08_tx_sweep_on_sim(radar_cfg, fixture_cfg, make_capture):
    points = []
    for g in (23, 29, 35, 41, 47):
        amp = 1e-3 * 10 ** ((g - 35) / 20)
        _, z = make_capture(radar_cfg, n_pulses=300, noise_power=1e-6, seed=g,
                            leakage={"delay": 159.0, "amp": amp / 5}, loopback={"delay": 173.25, "amp": amp})
        points.append({"sweep": "tx", "gain": g, "capture": z})
    reduced, ms = analyze_gain(points, fixture_cfg, {"base_tx_gain": 35, "base_rx_gain": 40})
    m = by_name(ms)
    assert m[("tx_sweep_slope_db_per_db", None)].value == pytest.approx(1.0, abs=0.02)
    assert m[("dbfs_at_base_gain", None)].value == pytest.approx(-60.0, abs=0.1)
    assert ("p1db_tx_gain", None) not in m          # full tier only
    assert list(reduced["gain"].values) == [23, 29, 35, 41, 47]
    _, ms = analyze_gain(points, fixture_cfg, {"base_tx_gain": 35, "base_rx_gain": 40, "full": True})
    m = by_name(ms)
    assert ("p1db_tx_gain", None) not in m and m[("linear_to_tx_gain", None)].value == 47
    assert m[("linear_range_db", None)].value > 30


def test_knee():
    assert knee([0.5, 0.6, 0.7, 0.8], [0.0, 0.01, 0.03, 0.001]) == 0.6
    assert np.isnan(knee([0.5], [0.5]))


def test_t09_duty_points_from_logs(radar_cfg, make_capture):
    base = deep_merge(radar_cfg, {"CHIRP": {"rx_duration": 100e-6, "tx_duration": 10e-6}})
    points = []
    for duty, errs in ((0.5, []), (0.75, list(range(100, 130))), (0.9, list(range(50, 250)))):
        cfg = deep_merge(base, {"CHIRP": {"pulse_rep_int": duty_to_pri(base, duty)}})
        prefix, _ = make_capture(cfg, n_pulses=1000, presums=10, errors=errs, noise_power=1e-6)
        points.append({"duty": duty, "log": open(prefix + "_uhd_stdout.log").read(),
                       "config": cfg | {"CHIRP": cfg["CHIRP"] | {"num_pulses": 1000}}})
    reduced, ms = analyze_duty(points, {"report_at": {0.5: "error_rate_at_50pct", 0.75: "error_rate_at_75pct"},
                                        "full": True})
    m = by_name(ms)
    assert m[("error_rate_at_50pct", None)].value == 0
    assert m[("error_rate_at_75pct", None)].value == pytest.approx(30 / 1030)
    assert m[("records_written_equals_requested", "duty_90")].value is True
    assert m[("max_duty_with_error_rate_below_2pct", None)].value == 0.5
    assert m[("process_killed", None)].value is False
    assert reduced["pri"].values[0] == pytest.approx(200e-6)


def test_t09_killed_point_without_summary(radar_cfg):
    from tests.hil.orca_hil.analysis.duty import analyze_duty
    cfg = {**radar_cfg, "CHIRP": {**radar_cfg["CHIRP"], "num_pulses": 1000}}
    reduced, ms = analyze_duty([{"duty": 1.0, "log": "", "config": cfg, "killed": True}])
    m = {(x.name, x.case): x for x in ms}
    assert m[("process_killed", None)].value is True
    assert np.isnan(reduced["attempted"].values[0])
