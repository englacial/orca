import numpy as np
import pytest

from tests.hil.orca_hil.analysis.coherence import (analyze_coherence, analyze_phase_stability,
                                                   coherence_ratios, detect_steps)
from tests.hil.orca_hil.analysis.freqresp import analyze_freq_response, smooth_fit
from tests.hil.orca_hil.configs import deep_merge

SIG = {"leakage": {"delay": 159.0, "amp": 2e-3}, "loopback": {"delay": 173.25, "amp": 1e-2}}
NOISE = 1e-4


def by_name(ms):
    return {(m.name, m.case): m for m in ms}


def pair(radar_cfg, make_capture, seed, **kw):
    """burst: 4000 pulses (1 s), presums 1; long: 40000 pulses (10 s), presums 40."""
    _, burst = make_capture(radar_cfg, n_pulses=4000, presums=1, noise_power=NOISE, seed=seed, **SIG, **kw)
    _, long = make_capture(radar_cfg, n_pulses=40000, presums=40, noise_power=NOISE, seed=seed + 1, **SIG, **kw)
    return burst, long


def test_coherence_ratios_noise_bias_removed():
    rng = np.random.default_rng(0)
    s2 = 0.5
    x = 1.0 + np.sqrt(s2 / 2) * (rng.standard_normal(20000) + 1j * rng.standard_normal(20000))
    r = coherence_ratios(x, np.arange(20000) * 1e-3, s2, T=1.0, record_s=1e-3)
    assert len(r) == 20
    assert r.mean() == pytest.approx(1.0, abs=0.02)


def test_t06_coherent_signal(radar_cfg, fixture_cfg, make_capture):
    burst, long = pair(radar_cfg, make_capture, 61, drift_rad_s=1e-3)
    reduced, ms = analyze_coherence(burst, long, fixture_cfg)
    m = by_name(ms)
    assert abs(m[("power_loss_db", "1s")].value) < 0.05
    assert m[("power_loss_db", "100s")].inconclusive
    assert m[("spread_1sigma_db", "100s")].inconclusive
    assert m[("errors_in_captures", None)].value == 0
    T = reduced["integration_time"].values
    assert T[0] == pytest.approx(250e-6)
    assert np.all(np.diff(T) > 0)
    assert np.all(np.abs(reduced["power_loss_db"].values) < 0.1)


def test_t06_detects_phase_noise(radar_cfg, fixture_cfg, make_capture):
    burst, long = pair(radar_cfg, make_capture, 63, phase_noise_std=0.02)
    _, ms = analyze_coherence(burst, long, fixture_cfg)
    assert by_name(ms)[("power_loss_db", "1s")].value > 0.5


def test_t07_drift_variance_and_errors(radar_cfg, fixture_cfg, make_capture):
    drift = np.radians(0.5)     # rad/s
    errors = [3000, 3001, 17000, 30000]
    _, long = make_capture(radar_cfg, n_pulses=40000, presums=40, noise_power=NOISE, seed=71,
                           drift_rad_s=drift, errors=errors, **SIG)
    _, burst = make_capture(radar_cfg, n_pulses=4000, presums=1, noise_power=NOISE, seed=72, **SIG)
    reduced, ms = analyze_phase_stability(long, burst, fixture_cfg)
    m = by_name(ms)
    assert m[("phase_drift_mdeg_per_s", None)].value == pytest.approx(np.degrees(drift) * 1e3, rel=0.01)
    assert m[("phase_var_to_theory_ratio", None)].value == pytest.approx(1.0, abs=0.15)
    assert m[("phase_steps_count", None)].value == 0
    assert m[("errors_in_captures", None)].value == 4
    assert reduced["phase_rad"].sizes["record"] == 1000


def test_detect_steps():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1e-3, 5000)
    r[2500:] += 0.02
    steps, sigma = detect_steps(r)
    assert list(steps) == [2499]
    assert sigma == pytest.approx(np.sqrt(2) * 1e-3, rel=0.1)


def test_smooth_fit_ripple():
    f = np.array([150, 250, 350, 450, 600, 750, 900]) * 1e6
    y = -40 - 0.5 * np.sqrt(f / 1e6)
    _, ripple = smooth_fit(f, y)
    assert ripple < 1e-9
    y2 = y.copy()
    y2[3] += 2.0
    assert smooth_fit(f, y2)[1] > 1.0


def test_t12_frequency_sweep(radar_cfg, fixture_cfg, make_capture):
    caps = {}
    for f in (150e6, 450e6, 900e6):
        g = 10 ** (-0.03 * np.sqrt(f / 1e6) / 20)
        sig = {"leakage": SIG["leakage"], "loopback": {"delay": 173.25, "amp": 1e-2 * g}}
        caps[f] = make_capture(deep_merge(radar_cfg, {"RF0": {"freq": f}}), n_pulses=500,
                               noise_power=NOISE, seed=int(f / 1e6), **sig)[1]
    reduced, ms = analyze_freq_response(caps, fixture_cfg)
    m = by_name(ms)
    loop = reduced["loopback_power_dbfs"].values
    assert loop[0] - loop[-1] == pytest.approx(0.03 * (np.sqrt(900) - np.sqrt(150)), abs=0.05)
    ratio = [m[("loopback_to_leakage_db", c)].value for c in ("150MHz", "900MHz")]
    assert ratio[0] - ratio[1] == pytest.approx(loop[0] - loop[-1], abs=0.1)
    assert m[("ripple_db", None)].value < 0.05
