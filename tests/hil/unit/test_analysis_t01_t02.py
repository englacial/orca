import numpy as np
import pytest

from tests.hil.orca_hil.analysis.profile import analyze_profile
from tests.hil.orca_hil.analysis.sidelobes import analyze_sidelobes
from tests.hil.orca_hil.configs import deep_merge

from .conftest import chirp_energy


def by_name(ms):
    return {(m.name, m.case): m for m in ms}


def test_t01_recovers_injected_profile(radar_cfg, fixture_cfg, make_capture):
    a_l, a_b, noise = 3e-4 * np.exp(0.3j), 1e-3 * np.exp(1.1j), 1e-6
    _, z = make_capture(radar_cfg, n_pulses=2000, noise_power=noise, seed=7,
                        leakage={"delay": 159.0, "amp": a_l}, loopback={"delay": 173.25, "amp": a_b})
    reduced, ms = analyze_profile(z, fixture_cfg)
    m = by_name(ms)
    snr_true = 10 * np.log10(abs(a_b) ** 2 / (noise / chirp_energy(radar_cfg)))

    assert m[("leakage_peak_index", None)].value == pytest.approx(159.0, abs=0.05)
    assert m[("peak_separation_samples", None)].value == pytest.approx(14.25, abs=0.05)
    assert m[("peak_separation_samples", None)].params["expected"] == pytest.approx(14.15, abs=0.01)
    assert m[("loopback_to_leakage_ratio_db", None)].value == pytest.approx(
        20 * np.log10(abs(a_b / a_l)), abs=0.3)
    assert m[("loopback_peak_power_dbfs", None)].value == pytest.approx(-60.0, abs=0.1)
    assert m[("single_pulse_snr_db", None)].value == pytest.approx(snr_true, abs=0.5)
    assert m[("noise_floor_dbfs", None)].value == pytest.approx(
        10 * np.log10(noise / chirp_energy(radar_cfg)), abs=0.2)
    width = m[("mainlobe_3db_width_samples", None)]
    assert width.value == pytest.approx(width.params["ideal"], rel=0.02)
    assert reduced["profile_power_db"].dims == ("sample",)


@pytest.mark.parametrize("window", ["rectangular", "hamming", "kaiser10"])
def test_t02_noiseless_matches_ideal(radar_cfg, fixture_cfg, make_capture, window):
    cfg = deep_merge(radar_cfg, {"GENERATE": {"window": window}})
    _, z = make_capture(cfg, n_pulses=20, seed=2,
                        leakage={"delay": 159.0, "amp": 1e-3}, loopback={"delay": 173.25, "amp": 1e-2})
    reduced, ms = analyze_sidelobes({window: z}, fixture_cfg)
    m = by_name(ms)
    assert m[("pslr_minus_ideal_db", window)].value == pytest.approx(0.0, abs=0.1)
    assert m[("islr_minus_ideal_db", window)].value == pytest.approx(0.0, abs=0.1)
    assert m[("mainlobe_width_ratio", window)].value == pytest.approx(1.0, abs=0.01)
    assert m[("sidelobe_extent_minus_ideal_samples", window)].value == pytest.approx(0.0, abs=1.0)
    assert m[("first_echo_db", window)].value < -60
    assert list(reduced["window"].values) == [window]


def test_t02_rectangular_pslr_is_about_13db(radar_cfg, fixture_cfg, make_capture):
    cfg = deep_merge(radar_cfg, {"GENERATE": {"window": "rectangular"}})
    _, z = make_capture(cfg, n_pulses=20, loopback={"delay": 173.25, "amp": 1e-2},
                        leakage={"delay": 159.0, "amp": 1e-3})
    reduced, _ = analyze_sidelobes({"rectangular": z}, fixture_cfg)
    assert float(reduced["pslr_db"][0]) == pytest.approx(-13.3, abs=1.0)
