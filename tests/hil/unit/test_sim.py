import os

import numpy as np
import pytest
import xarray as xr

from tests.hil.orca_hil import sim, yamlio
from tests.hil.orca_hil.analysis.timing import expected_slow_time, time_offset_increases

import processing_dask as pdk

L = 1400  # 25 us at 56 MSPS


def test_files_match_pipeline_format(radar_cfg, make_capture):
    prefix, zpath = make_capture(radar_cfg, n_pulses=300, noise_power=1e-4)
    assert os.path.getsize(prefix + "_rx_samps.bin") == 300 * L * 8
    cfg = yamlio.load(prefix + "_config.yaml")
    assert cfg["CHIRP"]["num_pulses"] == 300
    ds = xr.open_zarr(zpath)
    assert dict(ds.sizes) == {"sample_idx": L, "pulse_idx": 300}
    assert pdk.check_if_error_data_exists(ds) == "error_data_included"


def test_errors_logged_like_main_cpp(radar_cfg, make_capture):
    errors = [10, 11, 50]
    prefix, zpath = make_capture(radar_cfg, n_pulses=200, errors=errors, noise_power=1e-4)
    log = open(prefix + "_uhd_stdout.log").read()
    start, errs, attempted = pdk.process_stdout_log(log)
    assert start is not None
    assert sorted(errs) == errors and set(errs.values()) == {"ERROR_CODE_LATE_COMMAND"}
    assert attempted == 203
    assert time_offset_increases(log) == [(12, 0.0005), (13, 0.0005), (52, 0.0005)]
    ds = xr.open_zarr(zpath)
    assert ds.sizes["pulse_idx"] == 200
    assert pdk.check_if_error_data_exists(ds) == "error_data_not_included"


def test_presums_average_and_reduce_noise(radar_cfg, make_capture):
    _, z1 = make_capture(radar_cfg, n_pulses=400, noise_power=1.0, seed=3)
    _, z10 = make_capture(radar_cfg, n_pulses=4000, presums=10, noise_power=1.0, seed=3)
    d1, d10 = xr.open_zarr(z1), xr.open_zarr(z10)
    assert d10.sizes["pulse_idx"] == 400
    p1 = float((abs(d1.radar_data) ** 2).mean())
    p10 = float((abs(d10.radar_data) ** 2).mean())
    assert 10 * np.log10(p1 / p10) == pytest.approx(10.0, abs=0.2)


def test_dithering_decorrelates_undithered_artifacts(radar_cfg, make_capture):
    dc = 0.01
    _, zd = make_capture(radar_cfg, n_pulses=2000, lo_leakage=dc, dithering=True, seed=1)
    _, zn = make_capture(radar_cfg, n_pulses=2000, lo_leakage=dc, dithering=False, seed=1)
    mean_d = abs(complex(xr.open_zarr(zd).radar_data.mean()))
    mean_n = abs(complex(xr.open_zarr(zn).radar_data.mean()))
    assert mean_n == pytest.approx(dc, rel=1e-4)
    assert mean_d < dc / 10


def test_signal_unaffected_by_dithering(radar_cfg, make_capture):
    lb = {"delay": 173.25, "amp": 1e-2}
    _, zd = make_capture(radar_cfg, n_pulses=50, loopback=lb, dithering=True)
    _, zn = make_capture(radar_cfg, n_pulses=50, loopback=lb, dithering=False)
    np.testing.assert_allclose(xr.open_zarr(zd).radar_data.values, xr.open_zarr(zn).radar_data.values,
                               atol=1e-7)


def test_scheduled_times_match_slow_time_spec(radar_cfg, make_capture):
    errors = [5, 6, 30]
    for presums in (1, 10):
        prefix, zpath = make_capture(radar_cfg, n_pulses=100, presums=presums, errors=errors)
        log = open(prefix + "_uhd_stdout.log").read()
        cfg = yamlio.load(prefix + "_config.yaml")
        n_rec = 100 // presums
        t_all, _ = sim.scheduled_times(100, errors, cfg["CHIRP"]["pulse_rep_int"])
        good = np.array([s for s in range(len(t_all)) if s not in errors])
        np.testing.assert_allclose(expected_slow_time(cfg, log, n_rec), t_all[good[::presums]], atol=1e-12)


def test_desync_after_error_scrambles_signal_phase(radar_cfg, make_capture):
    lb = {"delay": 173.25, "amp": 1e-2}
    _, z = make_capture(radar_cfg, n_pulses=400, loopback=lb, errors=[199], desync_after_error=True)
    x = xr.open_zarr(z).radar_data.values  # (samples, records)
    before = abs(x[:, :199].mean(axis=1)).max()
    after = abs(x[:, 200:].mean(axis=1)).max()
    assert after < before / 5
