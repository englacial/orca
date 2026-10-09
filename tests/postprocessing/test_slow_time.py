"""Specification for slow_time in processing_dask.save_radar_data_to_zarr.

slow_time[k] must be the scheduled time of the first error-free pulse in record
k, relative to pulse 0, including the time_offset increases logged after errors
(see tests/hil/orca_hil/analysis/timing.py). The current implementation uses
np.linspace, which ignores errors and has an endpoint off-by-one, so these are
strict xfails until it is fixed.
"""
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from tests.hil.orca_hil import configs, sim, yamlio
from tests.hil.orca_hil.analysis.timing import expected_slow_time

import processing_dask as pdk

XFAIL = pytest.mark.xfail(strict=True, reason="slow_time ignores errors and the linspace endpoint")


@pytest.fixture
def base_config():
    fixture = {"rf_base": {"tx_gain": 30, "rx_gain": 36}}
    return configs.merged_config(fixture, {"FILES": {"output_dir": "data"}})


@pytest.mark.parametrize("presums,errors", [
    pytest.param(1, [], id="no_errors_p1", marks=XFAIL),
    pytest.param(10, [], id="no_errors_p10", marks=XFAIL),
    pytest.param(1, [7, 8, 40, 41, 42], id="errors_p1", marks=XFAIL),
    pytest.param(10, [7, 8, 40, 41, 42], id="errors_p10", marks=XFAIL),
])
def test_slow_time_matches_schedule(tmp_path, base_config, presums, errors):
    n_pulses = 200
    prefix = sim.simulate_capture(base_config, out_dir=tmp_path, n_pulses=n_pulses, presums=presums,
                                  errors=errors, noise_power=1e-6, timestamp=datetime(2026, 1, 1))
    ds = xr.open_zarr(pdk.save_radar_data_to_zarr(prefix))
    cfg = yamlio.load(prefix + "_config.yaml")
    log = open(prefix + "_uhd_stdout.log").read()
    expected = expected_slow_time(cfg, log, ds.sizes["pulse_idx"])
    np.testing.assert_allclose(ds["slow_time"].values, expected, rtol=0, atol=1e-9)
