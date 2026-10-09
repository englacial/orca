"""Shared fixtures for software-only HIL harness tests (no hardware, no UHD)."""
import itertools
from datetime import datetime, timedelta

import numpy as np
import pytest

from tests.hil.orca_hil import HIL_DIR, configs, sim, yamlio

import processing_dask as pdk

FIXTURE_FILE = HIL_DIR / "fixtures" / "b206mini_pi5_loopback_50m.yaml"
_stamps = itertools.count()


@pytest.fixture(scope="session")
def fixture_cfg():
    return yamlio.load(FIXTURE_FILE)


@pytest.fixture(scope="session")
def radar_cfg(fixture_cfg):
    """Base radar config with gains filled in (the fixture may not be commissioned)."""
    return configs.merged_config(fixture_cfg, {"RF0": {"tx_gain": 30, "rx_gain": 36},
                                               "FILES": {"output_dir": "hil_scratch/unit"}})


@pytest.fixture
def make_capture(tmp_path):
    """simulate_capture + save_radar_data_to_zarr; returns (prefix, zarr_path)."""
    def _make(config, **kwargs):
        i = next(_stamps)
        out = tmp_path / f"cap{i}"
        stamp = datetime(2026, 1, 1) + timedelta(seconds=i)
        prefix = sim.simulate_capture(config, out_dir=out, timestamp=stamp, **kwargs)
        return prefix, pdk.save_radar_data_to_zarr(prefix)
    return _make


def chirp_energy(config):
    from tests.hil.orca_hil.analysis.common import reference_chirp
    return float(np.sum(np.abs(reference_chirp(config)) ** 2))
