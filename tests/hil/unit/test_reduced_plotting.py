import numpy as np
import xarray as xr

from tests.hil.orca_hil import plotting
from tests.hil.orca_hil.reduced import read_reduced, write_reduced


def test_reduced_roundtrip(tmp_path):
    ds = xr.Dataset({"noise_db": (("case", "n_stack"), np.random.default_rng(0).normal(size=(2, 3)),
                                  {"units": "dBFS"})},
                    coords={"case": ["a", "b"], "n_stack": ("n_stack", [1, 3, 10], {"units": "pulses"})})
    path = write_reduced(ds, tmp_path / "reduced.zarr.zip", test_id="t05", run_id="r", git_sha="abc",
                         fixture_id="fx", tier="fast", configs={"a": {"CHIRP": {"num_pulses": np.int64(5)}}})
    assert path.exists()
    back = read_reduced(path)
    xr.testing.assert_allclose(back["noise_db"], ds["noise_db"])
    assert back.attrs["test_id"] == "t05"
    assert back.attrs["configs"]["a"]["CHIRP"]["num_pulses"] == 5
    assert back["n_stack"].attrs["units"] == "pulses"


def test_summary_figure(tmp_path):
    fig, axes = plotting.summary_figure("t01", "Fig. 1c", "20260101T000000Z_abc1234_fast", "pass", 1, 2)
    x = np.arange(10)
    plotting.plot_current(axes[0, 0], x, x)
    plotting.plot_baseline(axes[0, 0], x, x + 1)
    plotting.shade_band(axes[0, 0], x, x - 1, x + 1)
    plotting.hband(axes[0, 1], 0, 1)
    out = plotting.save(fig, tmp_path / "summary.png")
    assert out.stat().st_size > 1000
    w, h = fig.get_size_inches() * fig.dpi
    assert (round(w), round(h)) == (1100, 660)
