import math
from types import SimpleNamespace

from tests.hil.orca_hil import golden


def test_compare():
    exp = [{"name": "a", "case": None, "value": 1.0}, {"name": "b", "case": "x", "value": float("nan")},
           {"name": "c", "case": None, "value": True}, {"name": "d", "case": None, "value": 2.0}]
    same = [dict(m) for m in exp]
    same[0]["value"] = 1.0 + 5e-7
    assert golden.compare("t01", exp, same) == []
    off = [dict(m) for m in exp[:3]] + [{"name": "e", "case": None, "value": 0}]
    off[0]["value"] = 1.01
    off[2]["value"] = False
    diffs = golden.compare("t01", exp, off)
    assert {d.split(":")[0] for d in diffs} == {"a", "c", "d", "e"}
    assert golden.compare("t01", exp[:1], off[:1], {"t01.a": {"abs": 0.1}}) == []


def test_package_prepare_roundtrip(tmp_path, radar_cfg, make_capture):
    prefix, z = make_capture(radar_cfg, n_pulses=50, noise_power=1e-6, seed=1,
                             loopback={"delay": 173.25, "amp": 1e-3})
    cap = SimpleNamespace(prefix=prefix, case="a_profile")
    tar = golden.package("t01", [cap], tmp_path)
    assert golden.sha256(tar) == golden.sha256(tar)
    zarrs = golden.prepare(tmp_path, "t01")
    assert list(zarrs) == ["a_profile"]
    import xarray as xr
    a, b = xr.open_zarr(z), xr.open_zarr(zarrs["a_profile"])
    assert math.isclose(float(abs(a["radar_data"]).sum()), float(abs(b["radar_data"]).sum()))
