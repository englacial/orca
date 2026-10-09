import pytest

from tests.hil.orca_hil.baselines import Baselines, aggregate


def test_aggregate():
    a = aggregate([1.0, 2.0, 3.0])
    assert a == {"mean": 2.0, "std": pytest.approx(1.0), "n": 3}
    assert aggregate([5.0]) == {"mean": 5.0, "std": 0.0, "n": 1}


def test_update_save_reload_keeps_comments(tmp_path):
    b = Baselines("fx", directory=tmp_path)
    assert not b.exists and b.get("t01", "x") is None
    b.update_scalar("t01", "loopback_peak_power_dbfs", [-31.2, -31.3], runs=["a", "b"])
    b.update_scalar("t05", "plateau_level_dbfs", [-100.0], runs=["a"], case="default")
    b.set("t03", "narrowband_lines", [{"freq": 1e6, "level_db": 12.0}])
    b.save(git_sha="abc1234")

    text = b.path.read_text()
    b.path.write_text(text.replace("t01:", "t01:\n  # re-baselined after adapter swap", 1))

    b2 = Baselines("fx", directory=tmp_path)
    assert b2.get("t01", "loopback_peak_power_dbfs")["n"] == 2
    assert b2.get("t05", "plateau_level_dbfs", case="default")["mean"] == -100.0
    assert b2.get("t05", "plateau_level_dbfs") is None
    assert b2.get_raw("t03", "narrowband_lines")[0]["freq"] == 1e6
    b2.update_scalar("t01", "noise_floor_dbfs", [-80.0], runs=["c"])
    b2.save(git_sha="def5678")
    out = b2.path.read_text()
    assert "# re-baselined after adapter swap" in out and "git_sha: def5678" in out
    assert b2.reference_path("t01").name == "t01_reference.zarr.zip"
    assert b2.reference_path("t01").parent.name == "fast"
    full = Baselines("fx", "full", directory=tmp_path)
    assert full.get("t01", "loopback_peak_power_dbfs") is None
    full.update_scalar("t01", "loopback_peak_power_dbfs", [-30.0], runs=["d"])
    full.save()
    assert Baselines("fx", directory=tmp_path).get("t01", "loopback_peak_power_dbfs")["n"] == 2
