import math

from tests.hil.orca_hil import yamlio
from tests.hil.orca_hil.baseline_update import child_args, collect_base_values, majority_lines, update_baselines


def test_child_args():
    args = ["tests/hil", "--hil", "--update-baseline", "--baseline-repeats", "5", "--tier", "fast"]
    assert child_args(args) == ["tests/hil", "--hil", "--update-baseline", "--tier", "fast"]
    assert child_args(["--baseline-repeats=3", "--hil"]) == ["--hil"]


def test_majority_lines():
    a = [{"freq": 6.0e6, "level_db": 20.0}, {"freq": 1.0e6, "level_db": 12.0}]
    b = [{"freq": 6.01e6, "level_db": 22.0}]
    c = [{"freq": 5.99e6, "level_db": 21.0}, {"freq": 3.0e6, "level_db": 15.0}]
    out = majority_lines([a, b, c])
    assert len(out) == 1
    assert out[0]["level_db"] == 21.0


def _run(tmp_path, name, metrics, tid="t01"):
    rd = tmp_path / name
    (rd / tid).mkdir(parents=True)
    yamlio.dump({"test_id": tid, "metrics": metrics}, rd / tid / "metrics.yaml")
    return rd


def test_collect_and_update(tmp_path):
    def m(v, status="info", kind="base", case=None):
        r = {"name": "loopback_peak_power_dbfs", "value": v, "kind": kind, "status": status}
        if case:
            r["case"] = case
        return r
    runs = [_run(tmp_path, f"r{i}", [m(v), m(1.0, kind="abs", status="pass"), m(v + 1, case="x")])
            for i, v in enumerate((-52.0, -52.2, -51.8))]
    runs.append(_run(tmp_path, "r3", [m(math.nan), m(-60.0, status="inconclusive")]))
    vals = collect_base_values(runs)
    assert sorted(v for v, _ in vals["t01"][("loopback_peak_power_dbfs", None)]) == [-52.2, -52.0, -51.8]
    b, _ = update_baselines("fx", runs, "abc", directory=tmp_path / "baselines")
    got = b.get("t01", "loopback_peak_power_dbfs")
    assert abs(got["mean"] + 52.0) < 1e-9 and abs(got["std"] - 0.2) < 1e-9 and got["n"] == 3
    assert b.get("t01", "loopback_peak_power_dbfs", case="x")["n"] == 3
    assert (tmp_path / "baselines" / "fx.yaml").exists()
