import csv
import subprocess

import pytest

from tests.hil.orca_hil import yamlio
from tests.hil.orca_hil.publish import HISTORY_COLUMNS, history_rows, publish
from tests.hil.orca_hil.report.build_site import build

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")


def make_run(root, run_id, tier, status, started, value=1.0, dirty=False):
    rd = root / run_id
    (rd / "t01").mkdir(parents=True)
    yamlio.dump({"run_id": run_id, "tier": tier, "fixture_id": "fx", "started_utc": started,
                 "ended_utc": started, "git_sha": "a" * 40, "git_branch": "b", "dirty": dirty,
                 "status": status, "fixture_suspect": False, "tests": {"t01": status},
                 "session_metrics": [{"name": "fast_tier_duration_s", "value": 400.0, "units": "s",
                                      "kind": "abs", "bound": {"type": "max", "value": 600}, "status": "pass"}]},
                rd / "run.yaml")
    yamlio.dump({"uhd_build_stamp": {"commit": "c" * 40, "dirty": False}}, rd / "configuration.yaml")
    yamlio.dump({"test_id": "t01", "status": status, "fixture_suspect": False, "duration_s": 3.0, "metrics": [
        {"name": "snr_db", "value": value, "units": "dB", "kind": "base", "bound": {"type": "lower", "tol": 2},
         "bound_lo": 0.5, "bound_hi": None, "baseline": {"mean": 2.0, "std": 0.1, "n": 3}, "status": status},
        {"name": "ok", "case": "a", "value": True, "units": "bool", "kind": "abs",
         "bound": {"type": "equals", "value": True}, "status": "pass"},
    ]}, rd / "t01" / "metrics.yaml")
    (rd / "t01" / "summary.png").write_bytes(PNG)
    (rd / "t01" / "reduced.zarr.zip").write_bytes(b"zip")
    return rd


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "results"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.name", "t"], check=True)
    return r


def test_history_rows(tmp_path):
    rows = history_rows(make_run(tmp_path, "r1", "fast", "pass", "2026-10-13T10:00:00Z"))
    assert [r["metric"] for r in rows] == ["snr_db", "ok", "fast_tier_duration_s"]
    assert rows[0]["baseline_mean"] == 2.0 and rows[0]["uhd_commit"] == "c" * 40
    assert rows[1]["bound_lo"] is True and rows[1]["bound_type"] == "equals"
    assert rows[2]["test_id"] == "session"


def test_publish_rules_and_append_only(tmp_path, repo):
    runs = tmp_path / "runs"
    fast_pass = make_run(runs, "r1_fast", "fast", "pass", "2026-10-13T10:00:00Z")
    fast_fail = make_run(runs, "r2_fast", "fast", "fail", "2026-10-13T11:00:00Z")
    full = make_run(runs, "r3_full", "full", "pass", "2026-10-13T12:00:00Z")
    assert publish(fast_pass, repo, push=False) == "hil: r1_fast fast pass"
    for rd in (fast_fail, full):
        publish(rd, repo, push=False)
    assert publish(fast_pass, repo, push=False) is None
    assert not (repo / "runs/r1_fast/t01/summary.png").exists()
    assert (repo / "runs/r2_fast/t01/summary.png").exists()
    assert not (repo / "runs/r2_fast/t01/reduced.zarr.zip").exists()
    assert (repo / "runs/r3_full/t01/reduced.zarr.zip").exists()
    with open(repo / "history/history-2026.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HISTORY_COLUMNS and len(rows) == 1 + 3 * 3
    log = subprocess.run(["git", "-C", str(repo), "log", "--format=%s"], capture_output=True, text=True).stdout
    assert log.split("\n")[0] == "hil: r3_full full pass"


def test_build_site(tmp_path, repo):
    runs = tmp_path / "runs"
    for i, (tier, status) in enumerate([("fast", "pass"), ("fast", "fail"), ("full", "pass")]):
        publish(make_run(runs, f"r{i}_{tier}", tier, status, f"2026-10-1{i + 1}T10:00:00Z", value=1.0 + i,
                         dirty=i == 1), repo, push=False)
    out = build(repo, tmp_path / "site")
    index = (out / "index.html").read_text()
    assert "r2_full" in index and 'href="runs/r1_fast.html#t01"' in index
    assert (out / "tests/t01.html").exists() and (out / "trends/fast/t01/snr_db.png").exists()
    run_page = (out / "runs/r2_full.html").read_text()
    assert "snr_db" in run_page and "r2_full/t01/summary.png" in run_page
    assert (out / "runs/r2_full/t01/summary.png").exists()
