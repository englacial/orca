import os
import time

import pytest

from tests.hil.orca_hil import scratch


def _fill(path, nbytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * nbytes)


def test_scratch_link(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    link = scratch.ensure_scratch_link(data, repo_root=repo)
    assert link.is_symlink() and link.resolve() == (data / "scratch").resolve()
    assert scratch.ensure_scratch_link(data, repo_root=repo) == link  # idempotent
    with pytest.raises(RuntimeError):
        scratch.ensure_scratch_link(tmp_path / "other", repo_root=repo)


def _make_case(root, name="a_profile"):
    case = root / "run" / "t01" / name
    for suffix in ("_rx_samps.bin", "_config.yaml", "_uhd_stdout.log"):
        _fill(case / f"20260101_000000{suffix}", 10)
    (case / "20260101_000000.zarr").mkdir()
    _fill(case / "20260101_000000.zarr" / "chunk", 10)
    _fill(case / "config.yaml", 5)
    _fill(case / "merged_rx_samps.bin", 10)
    return case


def test_release_on_pass_deletes_everything(tmp_path):
    case = _make_case(tmp_path / "scratch")
    assert scratch.release_case(case, True, tmp_path / "kept", "run", "t01") is None
    assert not case.exists() and not (tmp_path / "kept").exists()


def test_release_on_fail_keeps_raw_only(tmp_path):
    case = _make_case(tmp_path / "scratch")
    kept = scratch.release_case(case, False, tmp_path / "kept", "run", "t01")
    assert kept == tmp_path / "kept" / "run" / "t01" / "a_profile"
    assert sorted(p.name for p in kept.iterdir()) == [
        "20260101_000000_config.yaml", "20260101_000000_rx_samps.bin", "20260101_000000_uhd_stdout.log"]
    assert not case.exists()


def test_keep_raw_on_pass(tmp_path):
    case = _make_case(tmp_path / "scratch")
    assert scratch.release_case(case, True, tmp_path / "kept", "run", "t01", keep_raw=True).exists()


def test_quota_evicts_oldest_but_protects_current(tmp_path):
    kept = tmp_path / "kept"
    for run in ("20260101T000000Z_a", "20260102T000000Z_b", "20260103T000000Z_c"):
        _fill(kept / run / "t01" / "x.bin", 1000)
    evicted = scratch.enforce_quota(kept, 1500, protect="20260103T000000Z_c")
    assert [p.name for p in evicted] == ["20260101T000000Z_a", "20260102T000000Z_b"]
    assert [p.name for p in kept.iterdir()] == ["20260103T000000Z_c"]
    # The protected run stays even when it alone exceeds the quota
    assert scratch.enforce_quota(kept, 10, protect="20260103T000000Z_c") == []


def test_prune_runs(tmp_path):
    old, new = tmp_path / "runs" / "old", tmp_path / "runs" / "new"
    old.mkdir(parents=True)
    new.mkdir()
    now = time.time()
    os.utime(old, (now - 90 * 86400, now - 90 * 86400))
    assert scratch.prune_runs(tmp_path / "runs", 60, now=now) == [old]
    assert new.exists()
