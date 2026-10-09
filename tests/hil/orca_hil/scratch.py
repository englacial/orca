"""Scratch space, raw-data retention, and the kept-on-failure quota.

main.cpp needs FILES.output_dir to be repo-relative, so captures go under the
gitignored ``<repo>/hil_scratch`` symlink, which points at ``<data_root>/scratch``.
"""
import re
import shutil
import time
from pathlib import Path

from . import REPO_ROOT

SCRATCH_LINK_NAME = "hil_scratch"
# save_data prefix files: <YYYYMMDD_HHMMSS>[_p<N>]_{rx_samps.bin, config.yaml, uhd_stdout.log}
RAW_FILE_RE = re.compile(r"^\d{8}_\d{6}(_p\d+)?_(rx_samps\.bin|config\.yaml|uhd_stdout\.log)$")


def ensure_scratch_link(data_root, repo_root=REPO_ROOT) -> Path:
    """Create <data_root>/scratch and the <repo>/hil_scratch symlink if missing."""
    target = Path(data_root).expanduser() / "scratch"
    target.mkdir(parents=True, exist_ok=True)
    link = Path(repo_root) / SCRATCH_LINK_NAME
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"{link} points to {link.resolve()}, expected {target}")
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink")
    else:
        link.symlink_to(target)
    return link


def run_scratch_dir(run_id, repo_root=REPO_ROOT) -> Path:
    if not run_id:
        raise ValueError("empty run_id")
    return Path(repo_root) / SCRATCH_LINK_NAME / run_id


def case_dir(run_id, test_id, case, repo_root=REPO_ROOT) -> Path:
    """Unique per-capture directory (save_data prefixes only have 1 s resolution)."""
    return run_scratch_dir(run_id, repo_root) / test_id / case


def dir_size(path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def raw_files(directory):
    """Raw prefix files written by save_data (including split-file parts)."""
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and RAW_FILE_RE.match(p.name))


def release_case(case_path, passed, kept_root, run_id, test_id, keep_raw=False):
    """Clean up one capture directory after analysis.

    On pass (and without keep_raw): delete everything. Otherwise move the raw
    prefix files to <kept_root>/<run_id>/<test_id>/<case>/ and delete the rest
    (the .zarr can be regenerated from the raw files).
    Returns the kept directory, or None.
    """
    case_path = Path(case_path)
    kept = None
    if not passed or keep_raw:
        kept = Path(kept_root) / run_id / test_id / case_path.name
        kept.mkdir(parents=True, exist_ok=True)
        for f in raw_files(case_path):
            shutil.move(str(f), kept / f.name)
    shutil.rmtree(case_path, ignore_errors=True)
    return kept


def enforce_quota(kept_root, quota_bytes, protect=None):
    """Delete the oldest kept runs until kept_root is under quota.

    Run directories are named <UTC timestamp>_..., so name order is age order.
    The ``protect`` run (normally the current one) is never evicted.
    Returns the list of evicted run directories.
    """
    kept_root = Path(kept_root)
    if not kept_root.exists():
        return []
    runs = sorted(p for p in kept_root.iterdir() if p.is_dir())
    sizes = {p: dir_size(p) for p in runs}
    total = sum(sizes.values())
    evicted = []
    for run in runs:
        if total <= quota_bytes:
            break
        if run.name == protect:
            continue
        shutil.rmtree(run)
        total -= sizes[run]
        evicted.append(run)
    return evicted


def prune_runs(runs_root, retention_days, now=None):
    """Delete run directories whose mtime is older than retention_days."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []
    cutoff = (now or time.time()) - retention_days * 86400
    pruned = []
    for run in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if run.stat().st_mtime < cutoff:
            shutil.rmtree(run)
            pruned.append(run)
    return pruned
