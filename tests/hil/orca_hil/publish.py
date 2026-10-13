"""Publish a run directory to the results repository (HIL plan Section 9).

    python -m tests.hil.orca_hil.publish [--repo DIR] [--no-push] [--publish-plots] RUN_DIR...

Copies run.yaml, configuration.yaml and each test's metrics.yaml; plots for the full
tier, failed tests, or with --publish-plots; reduced.zarr.zip for the full tier only.
Appends one history row per metric (append-only) and commits. Runs already present
in the repo are skipped.
"""
import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

from . import HIL_DIR, yamlio

HISTORY_COLUMNS = ["run_id", "timestamp_utc", "git_sha", "dirty", "uhd_commit", "uhd_dirty", "tier",
                   "fixture_id", "test_id", "case", "metric", "value", "units", "kind", "bound_type",
                   "bound_lo", "bound_hi", "baseline_mean", "baseline_std", "status", "fixture_suspect"]
PUSH_ATTEMPTS = 3


def default_repo(fixture):
    host = fixture.get("host", {})
    if host.get("results_repo"):
        return Path(host["results_repo"]).expanduser()
    return Path(host["data_root"]).expanduser() / "results-repo"


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _row(run, uhd, test_id, m, suspect):
    bound = m.get("bound") or {}
    base = m.get("baseline") or {}
    lo, hi = m.get("bound_lo"), m.get("bound_hi")
    if lo is None and hi is None and bound.get("type") == "equals":
        lo = hi = bound.get("value")
    return {
        "run_id": run["run_id"], "timestamp_utc": run["started_utc"], "git_sha": run["git_sha"],
        "dirty": run.get("dirty", False), "uhd_commit": uhd.get("commit"), "uhd_dirty": uhd.get("dirty"),
        "tier": run["tier"], "fixture_id": run["fixture_id"], "test_id": test_id, "case": m.get("case"),
        "metric": m["name"], "value": m.get("value"), "units": m.get("units"), "kind": m.get("kind"),
        "bound_type": bound.get("type"), "bound_lo": lo, "bound_hi": hi,
        "baseline_mean": base.get("mean"), "baseline_std": base.get("std"), "status": m.get("status"),
        "fixture_suspect": suspect,
    }


def history_rows(run_dir):
    """One dict per metric (tests in order, then session metrics), keyed by HISTORY_COLUMNS."""
    run_dir = Path(run_dir)
    run = yamlio.load(run_dir / "run.yaml")
    conf = run_dir / "configuration.yaml"
    uhd = (yamlio.load(conf).get("uhd_build_stamp") or {}) if conf.exists() else {}
    rows = []
    for mpath in sorted(run_dir.glob("t*/metrics.yaml")):
        doc = yamlio.load(mpath)
        for m in doc.get("metrics", []):
            rows.append(_row(run, uhd, doc.get("test_id", mpath.parent.name), m, doc.get("fixture_suspect", False)))
    for m in run.get("session_metrics") or []:
        rows.append(_row(run, uhd, "session", m, run.get("fixture_suspect", False)))
    return rows


def append_history(repo, rows):
    """Append rows to history/history-<YYYY>.csv by each row's year; returns the files touched."""
    touched = []
    for year in sorted({r["timestamp_utc"][:4] for r in rows}):
        path = Path(repo) / "history" / f"history-{year}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(HISTORY_COLUMNS)
            w.writerows([[_fmt(r[c]) for c in HISTORY_COLUMNS] for r in rows if r["timestamp_utc"][:4] == year])
        touched.append(path)
    return touched


def copy_run(run_dir, repo, publish_plots=False):
    """Copy the published subset of run_dir into <repo>/runs/<run_id>/; returns that directory."""
    run_dir = Path(run_dir)
    run = yamlio.load(run_dir / "run.yaml")
    full = run["tier"] == "full"
    dest = Path(repo) / "runs" / run["run_id"]
    dest.mkdir(parents=True)
    for name in ("run.yaml", "configuration.yaml"):
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, dest / name)
    for mpath in sorted(run_dir.glob("t*/metrics.yaml")):
        tdir = mpath.parent
        out = dest / tdir.name
        out.mkdir()
        shutil.copy(mpath, out / "metrics.yaml")
        failed = run.get("tests", {}).get(tdir.name) == "fail"
        if full or failed or publish_plots:
            for png in sorted(tdir.glob("*.png")):
                shutil.copy(png, out / png.name)
        if full and (tdir / "reduced.zarr.zip").exists():
            shutil.copy(tdir / "reduced.zarr.zip", out / "reduced.zarr.zip")
    return dest


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def _has_upstream(repo):
    return git(repo, "rev-parse", "--abbrev-ref", "@{upstream}", check=False).returncode == 0


def publish(run_dir, repo, push=True, publish_plots=False):
    """Copy, append history and commit one run. Returns the commit message, or None if already published."""
    run_dir, repo = Path(run_dir), Path(repo)
    run = yamlio.load(run_dir / "run.yaml")
    if push and _has_upstream(repo):
        git(repo, "pull", "--rebase")
    if (repo / "runs" / run["run_id"]).exists():
        return None
    dest = copy_run(run_dir, repo, publish_plots)
    touched = append_history(repo, history_rows(run_dir))
    status = run["status"] + (" dirty" if run.get("dirty") else "")
    msg = f"hil: {run['run_id']} {run['tier']} {status}"
    git(repo, "add", str(dest), *map(str, touched))
    git(repo, "commit", "-q", "-m", msg)
    if push:
        for attempt in range(PUSH_ATTEMPTS):
            if git(repo, "push", "-u", "origin", "HEAD", check=False).returncode == 0:
                break
            if attempt == PUSH_ATTEMPTS - 1:
                raise RuntimeError(f"push to the results repo failed {PUSH_ATTEMPTS} times")
            git(repo, "pull", "--rebase")
    return msg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Publish HIL run directories to the results repo.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--repo", type=Path, help="results repo clone (default: from the run's fixture)")
    ap.add_argument("--no-push", action="store_true", help="commit locally only")
    ap.add_argument("--publish-plots", action="store_true", help="also publish fast-tier plots of passing tests")
    a = ap.parse_args(argv)
    for rd in sorted(a.run_dirs, key=lambda p: p.name):
        fixture_id = yamlio.load(rd / "run.yaml")["fixture_id"]
        repo = a.repo or default_repo(yamlio.load(HIL_DIR / "fixtures" / f"{fixture_id}.yaml"))
        msg = publish(rd, repo, push=not a.no_push, publish_plots=a.publish_plots)
        print(msg or f"{rd.name}: already published")


if __name__ == "__main__":
    main()
