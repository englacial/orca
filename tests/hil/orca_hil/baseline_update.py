"""--update-baseline: repeat the HIL session N times and aggregate [base] metrics.

Each repeat is an independent pytest session (its own run directory and radar
start-up), so the baseline std covers run-to-run variation.
"""
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import yamlio
from .analysis.noise import match_lines
from .baselines import Baselines

CHILD_ENV = "ORCA_HIL_BASELINE_CHILD"
RUN_DIR_RE = re.compile(r"HIL run directory: (\S+)")


def child_args(args):
    """pytest args for one repeat: drop --baseline-repeats [N]."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a == "--baseline-repeats":
            skip = True
        elif not a.startswith("--baseline-repeats="):
            out.append(a)
    return out


def run_repeats(args, n, cwd):
    """Run pytest n times (output streamed); returns ([run_dir], [exit codes])."""
    env = {**os.environ, CHILD_ENV: "1"}
    run_dirs, codes = [], []
    for i in range(n):
        print(f"\n===== baseline repeat {i + 1}/{n} =====", flush=True)
        proc = subprocess.Popen([sys.executable, "-m", "pytest", *child_args(args)], cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        run_dir = None
        for line in proc.stdout:
            sys.stdout.write(line)
            m = RUN_DIR_RE.search(line)
            if m:
                run_dir = Path(m.group(1))
        codes.append(proc.wait())
        if run_dir is not None:
            run_dirs.append(run_dir)
    return run_dirs, codes


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def collect_base_values(run_dirs):
    """{test_id: {(name, case): [(value, run_id)]}} for judgeable [base] records."""
    out = {}
    for rd in run_dirs:
        for mpath in sorted(Path(rd).glob("t*/metrics.yaml")):
            tid = mpath.parent.name
            for r in yamlio.load(mpath).get("metrics", []):
                if r["kind"] != "base" or r["status"] == "inconclusive" or not _finite(r["value"]):
                    continue
                out.setdefault(tid, {}).setdefault((r["name"], r.get("case")), []).append((r["value"], rd.name))
    return out


def majority_lines(line_lists, tol_hz=None):
    """Lines present in more than half of the repeats (frequency and level averaged)."""
    kw = {} if tol_hz is None else {"tol_hz": tol_hz}
    if not line_lists:
        return []
    out = []
    for ln in line_lists[-1]:
        hits = [ln] + [m[0] for other in line_lists[:-1] for m in match_lines(other, [ln], **kw)[1]]
        if len(hits) * 2 > len(line_lists):
            out.append({"freq": float(sum(h["freq"] for h in hits) / len(hits)),
                        "level_db": float(sum(h["level_db"] for h in hits) / len(hits))})
    return out


def _reduced_attr(run_dir, tid, key):
    from .reduced import read_reduced
    p = Path(run_dir) / tid / "reduced.zarr.zip"
    return read_reduced(p).attrs.get(key) if p.exists() else None


def update_baselines(fixture_id, run_dirs, git_sha, directory=None):
    """Write mean/std/n of every [base] metric, the T3/T4 line lists and reference zarrs."""
    b = Baselines(fixture_id) if directory is None else Baselines(fixture_id, directory)
    values = collect_base_values(run_dirs)
    for tid, metrics in values.items():
        for (name, case), vals in sorted(metrics.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
            b.update_scalar(tid, name, [v for v, _ in vals], [r for _, r in vals], case=case)

    lines = [x for x in (_reduced_attr(rd, "t03", "lines") for rd in run_dirs) if x is not None]
    if lines:
        b.set("t03", "narrowband_lines", majority_lines(lines))
    spurs = [x["a"] for x in (_reduced_attr(rd, "t04", "spurs") for rd in run_dirs) if x and "a" in x]
    if spurs:
        b.set("t04", "spurs", majority_lines(spurs))

    # Reference curves: the last repeat's reduced data for each test
    for tid in sorted({p.parent.name for rd in run_dirs for p in Path(rd).glob("t*/reduced.zarr.zip")}):
        src = next((Path(rd) / tid / "reduced.zarr.zip" for rd in reversed(run_dirs)
                    if (Path(rd) / tid / "reduced.zarr.zip").exists()))
        dest = b.reference_path(tid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
    b.save(git_sha=git_sha)
    return b, values


def main_from_pytest(config):
    """Entry point from conftest's pytest_cmdline_main; returns the exit code."""
    from .telemetry import git_info
    if git_info()["dirty"]:
        print("--update-baseline refuses to run on a dirty git tree", file=sys.stderr)
        return 4
    n = int(config.getoption("--baseline-repeats"))
    args = list(config.invocation_params.args)
    run_dirs, codes = run_repeats(args, n, cwd=config.invocation_params.dir)
    if not run_dirs:
        print("baseline update: no run directories produced", file=sys.stderr)
        return 1
    run = yamlio.load(run_dirs[-1] / "run.yaml")
    b, values = update_baselines(run["fixture_id"], run_dirs, run["git_sha"])
    n_metrics = sum(len(v) for v in values.values())
    print(f"\nbaseline update: {n_metrics} [base] metrics from {len(run_dirs)} runs -> {b.path}")
    for rd, c in zip(run_dirs, codes):
        print(f"  {rd.name}: exit {c}")
    return max(codes)
