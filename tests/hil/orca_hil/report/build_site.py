"""Static site for the HIL results repo (HIL plan Section 9.3).

    python -m tests.hil.orca_hil.report.build_site --results <results repo> --out _site

Pages: an index (latest runs and a test x run status grid), one page per test (metric
trends and a summary-plot gallery) and one page per run (metric tables, plots and the
configuration snapshot). Plain HTML and PNGs, no JavaScript.
"""
import argparse
import csv
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import plotting, yamlio
from ..plotting import plt

ORCA_COMMIT_URL = "https://github.com/englacial/orca/commit/{}"
GRID_RUNS = 30
TEST_NAMES = {
    "t00": "Preflight", "t01": "End-to-end pipeline", "t02": "Pulse compression sidelobes",
    "t03": "RX-only noise", "t04": "Spectral artifacts", "t05": "Dithering", "t06": "Coherent integration",
    "t07": "Long-term phase stability", "t08": "Gain linearity", "t09": "Duty cycle",
    "t10": "Error recovery", "t11": "Presumming", "t12": "Frequency response", "session": "Session",
}
TEST_DESCRIPTIONS = {
    "t00": "Checks the UHD install, the SDR and the loopback fixture before anything else runs. "
           "Other hardware tests are skipped if it fails.",
    "t01": "Runs the full capture pipeline (harness runner, run.py, Ctrl-C shutdown and split files) and checks "
           "that every pulse arrives intact, with a stable loopback peak and a low late-command error rate.",
    "t02": "Measures range sidelobes of the compressed loopback pulse for each window function and compares "
           "them with the window's ideal sidelobe level.",
    "t03": "Records with the transmitter off to check the noise floor is Gaussian and free of interference. "
           "A failure flags T4-T7 as fixture suspect.",
    "t04": "Looks for LO leakage, PLL spurs and chirp copies in the spectrum, with and without phase dithering "
           "and a software LO offset.",
    "t05": "Checks that coherent stacking lowers the noise floor as expected and that phase dithering "
           "suppresses coherent artifacts.",
    "t06": "Measures how coherent the loopback phase stays as integration time grows, from single pulses to "
           "minutes.",
    "t07": "Tracks the loopback phase over a 25 minute capture to measure slow drift and residual phase noise.",
    "t08": "Sweeps TX and RX gain and checks that received power follows the commanded gain until "
           "compression.",
    "t09": "Sweeps duty cycle and measures the late-command error rate, locating the knee above which "
           "pulses start to drop.",
    "t10": "Runs just above the duty-cycle knee so errors occur, then checks that the radar recovers and "
           "that phase stays continuous across dropped pulses.",
    "t11": "Checks that on-device presumming gives the same result as stacking single pulses offline.",
    "t12": "Sweeps the RF center frequency and measures the loopback power, leakage and noise floor at each.",
}
GALLERY_RUNS = 3
TEMPLATES = Path(__file__).parent / "templates"


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_history(results):
    rows = []
    for path in sorted(Path(results).glob("history/history-*.csv")):
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def load_runs(results):
    """Runs newest first, each with its tests' metrics and plots."""
    runs = []
    for rdir in sorted(Path(results).glob("runs/*/"), reverse=True):
        if not (rdir / "run.yaml").exists():
            continue
        run = yamlio.load(rdir / "run.yaml")
        run["dir"] = rdir
        conf = rdir / "configuration.yaml"
        run["configuration_text"] = conf.read_text() if conf.exists() else ""
        run["uhd"] = (yamlio.load(conf).get("uhd_build_stamp") or {}) if conf.exists() else {}
        run["started"] = datetime.strptime(run["started_utc"], "%Y-%m-%dT%H:%M:%SZ")
        run["test_docs"] = {}
        for mpath in sorted(rdir.glob("t*/metrics.yaml")):
            doc = yamlio.load(mpath)
            doc["plots"] = sorted(p.name for p in mpath.parent.glob("*.png"))
            doc["has_reduced"] = (mpath.parent / "reduced.zarr.zip").exists()
            run["test_docs"][mpath.parent.name] = doc
        runs.append(run)
    return runs


def fmt_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v:
            return "–"
        return f"{v:.4g}"
    return "–" if v is None else str(v)


def fmt_bound(m):
    b = m.get("bound") or {}
    lo, hi = m.get("bound_lo"), m.get("bound_hi")
    t = b.get("type")
    if t == "equals":
        return f"= {fmt_value(b.get('value'))}"
    if lo is not None and hi is not None:
        return f"{fmt_value(lo)} … {fmt_value(hi)}"
    if hi is not None:
        return f"≤ {fmt_value(hi)}"
    if lo is not None:
        return f"≥ {fmt_value(lo)}"
    return ""


def fmt_baseline(m):
    b = m.get("baseline")
    if not b:
        return ""
    return f"{fmt_value(b.get('mean'))} ± {fmt_value(b.get('std'))} (n={b.get('n')})"


def slug(*parts):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(p for p in parts if p)).strip("-")


def trend_series(history):
    """{(tier, test_id): {(metric, case): [row]}} for numeric, non-boolean metrics.

    A metric judged twice in one run (abs and base) keeps one point, the failing one if any.
    """
    out = defaultdict(lambda: defaultdict(dict))
    for r in history:
        v = _float(r["value"])
        if v is None or r["units"] == "bool" or r["value"] in ("true", "false"):
            continue
        r = dict(r, v=v, t=datetime.strptime(r["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        per_run = out[(r["tier"], r["test_id"])][(r["metric"], r["case"])]
        if r["run_id"] not in per_run or r["status"] == "fail":
            per_run[r["run_id"]] = r
    return {k: {m: list(runs.values()) for m, runs in v.items()} for k, v in out.items()}


def plot_trend(rows, title, path):
    rows = sorted(rows, key=lambda r: r["t"])
    with plt.rc_context(plotting.STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=100)
        t = [r["t"] for r in rows]
        lo = [_float(r["bound_lo"]) for r in rows]
        hi = [_float(r["bound_hi"]) for r in rows]
        if any(x is not None for x in lo + hi) and len(rows) > 0:
            vals = [r["v"] for r in rows]
            span = (max(vals) - min(vals)) or abs(vals[0]) or 1.0
            lo_f = [x if x is not None else min(vals) - 10 * span for x in lo]
            hi_f = [x if x is not None else max(vals) + 10 * span for x in hi]
            ax.fill_between(t, lo_f, hi_f, step="pre", color=plotting.BAND, alpha=0.7, lw=0, label="bound")
            ax.set_ylim(min(vals) - 0.6 * span, max(vals) + 0.6 * span)
        for prev, cur in zip(rows, rows[1:]):
            if prev["baseline_mean"] != cur["baseline_mean"] and cur["baseline_mean"]:
                ax.axvline(cur["t"], color=plotting.BASELINE, ls="--", lw=0.8)
            if prev["uhd_commit"] != cur["uhd_commit"]:
                ax.axvline(cur["t"], color=plotting.ACCENT, ls=":", lw=1.0)
                ax.text(cur["t"], 1.0, f" UHD {cur['uhd_commit'][:7]}", transform=ax.get_xaxis_transform(),
                        fontsize=7, color=plotting.ACCENT, va="top")
        ax.plot(t, [r["v"] for r in rows], color=plotting.BASELINE, lw=0.6, zorder=1)
        for r in rows:
            c = plotting.STATUS_COLORS.get(r["status"], plotting.ACCENT)
            dirty = r["dirty"] == "true"
            ax.scatter([r["t"]], [r["v"]], s=22, zorder=2, edgecolors=c,
                       facecolors="none" if dirty else c, linewidths=1.2)
        ax.set_title(title, loc="left")
        ax.set_ylabel(rows[0]["units"] or "")
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def build(results, out):
    results, out = Path(results), Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters.update(value=fmt_value, bound=fmt_bound, baseline=fmt_baseline)
    shutil.copy(TEMPLATES / "style.css", out / "style.css")

    history = load_history(results)
    runs = load_runs(results)
    test_ids = sorted({tid for r in runs for tid in r.get("tests", {})})
    common = {"test_names": TEST_NAMES, "commit_url": ORCA_COMMIT_URL, "test_ids": test_ids,
              "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

    # Run pages
    (out / "runs").mkdir()
    for run in runs:
        rid = run["run_id"]
        for tid, doc in run["test_docs"].items():
            for name in doc["plots"] + (["reduced.zarr.zip"] if doc["has_reduced"] else []):
                dest = out / "runs" / rid / tid / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(run["dir"] / tid / name, dest)
        (out / "runs" / f"{rid}.html").write_text(
            env.get_template("run.html").render(root="../", run=run, **common))

    # Test pages
    series = trend_series(history)
    for tid in test_ids:
        tiers = []
        for tier in ("fast", "full"):
            plots = []
            for (metric, case), rows in sorted(series.get((tier, tid), {}).items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
                name = slug(metric, case) + ".png"
                kinds = "/".join(sorted({r["kind"] for r in rows}))
                title = f"{metric}" + (f" [{case}]" if case else "") + f"  ({kinds})"
                plot_trend(rows, title, out / "trends" / tier / tid / name)
                plots.append({"src": f"../trends/{tier}/{tid}/{name}", "title": title})
            if plots:
                tiers.append({"tier": tier, "plots": plots})
        gallery = [{"run": r, "plot": f"../runs/{r['run_id']}/{tid}/summary.png"}
                   for r in runs if "summary.png" in r["test_docs"].get(tid, {}).get("plots", [])]
        (out / "tests").mkdir(exist_ok=True)
        (out / "tests" / f"{tid}.html").write_text(env.get_template("test.html").render(
            root="../", tid=tid, tiers=tiers, gallery=gallery[:GALLERY_RUNS], n_gallery=len(gallery),
            description=TEST_DESCRIPTIONS.get(tid, ""),
            runs=[r for r in runs if tid in r.get("tests", {})], **common))

    # Index
    latest = {tier: next((r for r in runs if r["tier"] == tier), None) for tier in ("fast", "full")}
    (out / "index.html").write_text(env.get_template("index.html").render(
        root="", latest=latest, grid_runs=runs[:GRID_RUNS], n_runs=len(runs), **common))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the HIL results site.")
    ap.add_argument("--results", type=Path, required=True, help="results repo checkout")
    ap.add_argument("--out", type=Path, default=Path("_site"))
    a = ap.parse_args(argv)
    print(f"site written to {build(a.results, a.out)}")


if __name__ == "__main__":
    main()
