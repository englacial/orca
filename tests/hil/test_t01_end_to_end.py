"""T1 - End-to-end loopback (Fig. 1c, Fig. 2, Sec. III).

Cases:
  a_profile  harness runner, 20000 pulses: pipeline checks + range profile
  b_runpy    `python run.py <cfg>`, 5000 pulses: the user-facing entry point
  c_sigint   num_pulses -1, SIGINT after 5 s: graceful shutdown
  d_split    run.py, 20000 pulses split into files of 5000, merged by run.py
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.hil.orca_hil import REPO_ROOT, plotting, radar, yamlio
from tests.hil.orca_hil.analysis.profile import analyze_profile
from tests.hil.orca_hil.metrics import Measurement
from tests.hil.orca_hil.reduced import write_reduced

import processing_dask as pdk

PARAMS = {
    "fast": {"n_profile": 20000, "n_runpy": 5000, "sigint_after_s": 5, "n_split": 20000, "split_every": 5000},
}
PARAMS["full"] = PARAMS["fast"]
SIGINT_LIMIT_S = 10


def run_runpy(cap, fixture, timeout_s):
    """Launch `python run.py <cfg>` from the repo root and attach a RunResult to cap."""
    cfg = yamlio.load(REPO_ROOT / cap.config_rel)
    t0 = time.time()
    p = subprocess.run([sys.executable, "run.py", str(cap.config_rel)], cwd=REPO_ROOT,
                       env=radar.uhd_env(fixture), capture_output=True, text=True, timeout=timeout_s)
    (cap.case_dir / "runpy_stdout.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    m = re.search(r"Copying data to (\S+)\.\.\.", p.stdout)
    res = radar.RunResult(cap.config_rel, returncode=p.returncode, duration_s=time.time() - t0)
    if m:
        res.prefix = str(REPO_ROOT / m.group(1))
        res.log_path = Path(res.prefix + "_uhd_stdout.log")
        res.summary = radar.parse_summary(res.log_path.read_text())
        if p.returncode == 0:
            res.zarr = pdk.save_radar_data_to_zarr(res.prefix)
    cap.result = res
    return cfg


def pipeline_checks(cap, n_requested):
    """Boolean checks common to every completed capture."""
    res = cap.result
    s = res.summary
    gen_cfg = yamlio.load(cap.case_dir / "config.yaml")
    cfg = yamlio.load(res.prefix + "_config.yaml") if res.prefix else None
    rec_len = int(round(gen_cfg["CHIRP"]["rx_duration"] * gen_cfg["GENERATE"]["sample_rate"]))
    presums = int(gen_cfg["CHIRP"].get("num_presums", 1))
    errors, written, attempted = s.get("error_count"), s.get("pulses_written"), s.get("pulses_attempted")
    bin_path = Path(res.prefix + "_rx_samps.bin") if res.prefix else None
    size = bin_path.stat().st_size if bin_path and bin_path.exists() else -1
    ds = xr.open_zarr(res.zarr) if res.zarr else None
    expected_type = "error_data_included" if errors == 0 else "error_data_not_included"
    c = cap.case
    return [
        Measurement("exit_status_ok", res.returncode == 0, case=c),
        Measurement("start_present", bool(s.get("start_present")), case=c),
        Measurement("error_rate", errors / attempted if attempted else float("nan"), case=c,
                    note=f"{errors} errors in {attempted} pulses"),
        Measurement("pulses_written_equals_requested", written == n_requested, case=c),
        Measurement("file_size_matches", size == n_requested // presums * rec_len * 8, case=c),
        Measurement("config_matches_generated", cfg == gen_cfg, case=c),
        Measurement("zarr_dims_ok", ds is not None and dict(ds.sizes) == {
            "sample_idx": rec_len, "pulse_idx": n_requested // presums}, case=c),
        Measurement("error_file_type_ok", ds is not None and pdk.check_if_error_data_exists(ds) == expected_type,
                    case=c, note=f"expected {expected_type}"),
    ]


def sigint_checks(cap):
    res = cap.result
    gen_cfg = yamlio.load(cap.case_dir / "config.yaml")
    rec_bytes = int(round(gen_cfg["CHIRP"]["rx_duration"] * gen_cfg["GENERATE"]["sample_rate"])) * 8
    size = os.path.getsize(res.prefix + "_rx_samps.bin")
    exit_s = res.exited_at - res.sigint_sent_at if res.sigint_sent_at else float("nan")
    s = res.summary
    return [
        Measurement("exit_status_ok", res.returncode == 0, case=cap.case),
        Measurement("sigint_exit_within_10s", (not res.killed) and exit_s <= SIGINT_LIMIT_S, case=cap.case,
                    note=f"exited {exit_s:.1f} s after SIGINT"),
        Measurement("records_positive", size > 0, case=cap.case),
        Measurement("file_size_multiple_of_record", size % rec_bytes == 0, case=cap.case),
        Measurement("rx_summary_present", None not in (s.get("error_count"), s.get("pulses_written"),
                                                       s.get("pulses_attempted")), case=cap.case),
    ]


def split_checks(cap):
    """run.py split + merge: one part per [CLOSE FILE] line, merged == concatenated parts."""
    res = cap.result
    parts = sorted(cap.case_dir.glob("rx_samps.bin.*"), key=lambda p: int(p.suffix[1:]))
    closes = res.summary.get("close_file_lines", [])
    merged = Path(res.prefix + "_rx_samps.bin").read_bytes() if res.prefix else b""
    concat = b"".join(p.read_bytes() for p in parts)
    return [
        Measurement("close_file_count_matches_parts", len(closes) == len(parts) > 1, case=cap.case,
                    note=f"{len(closes)} [CLOSE FILE] lines, {len(parts)} part files"),
        Measurement("merged_matches_parts", len(merged) > 0 and merged == concat, case=cap.case),
    ]


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t01", "Fig. 1c range profile", run_id, status)
    ax = axes[0, 0]
    x = reduced["sample"].values
    plotting.plot_current(ax, x, reduced["profile_power_db"].values, label="stacked profile")
    t1, t2, exp = reduced.attrs["t1"], reduced.attrs["t2"], reduced.attrs["expected_separation"]
    ax.axvline(t1, color=plotting.BASELINE, ls=":", lw=1, label=f"leakage {t1:.2f}")
    ax.axvline(t2, color=plotting.ACCENT, ls=":", lw=1, label=f"loopback {t2:.2f}")
    ax.axvline(t1 + exp, color=plotting.FAIL, ls="--", lw=0.8, label=f"expected loopback {t1 + exp:.2f}")
    ax.set_xlabel("compressed sample (56 MSPS)")
    ax.set_ylabel("power (dBFS)")
    sec = ax.secondary_xaxis("top", functions=(lambda s: (s - t1) / 56e6 * 1.98e8,
                                               lambda d: d / 1.98e8 * 56e6 + t1))
    sec.set_xlabel("effective cable length from leakage peak (m, vf 0.66)")
    ax.legend(loc="upper right")
    return plotting.save(fig, path)


def plot_peak_zoom(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t01", "peak zoom with two-peak model", run_id, status)
    ax = axes[0, 0]
    t1, t2 = reduced.attrs["t1"], reduced.attrs["t2"]
    plotting.plot_current(ax, reduced["sample_up"], reduced["peak_region_power_db"], label="profile (x8)")
    m = (reduced["sample"] > t1 - 30) & (reduced["sample"] < t2 + 40)
    ax.plot(reduced["sample"][m], reduced["model_power_db"][m], "o", ms=3, color=plotting.FAIL,
            label="two-peak model")
    ax.set_xlabel("compressed sample")
    ax.set_ylabel("power (dBFS)")
    ax.legend()
    return plotting.save(fig, path)


def plot_raw_pulse(zarr_path, run_id, status, path):
    x = xr.open_zarr(zarr_path)["radar_data"].isel(pulse_idx=0).values
    fig, axes = plotting.summary_figure("t01", "single raw pulse", run_id, status)
    ax = axes[0, 0]
    ax.plot(x.real, lw=0.6, label="I")
    ax.plot(x.imag, lw=0.6, label="Q")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude (FS)")
    ax.legend()
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t01
@pytest.mark.timeout(900)
def test_t01_end_to_end(hil, tier, fixture_cfg, radar_build, capture_manager, recorder):
    p = PARAMS[tier]
    fx = fixture_cfg
    out = hil.run_dir / "t01"

    # a: harness runner + range profile
    a = capture_manager.acquire("t01", "a_profile", {"CHIRP": {"num_pulses": p["n_profile"]}})
    recorder.record_all(pipeline_checks(a, p["n_profile"]))
    reduced, profile_ms = analyze_profile(a.zarr, fx)
    recorder.record_all(profile_ms)

    # b: run.py (rebuilds sdr/build with cmake/make from inside the conda env)
    b = capture_manager.new_case("t01", "b_runpy", {"CHIRP": {"num_pulses": p["n_runpy"]}})
    run_runpy(b, fx, timeout_s=300)
    capture_manager.archive(b)
    recorder.record_all(pipeline_checks(b, p["n_runpy"]))
    binary = REPO_ROOT / "sdr" / "build" / "radar"
    recorder.record("radar_binary_changed_by_runpy", radar.sha256(binary) != radar_build.sha256)

    # c: SIGINT shutdown
    c = capture_manager.acquire("t01", "c_sigint", {"CHIRP": {"num_pulses": -1}},
                                stop_after_s=p["sigint_after_s"])
    recorder.record_all(sigint_checks(c))

    # d: split files merged by run.py
    d = capture_manager.new_case("t01", "d_split", {"CHIRP": {"num_pulses": p["n_split"]},
                                                    "FILES": {"max_chirps_per_file": p["split_every"]}})
    final = d.config_rel.parent / "merged_rx_samps.bin"
    cfg = yamlio.load(REPO_ROOT / d.config_rel)
    cfg["RUN_MANAGER"]["final_save_loc"] = str(final)
    yamlio.dump(cfg, REPO_ROOT / d.config_rel)
    run_runpy(d, fx, timeout_s=300)
    capture_manager.archive(d)
    recorder.record_all(pipeline_checks(d, p["n_split"]))
    recorder.record_all(split_checks(d))

    status = "fail" if recorder.failures else recorder.status
    configs = {k.case: yamlio.load(k.case_dir / "config.yaml") for k in (a, b, c, d)}
    write_reduced(reduced, out / "reduced.zarr.zip", test_id="t01", run_id=hil.run_id,
                  git_sha=hil.git["sha"], fixture_id=hil.fixture_id, tier=tier, configs=configs)
    plot_summary(reduced, hil.run_id, status, out / "summary.png")
    plot_peak_zoom(reduced, hil.run_id, status, out / "peak_zoom.png")
    plot_raw_pulse(a.zarr, hil.run_id, status, out / "raw_pulse.png")
    recorder.finalize()
