"""T10 - Error recovery and phase continuity (Sec. III-A, VIII-A).

The PRI is chosen by a bracketing search for an error rate between 0.5% and 5%,
starting from the T9 baseline knee + 5 percentage points.
"""
import numpy as np
import pytest
import xarray as xr

from tests.hil.orca_hil import plotting, testkit, yamlio
from tests.hil.orca_hil.analysis.duty import duty_to_pri, point_stats
from tests.hil.orca_hil.analysis.recovery import analyze_recovery
from tests.hil.orca_hil.analysis.timing import expected_slow_time
from tests.hil.orca_hil.configs import merged_config

OVERRIDES = {"CHIRP": {"rx_duration": 100e-6, "num_presums": 10, "phase_dithering": True}}
PARAMS = {
    "fast": {"n_pulses": 50000, "max_probes": 2},
    "full": {"n_pulses": 500000, "max_probes": 4},
}
PROBE_PULSES = 10000
TARGET = (0.005, 0.05)
DEFAULT_START_DUTY = 0.85
SHARED_KEY = "t10_errors"


def find_duty(capture_manager, fixture, baselines, max_probes):
    """Bracketing search for a duty cycle with an error rate in TARGET."""
    base = merged_config(fixture, OVERRIDES)
    knee = testkit.baseline_mean(baselines, "t09", "max_duty_with_error_rate_below_2pct")
    duty = min(knee + 0.05, 1.0) if knee is not None else DEFAULT_START_DUTY
    probes = []
    for i in range(max_probes):
        ov = {"CHIRP": {**OVERRIDES["CHIRP"], "num_pulses": PROBE_PULSES, "pulse_rep_int": duty_to_pri(base, duty)}}
        cap = capture_manager.acquire("t10", f"probe{i}", ov, convert=False)
        rate = point_stats(cap.result.log_path.read_text(), testkit.case_configs([cap])[cap.case])["error_rate"]
        probes.append((duty, rate))
        if TARGET[0] <= rate <= TARGET[1]:
            break
        duty = min(duty + 0.05, 1.0) if rate < TARGET[0] else max(duty - 0.05, 0.1)
    else:
        duty = probes[-1][0]
    return duty, probes


def t10_capture(capture_manager, fixture, baselines, tier):
    cap = capture_manager.get_shared(SHARED_KEY)
    if cap is None:
        p = PARAMS[tier]
        duty, probes = find_duty(capture_manager, fixture, baselines, p["max_probes"])
        base = merged_config(fixture, OVERRIDES)
        ov = {"CHIRP": {**OVERRIDES["CHIRP"], "num_pulses": p["n_pulses"], "pulse_rep_int": duty_to_pri(base, duty)}}
        cap = capture_manager.acquire("t10", "errors", ov, shared_key=SHARED_KEY)
        cap.extra.update(duty=duty, probes=probes)
    return cap


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t10", "phase and power through error gaps", run_id, status, 2, 1,
                                        sharex=True)
    k = reduced["record"].values
    plotting.plot_current(axes[0, 0], k, reduced["peak_phase_rad"], label="loopback peak phase")
    plotting.plot_current(axes[1, 0], k, reduced["peak_power_db"], label="loopback peak power")
    for g in reduced["gap_record"].values:
        for ax in axes.flat:
            ax.axvline(g, color=plotting.FAIL, lw=0.5, alpha=0.6)
    axes[0, 0].set_ylabel("phase (rad, unwrapped)")
    axes[1, 0].set(ylabel="power (dBFS)", xlabel=f"record ({reduced.attrs['presums']} presums); red: error gaps")
    return plotting.save(fig, path)


def plot_jumps(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t10", "record-to-record phase jumps", run_id, status)
    ax = axes[0, 0]
    normal = reduced["normal_jump_rad"].values
    bins = np.linspace(-4 * normal.std() - 1e-9, 4 * normal.std() + 1e-9, 61) if normal.size else 30
    ax.hist(normal, bins=bins, density=True, color=plotting.BASELINE, label="normal neighbours")
    for j in reduced["gap_jump_rad"].values:
        ax.axvline(j, color=plotting.FAIL, lw=1)
    ax.set(xlabel="phase jump (rad)", ylabel="density", title="red: jumps across error gaps")
    ax.legend()
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t10
@pytest.mark.timeout(2400)
def test_t10_error_recovery(hil, tier, fixture_cfg, capture_manager, recorder, baselines):
    cap = t10_capture(capture_manager, fixture_cfg, baselines, tier)
    for duty, rate in cap.extra.get("probes", []):
        recorder.record("probe_duty", duty, note=f"error rate {rate:.4f}")
    reduced, ms = analyze_recovery(cap.zarr, fixture_cfg)
    recorder.record_all(ms)
    capture_manager.mark_shared_users([SHARED_KEY], not recorder.failures)
    testkit.finish(hil, recorder, tier, reduced, [cap], {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
        "jumps.png": lambda s, path: plot_jumps(reduced, hil.run_id, s, path),
    })


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t10
@pytest.mark.timeout(2400)
@pytest.mark.xfail(strict=True, reason="slow_time ignores errors and the linspace endpoint")
def test_t10_slow_time_matches_schedule(tier, fixture_cfg, capture_manager, baselines):
    cap = t10_capture(capture_manager, fixture_cfg, baselines, tier)
    ds = xr.open_zarr(cap.zarr)
    cfg = yamlio.load(cap.prefix + "_config.yaml")
    expected = expected_slow_time(cfg, ds.attrs["stdout_log"], ds.sizes["pulse_idx"])
    np.testing.assert_allclose(ds["slow_time"].values, expected, rtol=0, atol=1e-6)
