"""T9 - Duty cycle vs error rate (Fig. 9). 100% duty is not a supported operating point.

Duty cycle = max(tx_duration, rx_duration) / PRI, as in
tests/error_code_late_command_sweep.py (full duplex).
"""
import numpy as np
import pytest

from tests.hil.orca_hil import plotting, telemetry, testkit
from tests.hil.orca_hil.analysis.duty import analyze_duty, duty_to_pri
from tests.hil.orca_hil.configs import merged_config
from tests.hil.orca_hil.radar import RadarTimeout

OVERRIDES = {"CHIRP": {"rx_duration": 100e-6, "num_presums": 100}}   # config/orca_paper/duty_cycle_b205.yaml
PARAMS = {
    "fast": {"duties": [0.5, 0.75, 0.9], "n_pulses": 50000,
             "report_at": {0.5: "error_rate_at_50pct", 0.75: "error_rate_at_75pct"}},
    # Plan: 1e6 pulses per point; 5e5 keeps the sweep near its ~35 min estimate (1e6 takes ~1 h)
    "full": {"duties": [round(d, 2) for d in np.arange(0.2, 0.95 + 1e-9, 0.05)], "n_pulses": 500000,
             "report_at": {0.5: "error_rate_at_50pct"}},
}


def plot_summary(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t09", "Fig. 9 error rate vs duty cycle", run_id, status)
    ax = axes[0, 0]
    d = reduced["duty"].values * 100
    plotting.plot_current(ax, d, reduced["error_rate"].values * 100, marker="o", label="error rate")
    ax.axhline(2, color=plotting.BASELINE, ls=":", label="2%")
    ax.set(xlabel="duty cycle (%)", ylabel="late-command errors (% of pulses)")
    pri = reduced["pri"].values * 1e6
    sec = ax.secondary_xaxis("top", functions=(lambda x: 100.0 / np.maximum(x, 1e-3) * 100,
                                               lambda p: 100.0 / np.maximum(p, 1e-3) * 100))
    sec.set_xlabel("PRI (us) for rx_duration 100 us")
    ax.legend()
    return plotting.save(fig, path)


def plot_table(reduced, run_id, status, path):
    fig, axes = plotting.summary_figure("t09", "duty cycle sweep", run_id, status)
    ax = axes[0, 0]
    ax.axis("off")
    def n(x):
        return "-" if x is None or not np.isfinite(x) else f"{int(x)}"

    rows = [[f"{d * 100:.0f}", f"{p * 1e6:.1f}", n(e), n(a), f"{r * 100:.3f}", n(o)]
            for d, p, e, a, r, o in zip(reduced["duty"].values, reduced["pri"].values, reduced["errors"].values,
                                        reduced["attempted"].values, reduced["error_rate"].values,
                                        reduced["other_errors"].values)]
    ax.table(cellText=rows, colLabels=["duty %", "PRI us", "errors", "attempted", "error %", "non-late errors"],
             loc="center").scale(1, 1.4)
    return plotting.save(fig, path)


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t09
@pytest.mark.timeout(7200)
def test_t09_duty_cycle(hil, tier, fixture_cfg, capture_manager, recorder):
    p = PARAMS[tier]
    base = merged_config(fixture_cfg, OVERRIDES)
    points, caps, throttled = [], [], False
    for duty in p["duties"]:
        overrides = {"CHIRP": {**OVERRIDES["CHIRP"], "num_pulses": p["n_pulses"],
                               "pulse_rep_int": duty_to_pri(base, duty)}}
        killed = False
        try:
            cap = capture_manager.acquire("t09", f"duty_{round(duty * 100):d}", overrides, convert=False)
            log = cap.result.log_path.read_text()
            killed = cap.result.killed
        except RadarTimeout:
            cap = capture_manager._owned["t09"][-1]
            log = (cap.case_dir / "uhd_stdout.log").read_text()
            killed = True
        caps.append(cap)
        throttled |= bool((telemetry.throttled() or 0) & 0xF)
        points.append({"duty": duty, "log": log, "config": testkit.case_configs([cap])[cap.case],
                       "killed": killed})
        if killed:   # higher duty cycles only get worse
            break
    reduced, ms = analyze_duty(points, {"report_at": p["report_at"], "full": tier == "full"})
    recorder.record_all(ms)
    recorder.record("throttled_during_sweep", throttled)
    testkit.finish(hil, recorder, tier, reduced, caps, {
        "summary.png": lambda s, path: plot_summary(reduced, hil.run_id, s, path),
        "table.png": lambda s, path: plot_table(reduced, hil.run_id, s, path),
    })
