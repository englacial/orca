"""pytest plumbing for the ORCA hardware-in-the-loop tests.

Hardware tests are marked ``hardware`` and only run with ``--hil``. Without it
they are skipped, so hosted CI never touches hardware. See README.md.
"""
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.hil.orca_hil import REPO_ROOT, baseline_update, expect, fixtures, scratch, telemetry, watchdog, yamlio
from tests.hil.orca_hil.baselines import Baselines
from tests.hil.orca_hil.metrics import MetricRecorder
from tests.hil.orca_hil.thresholds import evaluate, load_thresholds

TEST_IDS = [f"t{i:02d}" for i in range(13)]
SUSPECT_IF_T03_FAILS = {"t04", "t05", "t06", "t07"}
NONSTD_BASELINE_MSG = "--update-baseline needs a commissioned fixture and no --setup-note"


def pytest_addoption(parser):
    g = parser.getgroup("hil", "ORCA hardware-in-the-loop")
    g.addoption("--hil", action="store_true", help="enable hardware tests")
    g.addoption("--tier", choices=("fast", "full"), default="fast", help="test tier (default fast)")
    g.addoption("--fixture", default=None,
                help="fixture id in tests/hil/fixtures/ or a path to a fixture file "
                     "(default: the only file in tests/hil/fixtures/)")
    g.addoption("--setup-note", default=None,
                help="describe an undeclared hardware change; the run is logged as non-standard")
    g.addoption("--powered-on", default="",
                help="comma-separated ids of the fixture's powered chain elements, confirming they are on")
    g.addoption("--update-baseline", action="store_true",
                help="repeat the run --baseline-repeats times and rewrite the [base] baselines")
    g.addoption("--baseline-repeats", type=int, default=3)
    g.addoption("--replay", action="store_true", help="enable replay tests (needs golden data)")
    g.addoption("--update-golden-expected", action="store_true",
                help="with --replay: rewrite the expected metrics in golden.yaml")
    g.addoption("--publish", action="store_true", help="push results to the results repo")
    g.addoption("--publish-plots", action="store_true", help="with --publish: include fast-tier plots of passing tests")
    g.addoption("--keep-raw", action="store_true", help="keep raw captures even for passing tests")
    g.addoption("--results-root", default=None, help="default <data_root>/runs")


def pytest_cmdline_main(config):
    """--update-baseline: run the session --baseline-repeats times, then aggregate."""
    if (config.getoption("--update-baseline") and config.getoption("--hil")
            and not os.environ.get(baseline_update.CHILD_ENV)):
        fixture = fixtures.load(fixtures.resolve_path(config.getoption("--fixture")))
        if not fixtures.is_standard(fixture) or config.getoption("--setup-note"):
            print(NONSTD_BASELINE_MSG, file=sys.stderr)
            return 4
        return baseline_update.main_from_pytest(config)
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: needs the HIL fixture (run with --hil)")
    config.addinivalue_line("markers", "replay: replays golden captures (run with --replay)")
    config.addinivalue_line("markers", "fast: has a fast-tier variant")
    config.addinivalue_line("markers", "full: full tier only")
    for tid in TEST_IDS:
        config.addinivalue_line("markers", f"{tid}: HIL test {tid.upper()}")

    config._hil = None
    if not config.getoption("--hil"):
        return
    try:
        fx_path = fixtures.resolve_path(config.getoption("--fixture"))
        fixture = fixtures.load(fx_path)
    except fixtures.FixtureError as e:
        raise pytest.UsageError(str(e))
    setup_note = config.getoption("--setup-note")
    standard = fixtures.is_standard(fixture) and not setup_note
    check_setup(fixture, config.getoption("--powered-on"))
    try:
        expectation = expect.build(fixture, setup_note)
    except fixtures.FixtureError as e:
        raise pytest.UsageError(str(e))
    data_root = Path(fixture["host"]["data_root"]).expanduser()
    results_root = Path(config.getoption("--results-root") or data_root / "runs").expanduser()
    tier = config.getoption("--tier")
    git = telemetry.git_info()
    started = datetime.now(timezone.utc)
    if config.getoption("--update-baseline") and git["dirty"]:
        raise pytest.UsageError("--update-baseline refuses to run on a dirty git tree")
    if config.getoption("--update-baseline") and not standard:
        raise pytest.UsageError(NONSTD_BASELINE_MSG)
    run_id = (f"{started:%Y%m%dT%H%M%SZ}_{(git['sha'] or 'nogit')[:7]}{'-dirty' if git['dirty'] else ''}"
              f"_{tier}_{fixture['fixture_id']}{'' if standard else '-nonstd'}")

    retention = (fixture.get("storage") or {}).get("run_retention_days")
    if retention:
        scratch.prune_runs(results_root, retention)
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch.ensure_scratch_link(data_root)
    if not config.option.xmlpath:
        config.option.xmlpath = str(run_dir / "junit.xml")

    config._hil = SimpleNamespace(
        fixture=fixture, fixture_path=fx_path, fixture_id=fixture["fixture_id"], data_root=data_root,
        results_root=results_root, run_dir=run_dir, run_id=run_id, tier=tier, git=git,
        started=started, t0=time.time(), keep_raw=config.getoption("--keep-raw"),
        preflight_failed=None, fixture_suspect=False, test_status={}, configuration={},
        session_metrics=[], standard=standard, setup_note=setup_note,
        fixture_sha256=fixtures.digest(fixture), expectation=expectation, halted=None,
        setup=setup_summary(fixture, expectation),
    )


def check_setup(fixture, powered_on):
    """Refuse a session whose fixture is unsafe to run at its base settings or not confirmed."""
    dogs = fixtures.watchdogs(fixture)
    try:
        watchdog.check(dogs)
    except watchdog.WatchdogError as e:
        raise pytest.UsageError(f"{e}. Start the watchdog before the session (see the fixture's notes)")
    powered = set(fixtures.powered_elements(fixture))
    confirmed = {x.strip() for x in powered_on.split(",") if x.strip()}
    if powered != confirmed:
        raise pytest.UsageError(f"powered chain elements {sorted(powered)}: confirm each is switched on with "
                                f"--powered-on {','.join(sorted(powered))} (got {sorted(confirmed)})")
    rf, att = fixture["rf_base"], (fixture.get("safety") or {}).get("attenuator_check") or {}
    errors = []
    for label, gain in (("rf_base", rf.get("tx_gain")), ("attenuator_check", att.get("tx_gain"))):
        if gain is not None:
            try:
                errs, _ = fixtures.budget_problems(fixture, gain, rf["freq"])
            except fixtures.FixtureError as e:
                raise pytest.UsageError(str(e))
            errors += [f"{label} tx_gain {gain:g}: {e}" for e in errs]
    if errors:
        raise pytest.UsageError("power budget:\n  " + "\n  ".join(errors))


def setup_summary(fixture, expectation):
    rf = fixture["rf_base"]
    out = {"expectation": expectation.summary() if expectation else None,
           "powered_elements": fixtures.powered_elements(fixture)}
    if rf.get("tx_gain") is not None and (fixture.get("calibration") or {}).get("tx_power_table"):
        out["power_budget_at_rf_base"] = [
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()}
            for st in fixtures.power_budget(fixture, rf["tx_gain"], rf["freq"])]
        out["max_safe_tx_gain_at_rf_base_freq"] = fixtures.max_safe_tx_gain(fixture, rf["freq"])
    return out


def _test_id(item):
    for tid in TEST_IDS:
        if item.get_closest_marker(tid):
            return tid
    return None


def pytest_collection_modifyitems(config, items):
    hil = config.getoption("--hil")
    skip_hw = pytest.mark.skip(reason="hardware test: run with --hil")
    skip_replay = pytest.mark.skip(reason="replay test: run with --replay")
    keep, deselected = [], []
    for item in items:
        if item.get_closest_marker("replay") and not config.getoption("--replay"):
            item.add_marker(skip_replay)
        if item.get_closest_marker("hardware"):
            if not hil:
                item.add_marker(skip_hw)
            elif config.getoption("--tier") == "fast" and not item.get_closest_marker("fast"):
                deselected.append(item)
                continue
        keep.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    # T0 first; otherwise keep collection order
    keep.sort(key=lambda it: 0 if it.get_closest_marker("t00") else 1)
    items[:] = keep


def pytest_runtest_setup(item):
    ctx = item.config._hil
    if ctx and item.get_closest_marker("hardware") and ctx.halted:
        pytest.skip(f"session halted: {ctx.halted}")
    if ctx and item.get_closest_marker("hardware") and ctx.expectation is not None:
        if _test_id(item) in ctx.expectation.skip_tests:
            pytest.skip(f"skip_tests in fixture {ctx.fixture_id}")
    if ctx and item.get_closest_marker("hardware") and not item.get_closest_marker("t00"):
        if ctx.preflight_failed:
            pytest.skip(f"preflight failed: {ctx.preflight_failed}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    setattr(item, f"rep_{rep.when}", rep)
    ctx = item.config._hil
    tid = _test_id(item)
    if ctx and call.excinfo is not None and call.excinfo.errisinstance(watchdog.WatchdogError):
        ctx.halted = ctx.halted or str(call.excinfo.value)[:300]   # no further captures this session
    if ctx and tid and item.get_closest_marker("hardware"):
        if rep.failed:
            ctx.test_status[tid] = "fail"
        elif rep.skipped:
            ctx.test_status.setdefault(tid, "skip")
        elif rep.when == "call" and ctx.test_status.get(tid) != "fail":
            ctx.test_status[tid] = "pass"
        if rep.failed and tid == "t00":
            ctx.preflight_failed = (rep.longreprtext.strip().splitlines() or ["error"])[-1][:300]
        if rep.failed and tid == "t03":
            ctx.fixture_suspect = True
    return rep


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def hil(request):
    ctx = request.config._hil
    if ctx is None:
        pytest.skip("hardware test: run with --hil")
    return ctx


@pytest.fixture(scope="session")
def fixture_cfg(hil):
    return hil.fixture


@pytest.fixture(scope="session")
def tier(hil):
    return hil.tier


@pytest.fixture(scope="session")
def run_dir(hil):
    return hil.run_dir


@pytest.fixture(scope="session")
def sdr_lock(hil):
    from tests.hil.orca_hil.lock import SdrLock, SdrLockHeld
    lock = SdrLock(hil.data_root / "orca-hil.lock")
    try:
        lock.acquire()
    except SdrLockHeld as e:
        pytest.exit(str(e), returncode=4)
    # With the lock held, leftover scratch from crashed sessions is safe to delete
    root = REPO_ROOT / scratch.SCRATCH_LINK_NAME
    for old in root.iterdir():
        if old.is_dir() and old.name != hil.run_id:
            shutil.rmtree(old, ignore_errors=True)
    yield lock
    lock.release()


@pytest.fixture(scope="session")
def radar_build(hil, sdr_lock):
    """Build sdr/build/radar once per session and write configuration.yaml."""
    from tests.hil.orca_hil import radar
    fx = hil.fixture
    build = radar.build_radar(fx, log_path=hil.run_dir / "build.log")
    ver = radar.run_uhd_tool(fx, "uhd_config_info", ["--version"])
    hil.configuration.update({
        "git": hil.git,
        "radar_binary_sha256": build.sha256,
        "libuhd_resolved": build.libuhd,
        "uhd_version": ver.stdout.strip().splitlines()[0] if ver.stdout.strip() else None,
        "uhd_build_stamp": radar.build_stamp(fx),
        "host": telemetry.host_snapshot(fx, hil.data_root),
        "fixture": fx,
        "tier": hil.tier,
        "pytest_args": sys.argv[1:],
        "setup": hil.setup,
    })
    write_configuration(hil)
    return build


def write_configuration(hil):
    yamlio.dump(hil.configuration, hil.run_dir / "configuration.yaml")


@pytest.fixture(scope="session")
def capture_manager(hil, sdr_lock, radar_build):
    from tests.hil.orca_hil.capture import CaptureManager
    mgr = CaptureManager(hil.fixture, hil.run_id, hil.run_dir, hil.data_root / "kept",
                         keep_raw=hil.keep_raw)
    yield mgr
    mgr.release_all()


@pytest.fixture(scope="session")
def thresholds():
    return load_thresholds()


@pytest.fixture(scope="session")
def baselines(hil):
    """This fixture's baselines; for an exploratory fixture, its commissioned reference's."""
    ref = hil.expectation.ref if hil.expectation is not None and hil.expectation.ref else hil.fixture
    return Baselines(ref["fixture_id"], hil.tier)


@pytest.fixture
def recorder(request, hil, thresholds, baselines):
    """Per-test MetricRecorder with telemetry sampling; releases captures afterwards."""
    tid = _test_id(request.node)
    out_dir = hil.run_dir / tid
    out_dir.mkdir(parents=True, exist_ok=True)
    suspect = hil.fixture_suspect and tid in SUSPECT_IF_T03_FAILS
    rec = MetricRecorder(tid, thresholds, baselines, out_dir=out_dir, fixture_suspect=suspect,
                         ignore_base=request.config.getoption("--update-baseline"),
                         expectation=hil.expectation)
    sampler = telemetry.TelemetrySampler(out_dir / "telemetry.csv").start()
    rec.telemetry = sampler

    def add_telemetry(r):
        s = sampler.stop()
        r.record("soc_temp_max_c", s["soc_temp_max_c"])
        r.record("throttled_during_test", s["throttled_during_test"])

    rec.pre_finalize.append(add_telemetry)
    yield rec
    if not rec.finalized:
        rec.finalize(fail=False)
    passed = getattr(request.node, "rep_call", None) is not None and request.node.rep_call.passed
    mgr = request.node.funcargs.get("capture_manager")
    if mgr is not None:
        mgr.release_test(tid, passed)


# ---------------------------------------------------------------- session end

def pytest_sessionfinish(session, exitstatus):
    ctx = session.config._hil
    if ctx is None:
        return
    if any(el.get("disable_command") for el in ctx.fixture["hardware"]["chain"]):
        try:     # leave amplifier enable lines inactive, whatever happened in the session
            from tests.hil.orca_hil.radar import uhd_env
            watchdog.disable_amplifiers(ctx.fixture, print, uhd_env(ctx.fixture))
        except watchdog.WatchdogError as e:
            print(f"\nDISABLE FAILED: {e}", file=sys.stderr)
            ctx.halted = ctx.halted or f"disable at session end failed: {e}"
    duration = time.time() - ctx.t0
    th = load_thresholds().get("session", {})
    session_metrics = []
    if ctx.tier == "fast" and "fast_tier_duration_s" in th:
        check = th["fast_tier_duration_s"][0]
        ev = evaluate(duration, check)
        session_metrics.append({"name": "fast_tier_duration_s", "value": round(duration, 1),
                                "units": check.units, "kind": check.kind, "bound": ev.bound,
                                "status": ev.status})
        if ev.status == "fail":
            session.exitstatus = 1
    if ctx.halted:
        session.exitstatus = session.exitstatus or 1
    statuses = set(ctx.test_status.values())
    overall = ("fail" if ctx.halted or "fail" in statuses or any(m["status"] == "fail" for m in session_metrics)
               else "pass" if statuses else "empty")
    yamlio.dump({
        "run_id": ctx.run_id, "tier": ctx.tier, "fixture_id": ctx.fixture_id,
        "started_utc": ctx.started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": ctx.git["sha"], "git_branch": ctx.git["branch"], "dirty": ctx.git["dirty"],
        "fixture_status": ctx.fixture["status"], "standard": ctx.standard,
        "fixture_extends": ctx.fixture.get("extends"), "fixture_judge": ctx.fixture.get("judge"),
        "fixture_sha256": ctx.fixture_sha256, "setup_note": ctx.setup_note,
        "expectation": ctx.setup["expectation"], "halted": ctx.halted,
        "pytest_args": sys.argv[1:], "status": overall, "fixture_suspect": ctx.fixture_suspect,
        "tests": dict(sorted(ctx.test_status.items())), "session_metrics": session_metrics,
    }, ctx.run_dir / "run.yaml")
    if session.config.getoption("--publish"):
        from tests.hil.orca_hil import publish
        try:
            msg = publish.publish(ctx.run_dir, publish.default_repo(ctx.fixture),
                                  publish_plots=session.config.getoption("--publish-plots"))
            print(f"\npublished: {msg}")
        except Exception as e:  # results are still in the run directory; publish later with publish.py
            print(f"\npublish failed: {e}", file=sys.stderr)
            session.exitstatus = session.exitstatus or 1


def pytest_report_header(config):
    ctx = config._hil
    if ctx is not None and not ctx.standard:
        why = ctx.setup_note or f"{ctx.fixture['status']} fixture, judge: {ctx.fixture.get('judge')}"
        return [f"HIL NON-STANDARD SETUP {ctx.fixture_id}: {why}"]
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    ctx = config._hil
    if ctx is not None:
        if not ctx.standard:
            why = ctx.setup_note or f"{ctx.fixture['status']} fixture {ctx.fixture_id}"
            terminalreporter.write_sep("-", f"NON-STANDARD SETUP: {why}")
        if ctx.halted:
            terminalreporter.write_sep("!", f"SESSION HALTED: {ctx.halted}")
        terminalreporter.write_sep("-", f"HIL run directory: {ctx.run_dir}")
