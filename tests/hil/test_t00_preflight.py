"""T0 - Preflight and fixture health (Table II). Gates all other hardware tests."""
import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.hil.conftest import write_configuration
from tests.hil.orca_hil import radar
from tests.hil.orca_hil.analysis.common import db10
from tests.hil.orca_hil.analysis.profile import locate_peaks
from tests.hil.orca_hil.analysis.common import open_capture

PARAMS = {
    "fast": {"warmup_s": 60, "required_free_gb": 3, "timeout": 600},
    "full": {"warmup_s": 300, "required_free_gb": 15, "timeout": 1200},
}
ATTEN_CHECK_PULSES = 2000
CXX_HW_TESTS = "detectChannels|setRFParams|refLoLockDetect"
USB_ERROR_RE = re.compile(r"usb.*(error|fail|over-current|unable|reset|disconnect)", re.I)


def uhd_checks(fixture, build):
    prefix = radar.uhd_prefix(fixture)
    out = {}
    out["radar_links_custom_uhd"] = bool(build.libuhd) and Path(build.libuhd).resolve().is_relative_to(prefix.resolve())

    conda_lib = Path(os.environ.get("CONDA_PREFIX", "/nonexistent")) / "lib"
    conda_uhd = sorted(conda_lib.glob("libuhd.so*"))
    dpkg = subprocess.run(["dpkg-query", "-W", "-f", "${Package} ${Status}\n", "libuhd*"],
                          capture_output=True, text=True).stdout
    apt_uhd = [ln for ln in dpkg.splitlines() if "install ok installed" in ln]
    out["single_uhd_installed"] = not conda_uhd and not apt_uhd

    stamp = radar.build_stamp(fixture) or {}
    src = Path(fixture["uhd"]["source_dir"]).expanduser()
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out["uhd_stamp_matches_expected_and_head"] = bool(stamp) and (
        stamp.get("commit") == fixture["uhd"]["expected_commit"] == head)

    ver = radar.run_uhd_tool(fixture, "uhd_config_info", ["--version"]).stdout.strip()
    out["uhd_version_matches_stamp"] = bool(ver) and ver.splitlines()[0] == stamp.get("uhd_version")

    images = Path(radar.uhd_env(fixture)["UHD_IMAGES_DIR"])
    out["uhd_images_present"] = (images / fixture["uhd"]["fpga_image"]).is_file()
    details = {"conda_libuhd": [str(p) for p in conda_uhd], "apt_libuhd": apt_uhd, "uhd_source_head": head}
    return out, details


def device_checks(fixture):
    find = radar.run_uhd_tool(fixture, "uhd_find_devices", ["--args", "type=b200"])
    serials = re.findall(r"serial:\s*(\S+)", find.stdout)
    products = re.findall(r"product:\s*(\S+)", find.stdout)
    probe = radar.run_uhd_tool(fixture, "uhd_usrp_probe", timeout=180).stdout
    antennas = re.findall(r"Antennas:\s*(.+)", probe)
    info = {
        "serials_found": serials, "products_found": products,
        "fw_version": (re.findall(r"FW Version:\s*(\S+)", probe) or [None])[0],
        "fpga_version": (re.findall(r"FPGA Version:\s*(\S+)", probe) or [None])[0],
        "antennas": antennas,
        "frontend_sensors": re.findall(r"Sensors:\s*(.+)", probe),
    }
    rx_ants = {a.strip() for line in antennas for a in line.split(",")}
    checks = {
        "sdr_serial_match": serials == [str(fixture["hardware"]["sdr_serial"])] and products == ["B206mini"],
        "antennas_present": {"TX/RX", "RX2"} <= rx_ants,
    }
    return checks, info


def usb_kernel_errors():
    out = subprocess.run(["journalctl", "-k", "--since", "-1h", "--no-pager", "-q"],
                         capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if USB_ERROR_RE.search(ln)]


def run_cxx_hw_tests(fixture):
    p = subprocess.run(["ctest", "--test-dir", "sdr/build", "-R", CXX_HW_TESTS, "--output-on-failure"],
                       cwd=radar.REPO_ROOT, env=radar.uhd_env(fixture), capture_output=True, text=True,
                       timeout=300)
    return p.returncode == 0, p.stdout[-3000:]


@pytest.mark.hardware
@pytest.mark.fast
@pytest.mark.full
@pytest.mark.t00
@pytest.mark.timeout(1500)
def test_t00_preflight(hil, tier, fixture_cfg, radar_build, capture_manager, recorder):
    from tests.hil.orca_hil import telemetry
    params = PARAMS[tier]
    fx = fixture_cfg

    checks, details = uhd_checks(fx, radar_build)
    for name, ok in checks.items():
        recorder.record(name, ok)

    dev_checks, dev_info = device_checks(fx)
    for name, ok in dev_checks.items():
        recorder.record(name, ok)
    usb = [d for d in telemetry.usb_devices() if d["serial"] == str(fx["hardware"]["sdr_serial"])]
    recorder.record("usb_link_mbps", usb[0]["speed_mbps"] if usb else 0)
    flags = telemetry.throttled() or 0
    recorder.record("throttled_flags", flags & 0xF)
    recorder.record("throttled_since_boot_flags", flags >> 16)
    recorder.record("free_disk_gb", telemetry.free_disk_gb(hil.data_root),
                    params={"required": params["required_free_gb"]})
    usb_errs = usb_kernel_errors()
    recorder.record("usb_kernel_errors_1h", len(usb_errs))

    cxx_ok, cxx_out = run_cxx_hw_tests(fx)
    recorder.record("cxx_hardware_tests_passed", cxx_ok)

    hil.configuration.update({"sdr": dev_info, "uhd_checks": details, "usb_kernel_errors_1h": usb_errs})
    write_configuration(hil)
    (hil.run_dir / "t00" / "cxx_hardware_tests.log").write_text(cxx_out)

    # Abort before any RF test if the software/device checks failed
    if recorder.failures:
        recorder.finalize()

    # Attenuator-present check, at low power before any other TX test
    att = fx["safety"]["attenuator_check"]
    if att.get("tx_gain") is None:
        recorder.record("attenuator_check_peak_dbfs", float("nan"), inconclusive=True,
                        note="fixture not commissioned")
    else:
        cap = capture_manager.acquire("t00", "attenuator_check", {
            "RF0": {"tx_gain": att["tx_gain"]}, "CHIRP": {"num_pulses": ATTEN_CHECK_PULSES}})
        _, _, fit, _ = locate_peaks(open_capture(cap.zarr), fx)
        peak = float(db10(abs(fit.a2) ** 2))
        recorder.record("attenuator_check_peak_dbfs", peak)
        expected = att.get("expected_peak_dbfs")
        if expected is not None and peak > expected + att["tolerance_db"]:
            recorder.finalize(fail=False)
            pytest.exit(f"ABORT: loopback peak {peak:.1f} dBFS at tx_gain {att['tx_gain']} is more than "
                        f"{att['tolerance_db']} dB above the expected {expected} dBFS; "
                        "the 30 dB attenuator may be missing", returncode=3)

    # Warm-up: continuous TX at base settings. Presums keep the data volume small.
    if fx["rf_base"].get("tx_gain") is None:
        recorder.record("soc_temp_after_warmup_c", float("nan"), inconclusive=True,
                        note="fixture not commissioned")
    else:
        capture_manager.acquire("t00", "warmup", {"CHIRP": {"num_pulses": -1, "num_presums": 100}},
                                stop_after_s=params["warmup_s"])
        recorder.record("soc_temp_after_warmup_c", telemetry.soc_temp_c())

    recorder.finalize()
