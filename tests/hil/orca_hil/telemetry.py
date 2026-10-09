"""Host/SDR snapshot (configuration.yaml) and a background telemetry sampler."""
import csv
import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from . import REPO_ROOT

ETTUS_VENDOR_ID = "2500"
PACKAGES = ("numpy", "scipy", "xarray", "dask", "zarr", "matplotlib", "ruamel.yaml", "pytest")


def _run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kw).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(repo=REPO_ROOT) -> dict:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    return {"sha": sha, "branch": branch, "dirty": bool(status)}


def soc_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except OSError:
        return None


def throttled() -> int | None:
    """vcgencmd get_throttled bit field (bits 0-3: now, bits 16-19: since boot)."""
    out = _run(["vcgencmd", "get_throttled"])
    m = re.search(r"0x([0-9a-fA-F]+)", out or "")
    return int(m.group(1), 16) if m else None


def cpu_governor() -> str | None:
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return None


def usb_devices(vendor=ETTUS_VENDOR_ID) -> list[dict]:
    """USB devices from sysfs with the given vendor id: product, speed (Mb/s), serial."""
    devs = []
    for d in Path("/sys/bus/usb/devices").glob("*"):
        try:
            if (d / "idVendor").read_text().strip() != vendor:
                continue
            devs.append({"path": d.name, "product_id": (d / "idProduct").read_text().strip(),
                         "speed_mbps": int((d / "speed").read_text().strip()),
                         "serial": (d / "serial").read_text().strip() if (d / "serial").exists() else None})
        except (OSError, ValueError):
            continue
    return devs


def free_disk_gb(path) -> float:
    return shutil.disk_usage(Path(path).expanduser()).free / 1e9


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for pkg in PACKAGES:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def host_snapshot(fixture, data_root) -> dict:
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n")
    except OSError:
        model = platform.machine()
    return {
        "pi_model": model,
        "kernel": platform.release(),
        "cpu_governor": cpu_governor(),
        "throttled": throttled(),
        "soc_temp_c": soc_temp_c(),
        "free_disk_gb": round(free_disk_gb(data_root), 1),
        "usb_devices": usb_devices(),
        "packages": package_versions(),
    }


class TelemetrySampler:
    """Samples SoC temperature, throttle state, CPU load and memory every ``interval`` s."""
    FIELDS = ("time", "soc_temp_c", "throttled", "cpu_percent", "mem_percent")

    def __init__(self, path, interval=5.0):
        self.path = Path(path)
        self.interval = interval
        self.rows = []
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        row = {"time": round(time.time(), 3), "soc_temp_c": soc_temp_c(), "throttled": throttled(),
               "cpu_percent": psutil.cpu_percent(), "mem_percent": psutil.virtual_memory().percent}
        self.rows.append(row)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        psutil.cpu_percent()  # prime
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._sample()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        return self.summary()

    def summary(self) -> dict:
        temps = [r["soc_temp_c"] for r in self.rows if r["soc_temp_c"] is not None]
        flags = [r["throttled"] for r in self.rows if r["throttled"] is not None]
        return {"soc_temp_max_c": max(temps) if temps else None,
                "throttled_during_test": any(f & 0xF for f in flags)}
