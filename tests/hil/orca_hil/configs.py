"""Radar config builder: base structure + fixture rf_base + per-test overrides."""
import copy
import os
from pathlib import Path

from . import REPO_ROOT, yamlio

# Section 2.3 of the HIL plan: values applied over config/default.yaml's structure.
BASE_OVERRIDES = {
    "GENERATE": {
        "sample_rate": 56e6, "chirp_type": "linear", "chirp_bandwidth": 20e6,
        "lo_offset_sw": 12.5e6, "window": "hamming", "chirp_length": 10e-6,
        "pulse_length": 10e-6, "out_file": "chirp.bin", "show_plot": False,
    },
    "RF0": {
        "rx_rate": 56e6, "tx_rate": 56e6, "freq": 330e6, "lo_offset": 0, "bw": 56e6,
        "tx_ant": "TX/RX", "rx_ant": "RX2", "transmit": True,
    },
    "CHIRP": {
        "time_offset": 1, "tx_duration": 10e-6, "rx_duration": 25e-6,
        "pulse_rep_int": 250e-6, "tx_lead": 0, "num_pulses": None, "num_presums": 1,
        "phase_dithering": True,
    },
    "FILES": {
        "chirp_loc": "chirp.bin", "save_loc": "rx_samps.bin", "gps_loc": "gps_log.txt",
        "max_chirps_per_file": -1,
    },
    "RUN_MANAGER": {"final_save_loc": None, "save_partial_files": False, "save_gps": False},
}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (update or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def base_config() -> dict:
    default = yamlio.load(REPO_ROOT / "config" / "default.yaml")
    return deep_merge(default, BASE_OVERRIDES)


def merged_config(fixture: dict, overrides: dict) -> dict:
    rf_base = {k: v for k, v in (fixture.get("rf_base") or {}).items() if v is not None}
    return deep_merge(deep_merge(base_config(), {"RF0": rf_base}), overrides)


def _is_integer(x, tol=1e-9):
    return abs(x - round(x)) <= tol


def validate(cfg: dict, fixture: dict):
    errors = []
    gen, rf, chirp, files = cfg["GENERATE"], cfg["RF0"], cfg["CHIRP"], cfg["FILES"]
    tx_gain_max = (fixture.get("safety") or {}).get("tx_gain_max")
    fs = float(gen["sample_rate"])

    for key in ("tx_gain", "rx_gain"):
        if rf.get(key) is None:
            errors.append(f"RF0.{key} is not set (fixture not commissioned and no override)")
    if rf.get("tx_gain") is not None and tx_gain_max is not None and rf["tx_gain"] > tx_gain_max:
        errors.append(f"RF0.tx_gain {rf['tx_gain']} exceeds safety.tx_gain_max {tx_gain_max}")
    if gen["out_file"] != files["chirp_loc"]:
        errors.append(f"GENERATE.out_file {gen['out_file']!r} != FILES.chirp_loc {files['chirp_loc']!r}")
    for section, key in (("CHIRP", "rx_duration"), ("GENERATE", "chirp_length"),
                         ("GENERATE", "pulse_length")):
        n = float(cfg[section][key]) * fs
        if not _is_integer(n):
            errors.append(f"{section}.{key} x sample_rate = {n} is not an integer")
    for key in ("rx_rate", "tx_rate"):
        if float(rf[key]) != fs:
            errors.append(f"RF0.{key} {rf[key]} != GENERATE.sample_rate {fs}")
    if cfg["DEVICE"].get("cpu_format") != "fc32":
        errors.append("DEVICE.cpu_format must be 'fc32'")
    if chirp.get("num_pulses") is None:
        errors.append("CHIRP.num_pulses is not set")
    out = files.get("output_dir")
    if not out or os.path.isabs(out) or ".." in Path(out).parts:
        errors.append(f"FILES.output_dir {out!r} must be a repo-relative path")
    if errors:
        raise ConfigError("invalid radar config:\n  " + "\n  ".join(errors))


def build_config(fixture: dict, overrides: dict, case_dir) -> Path:
    """Write <case_dir>/config.yaml and return its path relative to the repo root.

    ``case_dir`` must be inside the repo (normally under the hil_scratch symlink),
    because main.cpp resolves the config, chirp and output paths relative to the
    repo root.
    """
    case_dir = Path(case_dir)
    rel_dir = case_dir if not case_dir.is_absolute() else case_dir.relative_to(REPO_ROOT)
    cfg = merged_config(fixture, overrides)
    cfg["FILES"]["output_dir"] = str(rel_dir)
    validate(cfg, fixture)
    (REPO_ROOT / rel_dir).mkdir(parents=True, exist_ok=True)
    rel_cfg = rel_dir / "config.yaml"
    yamlio.dump(cfg, REPO_ROOT / rel_cfg)
    return rel_cfg
