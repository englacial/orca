"""YAML helpers (ruamel.yaml only) with numpy-aware conversion."""
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


def to_builtin(obj):
    """Recursively convert numpy scalars/arrays and Paths to plain Python types."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load(path, roundtrip=False):
    """Load a YAML file. ``roundtrip=True`` keeps comments for later re-dumping."""
    yaml = YAML() if roundtrip else YAML(typ="safe")
    with open(path) as f:
        return yaml.load(f)


def dump(obj, path, roundtrip=False):
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(obj if roundtrip else to_builtin(obj), f)
