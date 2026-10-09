"""ORCA hardware-in-the-loop test harness library.

Importing this package puts ORCA's ``preprocessing/`` and ``postprocessing/``
directories on ``sys.path`` (they are not packages, and ``processing_dask``
imports ``processing`` as a top-level module).
"""
import subprocess
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=here,
                             capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return here.parents[2]  # tests/hil/orca_hil -> repo root


REPO_ROOT = _find_repo_root()

for _sub in ("preprocessing", "postprocessing"):
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

HIL_DIR = REPO_ROOT / "tests" / "hil"
