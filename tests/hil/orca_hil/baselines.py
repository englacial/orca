"""Per-fixture golden baselines (tests/hil/baselines/<fixture_id>.yaml).

Entries are kept per tier (top-level ``fast:`` / ``full:``), since a test's
parameters, and so some metric values, differ between tiers. Scalar metrics are
stored as {mean, std, n, runs} under the test id, keyed by metric name, or by
"<case>/<name>" for per-case metrics. Comments in the file
are preserved when it is updated. Non-scalar entries (e.g. a list of
narrowband lines) are stored as-is and read with ``get_raw``.
"""
from datetime import datetime, timezone

import numpy as np
from ruamel.yaml.comments import CommentedMap

from . import HIL_DIR, yamlio

BASELINE_DIR = HIL_DIR / "baselines"


def baseline_key(name, case=None):
    return f"{case}/{name}" if case else name


def aggregate(values):
    """mean/std/n of repeated measurements (sample std; 0 for a single value)."""
    v = np.asarray([float(x) for x in values], dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "n": int(v.size)}


class Baselines:
    def __init__(self, fixture_id, tier="fast", directory=BASELINE_DIR):
        self.fixture_id = fixture_id
        self.tier = tier
        self.path = directory / f"{fixture_id}.yaml"
        self.reference_dir = directory / fixture_id / tier
        if self.path.exists():
            self.data = yamlio.load(self.path, roundtrip=True) or CommentedMap()
        else:
            self.data = CommentedMap(fixture_id=fixture_id)
        if tier not in self.data:
            self.data[tier] = CommentedMap()
        self.section = self.data[tier]

    @property
    def exists(self):
        return self.path.exists() and bool(self.section)

    def get_raw(self, test_id, key):
        return (self.section.get(test_id) or {}).get(key)

    def get(self, test_id, name, case=None):
        """Return {mean, std, n} for a scalar metric, or None if there is no baseline."""
        entry = self.get_raw(test_id, baseline_key(name, case))
        if isinstance(entry, dict) and "mean" in entry:
            return {"mean": entry["mean"], "std": entry.get("std", 0.0), "n": entry.get("n", 1)}
        return None

    def set(self, test_id, key, value):
        if test_id not in self.section:
            self.section[test_id] = CommentedMap()
        self.section[test_id][key] = yamlio.to_builtin(value)

    def update_scalar(self, test_id, name, values, runs, case=None):
        entry = aggregate(values)
        entry["runs"] = list(runs)
        self.set(test_id, baseline_key(name, case), entry)

    def save(self, git_sha=None):
        self.data["fixture_id"] = self.fixture_id
        self.data["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.data["git_sha"] = git_sha
        yamlio.dump(self.data, self.path, roundtrip=True)

    def reference_path(self, test_id):
        return self.reference_dir / f"{test_id}_reference.zarr.zip"
