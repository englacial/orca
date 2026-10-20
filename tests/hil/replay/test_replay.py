"""Replay golden captures through each test's analyze() (HIL plan Section 8.2).

    pytest tests/hil/replay --replay [--update-golden-expected]
"""
import pytest

from tests.hil.orca_hil import golden

GOLDEN = golden.load()
pytestmark = pytest.mark.replay


@pytest.fixture(scope="session")
def golden_dir():
    if GOLDEN is None:
        pytest.skip("no tests/hil/golden.yaml")
    return golden.fetch(GOLDEN)


@pytest.mark.parametrize("test_id", sorted((GOLDEN or {}).get("assets_by_test", {})))
def test_replay(test_id, golden_dir, request):
    _, ms = golden.analyze(test_id, golden.prepare(golden_dir, test_id), GOLDEN["fixture"],
                           GOLDEN["params"].get(test_id))
    got = golden.metric_values(ms)
    if request.config.getoption("--update-golden-expected"):
        doc = golden.load()
        doc["expected"][test_id] = got
        golden.yamlio.dump(doc, golden.GOLDEN_YAML)
        return
    diffs = golden.compare(test_id, GOLDEN["expected"][test_id], got, GOLDEN.get("tolerances"))
    assert not diffs, f"{test_id} replay differs from golden.yaml:\n  " + "\n  ".join(diffs)
