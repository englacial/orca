"""Fixture loading, extends, chain validation and chain values."""
import pytest

from tests.hil.orca_hil import fixtures, golden, yamlio
from tests.hil.orca_hil.analysis.common import expected_separation_samples

BASE_ID = "b206mini_pi5_loopback_50m"
LNA = {"id": "lna", "type": "amplifier", "role": "lna", "gain_db": 15, "nf_db": 0.6, "max_input_dbm": 10,
       "powered": True}


def write(path, doc):
    yamlio.dump(doc, path)
    return path


@pytest.fixture
def base():
    return fixtures.load(fixtures.resolve_path(BASE_ID))


@pytest.fixture
def child_doc(base):
    return {"fixture_id": "rx_lna", "extends": BASE_ID, "description": "LNA before RX",
            "hardware": {"chain": base["hardware"]["chain"] + [LNA]}}


def test_base_fixture_is_commissioned_with_cable_delay(base):
    assert base["status"] == "commissioned"
    assert "judge" not in base and "extends" not in base
    assert fixtures.is_standard(base)
    assert fixtures.chain_delay_s(base) == pytest.approx(50.0 / (0.66 * fixtures.C0))


def test_expected_separation_unchanged_from_cable_formula(base):
    fs = 56e6
    assert expected_separation_samples(base, fs) == pytest.approx(50.0 / (0.66 * 299792458.0) * fs)


def test_child_inherits_parent_and_replaces_chain(tmp_path, base, child_doc):
    fx = fixtures.load(write(tmp_path / "rx_lna.yaml", child_doc))
    assert fx["fixture_id"] == "rx_lna" and fx["extends"] == BASE_ID
    assert fx["host"] == base["host"] and fx["rf_base"] == base["rf_base"]
    assert fx["hardware"]["sdr_serial"] == base["hardware"]["sdr_serial"]
    assert [e["id"] for e in fx["hardware"]["chain"]] == ["pad30", "cable50", "lna"]
    assert fx["status"] == "exploratory" and fx["judge"] == "info"   # status is not inherited
    assert not fixtures.is_standard(fx)
    assert fixtures.parent(fx)["fixture_id"] == BASE_ID
    assert len(fx["source_files"]) == 2


def test_nested_mapping_override_keeps_siblings(tmp_path, base, child_doc):
    child_doc["safety"] = {"tx_gain_max": 40.0}
    fx = fixtures.load(write(tmp_path / "c.yaml", child_doc))
    assert fx["safety"]["tx_gain_max"] == 40.0
    assert fx["safety"]["attenuator_check"] == base["safety"]["attenuator_check"]


def test_extends_by_relative_path_and_element_delay(tmp_path, child_doc):
    write(tmp_path / "rx_lna.yaml", child_doc)
    doc = {"fixture_id": "rx_lna_bpf", "extends": "rx_lna.yaml", "judge": "expect",
           "hardware": {"chain": child_doc["hardware"]["chain"]
                        + [{"id": "bpf", "type": "filter", "loss_db": 1.5, "delay_ns": 5}]}}
    sub = tmp_path / "sub"
    sub.mkdir()
    doc["extends"] = "../rx_lna.yaml"
    fx = fixtures.load(write(sub / "rx_lna_bpf.yaml", doc))
    assert fx["extends"] == "rx_lna" and fx["judge"] == "expect"
    assert len(fx["source_files"]) == 3
    parent = fixtures.parent(fx)
    assert fixtures.chain_delay_s(fx) - fixtures.chain_delay_s(parent) == pytest.approx(5e-9)


def test_extends_cycle(tmp_path):
    a = {"fixture_id": "a", "extends": "b.yaml", "hardware": {"chain": []}}
    b = {"fixture_id": "b", "extends": "a.yaml", "hardware": {"chain": []}}
    write(tmp_path / "a.yaml", a)
    write(tmp_path / "b.yaml", b)
    with pytest.raises(fixtures.FixtureError, match="cycle"):
        fixtures.load(tmp_path / "a.yaml")


@pytest.mark.parametrize("change, match", [
    (lambda d: d["hardware"]["chain"].append({"id": "x", "type": "balun"}), "unknown type"),
    (lambda d: d["hardware"]["chain"].append({"id": "x", "type": "limiter", "loss_db": 0.5}), "missing"),
    (lambda d: d["hardware"]["chain"].append({**LNA, "id": "y", "gain": 3}), "unknown fields"),
    (lambda d: d["hardware"]["chain"].append({**LNA, "id": "y", "role": "driver"}), "role"),
    (lambda d: d["hardware"]["chain"].append({**LNA, "id": "y", "source": "guess"}), "source"),
    (lambda d: d["hardware"]["chain"].append(dict(LNA)), "duplicate"),
    (lambda d: d["hardware"]["chain"].append(
        {**LNA, "id": "y", "gain_db": {"freq_mhz": [300, 360], "value": [15]}}), "equal"),
    (lambda d: d["hardware"]["chain"].append(
        {**LNA, "id": "y", "gain_db": {"freq_mhz": [360, 300], "value": [15, 14]}}), "increase"),
    (lambda d: d["hardware"]["chain"].append({**LNA, "id": "y", "gain_db": "15 dB"}), "finite numbers"),
    (lambda d: d.update(status="commissioned", judge="info"), "only to exploratory"),
    (lambda d: d.update(judge="strict"), "judge"),
    (lambda d: d.update(fixture_id="rx lna"), "fixture_id"),
    (lambda d: d.update(status="draft"), "status"),
])
def test_validation_errors(tmp_path, child_doc, change, match):
    change(child_doc)
    with pytest.raises(fixtures.FixtureError, match=match):
        fixtures.load(write(tmp_path / "bad.yaml", child_doc))


def test_frequency_table_interpolation():
    table = {"freq_mhz": [300, 360], "value": [16.0, 14.0]}
    assert fixtures.value_at(table, 330e6) == pytest.approx(15.0)
    assert fixtures.value_at(table, 100e6) == pytest.approx(16.0)   # clamped
    assert fixtures.value_at(12.5, 330e6) == 12.5
    assert fixtures.value_at(None, 330e6) == 0.0


def test_resolve_path(tmp_path, child_doc):
    assert fixtures.resolve_path(BASE_ID).name == f"{BASE_ID}.yaml"
    assert fixtures.resolve_path().name == f"{BASE_ID}.yaml"     # the only file in fixtures/
    p = write(tmp_path / "rx_lna.yaml", child_doc)
    assert fixtures.resolve_path(str(p)) == p.resolve()
    with pytest.raises(fixtures.FixtureError, match="not found"):
        fixtures.resolve_path("no_such_fixture")


def test_digest_tracks_content_not_order(tmp_path, base, child_doc):
    reordered = dict(reversed(list(base.items())))
    assert fixtures.digest(reordered) == fixtures.digest(base)
    assert fixtures.digest(fixtures.load(write(tmp_path / "c.yaml", child_doc))) != fixtures.digest(base)


def test_legacy_snapshot_converts_to_same_delay(base):
    legacy = (golden.load() or {}).get("fixture")
    if legacy is None or "chain" in legacy["hardware"]:
        pytest.skip("golden.yaml has no legacy fixture snapshot")
    conv = fixtures.from_legacy(legacy)
    assert [e["type"] for e in conv["hardware"]["chain"]] == ["attenuator", "cable"]
    fixtures.validate({**conv, "status": "commissioned"})
    assert fixtures.chain_delay_s(conv) == pytest.approx(fixtures.chain_delay_s(base))
    assert "cable" in legacy["hardware"]    # input not modified
