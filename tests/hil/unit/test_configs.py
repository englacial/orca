import shutil
from pathlib import Path

import pytest

from tests.hil.orca_hil import REPO_ROOT, configs, yamlio


@pytest.fixture
def fixture():
    return {"safety": {"tx_gain_max": 75},
            "rf_base": {"freq": 330e6, "tx_gain": 40, "rx_gain": 36}}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = configs.deep_merge(base, {"a": {"b": 9}, "e": 4})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3, "e": 4}
    assert base["a"]["b"] == 1


def test_layers_apply_in_order(fixture):
    cfg = configs.merged_config(fixture, {"RF0": {"tx_gain": 50}, "CHIRP": {"num_pulses": 10}})
    assert cfg["GENERATE"]["chirp_bandwidth"] == 20e6          # base
    assert cfg["DEVICE"]["cpu_format"] == "fc32"               # default.yaml structure
    assert cfg["RF0"]["rx_gain"] == 36                         # fixture
    assert cfg["RF0"]["tx_gain"] == 50                         # override
    assert cfg["CHIRP"]["rx_duration"] * cfg["GENERATE"]["sample_rate"] == pytest.approx(1400)


def test_build_config_writes_repo_relative(fixture, tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copy(REPO_ROOT / "config" / "default.yaml", tmp_path / "config" / "default.yaml")
    monkeypatch.setattr(configs, "REPO_ROOT", tmp_path)
    rel = configs.build_config(fixture, {"CHIRP": {"num_pulses": 100}}, tmp_path / "hil_scratch" / "case")
    assert rel == Path("hil_scratch/case/config.yaml")
    cfg = yamlio.load(tmp_path / rel)
    assert cfg["FILES"]["output_dir"] == "hil_scratch/case"
    assert cfg["CHIRP"]["num_pulses"] == 100


@pytest.mark.parametrize("overrides,msg", [
    ({"RF0": {"tx_gain": 80}}, "tx_gain_max"),
    ({"GENERATE": {"out_file": "other.bin"}}, "chirp_loc"),
    ({"CHIRP": {"rx_duration": 25.01e-6}}, "rx_duration"),
    ({"DEVICE": {"cpu_format": "sc16"}}, "cpu_format"),
    ({"CHIRP": {"num_pulses": None}}, "num_pulses"),
    ({"GENERATE": {"sample_rate": 28e6}}, "rx_rate"),
])
def test_validation_errors(fixture, overrides, msg):
    cfg = configs.merged_config(fixture, configs.deep_merge({"CHIRP": {"num_pulses": 10},
                                                             "FILES": {"output_dir": "x"}}, overrides))
    with pytest.raises(configs.ConfigError, match=msg):
        configs.validate(cfg, fixture)


@pytest.mark.parametrize("out", ["/abs/path", "../outside", ""])
def test_output_dir_must_be_repo_relative(fixture, out):
    cfg = configs.merged_config(fixture, {"CHIRP": {"num_pulses": 10}, "FILES": {"output_dir": out}})
    with pytest.raises(configs.ConfigError, match="output_dir"):
        configs.validate(cfg, fixture)


def test_uncommissioned_fixture_needs_gain_override():
    fx = {"safety": {"tx_gain_max": 75}, "rf_base": {"freq": 330e6, "tx_gain": None, "rx_gain": None}}
    cfg = configs.merged_config(fx, {"CHIRP": {"num_pulses": 10}, "FILES": {"output_dir": "x"}})
    cfg["RF0"].pop("tx_gain")
    with pytest.raises(configs.ConfigError, match="tx_gain"):
        configs.validate(cfg, fx)
