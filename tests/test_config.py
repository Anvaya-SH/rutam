import pytest

from rutam import RUTAMConfig
from rutam.config import GraphConfig, MemoryConfig, RuntimeConfig
from rutam.exceptions import ValidationError


def test_defaults_and_deterministic_serialization(tmp_path):
    config = RUTAMConfig()
    assert config.backbone == "LiquidAI/LFM3-1.7B-Base"
    assert config.graph.embedding_dim == 128
    assert config.graph.initial_nodes == 0
    assert config.memory.require_provenance
    path = tmp_path / "config.yaml"
    path.write_text(config.to_yaml())
    assert RUTAMConfig.from_yaml(path) == config
    assert RUTAMConfig.from_yaml(path).to_yaml() == config.to_yaml()
    assert RUTAMConfig.from_dict(config.to_dict()) == config


def test_partial_yaml_and_nested_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("graph:\n  embedding_dim: 8\nmemory:\n  require_provenance: false\n")
    config = RUTAMConfig.from_yaml(path, overrides={"graph": {"max_nodes": 12}})
    assert config.graph.embedding_dim == 8
    assert config.graph.max_nodes == 12
    assert config.graph.max_edges == 500_000
    assert config.memory.require_provenance is False
    assert RUTAMConfig().graph.max_nodes == 100_000


@pytest.mark.parametrize(
    "data",
    [
        {"graph": {"embedding_dim": 0}},
        {"graph": {"embedding_dim": True}},
        {"graph": {"embedding_dim": 1.5}},
        {"graph": {"max_nodes": -1}},
        {"graph": {"max_edges": 0}},
        {"graph": {"directed": "true"}},
        {"graph": {"auto_save": 1}},
        {"graph": {"initial_nodes": 1}},
        {"graph": {"path": ""}},
        {"graph": {"unknown": 7}},
        {"graph": None},
        {"memory": {"require_provenance": "false"}},
        {"memory": {"validation_enabled": 1}},
        {"runtime": {"device": "cuda"}},
        {"runtime": {"dtype": "float16"}},
        {"unknown": 1},
        {"backbone": ""},
        {"project_name": 42},
        [],
    ],
)
def test_invalid_configurations(data):
    with pytest.raises(ValidationError):
        RUTAMConfig.from_dict(data)


@pytest.mark.parametrize(
    "override",
    [
        {"graph": {"embedding_dim": -4}},
        {"graph": {"typo": 3}},
        {"typo": 3},
        {"graph": 3},
        {"memory": {"require_provenance": 1}},
        [],
    ],
)
def test_invalid_overrides(override):
    with pytest.raises(ValidationError):
        RUTAMConfig().with_overrides(override)


def test_direct_config_construction_is_validated():
    assert RUTAMConfig(graph=GraphConfig(embedding_dim=4)).graph.embedding_dim == 4
    assert MemoryConfig(require_provenance=False).require_provenance is False
    assert RuntimeConfig().device == "cpu"
    with pytest.raises(ValidationError):
        RUTAMConfig(graph={})


@pytest.mark.parametrize("text", ["graph: [", "[1, 2]", "!!python/object:os.system {}"])
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        RUTAMConfig.from_yaml(path)


def test_empty_and_missing_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    assert RUTAMConfig.from_yaml(path) == RUTAMConfig()
    with pytest.raises(ValidationError):
        RUTAMConfig.from_yaml(tmp_path / "missing.yaml")
