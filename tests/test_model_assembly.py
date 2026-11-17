"""Unit tests for RUTAM v0.1 model assembly, factory, registry, modes, and runtime subsystems."""

from __future__ import annotations

import pytest
import torch

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.operations import Node
from rutam.memory.provenance import AuthorizationContext
from rutam.model.architecture_manifest import generate_architecture_manifest
from rutam.model.layer_integration import SyntheticBackbone
from rutam.model.model_factory import RUTAMModelFactory
from rutam.model.model_registry import ModelRegistry, default_registry
from rutam.model.rutam_model import ModelMode, RUTAMForCausalLM
from rutam.runtime.capabilities import RuntimeCapabilities
from rutam.runtime.inference import RUTAMInferenceEngine
from rutam.runtime.memory_runtime import MemoryRuntime
from rutam.runtime.session import RUTAMSession


def _create_small_config(mode: str = "full_rutam") -> RUTAMConfig:
    """Helper creating a fast lightweight configuration for unit tests."""
    return RUTAMConfig().with_overrides({
        "graph": {"embedding_dim": 16},
        "encoder": {"input_dim": 16, "hidden_dim": 16, "num_relation_types": 4},
        "selector": {"query_dim": 16, "hidden_dim": 16, "score_dim": 16, "top_k": 2},
        "query_head": {
            "backbone_hidden_dim": 16,
            "graph_hidden_dim": 16,
            "query_dim": 16,
            "intermediate_dim": 16,
        },
        "traversal": {
            "hidden_dim": 16,
            "path_hidden_dim": 16,
            "query_dim": 16,
            "num_relation_types": 4,
        },
        "graph_fusion": {"hidden_dim": 16, "graph_dim": 16, "num_attention_heads": 2},
        "recurrent": {
            "enabled": True,
            "hidden_dim": 16,
            "graph_dim": 16,
            "num_heads": 2,
            "intermediate_dim": 32,
            "max_loops": 3,
            "min_loops": 1,
            "default_loops": 2,
            "dynamic_halting": False,
        },
        "memory_write": {
            "backbone_hidden_dim": 16,
            "hidden_dim": 16,
            "projection_dim": 16,
        },
        "rutam_model": {"backbone_frozen": False, "return_memory_diagnostics": True},
    }).with_mode(mode)


def test_model_modes_configuration() -> None:
    """Requirement 1: Verify config with_mode accurately configures component flags."""
    base = RUTAMConfig()

    # Baseline
    cfg_base = base.with_mode("baseline")
    assert cfg_base.rutam_model.graph_enabled is False
    assert cfg_base.rutam_model.graph_fusion_enabled is False
    assert cfg_base.rutam_model.recurrent_loops_enabled is False
    assert cfg_base.recurrent.enabled is False
    assert cfg_base.rutam_model.mode == "baseline"

    # Graph native
    cfg_gn = base.with_mode("graph_native")
    assert cfg_gn.rutam_model.graph_enabled is True
    assert cfg_gn.rutam_model.graph_fusion_enabled is True
    assert cfg_gn.rutam_model.recurrent_loops_enabled is False
    assert cfg_gn.recurrent.enabled is False
    assert cfg_gn.rutam_model.mode == "graph_native"

    # Recurrent
    cfg_rec = base.with_mode("recurrent")
    assert cfg_rec.rutam_model.graph_enabled is False
    assert cfg_rec.rutam_model.graph_fusion_enabled is False
    assert cfg_rec.rutam_model.recurrent_loops_enabled is True
    assert cfg_rec.recurrent.enabled is True
    assert cfg_rec.rutam_model.mode == "recurrent"

    # Full RUTAM
    cfg_full = base.with_mode("full_rutam")
    assert cfg_full.rutam_model.graph_enabled is True
    assert cfg_full.rutam_model.graph_fusion_enabled is True
    assert cfg_full.rutam_model.recurrent_loops_enabled is True
    assert cfg_full.recurrent.enabled is True
    assert cfg_full.rutam_model.mode == "full_rutam"

    # Invalid mode
    with pytest.raises(ValidationError):
        base.with_mode("invalid_mode")


def test_model_factory_no_duplicate_instances() -> None:
    """Requirement 2: Model factory builds all modules without duplicate or dangling instances."""
    cfg = _create_small_config("full_rutam")
    model = RUTAMModelFactory.create_model(cfg)

    assert isinstance(model, RUTAMForCausalLM)
    # Ensure memory controller shares the exact same submodule instances
    assert model.memory_controller.encoder is model.encoder
    assert model.memory_controller.selector is model.selector
    assert model.memory_controller.query_head is model.query_head
    assert model.memory_controller.traversal is model.traversal
    assert model.memory_controller.write_head is model.write_head
    assert model.memory_controller.validator is model.validator
    assert model.memory_controller.graph is model.graph
    assert model.bridge.graph is model.graph
    assert model.recurrent is not None


def test_model_factory_attach_backbone() -> None:
    """Requirement 2: Model factory attaches backbone satisfying BackboneProtocol."""
    cfg = _create_small_config("graph_native")
    backbone = SyntheticBackbone(hidden_dim=16, vocab_size=50, num_layers=2)

    model = RUTAMModelFactory.attach_backbone(backbone, config=cfg)
    assert model.backbone is backbone
    assert model.is_graph_native

    # Test invalid backbone rejected
    class NotABackbone:
        pass

    with pytest.raises(ValidationError):
        RUTAMModelFactory.attach_backbone(NotABackbone(), config=cfg)  # type: ignore[arg-type]


def test_all_four_model_modes_execution() -> None:
    """Requirement 1: Test forward behavior across all 4 modes."""
    # 1. Baseline
    cfg_base = _create_small_config("baseline")
    model_base = RUTAMModelFactory.create_model(cfg_base)
    assert model_base.mode == ModelMode.BASELINE.value
    assert model_base.is_baseline
    assert not model_base.is_graph_enabled
    assert not model_base.is_recurrent_enabled

    input_ids = torch.randint(0, 50, (2, 4), dtype=torch.long)
    out_base = model_base(input_ids)
    assert out_base.logits.shape == (2, 4, 1000)
    assert out_base.fusion_output is None
    assert out_base.recurrent_output is None
    assert len(out_base.write_proposals) == 0

    # 2. Graph native
    cfg_gn = _create_small_config("graph_native")
    model_gn = RUTAMModelFactory.create_model(cfg_gn)
    assert model_gn.mode == ModelMode.GRAPH_NATIVE.value
    assert model_gn.is_graph_native
    assert model_gn.is_graph_enabled
    assert not model_gn.is_recurrent_enabled

    # Seed nodes into graph
    model_gn.graph.add_node(
        Node(
            1,
            "Entity1",
            "Entity 1 content",
            embedding=tuple(torch.randn(16).tolist()),
            provenance="prov:init",
        )
    )
    model_gn.graph.add_node(
        Node(
            2,
            "Entity2",
            "Entity 2 content",
            embedding=tuple(torch.randn(16).tolist()),
            provenance="prov:init",
        )
    )

    out_gn = model_gn(input_ids)
    assert out_gn.fusion_output is not None
    assert out_gn.recurrent_output is None

    # 3. Recurrent
    cfg_rec = _create_small_config("recurrent")
    model_rec = RUTAMModelFactory.create_model(cfg_rec)
    assert model_rec.mode == ModelMode.RECURRENT.value
    assert model_rec.is_recurrent
    assert not model_rec.is_graph_enabled
    assert model_rec.is_recurrent_enabled

    out_rec = model_rec(input_ids)
    assert out_rec.fusion_output is None
    assert out_rec.recurrent_output is not None
    assert out_rec.recurrent_output.num_loops >= 1

    # 4. Full RUTAM
    cfg_full = _create_small_config("full_rutam")
    model_full = RUTAMModelFactory.create_model(cfg_full)
    assert model_full.mode == ModelMode.FULL_RUTAM.value
    assert model_full.is_full_rutam
    assert model_full.is_graph_enabled
    assert model_full.is_recurrent_enabled

    # Seed nodes into graph
    model_full.graph.add_node(
        Node(
            1,
            "Entity1",
            "Entity 1 content",
            embedding=tuple(torch.randn(16).tolist()),
            provenance="prov:init",
        )
    )
    model_full.graph.add_node(
        Node(
            2,
            "Entity2",
            "Entity 2 content",
            embedding=tuple(torch.randn(16).tolist()),
            provenance="prov:init",
        )
    )

    out_full = model_full(input_ids)
    assert out_full.fusion_output is not None
    assert out_full.recurrent_output is not None


def test_dynamic_mode_switching() -> None:
    """Verify runtime dynamic mode switching on a model instance."""
    cfg = _create_small_config("baseline")
    model = RUTAMModelFactory.create_model(cfg)
    assert model.mode == "baseline"

    model.set_mode("full_rutam")
    assert model.mode == "full_rutam"
    assert model.is_full_rutam
    assert model.is_graph_enabled
    assert model.is_recurrent_enabled

    model.set_mode("graph_native")
    assert model.mode == "graph_native"
    assert model.is_graph_native

    with pytest.raises(ValidationError):
        model.set_mode("unsupported_mode")


def test_model_registry() -> None:
    """Verify ModelRegistry configuration registration, versioning, and lookup."""
    registry = ModelRegistry(populate_defaults=True)

    # Check pre-registered models
    assert registry.has_model("baseline")
    assert registry.has_model("graph_native")
    assert registry.has_model("recurrent")
    assert registry.has_model("full_rutam")
    assert registry.has_model("rutam-v0.1-full", version="0.1.0")

    models = registry.list_models()
    assert "full_rutam" in models
    assert "rutam-v0.1-baseline" in models

    versions = registry.list_versions("full_rutam")
    assert "0.1.0" in versions

    cfg = registry.get("full_rutam")
    assert isinstance(cfg, RUTAMConfig)
    meta = registry.get_metadata("full_rutam")
    assert meta["mode"] == "full_rutam"

    # Register custom config
    custom_cfg = _create_small_config("recurrent")
    registry.register("custom-model", custom_cfg, version="1.0.0", metadata={"tier": "test"})
    assert registry.has_model("custom-model", version="1.0.0")

    # Creating model from registry
    model = registry.create_model("custom-model", version="1.0.0")
    assert model.is_recurrent

    # Error cases
    with pytest.raises(ValidationError):
        registry.register("custom-model", custom_cfg, version="1.0.0", overwrite=False)
    with pytest.raises(ValidationError):
        registry.get("unknown_model")

    # Global registry exports check
    assert default_registry.has_model("full_rutam")


def test_architecture_manifest() -> None:
    """Verify architecture manifest generation and serializability."""
    cfg = _create_small_config("full_rutam")
    model = RUTAMModelFactory.create_model(cfg)

    manifest = generate_architecture_manifest(model)
    assert manifest.version == "0.1.0"
    assert manifest.mode == "full_rutam"
    assert "backbone" in manifest.to_dict()
    assert "components" in manifest.to_dict()
    assert "capabilities" in manifest.to_dict()
    assert manifest.capabilities["graph_memory"] is True
    assert manifest.capabilities["recurrent_reasoning"] is True

    # JSON serialization
    json_str = manifest.to_json()
    assert '"mode": "full_rutam"' in json_str


def test_runtime_capabilities() -> None:
    """Verify RuntimeCapabilities evaluation and requirement verification."""
    cfg_base = _create_small_config("baseline")
    caps_base = RuntimeCapabilities.from_config(cfg_base)
    assert caps_base.supports_graph_memory is False
    assert caps_base.supports_recurrent_reasoning is False
    assert caps_base.has("generation") is True

    # Base verification
    assert caps_base.verify("generation") is True
    with pytest.raises(ValidationError):
        caps_base.verify("graph_memory")

    # Full RUTAM verification
    cfg_full = _create_small_config("full_rutam")
    caps_full = RuntimeCapabilities.from_config(cfg_full)
    assert caps_full.supports_graph_memory is True
    assert caps_full.supports_recurrent_reasoning is True
    assert caps_full.verify(["graph_memory", "recurrent_reasoning"]) is True


def test_inference_engine_validation_and_generation() -> None:
    """Verify RUTAMInferenceEngine input checks and autoregressive generation loop."""
    cfg = _create_small_config("baseline")
    model = RUTAMModelFactory.create_model(cfg)
    engine = RUTAMInferenceEngine(model, cfg)

    # Input validation tests
    with pytest.raises(ValidationError):
        engine.validate_inputs("invalid")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        engine.validate_inputs(torch.randn(2, 4))  # float instead of long
    with pytest.raises(ValidationError):
        engine.validate_inputs(torch.randint(0, 10, (2, 4, 3), dtype=torch.long))  # 3D
    with pytest.raises(ValidationError):
        engine.validate_inputs(
            torch.randint(0, 10, (2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 5), dtype=torch.bool),  # Mismatched shape
        )

    # Valid forward pass with telemetry
    input_ids = torch.randint(0, 50, (2, 3), dtype=torch.long)
    out = engine.forward(input_ids)
    assert out.logits.shape == (2, 3, 1000)
    diag = engine.get_last_diagnostics()
    assert "engine_latency_ms" in diag
    assert diag["total_tokens"] == 6

    # Autoregressive generation
    gen = engine.generate(input_ids, max_new_tokens=4, temperature=0.0)
    assert gen.shape == (2, 7)  # 3 prompt + 4 new tokens


def test_memory_runtime_lifecycle() -> None:
    """Requirement 3: Test MemoryRuntime read, propose, validate, and explicit commit."""
    cfg = _create_small_config("graph_native")
    model = RUTAMModelFactory.create_model(cfg)
    runtime = MemoryRuntime(model)
    runtime_admin = AuthorizationContext(role="admin")

    assert runtime.is_available is True
    assert runtime.get_node_count(authorization_context=runtime_admin) == 0

    # Seed initial node
    model.graph.add_node(
        Node(
            1,
            "Concept",
            "Concept content",
            embedding=tuple(torch.randn(16).tolist()),
            provenance="prov:init",
        )
    )
    assert runtime.get_node_count(authorization_context=runtime_admin) == 1

    # Propose memory updates
    input_ids = torch.randint(0, 50, (1, 4), dtype=torch.long)
    proposals = runtime.propose(input_ids)
    assert isinstance(proposals, tuple)

    # Validate proposals
    if proposals:
        validation_res = runtime.validate(proposals)
        assert validation_res is not None

        # Commit proposals
        commit_res = runtime.commit(proposals, require_valid=True)
        assert commit_res is not None
        assert commit_res.new_graph_version >= 1


def test_session_multi_turn() -> None:
    """Requirement 3: Test RUTAMSession stateful multi-turn execution and history."""
    cfg = _create_small_config("full_rutam")
    session = RUTAMSession.create(cfg, mode="full_rutam")

    assert session.turn_index == 0
    assert len(session.history) == 0

    input_ids = torch.randint(0, 50, (1, 4), dtype=torch.long)

    # Turn 1: standard step
    res1 = session.step(input_ids)
    assert res1.turn_index == 1
    assert len(session.history) == 1

    # Turn 2: step with generation
    res2 = session.step(input_ids, max_new_tokens=3)
    assert res2.turn_index == 2
    assert res2.generated_ids is not None
    assert res2.generated_ids.shape == (1, 7)

    # Turn 3: step with propose and auto-commit
    res3 = session.step(input_ids, generate_proposals=True, auto_commit=True)
    assert res3.turn_index == 3
    assert len(session.history) == 3

    # Reset history
    session.reset_history()
    assert session.turn_index == 0
    assert len(session.history) == 0
