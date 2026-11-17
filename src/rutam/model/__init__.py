"""Native neural model subsystems, without loading or modifying a backbone."""

from rutam.config import GraphFusionConfig, LFM3AdapterConfig, RecurrentConfig, RutamModelConfig
from rutam.model.adapter_config import LFM3AdapterConfig as AdapterConfigReExport
from rutam.model.adapter_output import AdapterOutput, BackboneInspectionReport
from rutam.model.architecture_inspection import inspect_backbone
from rutam.model.architecture_manifest import (
    ArchitectureManifest,
    generate_architecture_manifest,
)
from rutam.model.backbone_protocol import BackboneProtocol
from rutam.model.forward_context import ForwardContext
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.fusion_gate import RUTAMFusionGate
from rutam.model.fusion_output import FusionOutput
from rutam.model.fusion_projection import FusionProjections
from rutam.model.graph_attention import RUTAMGraphAttention
from rutam.model.graph_attention_bridge import GraphAttentionBridge
from rutam.model.graph_integration import GraphMemoryBridge
from rutam.model.layer_integration import (
    IntegratedLayerWrapper,
    LayerIntegrationHook,
    SyntheticBackbone,
    SyntheticDecoderLayer,
)
from rutam.model.lfm3_bridge import RUTAMLFM3Bridge, inspect_lfm3_stack
from rutam.model.lfm3_adapter import RUTAMLFM3Adapter
from rutam.model.loop_controller import DynamicLoopController
from rutam.model.loop_embeddings import LoopPositionEmbedding
from rutam.model.model_factory import RUTAMModelFactory
from rutam.model.model_registry import (
    ModelRegistry,
    ModelRegistryEntry,
    create_registered_model,
    default_registry,
    get_model_config,
    list_registered_models,
    register_model,
)
from rutam.model.parameter_groups import ParameterGroupManager
from rutam.model.query_gating import QueryDecisionGate, query_decision_loss
from rutam.model.query_head import RUTAMNeuralQueryHead, query_norm_loss
from rutam.model.query_output import (
    QueryGateOutput,
    QueryHeadOutput,
    QueryPoolingOutput,
    QuerySelectionOutput,
)
from rutam.model.query_pooling import QueryPooling
from rutam.model.recurrent import RUTAMRecurrentEngine
from rutam.model.recurrent_block import RUTAMRecurrentBlock
from rutam.model.recurrent_gate import RecurrentResidualGate
from rutam.model.recurrent_integration import RUTAMRecurrentIntegration
from rutam.model.recurrent_output import RecurrentOutput
from rutam.model.rutam_forward import execute_rutam_forward
from rutam.model.rutam_model import ModelMode, RUTAMForCausalLM
from rutam.model.rutam_outputs import RUTAMCausalLMOutput

__all__ = [
    "AdapterConfigReExport",
    "AdapterOutput",
    "ArchitectureManifest",
    "BackboneInspectionReport",
    "BackboneProtocol",
    "DynamicLoopController",
    "ForwardContext",
    "FusionOutput",
    "FusionProjections",
    "GraphAttentionBridge",
    "GraphFusionConfig",
    "GraphMemoryBridge",
    "IntegratedLayerWrapper",
    "LFM3AdapterConfig",
    "LayerIntegrationHook",
    "LoopPositionEmbedding",
    "ModelMode",
    "ModelRegistry",
    "ModelRegistryEntry",
    "ParameterGroupManager",
    "QueryDecisionGate",
    "QueryGateOutput",
    "QueryHeadOutput",
    "QueryPooling",
    "QueryPoolingOutput",
    "QuerySelectionOutput",
    "RUTAMCausalLMOutput",
    "RUTAMForCausalLM",
    "RUTAMFusionGate",
    "RUTAMGraphAttention",
    "RUTAMGraphFusion",
    "RUTAMLFM3Bridge",
    "RUTAMLFM3Adapter",
    "RUTAMModelFactory",
    "RUTAMNeuralQueryHead",
    "RUTAMRecurrentBlock",
    "RUTAMRecurrentEngine",
    "RUTAMRecurrentIntegration",
    "RecurrentConfig",
    "RecurrentOutput",
    "RecurrentResidualGate",
    "RutamModelConfig",
    "SyntheticBackbone",
    "SyntheticDecoderLayer",
    "create_registered_model",
    "default_registry",
    "execute_rutam_forward",
    "generate_architecture_manifest",
    "get_model_config",
    "inspect_backbone",
    "inspect_lfm3_stack",
    "list_registered_models",
    "query_decision_loss",
    "query_norm_loss",
    "register_model",
]
