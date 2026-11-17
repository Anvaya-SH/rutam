"""Model-specific graph bridge for the published LiquidAI LFM3 decoder stack."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import torch
from torch import nn

from rutam.config import RUTAMConfig, positive_int
from rutam.exceptions import ValidationError
from rutam.graph.encoder import GraphEncoderOutput, RUTAMGraphEncoder
from rutam.graph.operations import Graph
from rutam.graph.state import GraphState
from rutam.graph.storage import GraphStorage
from rutam.graph.traversal import RUTAMGraphTraversal
from rutam.memory.provenance import AuthorizationContext
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.fusion_output import FusionOutput
from rutam.model.graph_integration import GraphMemoryBridge
from rutam.model.query_head import RUTAMNeuralQueryHead
from rutam.graph.selector import RUTAMGraphSelector


@dataclass(slots=True)
class _BridgeRequest:
    state: GraphState
    visible_state: GraphState
    attention_mask: torch.Tensor | None
    prefix_hidden: torch.Tensor | None = None
    fusion_outputs: list[FusionOutput] | None = None

    def __post_init__(self) -> None:
        if self.fusion_outputs is None:
            self.fusion_outputs = []


def inspect_lfm3_stack(backbone: nn.Module) -> tuple[nn.Module, int, int]:
    """Resolve and fail closed unless the supplied module has the LFM3 layer-2 contract.

    The check is deliberately structural so importing RUTAM does not require
    Transformers. A loaded Transformers LFM3 model has the same class names and
    config fields described by the pinned bridge specification.
    """
    if not isinstance(backbone, nn.Module):
        raise ValidationError("LFM3 backbone must be a torch.nn.Module")
    config = getattr(backbone, "config", None)
    if getattr(config, "model_type", None) != "lfm3":
        raise ValidationError("Expected an LFM3 config with model_type='lfm3'")

    backbone_name = type(backbone).__name__
    if backbone_name == "Lfm3ForCausalLM":
        stack = getattr(backbone, "model", None)
        if type(stack).__name__ != "Lfm3Model":
            raise ValidationError("Lfm3ForCausalLM must expose its Lfm3Model at .model")
    elif backbone_name == "Lfm3Model":
        stack = backbone
    else:
        raise ValidationError(
            "Expected Lfm3ForCausalLM or an explicitly supplied Lfm3Model, "
            f"got {backbone_name}"
        )

    layers = getattr(stack, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise ValidationError("LFM3 stack must expose .layers")
    layer_types = getattr(config, "layer_types", None)
    if (
        len(layers) != 24
        or getattr(config, "num_hidden_layers", None) != 24
        or not isinstance(layer_types, (list, tuple))
        or len(layer_types) != 24
    ):
        raise ValidationError("Unsupported LFM3 stack: expected 24 layers and layer_types")
    if layer_types[2] != "full_attention":
        raise ValidationError("LFM3 layer 2 must be the first full_attention layer")

    layer = layers[2]
    if type(layer).__name__ != "Lfm3DecoderLayer":
        raise ValidationError("LFM3 layer 2 must be an Lfm3DecoderLayer")
    attention = getattr(layer, "self_attn", None)
    if type(attention).__name__ != "Lfm3Attention":
        raise ValidationError("LFM3 layer 2 must expose an Lfm3Attention as self_attn")

    hidden_size = getattr(config, "hidden_size", None)
    positive_int(hidden_size, "LFM3 config.hidden_size")
    return layer, 2, hidden_size


class RUTAMLFM3Bridge(nn.Module):
    """Run the real LFM3 forward/generate path and fuse RUTAM memory after layer 2.

    The bridge hooks the completed decoder block, not its inner attention module.
    LFM3 remains responsible for embeddings, position/rotary data, hybrid-layer
    dispatch, cache updates, logits, labels and generation. A request pins one
    graph snapshot for its whole forward or generate call.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        config: RUTAMConfig | None = None,
        graph: Graph | None = None,
        storage: GraphStorage | None = None,
        encoder: RUTAMGraphEncoder | None = None,
        selector: RUTAMGraphSelector | None = None,
        query_head: RUTAMNeuralQueryHead | None = None,
        traversal: RUTAMGraphTraversal | None = None,
        fusion: RUTAMGraphFusion | None = None,
    ) -> None:
        super().__init__()
        layer, layer_index, hidden_size = inspect_lfm3_stack(backbone)
        base_config = RUTAMConfig() if config is None else config
        if not isinstance(base_config, RUTAMConfig):
            raise ValidationError("config must be a RUTAMConfig")

        overrides: dict[str, dict[str, Any]] = {}
        query_overrides: dict[str, Any] = {}
        if base_config.query_head.backbone_hidden_dim != hidden_size:
            query_overrides["backbone_hidden_dim"] = hidden_size
        # Decoder generation can include left-padding tokens with no valid prefix.
        # The query head's explicit zero fallback keeps those rows as identity paths.
        if query_head is None and base_config.query_head.empty_sequence_policy == "error":
            query_overrides["empty_sequence_policy"] = "zeros"
        if query_overrides:
            overrides["query_head"] = query_overrides
        if base_config.graph_fusion.hidden_dim != hidden_size:
            overrides["graph_fusion"] = {"hidden_dim": hidden_size}
        self.config = base_config.with_overrides(overrides) if overrides else base_config
        self.backbone = backbone
        self.graph = graph if graph is not None else Graph(self.config)
        if not isinstance(self.graph, Graph):
            raise ValidationError("graph must be a Graph instance")
        if self.graph.config.graph.embedding_dim != self.config.graph.embedding_dim:
            raise ValidationError("graph embedding dimension does not match RUTAM config")
        self.storage = storage
        self.encoder = encoder or RUTAMGraphEncoder(self.config.encoder)
        self.selector = selector or RUTAMGraphSelector(self.config.selector)
        self.query_head = query_head or RUTAMNeuralQueryHead(
            self.config.query_head,
            selector_config=self.config.selector,
        )
        self.traversal = traversal
        if self.traversal is None and self.config.rutam_model.graph_traversal_enabled:
            self.traversal = RUTAMGraphTraversal(self.config.traversal)
        self.fusion = fusion or RUTAMGraphFusion(self.config.graph_fusion)
        if not isinstance(self.encoder, RUTAMGraphEncoder):
            raise ValidationError("encoder must be a RUTAMGraphEncoder")
        if not isinstance(self.selector, RUTAMGraphSelector):
            raise ValidationError("selector must be a RUTAMGraphSelector")
        if not isinstance(self.query_head, RUTAMNeuralQueryHead):
            raise ValidationError("query_head must be a RUTAMNeuralQueryHead")
        if self.traversal is not None and not isinstance(self.traversal, RUTAMGraphTraversal):
            raise ValidationError("traversal must be a RUTAMGraphTraversal or None")
        if not isinstance(self.fusion, RUTAMGraphFusion):
            raise ValidationError("fusion must be a RUTAMGraphFusion")
        if self.encoder.config.input_dim != self.config.graph.embedding_dim:
            raise ValidationError("encoder.input_dim must match graph.embedding_dim")
        graph_hidden_dim = self.encoder.config.hidden_dim
        if self.selector.config.hidden_dim != graph_hidden_dim:
            raise ValidationError("selector.hidden_dim must match encoder.hidden_dim")
        if self.query_head.config.graph_hidden_dim != graph_hidden_dim:
            raise ValidationError("query_head.graph_hidden_dim must match encoder.hidden_dim")
        if self.query_head.config.query_dim != self.selector.config.query_dim:
            raise ValidationError("query_head.query_dim must match selector.query_dim")
        if self.query_head.config.backbone_hidden_dim != hidden_size:
            raise ValidationError("query_head.backbone_hidden_dim must match LFM3 hidden size")
        if self.query_head.config.empty_sequence_policy != "zeros":
            raise ValidationError(
                "query_head.empty_sequence_policy must be 'zeros' for padded LFM3 inputs"
            )
        if self.fusion.config.hidden_dim != hidden_size:
            raise ValidationError("graph_fusion.hidden_dim must match LFM3 config.hidden_size")
        if self.fusion.config.graph_dim != graph_hidden_dim:
            raise ValidationError("graph_fusion.graph_dim must match encoder.hidden_dim")
        if self.traversal is not None and (
            self.traversal.config.hidden_dim != graph_hidden_dim
            or self.traversal.config.query_dim != self.query_head.config.query_dim
        ):
            raise ValidationError(
                "traversal hidden/query dimensions must match encoder and query head"
            )
        self.graph_memory = GraphMemoryBridge(
            self.graph,
            storage=self.storage,
            use_cache=self.config.rutam_model.use_cached_graph_states,
        )
        self.fusion_layer_index = layer_index
        self.hidden_size = hidden_size
        self._request: ContextVar[_BridgeRequest | None] = ContextVar(
            f"rutam_lfm3_request_{id(self)}", default=None
        )
        self._hook_handle = layer.register_forward_hook(self._replace_layer_output)
        self.last_graph_version = 0
        self.last_fusion_outputs: tuple[FusionOutput, ...] = ()

        if self.config.rutam_model.backbone_frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        if not self.config.lfm3_adapter.graph_modules_trainable:
            for module in (self.encoder, self.selector, self.query_head, self.traversal, self.fusion):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)

        if (
            self.config.rutam_model.sparse_selection_enabled
            and self.selector.config.selection_mode == "hard"
            and not self.selector.config.return_attention
            and self.config.rutam_model.max_sparse_nodes < self.selector.config.top_k
        ):
            self.close()
            raise ValidationError("max_sparse_nodes must be at least selector.top_k")

    def close(self) -> None:
        """Remove the decoder hook. Calling this more than once is safe."""
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            handle.remove()
            self._hook_handle = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args: Any,
        graph_state_snapshot: GraphState | None = None,
        node_mask: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to LFM3 unchanged except for the graph fusion hook at layer 2."""
        state = self._resolve_state(graph_state_snapshot)
        mask = kwargs.get("attention_mask")
        request = self._make_request(state, node_mask, authorization_context, mask)
        token = self._request.set(request)
        try:
            if input_ids is None:
                return self.backbone(*args, **kwargs)
            return self.backbone(input_ids, *args, **kwargs)
        finally:
            self.last_graph_version = state.graph_version
            self.last_fusion_outputs = tuple(request.fusion_outputs or ())
            self._request.reset(token)

    def generate(
        self,
        *args: Any,
        graph_state_snapshot: GraphState | None = None,
        node_mask: torch.Tensor | None = None,
        authorization_context: AuthorizationContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate generation while pinning graph state across prefill and cached decode."""
        generate = getattr(self.backbone, "generate", None)
        if not callable(generate):
            raise ValidationError("The supplied LFM3 model does not expose generate()")
        generation_config = kwargs.get("generation_config") or getattr(
            self.backbone, "generation_config", None
        )
        config_value = (
            generation_config.get
            if isinstance(generation_config, dict)
            else lambda key, default: getattr(generation_config, key, default)
        )
        num_beams = kwargs.get("num_beams")
        if num_beams is None:
            num_beams = config_value("num_beams", 1)
        num_return_sequences = kwargs.get("num_return_sequences")
        if num_return_sequences is None:
            num_return_sequences = config_value("num_return_sequences", 1)
        if num_beams != 1 or num_return_sequences != 1:
            raise ValidationError(
                "RUTAM LFM3 cached generation currently supports one sequence per input; "
                "beam/cache reordering is not integrated"
            )
        state = self._resolve_state(graph_state_snapshot)
        request = self._make_request(
            state,
            node_mask,
            authorization_context,
            kwargs.get("attention_mask"),
        )
        token = self._request.set(request)
        try:
            return generate(*args, **kwargs)
        finally:
            self.last_graph_version = state.graph_version
            self.last_fusion_outputs = tuple(request.fusion_outputs or ())
            self._request.reset(token)

    def _resolve_state(self, snapshot: GraphState | None) -> GraphState:
        state = self.graph_memory.get_active_state() if snapshot is None else snapshot
        if not isinstance(state, GraphState):
            raise ValidationError("graph_state_snapshot must be a GraphState")
        state.validate()
        return state

    def _make_request(
        self,
        state: GraphState,
        node_mask: torch.Tensor | None,
        authorization_context: AuthorizationContext | None,
        attention_mask: torch.Tensor | None,
    ) -> _BridgeRequest:
        if authorization_context is not None and not isinstance(
            authorization_context, AuthorizationContext
        ):
            raise ValidationError("authorization_context must be an AuthorizationContext")
        now = datetime.now(UTC)

        def currently_valid(record: dict[str, Any]) -> bool:
            try:
                valid_from = record.get("valid_from")
                valid_until = record.get("valid_until")
                return (
                    (valid_from is None or datetime.fromisoformat(valid_from) <= now)
                    and (valid_until is None or datetime.fromisoformat(valid_until) > now)
                )
            except (TypeError, ValueError):
                # Invalid validity metadata must never make a memory visible.
                return False

        scopes = (
            authorization_context.authorized_scopes
            if authorization_context is not None
            else ("default",)
        )
        can_read = authorization_context is None or "read" in authorization_context.permissions
        visible = []
        for record in state.node_metadata:
            visible.append(
                can_read
                and record.get("scope", "default") in scopes
                and "read" in record.get("permissions", ("read", "write"))
                and record.get("verification_status", "unverified")
                not in ("retracted", "superseded", "expired")
                and currently_valid(record)
            )
        visible_mask = torch.tensor(visible, dtype=torch.bool, device="cpu")
        if node_mask is not None:
            if (
                not isinstance(node_mask, torch.Tensor)
                or node_mask.dtype != torch.bool
                or node_mask.device.type != "cpu"
                or node_mask.shape != state.node_ids.shape
            ):
                raise ValidationError("node_mask must be a CPU bool tensor with shape [N]")
            visible_mask &= node_mask

        edge_visible = (
            visible_mask[state.edge_index[0]] & visible_mask[state.edge_index[1]]
            if state.edge_index.shape[1]
            else torch.empty(0, dtype=torch.bool, device="cpu")
        )
        active_edges = []
        for index, record in enumerate(state.edge_metadata):
            active_edges.append(
                can_read
                and record.get("scope", "default") in scopes
                and bool(edge_visible[index])
                and record.get("status", "active")
                not in ("retracted", "superseded", "expired", "inactive")
                and currently_valid(record)
            )
        edge_mask = torch.tensor(active_edges, dtype=torch.bool, device="cpu")
        indices = torch.nonzero(visible_mask, as_tuple=False).flatten()
        visible_state, _ = state.extract_subgraph(indices, edge_mask=edge_mask)
        return _BridgeRequest(state, visible_state, attention_mask)

    def _replace_layer_output(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> torch.Tensor:
        request = self._request.get()
        if request is None:
            return output
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValidationError("LFM3 layer 2 must return a tensor shaped [B,T,H]")
        if output.shape[-1] != self.hidden_size:
            raise ValidationError("LFM3 layer 2 hidden width does not match its config")
        if output.device.type != "cpu":
            raise ValidationError("RUTAM graph fusion currently supports CPU LFM3 execution only")
        if (
            not self.config.rutam_model.graph_enabled
            or not self.config.rutam_model.graph_fusion_enabled
            or request.visible_state.node_ids.numel() == 0
        ):
            return output

        if request.prefix_hidden is None or output.shape[1] > 1:
            prefix = output.float()
            first_token = 0
        else:
            if output.shape[0] != request.prefix_hidden.shape[0]:
                raise ValidationError("Cached LFM3 decoding changed the request batch size")
            first_token = request.prefix_hidden.shape[1]
            prefix = torch.cat((request.prefix_hidden, output.float()), dim=1)
        request.prefix_hidden = prefix
        fused = self._fuse_prefixes(request, prefix, first_token)
        return fused.to(dtype=output.dtype, device=output.device)

    def _fuse_prefixes(
        self,
        request: _BridgeRequest,
        prefix: torch.Tensor,
        first_token: int,
    ) -> torch.Tensor:
        from rutam.model.rutam_forward import _expand_neighborhood

        state = request.visible_state
        batch, length, _ = prefix.shape
        mask = self._query_mask(request.attention_mask, batch, length)
        sparse = (
            self.config.rutam_model.sparse_selection_enabled
            and self.selector.config.selection_mode == "hard"
            and not self.selector.config.return_attention
        )
        encoded_all: GraphEncoderOutput | None = None
        projected_candidates: torch.Tensor | None = None
        if sparse:
            projected_candidates = self.encoder.node_projection(state.node_embeddings)
            summary = (
                projected_candidates.mean(dim=0)
                if projected_candidates.shape[0]
                else projected_candidates.new_zeros(self.config.encoder.hidden_dim)
            )
        else:
            encoded_all = self.encoder(state)
            summary = encoded_all.graph_summary

        fused_tokens: list[torch.Tensor] = []
        for token_index in range(first_token, length):
            prefix_mask = mask[:, : token_index + 1] if mask is not None else None
            selection_out = self.query_head.select_graph(
                hidden_states=prefix[:, : token_index + 1],
                selector=self.selector,
                node_states=state.node_embeddings if sparse else encoded_all,
                node_ids=state.node_ids,
                attention_mask=prefix_mask,
                graph_summary=summary,
                memory_available=torch.ones(batch, dtype=torch.bool, device="cpu"),
                node_mask=(
                    torch.ones(state.node_ids.shape, dtype=torch.bool, device="cpu")
                    if sparse
                    else None
                ),
                candidate_projection=self.encoder.node_projection if sparse else None,
            )
            query = selection_out.query
            selection = selection_out.selection
            reasoning_context = None
            encoded_for_traversal = encoded_all
            traversal_state = state

            if sparse:
                seed_indices = selection.selected_indices[selection.selection_mask]
                sub_nodes = _expand_neighborhood(
                    seed_indices,
                    state.edge_index,
                    state.node_ids.shape[0],
                    max_hops=self.config.rutam_model.sparse_max_hops,
                    max_nodes=self.config.rutam_model.max_sparse_nodes,
                )
                sub_state, retained = state.extract_subgraph(sub_nodes)
                sub_encoded = self.encoder(sub_state, pool_graph=True)
                inverse = torch.full((state.node_ids.shape[0],), -1, dtype=torch.int64)
                if retained.numel():
                    inverse[retained] = torch.arange(retained.numel(), dtype=torch.int64)
                sub_indices = inverse[selection.selected_indices.clamp_min(0)]
                valid = selection.selection_mask & (sub_indices >= 0)
                if sub_encoded.node_states.shape[0]:
                    selected_states = sub_encoded.node_states[sub_indices.clamp_min(0)]
                    selected_states = torch.where(valid[..., None], selected_states, 0)
                else:
                    selected_states = sub_encoded.node_states.new_zeros(
                        (*selection.selected_indices.shape, self.config.encoder.hidden_dim)
                    )
                selection = replace(selection, selected_states=selected_states)
                encoded_for_traversal = sub_encoded
                traversal_state = sub_state
                traversal_indices = torch.where(valid, sub_indices, -1)
            else:
                traversal_indices = selection.selected_indices

            if (
                self.traversal is not None
                and traversal_state.edge_index.shape[1] > 0
                and encoded_for_traversal is not None
            ):
                traversal_out = self.traversal.from_state(
                    query_vectors=query.query_vectors,
                    encoded=encoded_for_traversal,
                    state=traversal_state,
                    seed_indices=traversal_indices,
                    seed_mask=selection.selection_mask,
                )
                reasoning_context = traversal_out.reasoning_context

            token_mask = mask[:, token_index : token_index + 1] if mask is not None else None
            fusion_out = self.fusion(
                hidden_states=prefix[:, token_index : token_index + 1],
                graph_states=selection.selected_states,
                graph_mask=selection.selection_mask,
                reasoning_context=reasoning_context,
                query_gate=query.query_gate,
                token_attention_mask=token_mask,
            )
            fused_tokens.append(fusion_out.fused_hidden_states)
            assert request.fusion_outputs is not None
            request.fusion_outputs.append(fusion_out)

        if not fused_tokens:
            return prefix[:, first_token:]
        return torch.cat(fused_tokens, dim=1)

    @staticmethod
    def _query_mask(
        attention_mask: torch.Tensor | None,
        batch: int,
        length: int,
    ) -> torch.Tensor | None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            return None
        if attention_mask.shape[0] != batch:
            raise ValidationError("LFM3 attention mask batch size does not match hidden states")
        mask = attention_mask[:, :length]
        if mask.shape[1] < length:
            extension = torch.ones(
                (batch, length - mask.shape[1]), dtype=mask.dtype, device=mask.device
            )
            mask = torch.cat((mask, extension), dim=1)
        return mask.to(device="cpu", dtype=torch.bool)
