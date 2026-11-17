"""Unified forward execution pipeline combining backbone, graph querying, traversal, and fusion."""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from rutam.config import RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.graph.encoder import RUTAMGraphEncoder
from rutam.graph.selector import RUTAMGraphSelector
from rutam.graph.state import GraphState
from rutam.graph.traversal import RUTAMGraphTraversal
from rutam.model.backbone_protocol import BackboneProtocol
from rutam.model.fusion import RUTAMGraphFusion
from rutam.model.graph_integration import GraphMemoryBridge
from rutam.model.query_head import RUTAMNeuralQueryHead
from rutam.model.recurrent_integration import RUTAMRecurrentIntegration
from rutam.model.rutam_outputs import RUTAMCausalLMOutput

if TYPE_CHECKING:
    from rutam.memory.provenance import AuthorizationContext
    from rutam.memory.write_head import RUTAMMemoryWriteHead


def _expand_neighborhood(
    seed_indices: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    max_hops: int = 2,
    max_nodes: int = 64,
    node_mask: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand seed node indices to their k-hop relational neighborhood, bounded by max_nodes."""
    if seed_indices.numel() == 0 or num_nodes == 0:
        return torch.empty(0, dtype=torch.int64, device=seed_indices.device)

    seeds = seed_indices.flatten()
    valid_seeds = seeds[(seeds >= 0) & (seeds < num_nodes)]
    if node_mask is not None:
        valid_seeds = valid_seeds[node_mask[valid_seeds]]

    visited = set(valid_seeds.tolist())
    if not visited or edge_index.shape[1] == 0:
        return torch.tensor(
            sorted(visited)[:max_nodes], dtype=torch.int64, device=seed_indices.device
        )

    active_edges = edge_index[:, edge_mask] if edge_mask is not None else edge_index
    if node_mask is not None and active_edges.shape[1] > 0:
        node_allowed = node_mask[active_edges[0]] & node_mask[active_edges[1]]
        active_edges = active_edges[:, node_allowed]

    adj: dict[int, list[int]] = {}
    if active_edges.shape[1] > 0:
        srcs = active_edges[0].tolist()
        dsts = active_edges[1].tolist()
        for s, d in zip(srcs, dsts, strict=True):
            adj.setdefault(s, []).append(d)
            adj.setdefault(d, []).append(s)

    current_frontier = set(visited)
    for _ in range(max_hops):
        next_frontier = set()
        for node in current_frontier:
            for neighbor in adj.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    if len(visited) >= max_nodes:
                        break
            if len(visited) >= max_nodes:
                break
        if len(visited) >= max_nodes or not next_frontier:
            break
        current_frontier = next_frontier

    return torch.tensor(
        sorted(visited)[:max_nodes], dtype=torch.int64, device=seed_indices.device
    )


def execute_rutam_forward(
    *,
    backbone: BackboneProtocol,
    encoder: RUTAMGraphEncoder | None = None,
    selector: RUTAMGraphSelector | None = None,
    query_head: RUTAMNeuralQueryHead | None = None,
    traversal: RUTAMGraphTraversal | None = None,
    fusion: RUTAMGraphFusion | None = None,
    recurrent: RUTAMRecurrentIntegration | None = None,
    write_head: RUTAMMemoryWriteHead | None = None,
    bridge: GraphMemoryBridge,
    config: RUTAMConfig,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    memory_available: torch.Tensor | None = None,
    node_mask: torch.Tensor | None = None,
    graph_state_snapshot: GraphState | None = None,
    generate_proposals: bool = False,
    authorization_context: AuthorizationContext | None = None,
    source_references: Any = None,
    diagnostics: dict[str, Any] | None = None,
) -> RUTAMCausalLMOutput:
    """Execute causal LM forward pass with integrated graph memory querying and fusion."""
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.dtype != torch.long
        or input_ids.device.type != "cpu"
        or input_ids.ndim != 2
    ):
        raise ValidationError("input_ids must be a rank-2 CPU torch.long tensor [B,T]")

    if input_ids.ndim != 2 or 0 in input_ids.shape:
        raise ValidationError("input_ids must have nonempty shape [B,T]")
    diag = dict(diagnostics or {})
    b, t = input_ids.shape
    diag["batch_size"] = b
    diag["seq_len"] = t

    # 1. State discovery and active snapshot
    state = bridge.get_active_state() if graph_state_snapshot is None else graph_state_snapshot
    if not isinstance(state, GraphState):
        raise ValidationError("graph_state_snapshot must be a GraphState")
    state.validate()
    if node_mask is not None and (
        not isinstance(node_mask, torch.Tensor)
        or node_mask.dtype != torch.bool
        or node_mask.device.type != "cpu"
        or node_mask.shape != state.node_ids.shape
    ):
        raise ValidationError("node_mask must be a CPU bool tensor with shape [N]")
    now = datetime.now(UTC)
    scopes = (
        authorization_context.authorized_scopes
        if authorization_context is not None
        else ("default",)
    )
    can_read = authorization_context is None or "read" in authorization_context.permissions
    visibility = []
    for record in state.node_metadata:
        valid_from = record.get("valid_from")
        valid_until = record.get("valid_until")
        visibility.append(
            can_read
            and record.get("scope", "default") in scopes
            and "read" in record.get("permissions", ("read", "write"))
            and record.get("verification_status", "unverified") not in ("retracted", "superseded")
            and (valid_from is None or datetime.fromisoformat(valid_from) <= now)
            and (valid_until is None or datetime.fromisoformat(valid_until) > now)
        )
    visible_mask = torch.tensor(visibility, dtype=torch.bool)
    node_mask = visible_mask if node_mask is None else node_mask & visible_mask
    has_nodes = bool(node_mask.any()) and config.rutam_model.graph_enabled
    diag["has_nodes"] = has_nodes
    diag["graph_version"] = state.graph_version

    edge_visibility = []
    edge_endpoints_visible = (
        node_mask[state.edge_index[0]] & node_mask[state.edge_index[1]]
        if state.edge_index.shape[1] > 0
        else torch.empty(0, dtype=torch.bool, device="cpu")
    )
    for edge_idx, record in enumerate(state.edge_metadata):
        valid_from = record.get("valid_from")
        valid_until = record.get("valid_until")
        edge_visibility.append(
            can_read
            and record.get("scope", "default") in scopes
            and bool(edge_endpoints_visible[edge_idx])
            and record.get("status", "active") not in ("retracted", "superseded", "inactive")
            and (valid_from is None or datetime.fromisoformat(valid_from) <= now)
            and (valid_until is None or datetime.fromisoformat(valid_until) > now)
        )
    edge_mask = (
        torch.tensor(edge_visibility, dtype=torch.bool, device="cpu")
        if state.edge_index.shape[1] > 0
        else torch.empty(0, dtype=torch.bool, device="cpu")
    )

    # 2. Token input embedding
    embed_tokens = backbone.get_input_embeddings()
    x = embed_tokens(input_ids)

    # 3. Backbone layers and integration split
    layers = backbone.get_layers()
    num_layers = len(layers)
    query_idx = min(config.rutam_model.query_layer_idx, num_layers - 1)
    fusion_idx = min(config.rutam_model.fusion_layer_idx, num_layers - 1)

    # Pass through initial backbone layers up to fusion/query point
    split_idx = max(query_idx, fusion_idx)
    for i in range(split_idx + 1):
        parameters = inspect.signature(layers[i].forward).parameters
        accepts_mask = "attention_mask" in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        x = layers[i](x, attention_mask=attention_mask) if accepts_mask else layers[i](x)

    selection_out = query = selection = encoded = summary = None
    traversal_out = reasoning_ctx = fusion_out = None
    fused_tokens = []
    fusion_outputs = []
    prefix_selections = []
    prefix_contexts = []
    is_sparse = (
        config.rutam_model.sparse_selection_enabled
        and selector is not None
        and selector.config.selection_mode == "hard"
        and not selector.config.return_attention
    )
    if is_sparse and config.rutam_model.max_sparse_nodes < selector.config.top_k:
        raise ValidationError(
            "rutam_model.max_sparse_nodes must be at least selector.top_k for sparse selection"
        )
    diag["sparse_selection_active"] = is_sparse
    if has_nodes and encoder is not None:
        if is_sparse:
            # This summary conditions the causal query without running relational
            # message passing over the full graph. Candidate rows stay in their
            # persisted feature space until selector chunking projects them.
            visible_features = state.node_embeddings[node_mask]
            pooled_features = visible_features.mean(dim=0, keepdim=True)
            summary = encoder.node_projection.forward_for_selection(pooled_features)[0]
        else:
            encoded = bridge.get_active_encoding(
                encoder, state=state, node_mask=node_mask, edge_mask=edge_mask
            )
            summary = encoded.graph_summary
    # Prefix pooling prevents future tokens selecting memory for earlier logits.
    # Public query/traversal diagnostics describe the final prefix.
    diag["graph_query_strategy"] = "per_token_prefix"
    for token_idx in range(t):
        sub_state = sub_encoded = sub_indices = None
        if has_nodes and query_head is not None and selector is not None and encoder is not None:
            candidate_node_states = state.node_embeddings if is_sparse else encoded
            selection_out = query_head.select_graph(
                hidden_states=x[:, : token_idx + 1],
                selector=selector,
                node_states=candidate_node_states,
                node_ids=state.node_ids,
                attention_mask=attention_mask[:, : token_idx + 1]
                if attention_mask is not None
                else None,
                graph_summary=summary,
                memory_available=memory_available,
                node_mask=node_mask,
                candidate_projection=encoder.node_projection if is_sparse else None,
            )
            query = selection_out.query
            selection = selection_out.selection

            if is_sparse and selection is not None:
                seed_idx = selection.selected_indices[selection.selection_mask]
                # A shared subgraph must retain every batch row's selected seeds.
                # For large batches, the effective cap is therefore at least the
                # union of those seeds, while still independent of graph size.
                bounded_node_count = max(
                    config.rutam_model.max_sparse_nodes,
                    int(torch.unique(seed_idx).numel()),
                )
                sub_nodes = _expand_neighborhood(
                    seed_idx,
                    state.edge_index,
                    state.node_ids.shape[0],
                    max_hops=config.rutam_model.sparse_max_hops,
                    max_nodes=bounded_node_count,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                )
                sub_state, retained = state.extract_subgraph(sub_nodes, edge_mask=edge_mask)
                sub_encoded = encoder(sub_state, pool_graph=True)
                inv_map = torch.full(
                    (state.node_ids.shape[0],), -1, dtype=torch.int64, device="cpu"
                )
                if retained.shape[0] > 0:
                    inv_map[retained] = torch.arange(
                        retained.shape[0], dtype=torch.int64, device="cpu"
                    )
                sub_indices = inv_map[selection.selected_indices.clamp_min(0)]
                if sub_encoded.node_states.shape[0] > 0:
                    encoded_states = sub_encoded.node_states[sub_indices.clamp_min(0)]
                    valid_sub = selection.selection_mask & (sub_indices >= 0)
                    encoded_states = torch.where(valid_sub[..., None], encoded_states, 0)
                else:
                    encoded_states = sub_encoded.node_states.new_zeros(
                        (*selection.selected_indices.shape, config.encoder.hidden_dim)
                    )
                selection = replace(selection, selected_states=encoded_states)

        # 5. Optional multi-hop path traversal
        traversal_out = None
        reasoning_ctx = None
        if (
            has_nodes
            and selection_out is not None
            and traversal is not None
            and config.rutam_model.graph_traversal_enabled
            and query is not None
            and selection is not None
        ):
            if is_sparse:
                if sub_state.edge_index.shape[1] > 0 and sub_encoded is not None:
                    sub_seed_indices = torch.where(selection.selection_mask, sub_indices, -1)
                    traversal_out = traversal.from_state(
                        query_vectors=query.query_vectors,
                        encoded=sub_encoded,
                        state=sub_state,
                        seed_indices=sub_seed_indices,
                        seed_mask=selection.selection_mask,
                    )
                    reasoning_ctx = traversal_out.reasoning_context
            else:
                if state.edge_index.shape[1] > 0 and encoded is not None:
                    traversal_out = traversal.from_state(
                        query_vectors=query.query_vectors,
                        encoded=encoded,
                        state=state,
                        seed_indices=selection.selected_indices,
                        seed_mask=selection.selection_mask,
                        node_mask=node_mask,
                    )
                    reasoning_ctx = traversal_out.reasoning_context

        prefix_selections.append(selection)
        prefix_contexts.append(reasoning_ctx)

        # 6. Graph memory fusion
        fusion_out = None
        if (
            has_nodes
            and selection is not None
            and fusion is not None
            and config.rutam_model.graph_fusion_enabled
        ):
            fusion_out = fusion(
                hidden_states=x[:, token_idx : token_idx + 1],
                graph_states=selection.selected_states,
                graph_mask=selection.selection_mask,
                reasoning_context=reasoning_ctx,
                query_gate=query.query_gate if query is not None else None,
                token_attention_mask=attention_mask[:, token_idx : token_idx + 1]
                if attention_mask is not None
                else None,
            )
            fusion_outputs.append(fusion_out)
            fused_tokens.append(fusion_out.fused_hidden_states)
        else:
            fused_tokens.append(x[:, token_idx : token_idx + 1])

    x = torch.cat(fused_tokens, dim=1)
    if fusion_outputs:
        fusion_out = replace(
            fusion_outputs[-1],
            fused_hidden_states=x,
            fusion_gate=torch.cat(
                [out.fusion_gate.reshape(b, -1, 1) for out in fusion_outputs], dim=1
            ),
            attention_weights=None,
        )

    # 6.5 Optional recurrent reasoning computation
    recurrent_out = None
    if recurrent is not None and (
        config.recurrent.enabled or config.rutam_model.recurrent_loops_enabled
    ):
        # Selection, traversal and dynamic halting can depend on the prefix.
        # Running all positions with the final selection leaks future tokens.
        # Evaluate each prefix and retain only its final token; diagnostics refer
        # to the final prefix, consistent with query/traversal diagnostics.
        recurrent_tokens = []
        for token_idx, (prefix_selection, prefix_context) in enumerate(
            zip(prefix_selections, prefix_contexts, strict=True)
        ):
            prefix_x, recurrent_out = recurrent(
                hidden_states=x[:, : token_idx + 1],
                graph_states=(
                    prefix_selection.selected_states if prefix_selection is not None else None
                ),
                graph_mask=(
                    prefix_selection.selection_mask if prefix_selection is not None else None
                ),
                reasoning_context=prefix_context,
                attention_mask=(
                    attention_mask[:, : token_idx + 1] if attention_mask is not None else None
                ),
            )
            recurrent_tokens.append(prefix_x[:, -1:])
        x = torch.cat(recurrent_tokens, dim=1)

    # 7. Remaining backbone decoder layers
    for i in range(split_idx + 1, num_layers):
        parameters = inspect.signature(layers[i].forward).parameters
        accepts_mask = "attention_mask" in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        x = layers[i](x, attention_mask=attention_mask) if accepts_mask else layers[i](x)

    # 8. Output logits and shifted causal LM cross-entropy loss
    logits = backbone.get_output_logits(x)

    loss = None
    if labels is not None:
        if (
            not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.long
            or labels.shape != input_ids.shape
            or labels.device != input_ids.device
        ):
            raise ValidationError(
                "labels must be a torch.long tensor matching input_ids shape and device"
            )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if not bool((shift_labels != -100).any()):
            # Mean cross entropy over zero supervised positions is NaN. Return
            # a differentiable zero so training steps remain well-defined.
            loss = logits.sum() * 0.0
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

    # 9. Optional write head memory proposals
    write_out = None
    proposals: tuple[Any, ...] = ()
    if generate_proposals and write_head is not None and config.rutam_model.memory_write_enabled:
        source_refs = source_references
        if source_refs is None and authorization_context is not None:
            from rutam.memory.provenance import ProvenanceRecord

            source_refs = ProvenanceRecord(
                source_id=authorization_context.user_id or "model_inference",
                source_category="user_input",
                authorization_category="user_authorized",
            )
        write_out = write_head(
            hidden_states=x,
            attention_mask=attention_mask,
            graph_summary=summary,
            selected_memory_states=selection.selected_states if selection is not None else None,
            selected_memory_ids=selection.selected_node_ids if selection is not None else None,
            source_references=source_refs,
            memory_available=memory_available,
            source_graph_version=state.graph_version,
        )
        if write_out.proposals:
            proposals = tuple(p for batch in write_out.proposals for p in batch)

    if config.rutam_model.return_memory_diagnostics:
        diag["selection_active"] = selection_out is not None
        diag["traversal_active"] = traversal_out is not None
        diag["fusion_active"] = fusion_out is not None
        diag["recurrent_active"] = recurrent_out is not None
        diag["write_proposals_active"] = len(proposals) > 0
        diag["num_nodes"] = state.node_ids.shape[0]
        diag["num_visible_nodes"] = (
            state.node_ids.shape[0] if node_mask is None else int(node_mask.sum().item())
        )
        diag["num_edges"] = state.edge_index.shape[1]

    return RUTAMCausalLMOutput(
        logits=logits,
        loss=loss,
        hidden_states=x,
        graph_selection_output=selection_out,
        traversal_output=traversal_out,
        fusion_output=fusion_out,
        recurrent_output=recurrent_out,
        write_output=write_out,
        write_proposals=proposals,
        graph_version=state.graph_version,
        diagnostics=diag,
    )
