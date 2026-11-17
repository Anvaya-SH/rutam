"""Deterministic storage-to-neural-to-commit proof; no learned QA claims."""
import os
from pathlib import Path
import pytest
import torch
from test_rutam_forward import _create_small_model_and_graph
from rutam import Graph, GraphStorage, Node, RUTAMConfig
from rutam.exceptions import ValidationError
from rutam.model.rutam_model import RUTAMForCausalLM
from rutam.memory.transaction import GraphTransaction
from rutam.memory.provenance import AuthorizationContext, ProvenanceRecord
from rutam.memory.write_proposal import MemoryProposal


def test_storage_neural_commit_reload_and_empty_bypass(tmp_path):
    torch.manual_seed(41)
    template, graph = _create_small_model_and_graph()
    storage = GraphStorage(tmp_path)
    storage.save(Graph(graph.config))
    assert not storage.load().nodes
    storage.save(graph, expected_generation=storage.load()._storage_generation)
    loaded = storage.load()
    model = RUTAMForCausalLM(config=graph.config, graph=loaded, backbone=template.backbone).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        out = model(ids)
    assert out.graph_selection_output is not None
    assert out.fusion_output is not None and torch.isfinite(out.logits).all()
    original_version = loaded.version
    candidate = MemoryProposal(
        candidate_embedding=torch.randn(8), source_graph_version=original_version,
        provenance=ProvenanceRecord(source_id='fixture', authorization_category='user_input', evidence='controlled fixture'),
    )
    with GraphTransaction(loaded, storage=storage, authorization_context=AuthorizationContext(permissions=frozenset({'commit'}))) as tx:
        tx.apply_proposal(candidate)
        assert tx.commit().committed
    assert len(storage.load().nodes) == 5
    committed_version = storage.load().version
    with GraphTransaction(loaded, storage=storage) as tx:
        with pytest.raises(ValidationError, match='NOT_AUTHORIZED'):
            tx.apply_proposal(MemoryProposal())
    assert storage.load().version == committed_version
    storage.save(Graph(graph.config), unconditional=True)
    empty = RUTAMForCausalLM(config=graph.config, graph=storage.load(), backbone=template.backbone).eval()
    with torch.no_grad():
        result = empty(ids)
    assert result.fusion_output is None


def test_real_lfm3_local_integration():
    checkpoint = os.environ.get('RUTAM_LFM3_CHECKPOINT')
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.skip('NOT RUN: RUTAM_LFM3_CHECKPOINT does not identify a local LiquidAI/LFM3-1.7B-Base checkpoint; no weights downloaded')
    transformers = pytest.importorskip('transformers', reason='Real LFM3 integration requires transformers')
    from rutam.model.model_factory import RUTAMModelFactory
    backbone = transformers.AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    # Intentionally exercises the model-specific public factory path. No mock or protocol replacement.
    config = RUTAMConfig()
    graph = Graph(config)
    model = RUTAMModelFactory.create_lfm3_bridge(backbone, config=config, graph=graph).eval()
    ids = tokenizer('Architecture integration test', return_tensors='pt').input_ids
    with torch.no_grad():
        baseline = backbone(ids).logits
        empty_graph_output = model(ids).logits
    torch.testing.assert_close(empty_graph_output, baseline, rtol=0, atol=0)

    graph.add_node(Node(
        id=0,
        node_type='fact',
        content='controlled bridge fixture',
        provenance='local integration test',
        embedding=tuple(float(index + 1) / 128 for index in range(config.graph.embedding_dim)),
    ))
    with torch.no_grad():
        model.query_head.decision_gate.output_projection.weight.zero_()
        model.query_head.decision_gate.output_projection.bias.fill_(10.0)
        model.fusion.residual_scale.fill_(1.0)
        fused_output = model(ids).logits
    assert torch.isfinite(fused_output).all()
    assert len(model.last_fusion_outputs) == ids.shape[1]
    assert any(output.fusion_gate.max().item() > 0 for output in model.last_fusion_outputs)
    assert not torch.equal(fused_output, baseline)

    with torch.no_grad():
        generated = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=3,
            min_new_tokens=3,
            use_cache=True,
        )
    assert generated.shape[1] == ids.shape[1] + 3
    assert len(model.last_fusion_outputs) >= ids.shape[1] + 2
    model.close()
