"""Explicit relation schemas, NO_RELATION supervision and append-only migration."""

from dataclasses import replace

import pytest
import torch

from rutam import (
    Graph,
    RelationVocabulary,
    RUTAMGraphLinker,
    RUTAMRelationPredictor,
    relation_classification_loss,
)
from rutam.exceptions import ValidationError


def test_vocabulary_class_ids_and_graph_ids_are_distinct(link_graph: Graph) -> None:
    vocabulary = RelationVocabulary.from_state(link_graph.to_state())
    assert vocabulary.labels == ("NO_RELATION", "related_to", "depends_on", "supports")
    assert vocabulary.name(0) == "NO_RELATION" and vocabulary.graph_relation_id(0) is None
    assert vocabulary.class_id("supports") == 3 and vocabulary.graph_relation_id(3) == 2
    for value in (-1, 4):
        with pytest.raises(ValidationError):
            vocabulary.name(value)
    with pytest.raises(ValidationError):
        vocabulary.class_id("generated_unknown_label")
    with pytest.raises(ValidationError):
        vocabulary.validate_graph_mapping({"supports": 0})


def test_classifier_gradients_loss_and_no_relation(link_graph: Graph) -> None:
    torch.manual_seed(9)
    config = replace(link_graph.config.linker, use_structural_features=False)
    predictor = RUTAMRelationPredictor(config)
    nodes = torch.randn((6, 8), requires_grad=True)
    logits = predictor(nodes, torch.tensor([[0, 1], [1, 0]]))
    assert logits.shape == (2, 4)
    targets = torch.tensor([1, 0])
    actual = relation_classification_loss(logits, targets, link_labels=torch.tensor([1, 0]))
    torch.testing.assert_close(actual, torch.nn.functional.cross_entropy(logits, targets))
    actual.backward()
    assert nodes.grad.count_nonzero() > 0
    assert predictor.output_projection.weight.grad.count_nonzero() > 0
    with pytest.raises(ValidationError):
        relation_classification_loss(logits, torch.tensor([4, 0]))
    with pytest.raises(ValidationError):
        relation_classification_loss(logits, torch.tensor([1, 2]), link_labels=torch.tensor([1, 0]))
    assert torch.isfinite(
        relation_classification_loss(
            logits,
            torch.tensor([1, 2]),
            link_labels=torch.tensor([1, 0]),
            allow_unlinked_relations=True,
        )
    )


def test_explicit_schema_migration_preserves_old_ids_and_logits(link_graph: Graph) -> None:
    torch.manual_seed(15)
    model = RUTAMGraphLinker(link_graph.config.linker).eval()
    state = link_graph.to_state()
    nodes = torch.randn((6, 8))
    before = model.from_state(nodes, state)
    grown = model.migrate_relation_types(("contradicts", "precedes"))
    after = grown.from_state(nodes, state)
    assert grown.vocabulary.schema_version == 2
    assert grown.vocabulary.labels[:4] == model.vocabulary.labels
    torch.testing.assert_close(before.link_logits, after.link_logits)
    torch.testing.assert_close(before.relation_logits, after.relation_logits[:, :4])
    assert after.relation_logits.shape[-1] == 6
    with pytest.raises(ValidationError):
        model.migrate_relation_types(("supports",))


def test_padded_and_empty_relation_losses_are_finite() -> None:
    logits = torch.zeros((2, 3), requires_grad=True)
    loss = relation_classification_loss(
        logits, torch.tensor([-1, -1]), mask=torch.zeros(2, dtype=torch.bool)
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.count_nonzero() == 0
    with pytest.raises(ValidationError):
        relation_classification_loss(logits, torch.tensor([-1, 0]))


def test_checkpoint_schema_identity_is_validated_and_migration_updates_signature(
    link_graph: Graph,
) -> None:
    config = link_graph.config.linker
    model = RUTAMRelationPredictor(config)
    same = RUTAMRelationPredictor(config)
    same.load_state_dict(model.state_dict())
    wrong = RUTAMRelationPredictor(
        replace(config, relation_labels=tuple(reversed(config.relation_labels)))
    )
    with pytest.raises(RuntimeError, match="schema differs"):
        wrong.load_state_dict(model.state_dict())
    migrated = model.migrate(("contradicts",))
    assert not torch.equal(model.relation_schema_signature, migrated.relation_schema_signature)
    clone = RUTAMRelationPredictor(migrated.config)
    clone.load_state_dict(migrated.state_dict())


def test_migrated_linker_accepts_explicit_new_graph_relationship(link_graph: Graph) -> None:
    model = RUTAMGraphLinker(link_graph.config.linker)
    grown = model.migrate_relation_types(("contradicts",))
    link_graph.add_relationship(20, 10, "contradicts", provenance="test:explicit-schema-update")
    with pytest.raises(ValidationError, match="vocabulary"):
        model.from_state(torch.ones((6, 8)), link_graph.to_state())
    output = grown.from_state(torch.ones((6, 8)), link_graph.to_state())
    assert output.relation_logits.shape[-1] == 5
