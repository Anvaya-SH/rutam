"""Unit tests for DatasetValidator error detection and consistency checks."""

from rutam.data.dataset_validator import DatasetValidator
from rutam.data.example_schema import MemoryExampleRecord


def test_validator_detects_duplicate_example_id():
    validator = DatasetValidator()
    r1 = MemoryExampleRecord(
        example_id="dup_id",
        scenario_id="s1",
        scenario_family="f1",
        instruction="Instruction 1",
        context="Context 1",
        graph_before={"nodes": [], "edges": []},
        expected_write_decision=True,
        expected_operation="CREATE_NODE",
    )
    r2 = MemoryExampleRecord(
        example_id="dup_id",  # Same ID!
        scenario_id="s2",
        scenario_family="f2",
        instruction="Instruction 2",
        context="Context 2",
        graph_before={"nodes": [], "edges": []},
        expected_write_decision=True,
        expected_operation="CREATE_NODE",
    )

    is_valid = validator.validate_records([r1, r2])
    assert not is_valid
    assert any("Duplicate example_id" in err for err in validator.errors)


def test_validator_detects_dangling_edge():
    validator = DatasetValidator()
    # Edge points to node 99 which does not exist in graph_before
    g_dangling = {
        "nodes": [{"id": 0, "node_type": "entity", "content": "Entity"}],
        "edges": [{"source_id": 0, "target_id": 99, "relationship": "depends_on"}],
    }

    r = MemoryExampleRecord(
        example_id="dang_1",
        scenario_id="s1",
        scenario_family="f1",
        instruction="Do something",
        context="Context",
        graph_before=g_dangling,
        expected_write_decision=True,
        expected_operation="CREATE_NODE",
    )

    is_valid = validator.validate_records([r])
    assert not is_valid
    assert any("not in node IDs" in err for err in validator.errors)
