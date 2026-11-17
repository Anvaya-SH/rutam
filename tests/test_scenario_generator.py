"""Unit tests for ScenarioGenerator across all memory categories."""

from rutam.data.scenario_generator import ScenarioGenerator


def test_scenario_generator_target_counts():
    gen = ScenarioGenerator(seed=42)
    targets = {
        "creation": 18,
        "update": 14,
        "link": 14,
        "noop": 14,
        "duplicate": 10,
        "contradiction": 10,
        "multihop": 10,
        "authorization": 10,
    }
    records = gen.generate_all(targets)
    assert len(records) == 100

    ops = [r.expected_operation for r in records]
    assert ops.count("CREATE_NODE") == 18
    assert ops.count("UPDATE_NODE") == 14
    assert ops.count("PROPOSE_LINK") == 14


def test_fixed_seed_reproducibility():
    gen1 = ScenarioGenerator(seed=123)
    gen2 = ScenarioGenerator(seed=123)
    rec1 = gen1.generate_creation_scenarios(5)
    rec2 = gen2.generate_creation_scenarios(5)

    for r1, r2 in zip(rec1, rec2, strict=False):
        assert r1.example_id == r2.example_id
        assert r1.instruction == r2.instruction
        assert r1.graph_after_expected == r2.graph_after_expected


def test_category_specific_labels():
    gen = ScenarioGenerator(seed=42)

    noops = gen.generate_noop_scenarios(5)
    for n in noops:
        assert n.expected_operation == "NO_OP"
        assert n.expected_write_decision is False

    auths = gen.generate_authorization_scenarios(5)
    for a in auths:
        assert a.expected_validation_decision == "reject"
        assert a.expected_write_decision is False
