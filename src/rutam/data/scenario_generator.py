"""Deterministic scenario generators producing varied memory training instructions."""

from __future__ import annotations

import random

from rutam.data.example_schema import MemoryExampleRecord
from rutam.data.graph_scenarios import (
    make_edge,
    make_empty_graph_dict,
    make_graph_dict,
    make_node,
)


class ScenarioGenerator:
    """Generates deterministic, synthetic memory training records across 8 categories."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_all(
        self, target_counts: dict[str, int] | None = None
    ) -> list[MemoryExampleRecord]:
        """Generate full dataset matching category target distributions."""
        if target_counts is None:
            target_counts = {
                "creation": 180,
                "update": 140,
                "link": 140,
                "noop": 140,
                "duplicate": 100,
                "contradiction": 100,
                "multihop": 100,
                "authorization": 100,
            }

        records: list[MemoryExampleRecord] = []
        gen_map = {
            "creation": self.generate_creation_scenarios,
            "update": self.generate_update_scenarios,
            "link": self.generate_link_scenarios,
            "noop": self.generate_noop_scenarios,
            "duplicate": self.generate_duplicate_scenarios,
            "contradiction": self.generate_contradiction_scenarios,
            "multihop": self.generate_multihop_scenarios,
            "authorization": self.generate_authorization_scenarios,
        }

        for category, count in target_counts.items():
            fn = gen_map[category]
            cat_records = fn(count)
            records.extend(cat_records)

        return records

    def generate_creation_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        projects = ["Aegis", "Orion", "Helios", "Krypton", "Zephyr", "Apex", "Vortex", "Titan"]
        languages = ["Rust", "Python", "Go", "TypeScript", "Zig", "C++", "Kotlin"]
        templates = [
            (
                "Remember that project {p} uses {lang}.",
                "Project {p} uses {lang} as primary language.",
            ),
            (
                "Please note that {p} has been implemented in {lang}.",
                "Implementation language of {p} is {lang}.",
            ),
            (
                "Record that the core architecture for {p} relies on {lang}.",
                "{p} architecture relies on {lang}.",
            ),
        ]
        records = []
        for i in range(count):
            p = projects[i % len(projects)] + f"_{i // len(projects)}"
            lang = languages[(i + 1) % len(languages)]
            t_in, t_ctx = templates[i % len(templates)]
            instruction = t_in.format(p=p, lang=lang)
            context = t_ctx.format(p=p, lang=lang)

            g_before = make_empty_graph_dict()
            g_after = make_graph_dict(
                [make_node(0, "project", p), make_node(1, "language", lang)],
                [make_edge(0, 1, "written_in")],
            )

            records.append(
                MemoryExampleRecord(
                    example_id=f"create_{i:04d}",
                    scenario_id=f"scen_create_{p}",
                    scenario_family=f"family_create_{p}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=True,
                    expected_operation="CREATE_NODE",
                    expected_target_nodes=[],
                    expected_relationships=[
                        {"source": 0, "target": 1, "relationship": "written_in"}
                    ],
                    expected_validation_decision="accept",
                    authorization_context={
                        "authorized": True,
                        "role": "engineer",
                        "scope": "project_meta",
                    },
                    provenance=f"synthetic:creation_{i}",
                    graph_after_expected=g_after,
                    difficulty="easy",
                )
            )
        return records

    def generate_update_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        projects = ["Orion", "Aegis", "Nova", "Cosmos", "Pulse", "Zenith", "Valkyrie"]
        old_langs = ["Python", "Java", "Ruby", "PHP"]
        new_langs = ["Rust", "Go", "TypeScript", "Kotlin"]
        records = []
        for i in range(count):
            p = projects[i % len(projects)] + f"_{i // len(projects)}"
            old_l = old_langs[i % len(old_langs)]
            new_l = new_langs[i % len(new_langs)]
            instruction = (
                f"Project {p} has completed migration from {old_l} to {new_l}. Update our records."
            )
            context = (
                f"Authorized architectural change: {p} now runs on {new_l} instead of {old_l}."
            )

            g_before = make_graph_dict(
                [make_node(0, "project", p), make_node(1, "language", old_l)],
                [make_edge(0, 1, "written_in")],
            )
            g_after = make_graph_dict(
                [make_node(0, "project", p), make_node(2, "language", new_l)],
                [make_edge(0, 2, "written_in")],
            )

            records.append(
                MemoryExampleRecord(
                    example_id=f"update_{i:04d}",
                    scenario_id=f"scen_update_{p}",
                    scenario_family=f"family_update_{p}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=True,
                    expected_operation="UPDATE_NODE",
                    expected_target_nodes=[0],
                    expected_relationships=[
                        {"source": 0, "target": 2, "relationship": "written_in"}
                    ],
                    expected_validation_decision="accept",
                    authorization_context={
                        "authorized": True,
                        "role": "lead_architect",
                        "scope": "stack_update",
                    },
                    provenance=f"synthetic:update_{i}",
                    graph_after_expected=g_after,
                    difficulty="medium",
                )
            )
        return records

    def generate_link_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        modules = ["AuthService", "BillingEngine", "QueryRouter", "CacheLayer", "WorkerPool"]
        targets = ["PostgreSQL", "Redis", "Kafka", "ElasticSearch", "Vault"]
        rels = ["connects_to", "depends_on", "caches_in", "authenticates_with"]
        records = []
        for i in range(count):
            m = modules[i % len(modules)] + f"_{i // len(modules)}"
            t = targets[(i + 1) % len(targets)]
            r = rels[i % len(rels)]
            instruction = f"Establish link: {m} {r} {t}."
            context = f"Infrastructure manifest: module {m} maintains active connection to {t}."

            g_before = make_graph_dict(
                [make_node(0, "service", m), make_node(1, "resource", t)],
                [],
            )
            g_after = make_graph_dict(
                [make_node(0, "service", m), make_node(1, "resource", t)],
                [make_edge(0, 1, r)],
            )

            records.append(
                MemoryExampleRecord(
                    example_id=f"link_{i:04d}",
                    scenario_id=f"scen_link_{m}_{t}",
                    scenario_family=f"family_link_{m}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=True,
                    expected_operation="PROPOSE_LINK",
                    expected_target_nodes=[0, 1],
                    expected_relationships=[{"source": 0, "target": 1, "relationship": r}],
                    expected_validation_decision="accept",
                    authorization_context={
                        "authorized": True,
                        "role": "infra_admin",
                        "scope": "topology",
                    },
                    provenance=f"synthetic:link_{i}",
                    graph_after_expected=g_after,
                    difficulty="medium",
                )
            )
        return records

    def generate_noop_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        greetings = [
            "Good morning! How are you doing today?",
            "Can you tell me what 25 * 14 is?",
            "Just checking in, no action required.",
            "I wonder what the weather will be like tomorrow.",
            "Thanks for the explanation earlier, appreciate it!",
            "Let's test if you can echo this string back to me.",
        ]
        records = []
        for i in range(count):
            text = greetings[i % len(greetings)] + f" (req_{i})"
            instruction = text
            context = "Transient conversational turn with no durable factual updates."

            g_before = make_graph_dict(
                [make_node(0, "workspace", "GlobalContext")],
                [],
            )
            # persistent state remains unchanged
            g_after = g_before

            records.append(
                MemoryExampleRecord(
                    example_id=f"noop_{i:04d}",
                    scenario_id=f"scen_noop_{i}",
                    scenario_family=f"family_noop_{i % 10}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=False,
                    expected_operation="NO_OP",
                    expected_target_nodes=[],
                    expected_relationships=[],
                    expected_validation_decision="defer",
                    authorization_context={"authorized": False, "role": "guest", "scope": "none"},
                    provenance=f"synthetic:noop_{i}",
                    graph_after_expected=g_after,
                    difficulty="easy",
                )
            )
        return records

    def generate_duplicate_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        facts = [
            ("Service Alpha uses port 8080", "Alpha", 8080),
            ("Database Beta is hosted on us-east-1", "Beta", "us-east-1"),
            ("Cluster Gamma runs Kubernetes 1.28", "Gamma", "1.28"),
        ]
        records = []
        for i in range(count):
            desc, name, val = facts[i % len(facts)]
            instruction = f"Re-stating fact: {desc} (ref_{i})."
            context = f"Duplicate incoming announcement for existing memory of {name}."

            g_before = make_graph_dict(
                [
                    make_node(0, "entity", f"{name}_{i // len(facts)}"),
                    make_node(1, "property", str(val)),
                ],
                [make_edge(0, 1, "configured_with")],
            )
            g_after = g_before  # No new node created, duplicate detected

            records.append(
                MemoryExampleRecord(
                    example_id=f"dup_{i:04d}",
                    scenario_id=f"scen_dup_{name}_{i}",
                    scenario_family=f"family_dup_{name}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=False,
                    expected_operation="NO_OP",
                    expected_target_nodes=[0],
                    expected_relationships=[],
                    expected_validation_decision="reject",
                    authorization_context={
                        "authorized": True,
                        "role": "engineer",
                        "scope": "meta",
                    },
                    provenance=f"synthetic:duplicate_{i}",
                    graph_after_expected=g_after,
                    difficulty="medium",
                )
            )
        return records

    def generate_contradiction_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        entities = ["Service Apollo", "Cache Node Delta", "Gateway Proxy", "Queue Worker"]
        records = []
        for i in range(count):
            ent = entities[i % len(entities)] + f"_{i // len(entities)}"
            instruction = f"Unverified claim: {ent} is completely deprecated."
            context = f"Disputed status claim for {ent} contradicting authoritative live telemetry."

            g_before = make_graph_dict(
                [make_node(0, "service", ent), make_node(1, "status", "active_healthy")],
                [make_edge(0, 1, "has_status")],
            )
            g_after = g_before  # Requires validation before any destructive overwrite

            records.append(
                MemoryExampleRecord(
                    example_id=f"contra_{i:04d}",
                    scenario_id=f"scen_contra_{i}",
                    scenario_family=f"family_contra_{i % 8}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=False,
                    expected_operation="REQUEST_VALIDATION",
                    expected_target_nodes=[0],
                    expected_relationships=[],
                    expected_validation_decision="defer",
                    authorization_context={
                        "authorized": False,
                        "role": "external_source",
                        "scope": "unverified",
                    },
                    provenance=f"synthetic:contradiction_{i}",
                    graph_after_expected=g_after,
                    difficulty="hard",
                )
            )
        return records

    def generate_multihop_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        chains = [
            ("Microservice A", "Library B", "Runtime C", "depends_on", "runs_on"),
            ("Client App", "API Gateway", "Auth Service", "routes_through", "validates_with"),
            ("Data Pipeline", "Kafka Broker", "ClickHouse Sink", "publishes_to", "streams_into"),
        ]
        records = []
        for i in range(count):
            a, b, c, r1, r2 = chains[i % len(chains)]
            idx = i // len(chains)
            a_name = f"{a}_{idx}"
            b_name = f"{b}_{idx}"
            c_name = f"{c}_{idx}"
            instruction = f"Which component is transitively utilized by {a_name} through {b_name}?"
            context = f"Multi-hop dependency query across 2 hops: {a_name} -> {b_name} -> {c_name}."

            g_before = make_graph_dict(
                [
                    make_node(0, "component", a_name),
                    make_node(1, "component", b_name),
                    make_node(2, "component", c_name),
                ],
                [make_edge(0, 1, r1), make_edge(1, 2, r2)],
            )
            g_after = g_before

            records.append(
                MemoryExampleRecord(
                    example_id=f"multihop_{i:04d}",
                    scenario_id=f"scen_multihop_{idx}_{i % len(chains)}",
                    scenario_family=f"family_multihop_{i % len(chains)}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=False,
                    expected_operation="NO_OP",
                    expected_target_nodes=[0, 1, 2],
                    expected_relationships=[
                        {"source": 0, "target": 1, "relationship": r1},
                        {"source": 1, "target": 2, "relationship": r2},
                    ],
                    expected_validation_decision="accept",
                    authorization_context={
                        "authorized": True,
                        "role": "read_only",
                        "scope": "graph_query",
                    },
                    provenance=f"synthetic:multihop_{i}",
                    graph_after_expected=g_after,
                    difficulty="hard",
                )
            )
        return records

    def generate_authorization_scenarios(self, count: int) -> list[MemoryExampleRecord]:
        sensitive_items = [
            ("Master API Key", "sk-live-synthetic-secret-999"),
            ("Admin DB Password", "p@ssword-synthetic-admin-123"),
            ("Customer Personal Email", "person@synthetic-corp.local"),
        ]
        records = []
        for i in range(count):
            item, val = sensitive_items[i % len(sensitive_items)]
            instruction = f"Store secret credential: {item} = {val}."
            context = "Attempted write of restricted secret without persistence authorization."

            g_before = make_empty_graph_dict()
            g_after = g_before  # Strictly rejected!

            records.append(
                MemoryExampleRecord(
                    example_id=f"auth_{i:04d}",
                    scenario_id=f"scen_auth_{i}",
                    scenario_family=f"family_auth_{i % 5}",
                    instruction=instruction,
                    context=context,
                    graph_before=g_before,
                    expected_write_decision=False,
                    expected_operation="REQUEST_VALIDATION",
                    expected_target_nodes=[],
                    expected_relationships=[],
                    expected_validation_decision="reject",
                    authorization_context={
                        "authorized": False,
                        "role": "untrusted_agent",
                        "scope": "secrets_denied",
                    },
                    provenance=f"synthetic:authorization_{i}",
                    graph_after_expected=g_after,
                    difficulty="hard",
                )
            )
        return records
