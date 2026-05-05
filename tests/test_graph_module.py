"""
Tests for the GraphRAG module. OpenAI is mocked throughout — these tests
are about the structural correctness of extraction validation, dedup,
graph traversal, and pipeline composition, not LLM quality.
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from graph_module.entity_extractor import extract_entities_and_relations
from graph_module.graph_builder import build_graph_from_chunks
from graph_module.graph_store import NetworkXGraphStore, Neo4jGraphStore
from graph_module.graphrag_pipeline import graphrag_query
from graph_module.schemas import Entity, EntityExtractionResult, Relation
from graph_module.semantic_query import (
    find_cascading_risks,
    find_critical_dependencies,
    find_unmitigated_risks,
)
from graph_module.vector_store import VectorStore, chunk_text


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _mock_openai_with_chat_responses(chat_payloads: List[dict]) -> MagicMock:
    """Return a MagicMock OpenAI client whose chat.completions.create yields
    JSON-serialised payloads from `chat_payloads` in order. Embeddings
    return deterministic random vectors.
    """
    client = MagicMock()
    payloads = list(chat_payloads)

    def _chat_create(**kwargs):
        data = payloads.pop(0) if payloads else {"entities": [], "relations": []}
        if isinstance(data, dict):
            content = json.dumps(data)
        else:
            content = str(data)
        msg = MagicMock()
        msg.message = MagicMock()
        msg.message.content = content
        rsp = MagicMock()
        rsp.choices = [msg]
        return rsp

    def _embed_create(**kwargs):
        inputs = kwargs.get("input", [])
        rng = np.random.default_rng(seed=42)
        rsp = MagicMock()
        rsp.data = [MagicMock(embedding=rng.standard_normal(8).tolist()) for _ in inputs]
        return rsp

    client.chat.completions.create.side_effect = _chat_create
    client.embeddings.create.side_effect = _embed_create
    return client


# ──────────────────────────────────────────────────────────────────────
# Schema validation
# ──────────────────────────────────────────────────────────────────────


def test_extract_returns_valid_pydantic_result():
    payload = {
        "entities": [
            {"id": "Risk_GW", "type": "Risk", "name": "Groundwater Ingress",
             "attributes": {}, "source_doc": ""},
            {"id": "Phase_Drive", "type": "Phase", "name": "Tunnel Drive",
             "attributes": {}, "source_doc": ""},
        ],
        "relations": [
            {"source_id": "Risk_GW", "relation_type": "AFFECTS",
             "target_id": "Phase_Drive",
             "evidence_span": "groundwater ingress at the TBM face",
             "confidence": 0.92},
        ],
    }
    client = _mock_openai_with_chat_responses([payload])
    result = extract_entities_and_relations("any chunk text", client)

    assert isinstance(result, EntityExtractionResult)
    assert len(result.entities) == 2
    assert {e.type for e in result.entities} == {"Risk", "Phase"}
    assert result.relations[0].confidence == pytest.approx(0.92)
    assert result.relations[0].evidence_span.startswith("groundwater ingress")


def test_extract_handles_invalid_json_gracefully():
    client = MagicMock()
    msg = MagicMock(); msg.message = MagicMock(); msg.message.content = "not json"
    rsp = MagicMock(); rsp.choices = [msg]
    client.chat.completions.create.return_value = rsp
    out = extract_entities_and_relations("text", client)
    assert isinstance(out, EntityExtractionResult)
    assert out.entities == []


# ──────────────────────────────────────────────────────────────────────
# Dedup behaviour
# ──────────────────────────────────────────────────────────────────────


def test_graph_builder_deduplicates_aliases():
    """Three chunks each mention the same risk under slightly different
    names. After build, only one canonical Risk node should remain.
    """
    chunk_a = {
        "entities": [
            {"id": "Risk_GW1", "type": "Risk", "name": "Groundwater Ingress",
             "attributes": {"likelihood": "High"}, "source_doc": ""}
        ],
        "relations": [],
    }
    chunk_b = {
        "entities": [
            {"id": "Risk_GW2", "type": "Risk", "name": "groundwater ingress",
             "attributes": {"severity": "High"}, "source_doc": ""}
        ],
        "relations": [],
    }
    chunk_c = {
        "entities": [
            {"id": "Risk_GW3", "type": "Risk", "name": "Groundwater  ingress at face",
             "attributes": {"location": "TBM face"}, "source_doc": ""}
        ],
        "relations": [],
    }
    client = _mock_openai_with_chat_responses([chunk_a, chunk_b, chunk_c])
    store, _ = build_graph_from_chunks(
        ["chunk one", "chunk two", "chunk three"],
        client,
        max_workers=1,
    )
    risks = store.all_nodes_of_type("Risk")
    assert len(risks) == 1, f"Expected dedup to a single node, got {[r.id for r in risks]}"
    merged = risks[0]
    # attributes from all aliases should have been merged onto the canonical entity
    assert "likelihood" in merged.attributes
    assert "severity" in merged.attributes


# ──────────────────────────────────────────────────────────────────────
# Cascading-risk multi-hop on hand-built graph
# ──────────────────────────────────────────────────────────────────────


def _build_cascade_fixture() -> NetworkXGraphStore:
    g = NetworkXGraphStore()
    risks = [
        ("Risk_Groundwater", "Groundwater Ingress"),
        ("Risk_TBM_Stop",    "TBM Stoppage"),
        ("Risk_Delay",       "Programme Delay"),
        ("Risk_Cost",        "Cost Overrun"),
        ("Risk_Settlement",  "Surface Settlement"),
        ("Risk_Building",    "Third-Party Building Damage"),
    ]
    for rid, name in risks:
        g.add_entity(Entity(id=rid, type="Risk", name=name))
    g.add_entity(Entity(id="Mitigation_Grout", type="Mitigation", name="Permeation Grouting"))

    edges = [
        ("Risk_Groundwater", "TRIGGERS",   "Risk_TBM_Stop"),
        ("Risk_TBM_Stop",    "TRIGGERS",   "Risk_Delay"),
        ("Risk_Delay",       "TRIGGERS",   "Risk_Cost"),
        ("Risk_Groundwater", "RELATED_TO", "Risk_Settlement"),
        ("Risk_Settlement",  "TRIGGERS",   "Risk_Building"),
        ("Mitigation_Grout", "MITIGATES",  "Risk_Groundwater"),
    ]
    for s, rt, t in edges:
        g.add_relation(Relation(
            source_id=s, relation_type=rt, target_id=t,
            evidence_span="fixture edge", confidence=0.9,
        ))
    return g


def test_find_cascading_risks_walks_three_hops():
    g = _build_cascade_fixture()
    cascades = find_cascading_risks(g, "Risk_Groundwater", max_hops=3)
    reached = {c["entity"].id for c in cascades}
    # All five downstream risks should be reachable within 3 hops
    assert {"Risk_TBM_Stop", "Risk_Delay", "Risk_Cost",
            "Risk_Settlement", "Risk_Building"}.issubset(reached)
    # Verify the deepest path actually has 3 hops
    cost_hit = next(c for c in cascades if c["entity"].id == "Risk_Cost")
    assert cost_hit["hops"] == 3
    assert [step["relation"] for step in cost_hit["path"]] == ["TRIGGERS", "TRIGGERS", "TRIGGERS"]


def test_find_unmitigated_risks_excludes_mitigated():
    g = _build_cascade_fixture()
    unmit = {r.id for r in find_unmitigated_risks(g)}
    # Risk_Groundwater is mitigated; everything else is not
    assert "Risk_Groundwater" not in unmit
    assert "Risk_Building" in unmit


def test_find_critical_dependencies_partitions_by_type():
    g = NetworkXGraphStore()
    g.add_entity(Entity(id="Phase_Drive",     type="Phase",         name="Tunnel Drive"))
    g.add_entity(Entity(id="Phase_Shaft",     type="Phase",         name="Launch Shaft"))
    g.add_entity(Entity(id="Material_TBM",    type="Material",      name="EPB-TBM"))
    g.add_entity(Entity(id="Spec_Diameter",   type="Specification", name="1800 mm ID"))
    for s, rt, t in [
        ("Phase_Drive", "DEPENDS_ON", "Phase_Shaft"),
        ("Phase_Drive", "REQUIRES",   "Material_TBM"),
        ("Phase_Drive", "REQUIRES",   "Spec_Diameter"),
    ]:
        g.add_relation(Relation(source_id=s, relation_type=rt, target_id=t,
                                evidence_span="fix", confidence=0.9))

    deps = find_critical_dependencies(g, "Phase_Drive")
    assert [p.id for p in deps["depends_on_phases"]]      == ["Phase_Shaft"]
    assert [m.id for m in deps["requires_materials"]]     == ["Material_TBM"]
    assert [s.id for s in deps["requires_specifications"]] == ["Spec_Diameter"]


# ──────────────────────────────────────────────────────────────────────
# graphrag_query end-to-end with mocked OpenAI
# ──────────────────────────────────────────────────────────────────────


def test_graphrag_query_returns_both_answers_and_diagnostic():
    g = _build_cascade_fixture()

    # Pre-build a tiny vector store directly (bypassing embedding API)
    vs = VectorStore(
        chunks=[
            "Groundwater ingress at the TBM face is a major risk.",
            "Cost overruns frequently follow programme delays.",
        ],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    # Sequence of mocked LLM responses:
    #   1) vector-only answer
    #   2) entity extraction from the question (seeds traversal)
    #   3) graphrag combined-context answer
    chat_responses = [
        "VECTOR_ONLY_ANSWER",
        {
            "entities": [
                {"id": "Risk_Groundwater", "type": "Risk",
                 "name": "Groundwater Ingress", "attributes": {}, "source_doc": ""}
            ],
            "relations": [],
        },
        "GRAPHRAG_ANSWER",
    ]
    client = MagicMock()
    rsp_iter = iter(chat_responses)

    def _chat(**kwargs):
        payload = next(rsp_iter)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        msg = MagicMock(); msg.message = MagicMock(); msg.message.content = text
        rsp = MagicMock(); rsp.choices = [msg]
        return rsp

    def _embed(**kwargs):
        inputs = kwargs.get("input", [])
        rsp = MagicMock()
        # Match the existing 2-D vectors so cosine works deterministically
        rsp.data = [MagicMock(embedding=[1.0, 0.0]) for _ in inputs]
        return rsp

    client.chat.completions.create.side_effect = _chat
    client.embeddings.create.side_effect = _embed

    out = graphrag_query(
        "What downstream cost impact could groundwater ingress trigger?",
        vs, g, client, top_k=2, radius=2,
    )

    assert out["vector_answer"]   == "VECTOR_ONLY_ANSWER"
    assert out["graphrag_answer"] == "GRAPHRAG_ANSWER"
    seed_ids = {e.id for e in out["graph_seed_entities"]}
    assert "Risk_Groundwater" in seed_ids
    # The traversal should have brought in at least Risk_TBM_Stop and friends
    sub_ids = {e.id for e in out["graph_subgraph_entities"]}
    assert "Risk_TBM_Stop" in sub_ids
    assert "seeded traversal" in out["diagnostic"].lower() or "seeded" in out["diagnostic"].lower()


# ──────────────────────────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────────────────────────


def test_chunk_text_overlaps_and_covers_input():
    text = " ".join(f"w{i}" for i in range(1500))
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0].startswith("w0")
    # last chunk reaches the final word
    assert chunks[-1].endswith("w1499")


def test_neo4j_stub_raises():
    with pytest.raises(NotImplementedError):
        Neo4jGraphStore()
