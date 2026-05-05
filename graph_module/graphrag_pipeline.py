"""
Side-by-side GraphRAG vs Vector-RAG question answering.

Both paths see the SAME chunks, embedded the SAME way, and call the
SAME model with the SAME instruction shape — the only thing that
differs is the retrieval mechanism. That keeps the comparison clean:
any quality delta is attributable to retrieval, not prompt or model.

The returned dict surfaces a `diagnostic` block explaining where the
two paths diverged — which graph entities/edges contributed extra
context, and which vector chunks the GraphRAG path saw on top of the
vector hits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from .entity_extractor import extract_entities_and_relations
from .graph_store import GraphStore
from .schemas import Entity, Relation
from .vector_store import VectorStore


_ANSWER_SYSTEM = (
    "You are a senior civil engineer specialising in trenchless construction. "
    "Answer only from the supplied context. Use precise civil/geotechnical terminology. "
    "If the context does not contain the answer, say so explicitly. "
    "Be concise: 4-8 sentences."
)


def _format_chunks(hits) -> str:
    if not hits:
        return "(no relevant chunks retrieved)"
    blocks = []
    for idx, score, chunk in hits:
        blocks.append(f"[chunk {idx} | sim={score:.3f}]\n{chunk}")
    return "\n\n".join(blocks)


def _format_graph_context(entities: List[Entity], relations: List[Relation]) -> str:
    if not entities and not relations:
        return "(no graph context)"
    lines = ["ENTITIES:"]
    for e in entities:
        attr = f" attrs={e.attributes}" if e.attributes else ""
        lines.append(f"  - {e.id} ({e.type}): {e.name}{attr}")
    lines.append("RELATIONS:")
    for r in relations:
        lines.append(
            f"  - ({r.source_id}) -[{r.relation_type} | conf={r.confidence:.2f}]-> ({r.target_id})"
            f"   evidence: \"{r.evidence_span}\""
        )
    return "\n".join(lines)


def _answer(client, model: str, question: str, context_block: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user",   "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context_block}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _seed_entities_from_question(
    question: str, openai_client, graph: GraphStore, model: str,
) -> List[Entity]:
    """Extract entities from the question itself, then resolve them to
    nodes that actually exist in the graph (by id-prefix match or fuzzy
    name match through canonical-key comparison).
    """
    extraction = extract_entities_and_relations(question, openai_client, model=model)
    if not extraction.entities:
        return []
    name_lookup: Dict[str, Entity] = {e.name.lower(): e for e in graph.all_entities()}
    id_lookup: Dict[str, Entity] = {e.id: e for e in graph.all_entities()}
    seeds: List[Entity] = []
    for q_ent in extraction.entities:
        if q_ent.id in id_lookup:
            seeds.append(id_lookup[q_ent.id])
            continue
        match = name_lookup.get(q_ent.name.lower())
        if match is not None:
            seeds.append(match)
            continue
        for graph_ent in graph.all_entities():
            if q_ent.name.lower() in graph_ent.name.lower():
                seeds.append(graph_ent)
                break
    # de-dup while preserving order
    seen: Set[str] = set()
    unique = []
    for e in seeds:
        if e.id not in seen:
            seen.add(e.id)
            unique.append(e)
    return unique


def graphrag_query(
    question: str,
    vector_store: VectorStore,
    graph_store: GraphStore,
    openai_client,
    model: str = "gpt-4o",
    top_k: int = 4,
    radius: int = 2,
) -> Dict[str, Any]:
    """Run both retrieval paths and return both answers + a diagnostic.

    Returns:
      {
        "vector_answer":     str,
        "graphrag_answer":   str,
        "vector_hits":       [(idx, score, chunk_preview), ...],
        "graph_seed_entities":   [Entity, ...],
        "graph_subgraph_entities": [Entity, ...],
        "graph_subgraph_relations": [Relation, ...],
        "diagnostic":        str   # human-readable why-it-differs explanation
      }
    """
    # ── Vector path ────────────────────────────────────────────────
    hits = vector_store.search(question, openai_client, top_k=top_k)
    vector_context = _format_chunks(hits)
    vector_answer = _answer(openai_client, model, question, vector_context)

    # ── GraphRAG path ─────────────────────────────────────────────
    seed_entities = _seed_entities_from_question(question, openai_client, graph_store, model=model)
    sub = graph_store.query_subgraph([e.id for e in seed_entities], radius=radius)
    sub_entities = sub.all_entities()
    sub_relations = sub.all_relations()
    graph_context_block = _format_graph_context(sub_entities, sub_relations)
    combined = (
        "===== KNOWLEDGE GRAPH CONTEXT =====\n"
        f"{graph_context_block}\n\n"
        "===== RETRIEVED DOCUMENT CHUNKS =====\n"
        f"{vector_context}"
    )
    graphrag_answer = _answer(openai_client, model, question, combined)

    diagnostic = _build_diagnostic(hits, seed_entities, sub_entities, sub_relations)

    return {
        "vector_answer": vector_answer,
        "graphrag_answer": graphrag_answer,
        "vector_hits": [(idx, score, chunk[:240]) for idx, score, chunk in hits],
        "graph_seed_entities": seed_entities,
        "graph_subgraph_entities": sub_entities,
        "graph_subgraph_relations": sub_relations,
        "diagnostic": diagnostic,
    }


def _build_diagnostic(
    vector_hits,
    seed_entities: List[Entity],
    sub_entities: List[Entity],
    sub_relations: List[Relation],
) -> str:
    """Compose the 'why GraphRAG won/lost here' explanation.

    The diagnostic is structural: which entities seeded the traversal,
    how many hops it surfaced, and which traversal-derived facts the
    vector chunks could not have provided in isolation.
    """
    if not seed_entities:
        return (
            "GraphRAG did not seed any graph entities from this question, so it "
            "fell back to vector context only — no traversal advantage was "
            "available for this query."
        )

    seed_ids = {e.id for e in seed_entities}
    new_entity_ids = [e.id for e in sub_entities if e.id not in seed_ids]
    edge_summaries = []
    for r in sub_relations[:6]:
        edge_summaries.append(f"{r.source_id} -[{r.relation_type}]-> {r.target_id}")

    return (
        f"GraphRAG seeded traversal from {len(seed_entities)} question entity"
        f"{'s' if len(seed_entities) != 1 else ''}: "
        f"{', '.join(e.id for e in seed_entities)}. "
        f"It pulled in {len(new_entity_ids)} additional entities reachable via "
        f"{len(sub_relations)} relations — facts the vector path could only see "
        "if they happened to co-occur in a single retrieved chunk. "
        f"Sample edges traversed: {'; '.join(edge_summaries) or '(none)'}."
    )
