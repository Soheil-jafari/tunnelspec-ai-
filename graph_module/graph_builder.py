"""
Builds a populated GraphStore from a list of document chunks.

Pipeline per chunk: GPT-4o extraction → Pydantic validation → entity
deduplication via case-insensitive canonical name + rapidfuzz fuzzy
match (threshold 85) → relation rewriting onto canonical ids → graph
insertion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from rapidfuzz import fuzz

from .entity_extractor import extract_entities_and_relations
from .graph_store import GraphStore, NetworkXGraphStore
from .schemas import Entity, EntityExtractionResult, Relation


FUZZY_DEDUP_THRESHOLD = 85


def _canonical_key(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _find_canonical_id(
    entity: Entity,
    canonical_by_key: Dict[str, str],
) -> str:
    """Return the existing canonical id for this entity (or its own id if new).

    Match strategy: exact case-insensitive name first, then rapidfuzz
    token-set ratio against existing names within the same entity type.
    """
    key = _canonical_key(entity.name)
    if key in canonical_by_key:
        return canonical_by_key[key]
    best_id, best_score = entity.id, 0
    for existing_key, existing_id in canonical_by_key.items():
        if not existing_id.startswith(entity.type + "_"):
            continue
        score = fuzz.token_set_ratio(key, existing_key)
        if score > best_score:
            best_id, best_score = existing_id, score
    if best_score >= FUZZY_DEDUP_THRESHOLD:
        return best_id
    return entity.id


def _merge_into_store(
    store: GraphStore,
    canonical_by_key: Dict[str, str],
    result: EntityExtractionResult,
) -> None:
    """Apply one extraction result to the store with dedup + id rewriting."""
    id_remap: Dict[str, str] = {}
    for ent in result.entities:
        canon_id = _find_canonical_id(ent, canonical_by_key)
        id_remap[ent.id] = canon_id
        canonical_by_key[_canonical_key(ent.name)] = canon_id
        store.add_entity(ent.model_copy(update={"id": canon_id}))

    for rel in result.relations:
        src = id_remap.get(rel.source_id, rel.source_id)
        tgt = id_remap.get(rel.target_id, rel.target_id)
        if src not in {e.id for e in store.all_entities()}:
            continue
        if tgt not in {e.id for e in store.all_entities()}:
            continue
        store.add_relation(rel.model_copy(update={"source_id": src, "target_id": tgt}))


def build_graph_from_chunks(
    chunks: List[str],
    openai_client,
    model: str = "gpt-4o",
    source_doc: str = "",
    max_workers: int = 4,
) -> Tuple[GraphStore, List[EntityExtractionResult]]:
    """Run extraction across chunks in parallel, return populated graph store.

    Returns (store, per_chunk_results) — the per-chunk results are kept
    so the UI can surface raw extractions for transparency.
    """
    store: GraphStore = NetworkXGraphStore()
    canonical_by_key: Dict[str, str] = {}
    results: List[EntityExtractionResult] = [EntityExtractionResult()] * len(chunks)

    def _run(idx: int, chunk: str) -> Tuple[int, EntityExtractionResult]:
        return idx, extract_entities_and_relations(
            chunk, openai_client, model=model, source_doc=f"{source_doc}#chunk{idx}",
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run, i, c) for i, c in enumerate(chunks)]
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result

    # Merge sequentially after parallel extraction so dedup is deterministic.
    for result in results:
        _merge_into_store(store, canonical_by_key, result)

    return store, results
