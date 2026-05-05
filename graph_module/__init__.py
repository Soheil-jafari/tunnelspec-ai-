"""
TunnelSpec AI — GraphRAG module.

Adds a knowledge-graph reasoning layer alongside the existing extraction
pipeline. Components:

    schemas              — Pydantic models for entities and relations.
    vector_store         — Minimal numpy-backed embedding retrieval.
    entity_extractor     — GPT-4o structured-output entity/relation extraction.
    graph_store          — Abstract store + NetworkX implementation.
    graph_builder        — Chunk → entities → deduped graph orchestration.
    semantic_query       — Cascading-risk, unmitigated-risk, and dependency walks.
    graphrag_pipeline    — Side-by-side vector vs GraphRAG question answering.
"""
