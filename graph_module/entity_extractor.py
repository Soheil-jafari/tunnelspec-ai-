"""
GPT-4o entity & relation extraction with structured outputs.

The system prompt explains the tunnelling-domain taxonomy and demands
an evidence_span (verbatim snippet) plus confidence on every relation,
mirroring the project's existing 3-D rubric grounding philosophy.
"""

from __future__ import annotations

import json
from typing import Optional

from .schemas import EntityExtractionResult


_SYSTEM_PROMPT = """You are a knowledge-graph extractor for civil-engineering tender documents
in the trenchless / tunnelling domain (TBM, pipe jacking, micro-tunnelling, HDD, auger boring).

You read a document chunk and return a strict JSON object describing the entities
mentioned and the relations between them.

ENTITY TYPES (use exactly one per entity):
  Project        — the tender or works package as a whole.
  Risk           — anything that could go wrong (geotechnical, programme, commercial, HSE, third-party).
  Mitigation     — a measure that reduces a Risk.
  Phase          — a stage of works (e.g. Launch Shaft, Tunnel Drive, Reception Shaft, Reinstatement).
  Stakeholder    — a party (HBT, client, asset owner, statutory body, sub-contractor).
  Material       — a tangible product or equipment (segments, TBM, sheet piles, grout, pipe).
  Specification  — a measurable parameter (diameter, drive length, SPT N value, groundwater depth).
  Cost_Item      — a priceable element drawn from a BoQ or rates schedule.
  Constraint     — a non-negotiable rule (working hours, possessions, environmental limits).
  Location       — a place (street, asset, geographic feature).

RELATION TYPES (use exactly one per relation):
  AFFECTS         — A has a measurable impact on B.
  MITIGATES       — Mitigation reduces a Risk.
  DEPENDS_ON      — A cannot proceed without B (typically Phase → Phase).
  REQUIRES        — A needs B (typically Phase → Material/Specification).
  RESPONSIBLE_FOR — Stakeholder owns delivery of B.
  TRIGGERS        — A causes/escalates B (Risk → Risk for cascading chains).
  RELATED_TO      — Generic association when none of the above fit.
  PART_OF         — Compositional containment (Phase ⊂ Project, Cost_Item ⊂ Phase).

GROUNDING RULES (non-negotiable):
  1. Only extract what is explicitly supported by the chunk text. Do not infer.
  2. Each Relation MUST include `evidence_span`: a short verbatim snippet (≤ 30 words)
     copied from the chunk that justifies the relation. No paraphrase.
  3. Each Relation MUST include `confidence` ∈ [0.0, 1.0]:
        ≥ 0.85 — explicit and unambiguous in the chunk
        0.60 – 0.84 — strongly implied by the chunk
        < 0.60 — weak/inferential; prefer to omit unless the link is structurally important.
  4. Use stable snake_case ids prefixed by type, e.g. `Risk_Groundwater_Ingress`,
     `Phase_Tunnel_Drive`, `Material_EPB_TBM`. Reuse the same id across relations.
  5. Output strictly valid JSON matching the schema. No prose, no markdown fences.

OUTPUT JSON SHAPE:
{
  "entities": [
    {"id": "...", "type": "...", "name": "...", "attributes": {"key": "value"}, "source_doc": ""}
  ],
  "relations": [
    {"source_id": "...", "relation_type": "...", "target_id": "...",
     "evidence_span": "...", "confidence": 0.0}
  ]
}
"""


def extract_entities_and_relations(
    text_chunk: str,
    openai_client,
    model: str = "gpt-4o",
    source_doc: Optional[str] = None,
) -> EntityExtractionResult:
    """Run one extraction pass over a single chunk.

    Returns a validated EntityExtractionResult. Returns an empty result
    on parse/validation failure rather than raising — graph construction
    must remain resilient to a single bad chunk.
    """
    response = openai_client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"DOCUMENT CHUNK:\n{text_chunk}"},
        ],
    )
    raw = response.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return EntityExtractionResult()

    if source_doc:
        for ent in data.get("entities", []):
            ent.setdefault("source_doc", source_doc)

    try:
        return EntityExtractionResult.model_validate(data)
    except Exception:
        return EntityExtractionResult()
