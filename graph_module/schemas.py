"""
Pydantic schemas for the GraphRAG module.

Every Relation carries an evidence_span (the verbatim text snippet that
supports it) and a confidence float. This mirrors the project's
existing 3-D confidence rubric philosophy — claims must be grounded in
the source document, not asserted.
"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "Project",
    "Risk",
    "Mitigation",
    "Phase",
    "Stakeholder",
    "Material",
    "Specification",
    "Cost_Item",
    "Constraint",
    "Location",
]

RelationType = Literal[
    "AFFECTS",
    "MITIGATES",
    "DEPENDS_ON",
    "REQUIRES",
    "RESPONSIBLE_FOR",
    "TRIGGERS",
    "RELATED_TO",
    "PART_OF",
]


class Entity(BaseModel):
    id: str = Field(..., description="Stable identifier, e.g. Risk_Groundwater_Ingress.")
    type: EntityType
    name: str = Field(..., description="Human-readable canonical name.")
    attributes: Dict[str, str] = Field(default_factory=dict)
    source_doc: str = Field(default="", description="Originating document or chunk id.")


class Relation(BaseModel):
    source_id: str
    relation_type: RelationType
    target_id: str
    evidence_span: str = Field(
        ...,
        description="Verbatim snippet from the source text supporting this relation.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class EntityExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
