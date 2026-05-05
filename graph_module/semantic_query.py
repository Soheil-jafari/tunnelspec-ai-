"""
Three demo queries that illustrate why a graph layer adds reasoning
capability beyond vector retrieval.

  - find_cascading_risks    — multi-hop risk propagation.
  - find_unmitigated_risks  — risks with no incoming MITIGATES edge.
  - find_critical_dependencies — upstream phases plus required materials/specs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .graph_store import GraphStore
from .schemas import Entity


CASCADING_RELATIONS = ("TRIGGERS", "RELATED_TO")


def find_cascading_risks(
    graph: GraphStore,
    root_risk_id: str,
    max_hops: int = 3,
) -> List[Dict[str, Any]]:
    """BFS from a root risk along TRIGGERS / RELATED_TO edges.

    Returns one dict per downstream risk: {entity, hops, path} where
    `path` is a list of {source, relation, target, evidence_span,
    confidence} dicts describing how we got there.
    """
    root = graph.get_entity(root_risk_id)
    if root is None or root.type != "Risk":
        return []
    out: List[Dict[str, Any]] = []
    for entity, path in graph.multi_hop_search(root_risk_id, CASCADING_RELATIONS, max_hops=max_hops):
        if entity.type != "Risk":
            continue
        out.append({
            "entity": entity,
            "hops": len(path),
            "path": [
                {
                    "source": rel.source_id,
                    "relation": rel.relation_type,
                    "target": rel.target_id,
                    "evidence_span": rel.evidence_span,
                    "confidence": rel.confidence,
                }
                for rel in path
            ],
        })
    return out


def find_unmitigated_risks(graph: GraphStore) -> List[Entity]:
    """Return every Risk entity with no incoming MITIGATES edge."""
    risks = graph.all_nodes_of_type("Risk")
    mitigated_ids = {
        rel.target_id
        for rel in graph.all_relations()
        if rel.relation_type == "MITIGATES"
    }
    return [r for r in risks if r.id not in mitigated_ids]


def find_critical_dependencies(graph: GraphStore, phase_id: str) -> Dict[str, Any]:
    """Return upstream phase dependencies + required materials/specs for a Phase.

    Output:
      {
        "phase":               <Entity or None>,
        "depends_on_phases":   [<Entity>, ...],
        "requires_materials":  [<Entity>, ...],
        "requires_specifications": [<Entity>, ...],
      }
    """
    phase = graph.get_entity(phase_id)
    result: Dict[str, Any] = {
        "phase": phase,
        "depends_on_phases": [],
        "requires_materials": [],
        "requires_specifications": [],
    }
    if phase is None:
        return result

    for target, _ in graph.get_neighbors(phase_id, ("DEPENDS_ON",)):
        if target.type == "Phase":
            result["depends_on_phases"].append(target)

    for target, _ in graph.get_neighbors(phase_id, ("REQUIRES",)):
        if target.type == "Material":
            result["requires_materials"].append(target)
        elif target.type == "Specification":
            result["requires_specifications"].append(target)

    return result
