"""
Storage abstraction for the knowledge graph.

`GraphStore` is the contract; `NetworkXGraphStore` is the in-memory
implementation used for the demo. A `Neo4jGraphStore` placeholder
signals the intended production target without requiring a server
for the portfolio demonstration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .schemas import Entity, Relation


class GraphStore(ABC):
    """Abstract contract every graph backend must implement."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def add_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]: ...

    @abstractmethod
    def get_neighbors(
        self, entity_id: str, relation_types: Optional[Iterable[str]] = None
    ) -> List[Tuple[Entity, Relation]]: ...

    @abstractmethod
    def query_subgraph(self, entity_ids: Iterable[str], radius: int = 1) -> "GraphStore": ...

    @abstractmethod
    def multi_hop_search(
        self,
        start_id: str,
        relation_types: Iterable[str],
        max_hops: int = 3,
    ) -> List[Tuple[Entity, List[Relation]]]: ...

    @abstractmethod
    def all_nodes_of_type(self, entity_type: str) -> List[Entity]: ...

    @abstractmethod
    def all_entities(self) -> List[Entity]: ...

    @abstractmethod
    def all_relations(self) -> List[Relation]: ...


class NetworkXGraphStore(GraphStore):
    """networkx.MultiDiGraph-backed store.

    Entities are nodes (id keyed). Relations are edges; multiple edges
    of different relation_types between the same pair are allowed.
    """

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entities: Dict[str, Entity] = {}

    # ── mutation ──────────────────────────────────────────────────────
    def add_entity(self, entity: Entity) -> None:
        existing = self._entities.get(entity.id)
        if existing is not None:
            merged_attrs = {**existing.attributes, **entity.attributes}
            entity = existing.model_copy(update={"attributes": merged_attrs})
        self._entities[entity.id] = entity
        self._g.add_node(entity.id, type=entity.type, name=entity.name)

    def add_relation(self, relation: Relation) -> None:
        self._g.add_edge(
            relation.source_id,
            relation.target_id,
            key=relation.relation_type,
            relation_type=relation.relation_type,
            evidence_span=relation.evidence_span,
            confidence=relation.confidence,
        )

    # ── reads ────────────────────────────────────────────────────────
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_neighbors(
        self, entity_id: str, relation_types: Optional[Iterable[str]] = None
    ) -> List[Tuple[Entity, Relation]]:
        if entity_id not in self._g:
            return []
        wanted = set(relation_types) if relation_types else None
        out: List[Tuple[Entity, Relation]] = []
        for _, target_id, key, data in self._g.out_edges(entity_id, keys=True, data=True):
            rtype = data.get("relation_type", key)
            if wanted and rtype not in wanted:
                continue
            target = self._entities.get(target_id)
            if target is None:
                continue
            rel = Relation(
                source_id=entity_id,
                relation_type=rtype,
                target_id=target_id,
                evidence_span=data.get("evidence_span", ""),
                confidence=float(data.get("confidence", 0.0)),
            )
            out.append((target, rel))
        return out

    def query_subgraph(self, entity_ids: Iterable[str], radius: int = 1) -> "NetworkXGraphStore":
        seeds = [e for e in entity_ids if e in self._g]
        keep: set = set(seeds)
        frontier = set(seeds)
        for _ in range(max(0, radius)):
            next_frontier: set = set()
            for node in frontier:
                next_frontier.update(self._g.successors(node))
                next_frontier.update(self._g.predecessors(node))
            next_frontier -= keep
            keep.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        sub = NetworkXGraphStore()
        for nid in keep:
            ent = self._entities.get(nid)
            if ent is not None:
                sub.add_entity(ent)
        for u, v, key, data in self._g.edges(keys=True, data=True):
            if u in keep and v in keep:
                sub.add_relation(Relation(
                    source_id=u,
                    relation_type=data.get("relation_type", key),
                    target_id=v,
                    evidence_span=data.get("evidence_span", ""),
                    confidence=float(data.get("confidence", 0.0)),
                ))
        return sub

    def multi_hop_search(
        self,
        start_id: str,
        relation_types: Iterable[str],
        max_hops: int = 3,
    ) -> List[Tuple[Entity, List[Relation]]]:
        """BFS along edges whose type is in `relation_types`. Returns each
        reachable entity together with the path of relations taken to it.
        """
        if start_id not in self._g:
            return []
        wanted = set(relation_types)
        results: List[Tuple[Entity, List[Relation]]] = []
        visited = {start_id}
        queue: deque = deque([(start_id, [])])
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for target, rel in self.get_neighbors(node, wanted):
                if target.id in visited:
                    continue
                visited.add(target.id)
                new_path = path + [rel]
                results.append((target, new_path))
                queue.append((target.id, new_path))
        return results

    def all_nodes_of_type(self, entity_type: str) -> List[Entity]:
        return [e for e in self._entities.values() if e.type == entity_type]

    def all_entities(self) -> List[Entity]:
        return list(self._entities.values())

    def all_relations(self) -> List[Relation]:
        rels: List[Relation] = []
        for u, v, key, data in self._g.edges(keys=True, data=True):
            rels.append(Relation(
                source_id=u,
                relation_type=data.get("relation_type", key),
                target_id=v,
                evidence_span=data.get("evidence_span", ""),
                confidence=float(data.get("confidence", 0.0)),
            ))
        return rels

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Direct access to the underlying networkx graph (for visualisation)."""
        return self._g


class Neo4jGraphStore(GraphStore):
    """Production target — Cypher / Neo4j backed store.

    Defined here to make the abstraction explicit and document the
    intended path beyond the in-memory demo. Not implemented for the
    portfolio scope.
    """

    # TODO: Neo4jGraphStore — implement against neo4j-python driver,
    # mapping add_entity → MERGE (e:Type {id}), add_relation → MERGE
    # (a)-[:TYPE {evidence_span, confidence}]->(b), and multi_hop_search
    # to a parameterised variable-length Cypher query.

    def __init__(self, uri: str = "", user: str = "", password: str = "") -> None:
        raise NotImplementedError(
            "Neo4jGraphStore is a planned production backend; use NetworkXGraphStore for the demo."
        )

    def add_entity(self, entity: Entity) -> None: raise NotImplementedError
    def add_relation(self, relation: Relation) -> None: raise NotImplementedError
    def get_entity(self, entity_id: str) -> Optional[Entity]: raise NotImplementedError
    def get_neighbors(self, entity_id, relation_types=None): raise NotImplementedError
    def query_subgraph(self, entity_ids, radius=1): raise NotImplementedError
    def multi_hop_search(self, start_id, relation_types, max_hops=3): raise NotImplementedError
    def all_nodes_of_type(self, entity_type): raise NotImplementedError
    def all_entities(self): raise NotImplementedError
    def all_relations(self): raise NotImplementedError
