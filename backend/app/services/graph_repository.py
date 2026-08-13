"""Knowledge-graph persistence ports and implementations.

The in-memory implementation is the executable reference semantics used by
tests.  The Neo4j adapter persists the same snapshot contract without exposing
driver objects to the compiler.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from threading import RLock
from typing import Any, Iterable, Protocol


@dataclass(frozen=True, slots=True, order=True)
class EvidenceLineage:
    """A PDF-grounded evidence location.

    Coordinates are normalized to the unrotated page in the inclusive 0..1
    range.  A chunk is mandatory; an element is optional for text-only chunks.
    """

    document_id: str
    page: int
    bbox: tuple[float, float, float, float]
    chunk_id: str
    element_id: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("evidence.document_id must be non-empty")
        if not self.chunk_id.strip():
            raise ValueError("evidence.chunk_id must be non-empty")
        if self.page < 1:
            raise ValueError("evidence.page must be >= 1")
        if len(self.bbox) != 4:
            raise ValueError("evidence.bbox must contain four coordinates")
        x1, y1, x2, y2 = self.bbox
        if not all(0 <= value <= 1 for value in self.bbox):
            raise ValueError("evidence.bbox coordinates must be normalized to 0..1")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("evidence.bbox must have positive width and height")
        if self.element_id is not None and not self.element_id.strip():
            raise ValueError("evidence.element_id cannot be blank")

    @property
    def key(self) -> str:
        element = self.element_id or "-"
        box = ",".join(f"{coordinate:.6f}" for coordinate in self.bbox)
        return f"{self.document_id}:{self.page}:{self.chunk_id}:{element}:{box}"


@dataclass(frozen=True, slots=True)
class GraphNodeRecord:
    id: str
    knowledge_base_id: str
    kind: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.id, self.knowledge_base_id, self.kind, self.label)
        ):
            raise ValueError("graph node identity, kind, and label must be non-empty")


@dataclass(frozen=True, slots=True)
class GraphEdgeRecord:
    id: str
    knowledge_base_id: str
    source: str
    target: str
    relation: str
    evidence: tuple[EvidenceLineage, ...] = ()
    properties: dict[str, Any] = field(default_factory=dict)
    semantic: bool = False

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.id,
                self.knowledge_base_id,
                self.source,
                self.target,
                self.relation,
            )
        ):
            raise ValueError("graph edge identity and endpoints must be non-empty")
        if self.semantic and not self.evidence:
            raise ValueError("semantic graph edges require PDF-grounded evidence")
        if len({item.key for item in self.evidence}) != len(self.evidence):
            raise ValueError("graph edge evidence must not contain duplicates")


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    knowledge_base_id: str
    version: int
    nodes: tuple[GraphNodeRecord, ...]
    edges: tuple[GraphEdgeRecord, ...]

    def node(self, node_id: str) -> GraphNodeRecord | None:
        return next((node for node in self.nodes if node.id == node_id), None)


class GraphVersionConflict(RuntimeError):
    pass


class GraphRepository(Protocol):
    def snapshot(self, knowledge_base_id: str) -> GraphSnapshot: ...

    def commit_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        nodes: Iterable[GraphNodeRecord],
        edges: Iterable[GraphEdgeRecord],
        *,
        expected_version: int | None = None,
    ) -> GraphSnapshot: ...

    def remove_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        *,
        expected_version: int | None = None,
    ) -> GraphSnapshot: ...

    def has_document_contribution(
        self, knowledge_base_id: str, document_id: str
    ) -> bool: ...

    def remove_knowledge_base(self, knowledge_base_id: str) -> None: ...

    def evidence_for(
        self, knowledge_base_id: str, node_or_edge_id: str
    ) -> tuple[EvidenceLineage, ...]: ...

    def local_subgraph(
        self,
        knowledge_base_id: str,
        node_id: str,
        *,
        hops: int = 2,
        limit: int = 200,
    ) -> GraphSnapshot: ...

class InMemoryGraphRepository:
    """Per-document contributions with optimistic, incremental publication.

    Recompiling one document replaces only that document's contribution.
    Identical entity/assertion/relationship IDs from other documents are merged
    and their evidence is unioned, so removing a document cannot erase evidence
    still supplied by another document.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._versions: dict[str, int] = defaultdict(int)
        self._batches: dict[
            str, dict[str, tuple[tuple[GraphNodeRecord, ...], tuple[GraphEdgeRecord, ...]]]
        ] = defaultdict(dict)

    def snapshot(self, knowledge_base_id: str) -> GraphSnapshot:
        with self._lock:
            return self._aggregate(knowledge_base_id)

    def commit_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        nodes: Iterable[GraphNodeRecord],
        edges: Iterable[GraphEdgeRecord],
        *,
        expected_version: int | None = None,
    ) -> GraphSnapshot:
        node_items = tuple(nodes)
        edge_items = tuple(edges)
        self._validate_batch(knowledge_base_id, document_id, node_items, edge_items)
        with self._lock:
            current = self._versions[knowledge_base_id]
            if expected_version is not None and expected_version != current:
                raise GraphVersionConflict(
                    f"expected graph version {expected_version}, found {current}"
                )
            self._batches[knowledge_base_id][document_id] = (
                deepcopy(node_items),
                deepcopy(edge_items),
            )
            self._versions[knowledge_base_id] = current + 1
            return self._aggregate(knowledge_base_id)

    def remove_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        *,
        expected_version: int | None = None,
    ) -> GraphSnapshot:
        with self._lock:
            current = self._versions[knowledge_base_id]
            if expected_version is not None and expected_version != current:
                raise GraphVersionConflict(
                    f"expected graph version {expected_version}, found {current}"
                )
            batches = self._batches[knowledge_base_id]
            if document_id not in batches:
                return self._aggregate(knowledge_base_id)
            del batches[document_id]
            self._versions[knowledge_base_id] = current + 1
            return self._aggregate(knowledge_base_id)

    def has_document_contribution(
        self, knowledge_base_id: str, document_id: str
    ) -> bool:
        with self._lock:
            return document_id in self._batches.get(knowledge_base_id, {})

    def remove_knowledge_base(self, knowledge_base_id: str) -> None:
        with self._lock:
            self._batches.pop(knowledge_base_id, None)
            self._versions.pop(knowledge_base_id, None)

    def evidence_for(
        self, knowledge_base_id: str, node_or_edge_id: str
    ) -> tuple[EvidenceLineage, ...]:
        snapshot = self.snapshot(knowledge_base_id)
        evidence: dict[str, EvidenceLineage] = {}
        for edge in snapshot.edges:
            if (
                edge.id == node_or_edge_id
                or edge.source == node_or_edge_id
                or edge.target == node_or_edge_id
            ):
                for item in edge.evidence:
                    evidence[item.key] = item
        return tuple(evidence[key] for key in sorted(evidence))

    def local_subgraph(
        self,
        knowledge_base_id: str,
        node_id: str,
        *,
        hops: int = 2,
        limit: int = 200,
    ) -> GraphSnapshot:
        if hops < 0:
            raise ValueError("hops must be >= 0")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        snapshot = self.snapshot(knowledge_base_id)
        node_map = {node.id: node for node in snapshot.nodes}
        if node_id not in node_map:
            return GraphSnapshot(knowledge_base_id, snapshot.version, (), ())
        adjacency: dict[str, list[tuple[str, GraphEdgeRecord]]] = defaultdict(list)
        for edge in snapshot.edges:
            adjacency[edge.source].append((edge.target, edge))
            adjacency[edge.target].append((edge.source, edge))
        selected = {node_id}
        selected_edges: dict[str, GraphEdgeRecord] = {}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        while queue and len(selected) < limit:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbour, edge in sorted(
                adjacency[current], key=lambda pair: (pair[1].relation, pair[0])
            ):
                if len(selected) >= limit and neighbour not in selected:
                    break
                selected_edges[edge.id] = edge
                if neighbour not in selected:
                    selected.add(neighbour)
                    queue.append((neighbour, depth + 1))
        return GraphSnapshot(
            knowledge_base_id=knowledge_base_id,
            version=snapshot.version,
            nodes=tuple(node_map[item] for item in sorted(selected)),
            edges=tuple(selected_edges[item] for item in sorted(selected_edges)),
        )

    def _aggregate(self, knowledge_base_id: str) -> GraphSnapshot:
        nodes: dict[str, GraphNodeRecord] = {}
        edges: dict[str, GraphEdgeRecord] = {}
        for document_id in sorted(self._batches[knowledge_base_id]):
            batch_nodes, batch_edges = self._batches[knowledge_base_id][document_id]
            for node in batch_nodes:
                existing = nodes.get(node.id)
                if existing is None:
                    nodes[node.id] = deepcopy(node)
                else:
                    properties = {**existing.properties, **node.properties}
                    if existing.kind == "entity":
                        aliases = set(existing.properties.get("aliases", ()))
                        aliases.update(node.properties.get("aliases", ()))
                        properties["aliases"] = sorted(aliases)
                    nodes[node.id] = GraphNodeRecord(
                        id=node.id,
                        knowledge_base_id=knowledge_base_id,
                        kind=existing.kind,
                        label=existing.label,
                        properties=properties,
                    )
            for edge in batch_edges:
                existing = edges.get(edge.id)
                if existing is None:
                    edges[edge.id] = deepcopy(edge)
                    continue
                evidence = {item.key: item for item in existing.evidence}
                evidence.update({item.key: item for item in edge.evidence})
                edges[edge.id] = GraphEdgeRecord(
                    id=edge.id,
                    knowledge_base_id=knowledge_base_id,
                    source=edge.source,
                    target=edge.target,
                    relation=edge.relation,
                    evidence=tuple(evidence[key] for key in sorted(evidence)),
                    properties={**existing.properties, **edge.properties},
                    semantic=existing.semantic or edge.semantic,
                )
        referenced = {edge.source for edge in edges.values()} | {
            edge.target for edge in edges.values()
        }
        nodes = {
            node_id: node
            for node_id, node in nodes.items()
            if node.kind == "entity" or node_id in referenced
        }
        return GraphSnapshot(
            knowledge_base_id=knowledge_base_id,
            version=self._versions[knowledge_base_id],
            nodes=tuple(nodes[key] for key in sorted(nodes)),
            edges=tuple(edges[key] for key in sorted(edges)),
        )

    @staticmethod
    def _validate_batch(
        knowledge_base_id: str,
        document_id: str,
        nodes: tuple[GraphNodeRecord, ...],
        edges: tuple[GraphEdgeRecord, ...],
    ) -> None:
        if not knowledge_base_id.strip() or not document_id.strip():
            raise ValueError("knowledge_base_id and document_id must be non-empty")
        node_ids = {node.id for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("graph batch contains duplicate node IDs")
        for node in nodes:
            if node.knowledge_base_id != knowledge_base_id:
                raise ValueError("graph node belongs to another knowledge base")
        edge_ids = {edge.id for edge in edges}
        if len(edge_ids) != len(edges):
            raise ValueError("graph batch contains duplicate edge IDs")
        for edge in edges:
            if edge.knowledge_base_id != knowledge_base_id:
                raise ValueError("graph edge belongs to another knowledge base")
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError("graph edge endpoint is absent from the document batch")
            if any(item.document_id != document_id for item in edge.evidence):
                raise ValueError(
                    "a document contribution may only contain its own evidence"
                )


class Neo4jGraphRepository(InMemoryGraphRepository):
    """Neo4j-backed adapter with a lazily hydrated deterministic read mirror."""

    def __init__(self, driver: Any, *, database: str = "neo4j") -> None:
        super().__init__()
        self._driver = driver
        self._database = database
        self._hydrated: set[str] = set()

    def snapshot(self, knowledge_base_id: str) -> GraphSnapshot:
        self._ensure_hydrated(knowledge_base_id)
        return super().snapshot(knowledge_base_id)

    def commit_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        nodes: Iterable[GraphNodeRecord],
        edges: Iterable[GraphEdgeRecord],
        *,
        expected_version: int | None = None,
    ) -> GraphSnapshot:
        node_items = tuple(nodes)
        edge_items = tuple(edges)
        self._ensure_hydrated(knowledge_base_id)
        self._validate_batch(
            knowledge_base_id, document_id, node_items, edge_items
        )
        current = self._versions[knowledge_base_id]
        if expected_version is not None and expected_version != current:
            raise GraphVersionConflict(
                f"expected graph version {expected_version}, found {current}"
            )
        node_payload = [
            {
                "id": node.id,
                "kb": node.knowledge_base_id,
                "kind": node.kind,
                "label": node.label,
                "properties_json": json.dumps(
                    node.properties, ensure_ascii=False, sort_keys=True
                ),
            }
            for node in node_items
        ]
        edge_payload = [
            {
                "id": edge.id,
                "kb": edge.knowledge_base_id,
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "semantic": edge.semantic,
                "properties_json": json.dumps(
                    edge.properties, ensure_ascii=False, sort_keys=True
                ),
                "evidence_json": json.dumps(
                    [asdict(item) | {"key": item.key} for item in edge.evidence],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for edge in edge_items
        ]
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._write_document,
                knowledge_base_id,
                document_id,
                current,
                current + 1,
                node_payload,
                edge_payload,
            )
        return super().commit_document(
            knowledge_base_id,
            document_id,
            node_items,
            edge_items,
            expected_version=current,
        )

    def remove_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        *,
        expected_version: int | None = None,
    ) -> GraphSnapshot:
        self._ensure_hydrated(knowledge_base_id)
        current = self._versions[knowledge_base_id]
        if expected_version is not None and expected_version != current:
            raise GraphVersionConflict(
                f"expected graph version {expected_version}, found {current}"
            )
        if document_id not in self._batches.get(knowledge_base_id, {}):
            return super().remove_document(
                knowledge_base_id, document_id, expected_version=current
            )
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._remove_document,
                knowledge_base_id,
                document_id,
                current,
                current + 1,
            )
        return super().remove_document(
            knowledge_base_id,
            document_id,
            expected_version=current,
        )

    @staticmethod
    def _remove_document(
        tx: Any,
        knowledge_base_id: str,
        document_id: str,
        expected_version: int,
        version: int,
    ) -> None:
        version_record = tx.run(
            """
            MERGE (v:GraphVersion {knowledge_base_id: $kb})
            ON CREATE SET v.version = 0
            RETURN v.version AS version
            """,
            kb=knowledge_base_id,
        ).single()
        actual_version = int(version_record["version"])
        if actual_version != expected_version:
            raise GraphVersionConflict(
                f"expected graph version {expected_version}, found {actual_version}"
            )
        # The same cleanup the compiler runs when replacing a document's
        # contribution, without re-adding nodes/edges: drop its edges, remove it
        # from shared-node contribution lists, detach nodes with no remaining
        # contributions, then advance the graph version.
        tx.run(
            """
            MATCH ()-[r:GRAPH_EDGE {knowledge_base_id: $kb, document_id: $doc}]->()
            DELETE r
            """,
            kb=knowledge_base_id,
            doc=document_id,
        )
        tx.run(
            """
            MATCH (n:KnowledgeNode {knowledge_base_id: $kb})
            WHERE $doc IN coalesce(n.contribution_documents, [])
            SET n.contribution_documents =
                [item IN n.contribution_documents WHERE item <> $doc]
            """,
            kb=knowledge_base_id,
            doc=document_id,
        )
        tx.run(
            """
            MATCH (n:KnowledgeNode {knowledge_base_id: $kb})
            WHERE size(coalesce(n.contribution_documents, [])) = 0
            DETACH DELETE n
            """,
            kb=knowledge_base_id,
        )
        tx.run(
            """
            MERGE (v:GraphVersion {knowledge_base_id: $kb})
            SET v.version = $version
            """,
            kb=knowledge_base_id,
            version=version,
        )

    def remove_knowledge_base(self, knowledge_base_id: str) -> None:
        self._ensure_hydrated(knowledge_base_id)
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._remove_knowledge_base, knowledge_base_id
            )
        with self._lock:
            self._batches.pop(knowledge_base_id, None)
            self._versions.pop(knowledge_base_id, None)
            self._hydrated.discard(knowledge_base_id)

    @staticmethod
    def _remove_knowledge_base(tx: Any, knowledge_base_id: str) -> None:
        tx.run(
            "MATCH (n:KnowledgeNode {knowledge_base_id: $kb}) DETACH DELETE n",
            kb=knowledge_base_id,
        )
        tx.run(
            "MATCH (v:GraphVersion {knowledge_base_id: $kb}) DELETE v",
            kb=knowledge_base_id,
        )

    def _ensure_hydrated(self, knowledge_base_id: str) -> None:
        with self._lock:
            if knowledge_base_id in self._hydrated:
                return
            with self._driver.session(database=self._database) as session:
                version, node_rows, edge_rows = session.execute_read(
                    self._read_knowledge_base, knowledge_base_id
                )
            batches: dict[
                str, tuple[list[GraphNodeRecord], list[GraphEdgeRecord]]
            ] = {}
            node_map: dict[str, GraphNodeRecord] = {}
            contributions: dict[str, tuple[str, ...]] = {}
            for row in node_rows:
                properties = json.loads(row.get("properties_json") or "{}")
                node = GraphNodeRecord(
                    id=row["id"],
                    knowledge_base_id=knowledge_base_id,
                    kind=row["kind"],
                    label=row["label"],
                    properties=properties,
                )
                node_map[node.id] = node
                contributions[node.id] = tuple(row.get("contribution_documents") or ())
                for document_id in contributions[node.id]:
                    batches.setdefault(document_id, ([], []))[0].append(node)
            for row in edge_rows:
                raw_evidence = json.loads(row.get("evidence_json") or "[]")
                evidence = tuple(
                    EvidenceLineage(
                        document_id=item["document_id"],
                        page=int(item["page"]),
                        bbox=tuple(float(value) for value in item["bbox"]),
                        chunk_id=item["chunk_id"],
                        element_id=item.get("element_id"),
                    )
                    for item in raw_evidence
                )
                edge = GraphEdgeRecord(
                    id=row["id"],
                    knowledge_base_id=knowledge_base_id,
                    source=row["source"],
                    target=row["target"],
                    relation=row["relation"],
                    evidence=evidence,
                    properties=json.loads(row.get("properties_json") or "{}"),
                    semantic=bool(row.get("semantic")),
                )
                document_id = row["document_id"]
                batch_nodes, batch_edges = batches.setdefault(document_id, ([], []))
                for endpoint in (edge.source, edge.target):
                    if endpoint not in {node.id for node in batch_nodes}:
                        batch_nodes.append(node_map[endpoint])
                batch_edges.append(edge)
            self._batches[knowledge_base_id] = {
                document_id: (
                    tuple({node.id: node for node in nodes}.values()),
                    tuple(edges),
                )
                for document_id, (nodes, edges) in batches.items()
            }
            self._versions[knowledge_base_id] = int(version)
            self._hydrated.add(knowledge_base_id)

    @staticmethod
    def _read_knowledge_base(
        tx: Any, knowledge_base_id: str
    ) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
        version_record = tx.run(
            """
            OPTIONAL MATCH (v:GraphVersion {knowledge_base_id: $kb})
            RETURN coalesce(v.version, 0) AS version
            """,
            kb=knowledge_base_id,
        ).single()
        node_rows = [
            record["node"]
            for record in tx.run(
                """
                MATCH (n:KnowledgeNode {knowledge_base_id: $kb})
                RETURN {
                    id: n.id,
                    kind: n.kind,
                    label: n.label,
                    properties_json: n.properties_json,
                    contribution_documents: n.contribution_documents
                } AS node
                """,
                kb=knowledge_base_id,
            )
        ]
        edge_rows = [
            record["edge"]
            for record in tx.run(
                """
                MATCH (source:KnowledgeNode {knowledge_base_id: $kb})
                      -[r:GRAPH_EDGE {knowledge_base_id: $kb}]->
                      (target:KnowledgeNode {knowledge_base_id: $kb})
                RETURN {
                    id: r.id,
                    document_id: r.document_id,
                    source: source.id,
                    target: target.id,
                    relation: r.relation,
                    semantic: r.semantic,
                    properties_json: r.properties_json,
                    evidence_json: r.evidence_json
                } AS edge
                """,
                kb=knowledge_base_id,
            )
        ]
        return int(version_record["version"]), node_rows, edge_rows

    @staticmethod
    def _write_document(
        tx: Any,
        knowledge_base_id: str,
        document_id: str,
        expected_version: int,
        version: int,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        version_record = tx.run(
            """
            MERGE (v:GraphVersion {knowledge_base_id: $kb})
            ON CREATE SET v.version = 0
            RETURN v.version AS version
            """,
            kb=knowledge_base_id,
        ).single()
        actual_version = int(version_record["version"])
        if actual_version != expected_version:
            raise GraphVersionConflict(
                f"expected graph version {expected_version}, found {actual_version}"
            )
        tx.run(
            """
            MATCH ()-[r:GRAPH_EDGE {knowledge_base_id: $kb, document_id: $doc}]->()
            DELETE r
            """,
            kb=knowledge_base_id,
            doc=document_id,
        )
        tx.run(
            """
            MATCH (n:KnowledgeNode {knowledge_base_id: $kb})
            WHERE $doc IN coalesce(n.contribution_documents, [])
            SET n.contribution_documents =
                [item IN n.contribution_documents WHERE item <> $doc]
            """,
            kb=knowledge_base_id,
            doc=document_id,
        )
        tx.run(
            """
            MATCH (n:KnowledgeNode {knowledge_base_id: $kb})
            WHERE size(coalesce(n.contribution_documents, [])) = 0
            DETACH DELETE n
            """,
            kb=knowledge_base_id,
        )
        tx.run(
            """
            UNWIND $nodes AS item
            MERGE (n:KnowledgeNode {knowledge_base_id: item.kb, id: item.id})
            SET n.kind = item.kind,
                n.label = item.label,
                n.properties_json = item.properties_json,
                n.contribution_documents =
                    CASE WHEN $doc IN coalesce(n.contribution_documents, [])
                    THEN n.contribution_documents
                    ELSE coalesce(n.contribution_documents, []) + $doc END
            """,
            nodes=nodes,
            doc=document_id,
        )
        tx.run(
            """
            UNWIND $edges AS item
            MATCH (source:KnowledgeNode {knowledge_base_id: item.kb, id: item.source})
            MATCH (target:KnowledgeNode {knowledge_base_id: item.kb, id: item.target})
            CREATE (source)-[r:GRAPH_EDGE {
                knowledge_base_id: item.kb,
                document_id: $doc,
                id: item.id
            }]->(target)
            SET r.relation = item.relation,
                r.semantic = item.semantic,
                r.properties_json = item.properties_json,
                r.evidence_json = item.evidence_json
            """,
            edges=edges,
            doc=document_id,
        )
        tx.run(
            """
            MERGE (v:GraphVersion {knowledge_base_id: $kb})
            SET v.version = $version
            """,
            kb=knowledge_base_id,
            version=version,
        )
