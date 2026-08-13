"""Grounded knowledge-graph and LLM Wiki compiler.

The compiler accepts only a strict extraction envelope.  Model-supplied
evidence IDs are resolved against normalized chunks/elements and converted to
authoritative document/page/bbox/chunk lineage; callers cannot invent lineage
inside an extraction response.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.services.graph_repository import (
    EvidenceLineage,
    GraphEdgeRecord,
    GraphNodeRecord,
    GraphRepository,
    GraphSnapshot,
)
from app.services.wiki_repository import (
    WikiConclusion,
    WikiPageDraft,
    WikiPageRecord,
    WikiRepository,
    WikiVersionConflict,
)

SCHEMA_VERSION = "1.0"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _strict_keys(
    value: Any, *, required: set[str], optional: set[str], path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{path} contains unsupported fields: {sorted(extra)}")
    return value


def _string_list(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    result = tuple(_required_text(item, f"{path}[]") for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{path} must contain at least one item")
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicates")
    return result


def _normalized_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class NormalizedChunk:
    id: str
    document_id: str
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    element_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_source_location(
            self.id, self.document_id, self.page, self.bbox, "chunk"
        )
        if not self.text.strip():
            raise ValueError("normalized chunk text must be non-empty")


@dataclass(frozen=True, slots=True)
class NormalizedElement:
    id: str
    document_id: str
    chunk_id: str
    page: int
    bbox: tuple[float, float, float, float]
    kind: str
    text: str

    def __post_init__(self) -> None:
        _validate_source_location(
            self.id, self.document_id, self.page, self.bbox, "element"
        )
        if not self.chunk_id.strip() or not self.kind.strip():
            raise ValueError("normalized element requires chunk_id and kind")
        if not self.text.strip():
            raise ValueError("normalized element text must be non-empty")


def _validate_source_location(
    item_id: str,
    document_id: str,
    page: int,
    bbox: tuple[float, float, float, float],
    kind: str,
) -> None:
    if not item_id.strip() or not document_id.strip():
        raise ValueError(f"normalized {kind} requires id and document_id")
    EvidenceLineage(
        document_id=document_id,
        page=page,
        bbox=bbox,
        chunk_id=item_id if kind == "chunk" else "__validation__",
        element_id=item_id if kind == "element" else None,
    )


@dataclass(frozen=True, slots=True)
class EntityDeclaration:
    ref: str
    name: str
    kind: str
    aliases: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClaimDeclaration:
    ref: str
    subject_ref: str
    statement: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RelationDeclaration:
    ref: str
    source_ref: str
    target_ref: str
    predicate: str
    statement: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractionEnvelope:
    document_id: str
    entities: tuple[EntityDeclaration, ...]
    claims: tuple[ClaimDeclaration, ...]
    relations: tuple[RelationDeclaration, ...]


@dataclass(frozen=True, slots=True)
class GraphWikiCompilationResult:
    graph: GraphSnapshot
    wiki_pages: tuple[WikiPageRecord, ...]
    graph_version: int
    wiki_version: int


def extraction_json_schema() -> dict[str, Any]:
    """JSON Schema suitable for DeepSeek strict structured output."""

    evidence = {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "document_id", "entities", "claims", "relations"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "document_id": {"type": "string", "minLength": 1},
            "entities": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ref", "name", "kind", "aliases", "evidence_refs"],
                    "properties": {
                        "ref": {"type": "string", "pattern": _IDENTIFIER.pattern},
                        "name": {"type": "string", "minLength": 1},
                        "kind": {"type": "string", "pattern": _TYPE.pattern},
                        "aliases": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "evidence_refs": evidence,
                    },
                },
            },
            "claims": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "ref",
                        "subject_ref",
                        "statement",
                        "evidence_refs",
                    ],
                    "properties": {
                        "ref": {"type": "string", "pattern": _IDENTIFIER.pattern},
                        "subject_ref": {"type": "string", "pattern": _IDENTIFIER.pattern},
                        "statement": {"type": "string", "minLength": 1},
                        "evidence_refs": evidence,
                    },
                },
            },
            "relations": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "ref",
                        "source_ref",
                        "target_ref",
                        "predicate",
                        "statement",
                        "evidence_refs",
                    ],
                    "properties": {
                        "ref": {"type": "string", "pattern": _IDENTIFIER.pattern},
                        "source_ref": {"type": "string", "pattern": _IDENTIFIER.pattern},
                        "target_ref": {"type": "string", "pattern": _IDENTIFIER.pattern},
                        "predicate": {"type": "string", "pattern": _TYPE.pattern},
                        "statement": {"type": "string", "minLength": 1},
                        "evidence_refs": evidence,
                    },
                },
            },
        },
    }


def parse_extraction(
    payload: Mapping[str, Any],
    *,
    document_id: str,
    chunks: Sequence[NormalizedChunk],
    elements: Sequence[NormalizedElement],
) -> ExtractionEnvelope:
    """Validate a model response and prove every reference against source data."""

    root = _strict_keys(
        payload,
        required={"schema_version", "document_id", "entities", "claims", "relations"},
        optional=set(),
        path="$",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"$.schema_version must equal {SCHEMA_VERSION}")
    if _required_text(root["document_id"], "$.document_id") != document_id:
        raise ValueError("extraction document_id does not match the compile target")
    catalog = _EvidenceCatalog(document_id, chunks, elements)
    raw_entities = root["entities"]
    raw_claims = root["claims"]
    raw_relations = root["relations"]
    if not all(isinstance(items, list) for items in (raw_entities, raw_claims, raw_relations)):
        raise ValueError("entities, claims, and relations must be arrays")
    entities = tuple(
        _parse_entity(item, index, catalog)
        for index, item in enumerate(raw_entities)
    )
    entity_refs = {entity.ref for entity in entities}
    if len(entity_refs) != len(entities):
        raise ValueError("entity refs must be unique")
    claims = tuple(
        _parse_claim(item, index, catalog, entity_refs)
        for index, item in enumerate(raw_claims)
    )
    relations = tuple(
        _parse_relation(item, index, catalog, entity_refs)
        for index, item in enumerate(raw_relations)
    )
    assertion_refs = [item.ref for item in (*claims, *relations)]
    if len(set(assertion_refs)) != len(assertion_refs):
        raise ValueError("claim and relation refs must be globally unique")
    return ExtractionEnvelope(document_id, entities, claims, relations)


class _EvidenceCatalog:
    def __init__(
        self,
        document_id: str,
        chunks: Sequence[NormalizedChunk],
        elements: Sequence[NormalizedElement],
    ) -> None:
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("normalized chunk IDs must be unique")
        if len({element.id for element in elements}) != len(elements):
            raise ValueError("normalized element IDs must be unique")
        if {chunk.id for chunk in chunks} & {element.id for element in elements}:
            raise ValueError("chunk and element IDs must occupy separate namespaces")
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.elements = {element.id: element for element in elements}
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("normalized chunk belongs to another document")
        for element in elements:
            if element.document_id != document_id:
                raise ValueError("normalized element belongs to another document")
            if element.chunk_id not in self.chunks:
                raise ValueError(f"element {element.id} references an unknown chunk")
            if element.id not in self.chunks[element.chunk_id].element_ids:
                raise ValueError(
                    f"element {element.id} is absent from chunk.element_ids"
                )
        for chunk in chunks:
            unknown = set(chunk.element_ids) - set(self.elements)
            if unknown:
                raise ValueError(
                    f"chunk {chunk.id} references unknown elements: {sorted(unknown)}"
                )

    def require(self, refs: tuple[str, ...], path: str) -> None:
        unknown = set(refs) - set(self.chunks) - set(self.elements)
        if unknown:
            raise ValueError(f"{path} references unknown source IDs: {sorted(unknown)}")

    def lineage(self, refs: Iterable[str]) -> tuple[EvidenceLineage, ...]:
        result: dict[str, EvidenceLineage] = {}
        for ref in refs:
            element = self.elements.get(ref)
            if element is not None:
                lineage = EvidenceLineage(
                    document_id=element.document_id,
                    page=element.page,
                    bbox=element.bbox,
                    chunk_id=element.chunk_id,
                    element_id=element.id,
                )
            else:
                chunk = self.chunks[ref]
                lineage = EvidenceLineage(
                    document_id=chunk.document_id,
                    page=chunk.page,
                    bbox=chunk.bbox,
                    chunk_id=chunk.id,
                )
            result[lineage.key] = lineage
        return tuple(result[key] for key in sorted(result))


def _parse_entity(
    raw: Any, index: int, catalog: _EvidenceCatalog
) -> EntityDeclaration:
    path = f"$.entities[{index}]"
    item = _strict_keys(
        raw,
        required={"ref", "name", "kind", "aliases", "evidence_refs"},
        optional=set(),
        path=path,
    )
    ref = _typed_identifier(item["ref"], f"{path}.ref", _IDENTIFIER)
    kind = _typed_identifier(item["kind"], f"{path}.kind", _TYPE)
    aliases = _string_list(item["aliases"], f"{path}.aliases", allow_empty=True)
    evidence = _string_list(item["evidence_refs"], f"{path}.evidence_refs")
    catalog.require(evidence, f"{path}.evidence_refs")
    name = _required_text(item["name"], f"{path}.name")
    if not _normalized_alias(name):
        raise ValueError(f"{path}.name does not contain a usable entity name")
    return EntityDeclaration(ref, name, kind, aliases, evidence)


def _parse_claim(
    raw: Any, index: int, catalog: _EvidenceCatalog, entity_refs: set[str]
) -> ClaimDeclaration:
    path = f"$.claims[{index}]"
    item = _strict_keys(
        raw,
        required={"ref", "subject_ref", "statement", "evidence_refs"},
        optional=set(),
        path=path,
    )
    ref = _typed_identifier(item["ref"], f"{path}.ref", _IDENTIFIER)
    subject = _typed_identifier(
        item["subject_ref"], f"{path}.subject_ref", _IDENTIFIER
    )
    if subject not in entity_refs:
        raise ValueError(f"{path}.subject_ref references an unknown entity")
    evidence = _string_list(item["evidence_refs"], f"{path}.evidence_refs")
    catalog.require(evidence, f"{path}.evidence_refs")
    return ClaimDeclaration(
        ref,
        subject,
        _required_text(item["statement"], f"{path}.statement"),
        evidence,
    )


def _parse_relation(
    raw: Any, index: int, catalog: _EvidenceCatalog, entity_refs: set[str]
) -> RelationDeclaration:
    path = f"$.relations[{index}]"
    item = _strict_keys(
        raw,
        required={
            "ref",
            "source_ref",
            "target_ref",
            "predicate",
            "statement",
            "evidence_refs",
        },
        optional=set(),
        path=path,
    )
    ref = _typed_identifier(item["ref"], f"{path}.ref", _IDENTIFIER)
    source = _typed_identifier(
        item["source_ref"], f"{path}.source_ref", _IDENTIFIER
    )
    target = _typed_identifier(
        item["target_ref"], f"{path}.target_ref", _IDENTIFIER
    )
    if source not in entity_refs or target not in entity_refs:
        raise ValueError(f"{path} references an unknown source or target entity")
    predicate = _typed_identifier(
        item["predicate"], f"{path}.predicate", _TYPE
    )
    evidence = _string_list(item["evidence_refs"], f"{path}.evidence_refs")
    catalog.require(evidence, f"{path}.evidence_refs")
    return RelationDeclaration(
        ref,
        source,
        target,
        predicate,
        _required_text(item["statement"], f"{path}.statement"),
        evidence,
    )


def _typed_identifier(value: Any, path: str, pattern: re.Pattern[str]) -> str:
    result = _required_text(value, path)
    if not pattern.fullmatch(result):
        raise ValueError(f"{path} has an invalid format")
    return result


class WikiGraphCompiler:
    def __init__(
        self, graph_repository: GraphRepository, wiki_repository: WikiRepository
    ) -> None:
        self.graph_repository = graph_repository
        self.wiki_repository = wiki_repository

    def compile_document(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
        document_title: str,
        chunks: Sequence[NormalizedChunk],
        elements: Sequence[NormalizedElement],
        extraction: Mapping[str, Any],
        expected_graph_version: int | None = None,
        expected_wiki_version: int | None = None,
    ) -> GraphWikiCompilationResult:
        if not knowledge_base_id.strip() or not document_title.strip():
            raise ValueError("knowledge_base_id and document_title must be non-empty")
        envelope = parse_extraction(
            extraction,
            document_id=document_id,
            chunks=chunks,
            elements=elements,
        )
        # Detect a wiki concurrency conflict before changing graph state.
        current_wiki_version = self.wiki_repository.current_version(knowledge_base_id)
        if (
            expected_wiki_version is not None
            and expected_wiki_version != current_wiki_version
        ):
            raise WikiVersionConflict(
                f"expected wiki version {expected_wiki_version}, "
                f"found {current_wiki_version}"
            )
        current = self.graph_repository.snapshot(knowledge_base_id)
        nodes, edges = self._build_document_graph(
            knowledge_base_id,
            document_title,
            chunks,
            elements,
            envelope,
            current,
        )
        graph = self.graph_repository.commit_document(
            knowledge_base_id,
            document_id,
            nodes,
            edges,
            expected_version=expected_graph_version,
        )
        drafts = self._compile_wiki(graph)
        pages = self.wiki_repository.commit_generated(
            knowledge_base_id,
            graph.version,
            drafts,
            expected_version=expected_wiki_version,
        )
        return GraphWikiCompilationResult(
            graph=graph,
            wiki_pages=pages,
            graph_version=graph.version,
            wiki_version=self.wiki_repository.current_version(knowledge_base_id),
        )

    def rebuild_wiki(
        self, knowledge_base_id: str, *, graph: Any | None = None
    ) -> tuple[Any, ...]:
        """Regenerate wiki pages from the current graph snapshot.

        Used after a document is removed from the graph: pages whose entities
        lost all machine evidence disappear (unless human-locked), and surviving
        pages keep their grounded conclusions.
        """
        current = graph if graph is not None else self.graph_repository.snapshot(
            knowledge_base_id
        )
        drafts = self._compile_wiki(current)
        return self.wiki_repository.commit_generated(
            knowledge_base_id,
            current.version,
            drafts,
            expected_version=self.wiki_repository.current_version(
                knowledge_base_id
            ),
        )

    def _build_document_graph(
        self,
        knowledge_base_id: str,
        document_title: str,
        chunks: Sequence[NormalizedChunk],
        elements: Sequence[NormalizedElement],
        envelope: ExtractionEnvelope,
        current: GraphSnapshot,
    ) -> tuple[tuple[GraphNodeRecord, ...], tuple[GraphEdgeRecord, ...]]:
        catalog = _EvidenceCatalog(envelope.document_id, chunks, elements)
        entity_nodes, entity_map = self._merge_entities(
            knowledge_base_id, envelope.entities, current
        )
        nodes: dict[str, GraphNodeRecord] = {node.id: node for node in entity_nodes}
        edges: dict[str, GraphEdgeRecord] = {}
        document_node_id = f"document:{envelope.document_id}"
        nodes[document_node_id] = GraphNodeRecord(
            document_node_id,
            knowledge_base_id,
            "document",
            document_title,
            {"document_id": envelope.document_id},
        )
        page_ids: dict[int, str] = {}
        for page in sorted({chunk.page for chunk in chunks} | {item.page for item in elements}):
            page_id = f"page:{envelope.document_id}:{page}"
            page_ids[page] = page_id
            nodes[page_id] = GraphNodeRecord(
                page_id,
                knowledge_base_id,
                "page",
                f"{document_title} · 第 {page} 页",
                {"document_id": envelope.document_id, "page": page},
            )
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    _stable_id("edge", document_node_id, page_id, "CONTAINS"),
                    knowledge_base_id,
                    document_node_id,
                    page_id,
                    "CONTAINS",
                ),
            )
        for chunk in chunks:
            node_id = f"chunk:{envelope.document_id}:{chunk.id}"
            nodes[node_id] = GraphNodeRecord(
                node_id,
                knowledge_base_id,
                "chunk",
                chunk.text[:100],
                {
                    "document_id": envelope.document_id,
                    "chunk_id": chunk.id,
                    "page": chunk.page,
                    "bbox": list(chunk.bbox),
                    "text": chunk.text,
                },
            )
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    _stable_id("edge", page_ids[chunk.page], node_id, "CONTAINS"),
                    knowledge_base_id,
                    page_ids[chunk.page],
                    node_id,
                    "CONTAINS",
                ),
            )
        for element in elements:
            node_id = f"element:{envelope.document_id}:{element.id}"
            nodes[node_id] = GraphNodeRecord(
                node_id,
                knowledge_base_id,
                "asset" if element.kind in {"figure", "table", "formula"} else "chunk",
                element.text[:100],
                {
                    "document_id": envelope.document_id,
                    "element_id": element.id,
                    "chunk_id": element.chunk_id,
                    "page": element.page,
                    "bbox": list(element.bbox),
                    "element_kind": element.kind,
                    "text": element.text,
                },
            )
            parent = f"chunk:{envelope.document_id}:{element.chunk_id}"
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    _stable_id("edge", parent, node_id, "CONTAINS"),
                    knowledge_base_id,
                    parent,
                    node_id,
                    "CONTAINS",
                ),
            )
        for declaration in envelope.entities:
            entity_id = entity_map[declaration.ref]
            for lineage in catalog.lineage(declaration.evidence_refs):
                target = self._evidence_node_id(lineage)
                self._put_edge(
                    edges,
                    GraphEdgeRecord(
                        _stable_id("edge", entity_id, target, "MENTIONED_IN"),
                        knowledge_base_id,
                        entity_id,
                        target,
                        "MENTIONED_IN",
                        (lineage,),
                    ),
                )
        for claim in envelope.claims:
            evidence = catalog.lineage(claim.evidence_refs)
            subject = entity_map[claim.subject_ref]
            assertion_id = _stable_id(
                "claim", subject, "CLAIM", _normalized_alias(claim.statement)
            )
            nodes[assertion_id] = GraphNodeRecord(
                assertion_id,
                knowledge_base_id,
                "claim",
                claim.statement,
                {"statement": claim.statement, "assertion_kind": "claim"},
            )
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    _stable_id("edge", subject, assertion_id, "HAS_CLAIM"),
                    knowledge_base_id,
                    subject,
                    assertion_id,
                    "HAS_CLAIM",
                    evidence,
                    {"statement": claim.statement},
                    True,
                ),
            )
            self._add_evidence_edges(
                edges, knowledge_base_id, assertion_id, evidence
            )
        for relation in envelope.relations:
            evidence = catalog.lineage(relation.evidence_refs)
            source = entity_map[relation.source_ref]
            target = entity_map[relation.target_ref]
            assertion_id = _stable_id(
                "claim",
                source,
                target,
                relation.predicate,
                _normalized_alias(relation.statement),
            )
            nodes[assertion_id] = GraphNodeRecord(
                assertion_id,
                knowledge_base_id,
                "claim",
                relation.statement,
                {
                    "statement": relation.statement,
                    "assertion_kind": "relation",
                    "predicate": relation.predicate,
                    "target_entity_id": target,
                },
            )
            direct_id = _stable_id(
                "relation",
                source,
                target,
                relation.predicate,
                _normalized_alias(relation.statement),
            )
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    direct_id,
                    knowledge_base_id,
                    source,
                    target,
                    relation.predicate,
                    evidence,
                    {
                        "statement": relation.statement,
                        "assertion_id": assertion_id,
                    },
                    True,
                ),
            )
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    _stable_id("edge", source, assertion_id, "ASSERTS"),
                    knowledge_base_id,
                    source,
                    assertion_id,
                    "ASSERTS",
                    evidence,
                    {"statement": relation.statement},
                    True,
                ),
            )
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    _stable_id("edge", assertion_id, target, "OBJECT"),
                    knowledge_base_id,
                    assertion_id,
                    target,
                    "OBJECT",
                    evidence,
                    {"predicate": relation.predicate},
                    True,
                ),
            )
            self._add_evidence_edges(
                edges, knowledge_base_id, assertion_id, evidence
            )
        return (
            tuple(nodes[key] for key in sorted(nodes)),
            tuple(edges[key] for key in sorted(edges)),
        )

    def _merge_entities(
        self,
        knowledge_base_id: str,
        declarations: tuple[EntityDeclaration, ...],
        current: GraphSnapshot,
    ) -> tuple[tuple[GraphNodeRecord, ...], dict[str, str]]:
        existing_by_alias: dict[tuple[str, str], set[str]] = {}
        existing_nodes = {
            node.id: node for node in current.nodes if node.kind == "entity"
        }
        for node in existing_nodes.values():
            aliases = node.properties.get("normalized_aliases", ())
            for alias in aliases:
                existing_by_alias.setdefault((node.properties.get("entity_kind", ""), alias), set()).add(node.id)
        groups: list[list[EntityDeclaration]] = []
        for declaration in declarations:
            aliases = {
                _normalized_alias(item)
                for item in (declaration.name, *declaration.aliases)
            } - {""}
            overlapping: list[int] = []
            for index, group in enumerate(groups):
                if group[0].kind != declaration.kind:
                    continue
                group_aliases = {
                    _normalized_alias(item)
                    for item in group
                    for item in (item.name, *item.aliases)
                }
                if aliases & group_aliases:
                    overlapping.append(index)
            if not overlapping:
                groups.append([declaration])
            else:
                primary = overlapping[0]
                groups[primary].append(declaration)
                for index in reversed(overlapping[1:]):
                    groups[primary].extend(groups.pop(index))
        nodes: list[GraphNodeRecord] = []
        ref_map: dict[str, str] = {}
        for group in groups:
            kind = group[0].kind
            display_aliases = sorted(
                {item for declaration in group for item in (declaration.name, *declaration.aliases)},
                key=lambda item: (len(item), item.casefold()),
            )
            normalized_aliases = sorted(
                {_normalized_alias(item) for item in display_aliases} - {""}
            )
            declared_names = sorted(
                {declaration.name for declaration in group},
                key=lambda item: (-len(item), item.casefold()),
            )
            candidates = {
                entity_id
                for alias in normalized_aliases
                for entity_id in existing_by_alias.get((kind, alias), ())
            }
            if len(candidates) > 1:
                raise ValueError(
                    "entity aliases bridge multiple existing entities; manual merge required"
                )
            if candidates:
                entity_id = next(iter(candidates))
                existing = existing_nodes[entity_id]
                display_aliases = sorted(
                    set(existing.properties.get("aliases", ())) | set(display_aliases),
                    key=lambda item: (len(item), item.casefold()),
                )
                normalized_aliases = sorted(
                    set(existing.properties.get("normalized_aliases", ()))
                    | set(normalized_aliases)
                )
                label = existing.label
            else:
                entity_id = _stable_id(
                    "entity", knowledge_base_id, kind, normalized_aliases[0]
                )
                # Aliases participate in merging but never silently replace the
                # model's declared canonical name as the page title.
                label = declared_names[0]
            node = GraphNodeRecord(
                entity_id,
                knowledge_base_id,
                "entity",
                label,
                {
                    "entity_kind": kind,
                    "aliases": display_aliases,
                    "normalized_aliases": normalized_aliases,
                },
            )
            nodes.append(node)
            for declaration in group:
                ref_map[declaration.ref] = entity_id
        return tuple(nodes), ref_map

    @staticmethod
    def _evidence_node_id(lineage: EvidenceLineage) -> str:
        if lineage.element_id:
            return f"element:{lineage.document_id}:{lineage.element_id}"
        return f"chunk:{lineage.document_id}:{lineage.chunk_id}"

    def _add_evidence_edges(
        self,
        edges: dict[str, GraphEdgeRecord],
        knowledge_base_id: str,
        assertion_id: str,
        evidence: tuple[EvidenceLineage, ...],
    ) -> None:
        for lineage in evidence:
            target = self._evidence_node_id(lineage)
            self._put_edge(
                edges,
                GraphEdgeRecord(
                    _stable_id("evidence", assertion_id, lineage.key),
                    knowledge_base_id,
                    assertion_id,
                    target,
                    "SUPPORTED_BY",
                    (lineage,),
                    {"lineage_key": lineage.key},
                ),
            )

    @staticmethod
    def _put_edge(
        edges: dict[str, GraphEdgeRecord], edge: GraphEdgeRecord
    ) -> None:
        existing = edges.get(edge.id)
        if existing is None:
            edges[edge.id] = edge
            return
        evidence = {item.key: item for item in existing.evidence}
        evidence.update({item.key: item for item in edge.evidence})
        edges[edge.id] = GraphEdgeRecord(
            edge.id,
            edge.knowledge_base_id,
            edge.source,
            edge.target,
            edge.relation,
            tuple(evidence[key] for key in sorted(evidence)),
            {**existing.properties, **edge.properties},
            existing.semantic or edge.semantic,
        )

    def _compile_wiki(self, graph: GraphSnapshot) -> tuple[WikiPageDraft, ...]:
        nodes = {node.id: node for node in graph.nodes}
        entities = {node.id: node for node in graph.nodes if node.kind == "entity"}
        direct_relations = [
            edge
            for edge in graph.edges
            if edge.semantic
            and edge.source in entities
            and edge.target in entities
            and edge.properties.get("assertion_id")
        ]
        claims = [
            edge
            for edge in graph.edges
            if edge.semantic
            and edge.relation == "HAS_CLAIM"
            and edge.source in entities
        ]
        page_ids = {
            entity_id: _stable_id("wiki", graph.knowledge_base_id, entity_id)
            for entity_id in entities
        }
        page_slugs = {
            entity_id: self._slug(entities[entity_id].label, page_ids[entity_id])
            for entity_id in entities
        }
        related: dict[str, set[str]] = {entity_id: set() for entity_id in entities}
        for edge in direct_relations:
            related[edge.source].add(edge.target)
            related[edge.target].add(edge.source)
        drafts: list[WikiPageDraft] = []
        for entity_id, entity in sorted(entities.items(), key=lambda pair: pair[1].label):
            conclusions: list[WikiConclusion] = []
            claim_lines: list[str] = []
            relation_lines: list[str] = []
            incoming_lines: list[str] = []
            for edge in claims:
                if edge.source != entity_id:
                    continue
                statement = str(edge.properties["statement"])
                conclusions.append(
                    WikiConclusion(
                        edge.id,
                        statement,
                        "claim",
                        edge.evidence,
                    )
                )
                claim_lines.append(f"- {statement}{self._citation_suffix(edge.evidence)}")
            for edge in direct_relations:
                statement = str(edge.properties["statement"])
                target = entities[edge.target]
                source = entities[edge.source]
                if edge.source == entity_id:
                    conclusions.append(
                        WikiConclusion(
                            edge.id,
                            statement,
                            "relation",
                            edge.evidence,
                            edge.relation,
                            edge.target,
                        )
                    )
                    relation_lines.append(
                        f"- {statement}（[[{target.label}|{page_slugs[edge.target]}]]）"
                        f"{self._citation_suffix(edge.evidence)}"
                    )
                elif edge.target == entity_id:
                    conclusions.append(
                        WikiConclusion(
                            f"{edge.id}:incoming",
                            statement,
                            "relation",
                            edge.evidence,
                            edge.relation,
                            edge.source,
                        )
                    )
                    incoming_lines.append(
                        f"- {statement}（[[{source.label}|{page_slugs[edge.source]}]]）"
                        f"{self._citation_suffix(edge.evidence)}"
                    )
            if not conclusions:
                continue
            sections: dict[str, str] = {}
            if claim_lines:
                sections["关键结论"] = "\n".join(claim_lines)
            if relation_lines:
                sections["关联关系"] = "\n".join(relation_lines)
            if incoming_lines:
                sections["反向关联"] = "\n".join(incoming_lines)
            entity_kind = entity.properties.get("entity_kind", "ENTITY")
            summary = (
                f"{entity.label} 是知识库中的 {entity_kind} 实体；"
                f"当前页面汇集 {len(conclusions)} 条可回溯结论。"
            )
            drafts.append(
                WikiPageDraft(
                    id=page_ids[entity_id],
                    knowledge_base_id=graph.knowledge_base_id,
                    entity_id=entity_id,
                    slug=page_slugs[entity_id],
                    title=entity.label,
                    summary=summary,
                    sections=sections,
                    conclusions=tuple(conclusions),
                    related_page_ids=tuple(
                        sorted(page_ids[item] for item in related[entity_id])
                    ),
                )
            )
        valid_page_ids = {draft.id for draft in drafts}
        return tuple(
            WikiPageDraft(
                id=draft.id,
                knowledge_base_id=draft.knowledge_base_id,
                entity_id=draft.entity_id,
                slug=draft.slug,
                title=draft.title,
                summary=draft.summary,
                sections=draft.sections,
                conclusions=draft.conclusions,
                related_page_ids=tuple(
                    item for item in draft.related_page_ids if item in valid_page_ids
                ),
            )
            for draft in drafts
        )

    @staticmethod
    def _citation_suffix(evidence: tuple[EvidenceLineage, ...]) -> str:
        refs = "; ".join(
            f"{item.document_id} p.{item.page} · {item.chunk_id}"
            + (f" · {item.element_id}" if item.element_id else "")
            for item in evidence
        )
        return f" `[{refs}]`"

    @staticmethod
    def _slug(label: str, page_id: str) -> str:
        ascii_label = (
            unicodedata.normalize("NFKD", label)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )
        stem = re.sub(r"[^a-z0-9]+", "-", ascii_label).strip("-") or "entity"
        return f"{stem}-{page_id.rsplit(':', 1)[-1][:8]}"
