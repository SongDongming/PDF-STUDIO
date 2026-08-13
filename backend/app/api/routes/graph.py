from fastapi import APIRouter, HTTPException, Query, status

from app.api.routes._common import require_item
from app.schemas import EvidenceLineageView, GraphEdge, GraphNode, GraphView
from app.store import store

router = APIRouter(tags=["graph"])


@router.get(
    "/knowledge-bases/{knowledge_base_id}/graph",
    response_model=GraphView,
    operation_id="getKnowledgeGraph",
)
def get_graph(
    knowledge_base_id: str,
    query: str | None = None,
    node_id: str | None = None,
    hops: int = Query(default=2, ge=0, le=5),
    limit: int = Query(default=200, ge=1, le=1000),
) -> GraphView:
    kb = require_item("knowledge_bases", knowledge_base_id, "知识库")
    try:
        full_snapshot = store.graph_repository.snapshot(knowledge_base_id)
        snapshot = (
            store.graph_repository.local_subgraph(
                knowledge_base_id, node_id, hops=hops, limit=limit
            )
            if node_id
            else full_snapshot
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "graph_unavailable",
                "message": "知识图谱服务暂不可用，请稍后重试",
            },
        ) from exc
    nodes = list(snapshot.nodes)
    edges = list(snapshot.edges)
    total_node_count = len(full_snapshot.nodes)
    total_edge_count = len(full_snapshot.edges)
    if query:
        needle = query.casefold()
        selected = {
            node.id
            for node in nodes
            if needle in node.label.casefold()
            or needle in str(node.properties).casefold()
        }
        edges = [
            edge
            for edge in edges
            if edge.source in selected or edge.target in selected
        ]
        selected.update(
            endpoint
            for edge in edges
            for endpoint in (edge.source, edge.target)
        )
        nodes = [node for node in nodes if node.id in selected]
    elif not node_id:
        # The full graph also contains every page, chunk and OCR element.  Those
        # provenance nodes are useful for evidence traversal but produce an
        # unreadable default canvas.  Rank the semantic entity/claim layer by
        # connectivity; users can still reach provenance through node evidence.
        semantic_nodes = [
            node for node in nodes if node.kind in {"entity", "claim", "wiki"}
        ]
        semantic_ids = {node.id for node in semantic_nodes}
        semantic_edges = [
            edge
            for edge in edges
            if edge.source in semantic_ids and edge.target in semantic_ids
        ]
        degree = {node.id: 0 for node in semantic_nodes}
        for edge in semantic_edges:
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1
        nodes = sorted(
            semantic_nodes,
            key=lambda node: (
                -degree.get(node.id, 0),
                0 if node.kind == "entity" else 1,
                node.label.casefold(),
            ),
        )
        edges = semantic_edges
    if not nodes:
        root = GraphNode(
            id=f"kb:{knowledge_base_id}",
            label=kb["name"],
            kind="entity",
            properties={
                "root": True,
                "graph_version": snapshot.version,
                "empty": True,
            },
        )
        return GraphView(
            knowledge_base_id=knowledge_base_id,
            nodes=[root],
            edges=[],
            graph_version=full_snapshot.version,
            total_node_count=total_node_count,
            total_edge_count=total_edge_count,
            truncated=False,
        )
    truncated = len(nodes) > limit
    allowed = {node.id for node in nodes[:limit]}
    return GraphView(
        knowledge_base_id=knowledge_base_id,
        nodes=[
            GraphNode(
                id=node.id,
                label=node.label,
                kind=node.kind,
                properties={**node.properties, "root": False},
            )
            for node in nodes[:limit]
        ],
        edges=[
            GraphEdge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                relation=edge.relation,
                evidence_ids=[item.key for item in edge.evidence],
                properties=edge.properties,
            )
            for edge in edges
            if edge.source in allowed and edge.target in allowed
        ],
        graph_version=full_snapshot.version,
        total_node_count=total_node_count,
        total_edge_count=total_edge_count,
        truncated=truncated,
    )


@router.get(
    "/knowledge-bases/{knowledge_base_id}/graph/nodes/{node_id}/evidence",
    response_model=list[EvidenceLineageView],
    operation_id="getGraphNodeEvidence",
)
def get_node_evidence(
    knowledge_base_id: str, node_id: str
) -> list[EvidenceLineageView]:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    try:
        lineage = store.graph_repository.evidence_for(knowledge_base_id, node_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "graph_unavailable",
                "message": "知识图谱服务暂不可用，请稍后重试",
            },
        ) from exc
    return [
        EvidenceLineageView(
            document_id=item.document_id,
            page=item.page,
            bbox=item.bbox,
            chunk_id=item.chunk_id,
            element_id=item.element_id,
            pdf_region_url=(
                f"/api/v1/documents/{item.document_id}/regions/{item.element_id}"
                if item.element_id
                else f"/api/v1/documents/{item.document_id}/pages/{item.page}/image"
            ),
        )
        for item in lineage
    ]
