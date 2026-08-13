from fastapi import APIRouter

from app.api.routes._common import require_item
from app.schemas import (
    RagSettings,
    RetrievalHitView,
    RetrievalRequest,
    RetrievalResponse,
)
from app.store import store

router = APIRouter(tags=["retrieval"])


@router.post(
    "/knowledge-bases/{knowledge_base_id}/retrieval",
    response_model=RetrievalResponse,
    operation_id="retrieveKnowledgeBase",
)
async def retrieve(
    knowledge_base_id: str, payload: RetrievalRequest
) -> RetrievalResponse:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    rag = RagSettings(**store.settings["rag"])
    if payload.top_k is not None:
        rag = rag.model_copy(
            update={
                "dense_top_k": max(rag.dense_top_k, payload.top_k),
                "lexical_top_k": max(rag.lexical_top_k, payload.top_k),
                "rerank_top_k": payload.top_k,
            }
        )
    hits = await store.retrieve(
        query=payload.query,
        knowledge_base_id=knowledge_base_id,
        rag=rag,
        document_ids=payload.document_ids,
    )
    mode = "empty" if not hits else (
        "hybrid" if store.retriever.embedder is not None else "lexical_fallback"
    )
    return RetrievalResponse(
        query=payload.query,
        knowledge_base_id=knowledge_base_id,
        mode=mode,
        hits=[
            RetrievalHitView(
                chunk_id=hit.chunk.id,
                document_id=hit.chunk.document_id,
                document_title=hit.chunk.document_title,
                page=hit.chunk.page,
                text=hit.chunk.text,
                score=hit.score,
                bbox=hit.chunk.bbox,
                element_id=hit.chunk.element_id,
                asset_ids=hit.chunk.asset_ids,
                citation=hit.as_evidence().citation,
            )
            for hit in hits
        ],
    )
