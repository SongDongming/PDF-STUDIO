from fastapi import APIRouter, HTTPException, status

from app.api.routes._common import require_item
from app.schemas import (
    Citation,
    GraphBuildRequest,
    GraphBuildResponse,
    WikiPage,
    WikiPageSummary,
)
from app.services.providers import ProviderConfigurationError, ProviderError
from app.store import store

router = APIRouter(tags=["wiki"])


def _summary(page) -> WikiPageSummary:
    return WikiPageSummary(
        id=page.id,
        knowledge_base_id=page.knowledge_base_id,
        slug=page.slug,
        title=page.title,
        summary=page.summary,
        status=page.status,
        updated_at=page.updated_at,
    )


def _citation(lineage) -> Citation:
    document = store.get("documents", lineage.document_id)
    return Citation(
        id=f"citation:{lineage.key}",
        document_id=lineage.document_id,
        document_title=(
            str(document["title"]) if document else f"文档 {lineage.document_id}"
        ),
        page=lineage.page,
        bbox=lineage.bbox,
        element_id=lineage.element_id,
        excerpt=f"图谱证据 · {lineage.chunk_id}",
    )


@router.get(
    "/knowledge-bases/{knowledge_base_id}/wiki/pages",
    response_model=list[WikiPageSummary],
    operation_id="listWikiPages",
)
def list_wiki_pages(knowledge_base_id: str) -> list[WikiPageSummary]:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    return [
        _summary(page)
        for page in store.wiki_repository.list_pages(knowledge_base_id)
    ]


@router.get(
    "/knowledge-bases/{knowledge_base_id}/wiki/pages/{slug}",
    response_model=WikiPage,
    operation_id="getWikiPage",
)
def get_wiki_page(knowledge_base_id: str, slug: str) -> WikiPage:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    page = store.wiki_repository.get_page(knowledge_base_id, slug)
    if page is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "Wiki 页面不存在"},
        )
    evidence = {
        lineage.key: lineage
        for conclusion in page.conclusions
        for lineage in conclusion.evidence
    }
    return WikiPage(
        **_summary(page).model_dump(),
        markdown=page.markdown,
        citations=[
            _citation(evidence[key]) for key in sorted(evidence)
        ],
        related_page_ids=list(page.related_page_ids),
    )


@router.post(
    "/knowledge-bases/{knowledge_base_id}/wiki/build",
    response_model=GraphBuildResponse,
    operation_id="buildKnowledgeGraphAndWiki",
)
async def build_wiki(
    knowledge_base_id: str, payload: GraphBuildRequest
) -> GraphBuildResponse:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    try:
        result = await store.build_graph_and_wiki(
            knowledge_base_id, payload.document_ids
        )
    except ProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "deepseek_not_configured",
                "message": "模型服务端凭证未配置，不能构建有依据的 Wiki",
            },
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "graph_extraction_unavailable",
                "message": "模型图谱抽取服务暂不可用",
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "graph_build_rejected", "message": str(exc)},
        ) from exc
    store.update(
        "knowledge_bases",
        knowledge_base_id,
        {
            "status": "ready",
            "published_version": result["wiki_version"],
        },
    )
    return GraphBuildResponse(
        knowledge_base_id=knowledge_base_id,
        **result,
    )
