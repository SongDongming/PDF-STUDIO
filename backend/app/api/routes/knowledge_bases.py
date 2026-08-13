from fastapi import APIRouter, Response, status

from app.api.routes._common import require_item
from app.schemas import (
    JobCreate,
    JobView,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseView,
)
from app.services.job_execution import job_runner
from app.store import store

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


def as_view(item: dict) -> KnowledgeBaseView:
    count = sum(
        document["knowledge_base_id"] == item["id"]
        for document in store.list("documents")
    )
    return KnowledgeBaseView(**item, document_count=count)


@router.get("", response_model=list[KnowledgeBaseView], operation_id="listKnowledgeBases")
def list_knowledge_bases() -> list[KnowledgeBaseView]:
    return [as_view(item) for item in store.list("knowledge_bases")]


@router.post(
    "",
    response_model=KnowledgeBaseView,
    status_code=status.HTTP_201_CREATED,
    operation_id="createKnowledgeBase",
)
def create_knowledge_base(payload: KnowledgeBaseCreate) -> KnowledgeBaseView:
    item = store.create(
        "knowledge_bases",
        {
            **payload.model_dump(),
            "status": "draft",
            "published_version": 0,
        },
    )
    return as_view(item)


@router.get(
    "/{knowledge_base_id}",
    response_model=KnowledgeBaseView,
    operation_id="getKnowledgeBase",
)
def get_knowledge_base(knowledge_base_id: str) -> KnowledgeBaseView:
    return as_view(require_item("knowledge_bases", knowledge_base_id, "知识库"))


@router.patch(
    "/{knowledge_base_id}",
    response_model=KnowledgeBaseView,
    operation_id="updateKnowledgeBase",
)
def update_knowledge_base(
    knowledge_base_id: str, payload: KnowledgeBaseUpdate
) -> KnowledgeBaseView:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    item = store.update(
        "knowledge_bases", knowledge_base_id, payload.model_dump(exclude_none=True)
    )
    return as_view(item)


@router.delete(
    "/{knowledge_base_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deleteKnowledgeBase",
)
def delete_knowledge_base(knowledge_base_id: str) -> Response:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    store.remove_knowledge_base_cascade(knowledge_base_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{knowledge_base_id}/reconcile",
    operation_id="reconcileKnowledgeBase",
)
def reconcile_knowledge_base(knowledge_base_id: str) -> dict:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    return store.reconcile_graph_and_wiki(knowledge_base_id)


@router.post(
    "/{knowledge_base_id}/compile",
    response_model=JobView,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="compileKnowledgeBase",
)
def compile_knowledge_base(knowledge_base_id: str) -> JobView:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    payload = JobCreate(kind="rebuild_knowledge_base", knowledge_base_id=knowledge_base_id)
    item = store.create(
        "jobs",
        {
            **payload.model_dump(),
            "status": "queued",
            "stage": "waiting",
            "progress": 0,
            "error_code": None,
            "error_message": None,
            "retryable": False,
            "attempt": 1,
            "retry_of": None,
            "superseded_by": None,
            "result": {},
        },
    )
    store.update("knowledge_bases", knowledge_base_id, {"status": "compiling"})
    job_runner.submit(str(item["id"]))
    return JobView(**item)
