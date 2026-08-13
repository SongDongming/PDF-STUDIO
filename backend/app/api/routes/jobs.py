import logging

from fastapi import APIRouter, HTTPException, Response, status

from app.api.routes._common import require_item
from app.schemas import JobCreate, JobView
from app.services.job_execution import error_is_retryable, job_runner
from app.store import store

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = logging.getLogger("uvicorn.error")


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deleteJob",
)
def delete_job(job_id: str) -> Response:
    require_item("jobs", job_id, "任务")
    store.delete("jobs", job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("", response_model=list[JobView], operation_id="listJobs")
def list_jobs() -> list[JobView]:
    items = store.list("jobs")
    latest_ids: dict[tuple[str, str, str], str] = {}
    for item in items:
        group = _job_group(item)
        if group not in latest_ids:
            latest_ids[group] = str(item["id"])
    return [
        _job_view(item, is_current=latest_ids[_job_group(item)] == item["id"])
        for item in items
    ]


def _job_group(item: dict) -> tuple[str, str, str]:
    if item.get("document_id"):
        return ("document", str(item["document_id"]), str(item["kind"]))
    return (
        "knowledge_base",
        str(item.get("knowledge_base_id") or ""),
        str(item["kind"]),
    )


def _job_view(item: dict, *, is_current: bool = True) -> JobView:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    retryable = bool(
        item.get("retryable")
        or result.get("retryable")
        or error_is_retryable(item.get("error_code"))
    )
    return JobView(
        **{
            **item,
            "retryable": retryable
            and item.get("status") in {"failed", "partial"},
            "attempt": int(item.get("attempt") or 1),
            "retry_of": item.get("retry_of") or result.get("retry_of"),
            "is_current": is_current,
        }
    )


def _is_current_job(source: dict) -> bool:
    for item in store.list("jobs"):
        if _job_group(item) == _job_group(source):
            return item["id"] == source["id"]
    return True


@router.post(
    "",
    response_model=JobView,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="createJob",
)
def create_job(payload: JobCreate) -> JobView:
    if payload.knowledge_base_id:
        require_item("knowledge_bases", payload.knowledge_base_id, "知识库")
    if payload.document_id:
        require_item("documents", payload.document_id, "文档")
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
    job_runner.submit(str(item["id"]))
    return _job_view(item)


@router.get("/{job_id}", response_model=JobView, operation_id="getJob")
def get_job(job_id: str) -> JobView:
    item = require_item("jobs", job_id, "任务")
    return _job_view(item, is_current=_is_current_job(item))


@router.post("/{job_id}/run", response_model=JobView, operation_id="runJob")
def run_job(job_id: str) -> JobView:
    """Execute a queued prototype job synchronously.

    Production workers use the same compiler service asynchronously.  This
    endpoint gives local/LAN development an explicit executable path instead
    of silently leaving jobs in a decorative queued state.
    """

    item = require_item("jobs", job_id, "任务")
    if item["status"] not in {"queued", "failed"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "job_not_runnable", "message": "任务当前不可执行"},
        )
    if item["kind"] != "compile_document" or not item.get("document_id"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "job_runner_not_available",
                "message": "该任务类型尚无同步执行器",
            },
        )
    try:
        store.compile_document_now(item["document_id"], job_id)
    except Exception as exc:
        # The compiler normally records a sanitized stage/code.  If it left the
        # job non-terminal (e.g. an unexpected storage error), fail it explicitly
        # instead of leaving it stuck in a decorative queued/running state.
        logger.exception("run_job failed job=%s", job_id)
        refreshed = store.get("jobs", job_id)
        if refreshed is not None and refreshed.get("status") not in {
            "succeeded",
            "failed",
            "partial",
            "canceled",
        }:
            store.update(
                "jobs",
                job_id,
                {
                    "status": "failed",
                    "stage": "pipeline",
                    "error_code": "pipeline_internal_error",
                    "error_message": "知识库流水线发生内部错误",
                    "retryable": False,
                },
            )
    refreshed = require_item("jobs", job_id, "任务")
    return _job_view(refreshed, is_current=_is_current_job(refreshed))


@router.post("/{job_id}/cancel", response_model=JobView, operation_id="cancelJob")
def cancel_job(job_id: str) -> JobView:
    item = require_item("jobs", job_id, "任务")
    if item["status"] not in {"succeeded", "failed", "canceled"}:
        item = store.update(
            "jobs", job_id, {"status": "canceled", "stage": "canceled"}
        )
    return _job_view(item, is_current=_is_current_job(item))


@router.post(
    "/{job_id}/retry",
    response_model=JobView,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="retryJob",
)
def retry_job(job_id: str) -> JobView:
    source = require_item("jobs", job_id, "任务")
    source_view = _job_view(source, is_current=_is_current_job(source))
    if source_view.status not in {"failed", "partial"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "job_not_failed", "message": "只有失败任务才能重试"},
        )
    if not source_view.is_current:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "job_already_resolved",
                "message": "该历史失败已被后续任务覆盖，无需再次重试",
            },
        )
    if not source_view.retryable:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "job_not_retryable",
                "message": "该失败需要修改配置或文档，不能直接重试",
            },
        )
    item = store.create(
        "jobs",
        {
            "kind": source["kind"],
            "knowledge_base_id": source["knowledge_base_id"],
            "document_id": source["document_id"],
            "status": "queued",
            "stage": "waiting",
            "progress": 0,
            "error_code": None,
            "error_message": None,
            "retryable": False,
            "attempt": source_view.attempt + 1,
            "retry_of": job_id,
            "superseded_by": None,
            "result": {"retry_of": job_id},
        },
    )
    store.update("jobs", job_id, {"superseded_by": item["id"]})
    if source.get("document_id"):
        store.update(
            "documents", str(source["document_id"]), {"status": "queued"}
        )
    job_runner.submit(str(item["id"]))
    return _job_view(item)
