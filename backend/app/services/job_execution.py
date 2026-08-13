from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from threading import RLock
from typing import Any

from app.config import get_settings
from app.services.ingestion import CompilationError
from app.services.providers import ProviderConfigurationError, ProviderError


RETRYABLE_ERROR_CODES = frozenset(
    {
        "connection_error",
        "graph_extraction_unavailable",
        "ocr_timeout",
        "ocr_unavailable",
        "process_restarted",
        "provider_unavailable",
        "rate_limited",
        "service_unavailable",
        "temporary_failure",
    }
)


def error_is_retryable(code: str | None) -> bool:
    return bool(code and code in RETRYABLE_ERROR_CODES)


class LocalJobRunner:
    """Run the local product pipeline outside the HTTP request.

    The prototype metadata store is intentionally process-local and persisted as
    one transactional snapshot, so a separate Celery process cannot safely
    mutate it. A single-worker executor keeps mutations in the API process,
    serializes access to the one-GPU OCR service, and lets upload/compile
    endpoints return immediately.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(os.getenv("APP_LOCAL_JOB_CONCURRENCY", "1"))),
            thread_name_prefix="pdfwiki-job",
        )
        self._lock = RLock()
        self._scheduled: set[str] = set()
        # A single-worker executor means one hung pipeline blocks the queue.
        # Enforce a hard budget per job and abandon (fail) it when exceeded so
        # later jobs still run.
        self._timeout_seconds = float(
            os.getenv("APP_JOB_TIMEOUT_SECONDS", "1500")
        )

    @staticmethod
    def enabled() -> bool:
        return (
            get_settings().env != "test"
            and os.getenv("APP_COMPILE_MODE", "deferred").lower() == "sync"
        )

    def submit(self, job_id: str) -> bool:
        if not self.enabled():
            return False
        with self._lock:
            if job_id in self._scheduled:
                return True
            self._scheduled.add(job_id)
        self._executor.submit(self._execute, job_id)
        return True

    def _execute(self, job_id: str) -> None:
        from app.store import store

        # Run the job body in a daemon thread with a hard budget.  If the
        # pipeline hangs (e.g. a provider call), the job is failed and the
        # single executor thread is freed for the next queued job instead of
        # blocking the whole queue indefinitely.
        worker = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"pdfwiki-job-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        worker.join(timeout=self._timeout_seconds)
        if worker.is_alive():
            # Register the timeout BEFORE failing the job: the abandoned worker
            # thread is still running and would otherwise flip the job back to
            # running/succeeded when its late writes land.
            store.mark_job_timed_out(job_id)
            try:
                store.update(
                    "jobs",
                    job_id,
                    {
                        "status": "failed",
                        "stage": "timeout",
                        "error_code": "job_timeout",
                        "error_message": "任务执行超时，已中止，不会阻塞后续任务",
                        "retryable": False,
                    },
                )
            except Exception:
                pass
            with self._lock:
                self._scheduled.discard(job_id)

    def _run_job(self, job_id: str) -> None:
        from app.store import store

        try:
            job = store.get("jobs", job_id)
            if job is None or job.get("status") not in {"queued", "failed"}:
                return
            if job["kind"] == "compile_document" and job.get("document_id"):
                self._compile_document(job)
            elif job["kind"] == "rebuild_knowledge_base" and job.get(
                "knowledge_base_id"
            ):
                self._rebuild_knowledge_base(job)
            else:
                store.update(
                    "jobs",
                    job_id,
                    {
                        "status": "failed",
                        "stage": "unsupported",
                        "error_code": "job_runner_not_available",
                        "error_message": "该任务类型尚无可用执行器",
                        "retryable": False,
                    },
                )
        except CompilationError as exc:
            current = store.get("jobs", job_id) or {}
            store.update(
                "jobs",
                job_id,
                {
                    "status": "failed",
                    "stage": current.get("stage") or exc.stage,
                    "error_code": current.get("error_code") or exc.code,
                    "error_message": current.get("error_message") or str(exc),
                    "retryable": bool(
                        current.get("retryable", exc.retryable)
                    ),
                },
            )
        except ProviderConfigurationError as exc:
            self._mark_pipeline_failed(
                job_id,
                code="provider_not_configured",
                message=str(exc) or "模型服务未配置",
                retryable=False,
            )
        except ProviderError:
            self._mark_pipeline_failed(
                job_id,
                code="provider_unavailable",
                message="模型服务暂不可用，请稍后重试",
                retryable=True,
            )
        except (ConnectionError, TimeoutError):
            self._mark_pipeline_failed(
                job_id,
                code="connection_error",
                message="外部服务连接中断，请稍后重试",
                retryable=True,
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._mark_pipeline_failed(
                job_id,
                code="pipeline_internal_error",
                message="知识库流水线发生内部错误",
                retryable=False,
            )
        finally:
            with self._lock:
                self._scheduled.discard(job_id)

    @staticmethod
    def _mark_pipeline_failed(
        job_id: str, *, code: str, message: str, retryable: bool
    ) -> None:
        from app.store import store

        current = store.get("jobs", job_id)
        if current is None:
            return
        store.update(
            "jobs",
            job_id,
            {
                "status": "failed",
                "stage": current.get("stage") or "pipeline",
                "error_code": code,
                "error_message": message,
                "retryable": retryable,
            },
        )
        document_id = current.get("document_id")
        if document_id and current.get("stage") != "graph_wiki":
            store.update(
                "documents",
                str(document_id),
                {"status": "failed", "error_message": message},
            )
        knowledge_base_id = current.get("knowledge_base_id")
        if knowledge_base_id:
            ready_documents = [
                document
                for document in store.list("documents")
                if document.get("knowledge_base_id") == knowledge_base_id
                and document.get("status") == "ready"
            ]
            store.update(
                "knowledge_bases",
                knowledge_base_id,
                {"status": "partial" if ready_documents else "failed"},
            )

    @staticmethod
    def _compile_document(job: dict[str, Any]) -> None:
        from app.store import store

        job_id = str(job["id"])
        document_id = str(job["document_id"])
        payload = store.compile_document_now(document_id, job_id)
        document = store.get("documents", document_id)
        if document is None:
            return

        store.update(
            "jobs",
            job_id,
            {"status": "running", "stage": "graph_wiki", "progress": 96},
        )
        try:
            graph_result = asyncio.run(
                store.build_graph_and_wiki(
                    str(document["knowledge_base_id"]), [document_id]
                )
            )
        except ProviderConfigurationError as exc:
            store.update(
                "jobs",
                job_id,
                {
                    "status": "partial",
                    "stage": "graph_wiki",
                    "progress": 100,
                    "error_code": "provider_not_configured",
                    "error_message": str(exc) or "文档已入库，但 Wiki 模型未配置",
                    "retryable": False,
                    "result": {**payload, "knowledge_refresh": "skipped"},
                },
            )
            return
        except ProviderError:
            store.update(
                "jobs",
                job_id,
                {
                    "status": "partial",
                    "stage": "graph_wiki",
                    "progress": 100,
                    "error_code": "provider_unavailable",
                    "error_message": "文档已入库，但图谱与 Wiki 更新暂时失败",
                    "retryable": True,
                    "result": {**payload, "knowledge_refresh": "pending"},
                },
            )
            return

        store.update(
            "knowledge_bases",
            str(document["knowledge_base_id"]),
            {
                "status": "ready",
                "published_version": graph_result["wiki_version"],
            },
        )
        store.update(
            "jobs",
            job_id,
            {
                "status": "succeeded",
                "stage": "completed",
                "progress": 100,
                "error_code": None,
                "error_message": None,
                "retryable": False,
                "result": {**payload, "knowledge_refresh": graph_result},
            },
        )

    @staticmethod
    def _rebuild_knowledge_base(job: dict[str, Any]) -> None:
        from app.store import store

        job_id = str(job["id"])
        knowledge_base_id = str(job["knowledge_base_id"])
        documents = [
            document
            for document in reversed(store.list("documents"))
            if document.get("knowledge_base_id") == knowledge_base_id
            and document.get("status") != "archived"
        ]
        if not documents:
            store.update(
                "jobs",
                job_id,
                {
                    "status": "failed",
                    "stage": "planning",
                    "error_code": "knowledge_base_empty",
                    "error_message": "知识库中还没有可编译的 PDF",
                    "retryable": False,
                },
            )
            store.update(
                "knowledge_bases", knowledge_base_id, {"status": "draft"}
            )
            return

        pending = [
            document for document in documents if document.get("status") != "ready"
        ]
        child_job_ids: list[str] = []
        failed_document_ids: list[str] = []
        store.update(
            "jobs",
            job_id,
            {
                "status": "running",
                "stage": "planning",
                "progress": 2,
                "retryable": False,
            },
        )

        for index, document in enumerate(pending, start=1):
            child = store.create(
                "jobs",
                {
                    "kind": "compile_document",
                    "knowledge_base_id": knowledge_base_id,
                    "document_id": document["id"],
                    "status": "queued",
                    "stage": "waiting",
                    "progress": 0,
                    "error_code": None,
                    "error_message": None,
                    "retryable": False,
                    "attempt": 1,
                    "retry_of": None,
                    "superseded_by": None,
                    "result": {"parent_job_id": job_id},
                },
            )
            child_job_ids.append(str(child["id"]))
            store.update("documents", str(document["id"]), {"status": "queued"})

            # Mirror the child compile's granular stage/progress (rendering,
            # OCR page N/M, indexing, ...) onto the parent rebuild job so the
            # user sees exactly where the pipeline is instead of a black box.
            def report_child(stage: str, progress: int) -> None:
                base = 5 + round((index - 1) / max(len(pending), 1) * 70)
                composite = base + round(
                    progress / 100 * 70 / max(len(pending), 1)
                )
                try:
                    store.update(
                        "jobs",
                        job_id,
                        {
                            "status": "running",
                            "stage": f"编译 {index}/{len(pending)}：{stage}",
                            "progress": min(composite, 79),
                            "result": {
                                "child_job_ids": child_job_ids,
                                "failed_document_ids": failed_document_ids,
                            },
                        },
                    )
                except Exception:
                    pass

            try:
                store.compile_document_now(
                    str(document["id"]),
                    str(child["id"]),
                    progress_callback=report_child,
                )
            except Exception:
                current_child = store.get("jobs", str(child["id"])) or {}
                if current_child.get("status") != "failed":
                    LocalJobRunner._mark_pipeline_failed(
                        str(child["id"]),
                        code="pipeline_internal_error",
                        message="文档编译流水线发生内部错误",
                        retryable=False,
                    )
                failed_document_ids.append(str(document["id"]))
            store.update(
                "jobs",
                job_id,
                {
                    "status": "running",
                    "stage": "compiling_documents",
                    "progress": 5 + round(index / max(len(pending), 1) * 70),
                    "result": {
                        "child_job_ids": child_job_ids,
                        "failed_document_ids": failed_document_ids,
                    },
                },
            )

        ready_documents = [
            document
            for document in store.list("documents")
            if document.get("knowledge_base_id") == knowledge_base_id
            and document.get("status") == "ready"
        ]
        if not ready_documents:
            store.update(
                "jobs",
                job_id,
                {
                    "status": "failed",
                    "stage": "compiling_documents",
                    "progress": 100,
                    "error_code": "document_compilation_failed",
                    "error_message": "所有文档均未能完成编译",
                    "retryable": any(
                        bool((store.get("jobs", child_id) or {}).get("retryable"))
                        for child_id in child_job_ids
                    ),
                    "result": {
                        "child_job_ids": child_job_ids,
                        "failed_document_ids": failed_document_ids,
                    },
                },
            )
            store.update(
                "knowledge_bases", knowledge_base_id, {"status": "failed"}
            )
            return

        store.update(
            "jobs",
            job_id,
            {"status": "running", "stage": "graph_wiki", "progress": 82},
        )

        def report_graph_progress(percent: int, stage: str) -> None:
            try:
                store.update(
                    "jobs",
                    job_id,
                    {"status": "running", "stage": stage, "progress": percent},
                )
            except Exception:
                pass

        graph_result = asyncio.run(
            store.build_graph_and_wiki(
                knowledge_base_id, progress_callback=report_graph_progress
            )
        )
        final_status = "partial" if failed_document_ids else "succeeded"
        store.update(
            "knowledge_bases",
            knowledge_base_id,
            {
                "status": "partial" if failed_document_ids else "ready",
                "published_version": graph_result["wiki_version"],
            },
        )
        store.update(
            "jobs",
            job_id,
            {
                "status": final_status,
                "stage": "completed",
                "progress": 100,
                "error_code": (
                    "some_documents_failed" if failed_document_ids else None
                ),
                "error_message": (
                    f"{len(failed_document_ids)} 份文档编译失败，其余内容已发布"
                    if failed_document_ids
                    else None
                ),
                "retryable": bool(failed_document_ids),
                "result": {
                    "child_job_ids": child_job_ids,
                    "failed_document_ids": failed_document_ids,
                    "compiled_document_count": len(pending)
                    - len(failed_document_ids),
                    "knowledge_refresh": graph_result,
                },
            },
        )


job_runner = LocalJobRunner()
