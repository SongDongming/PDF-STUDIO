from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
import time
from threading import RLock
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.schemas import RagSettings
from app.services.assets import ObjectStorage, get_json, put_json, storage_from_environment
from app.services.state_persistence import ConcurrentStateUpdateError

from app.services.graph_repository import (
    EvidenceLineage,
    InMemoryGraphRepository,
    Neo4jGraphRepository,
)
from app.services.ingestion import (
    BoundingBox,
    CompilationError,
    DocumentCompiler,
    NormalizedElement as IngestionElement,
    PyMuPDFPageRenderer,
)
from app.services.indexing import DocumentIndexer
from app.services.memory import MemoryService
from app.services.ocr_client import PaddleOCRHttpClient
from app.services.repositories import CompilationRepository, DocumentSource, RepositoryError
from app.services.retrieval import HybridRetriever, RetrievalChunk, RetrievalHit
from app.services.wiki_graph import (
    NormalizedChunk,
    NormalizedElement as GraphNormalizedElement,
    WikiGraphCompiler,
    extraction_json_schema,
)
from app.services.wiki_repository import (
    InMemoryWikiRepository,
    WikiConclusion,
    WikiPageRecord,
)

logger = logging.getLogger("uvicorn.error")

_STATE_PERSIST_RETRIES = 5


def now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    """Development composition root plus thread-safe public repositories.

    Metadata remains in memory for the prototype, while uploaded source bytes
    and compilation artifacts are persisted through the real ObjectStorage
    adapter.  The service objects here are the same implementations used by
    production adapters, so route tests do not need fake answer or graph data.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # Serializes the knowledge-graph + wiki commit inside build_graph_and_wiki
        # so concurrent jobs (with APP_LOCAL_JOB_CONCURRENCY > 1) do not race on
        # graph/wiki versioned commits.
        self._graph_wiki_lock = RLock()
        self._persistence = None
        self._state_revision: int | None = None
        # Jobs whose execution budget was exceeded.  Their abandoned worker
        # threads may still be running, so writes that would flip them back to
        # running/succeeded are rejected (see ``update``).
        self._timed_out_jobs: set[str] = set()
        self.reset()
        runtime = get_settings()
        if runtime.env != "test" and os.getenv("APP_STATE_PERSISTENCE", "sql") != "off":
            from app.services.state_persistence import SqlAlchemyStatePersistence

            try:
                self._persistence = SqlAlchemyStatePersistence(
                    runtime.database_url, namespace="multimodal-pdf-workbench"
                )
            except Exception:
                # Durable state is best-effort at boot.  If the backing store is
                # down, run ephemeral in-memory and keep serving (health reports
                # the degraded mode) instead of crashing the whole API.
                self._persistence = None
                logger.exception(
                    "state persistence unavailable; running ephemeral in-memory"
                )
            self._restore_from_persistence()

    def _restore_from_persistence(self) -> None:
        if self._persistence is None:
            return
        try:
            snapshot = self._persistence.load()
        except Exception:
            logger.exception("could not load persisted state; starting empty")
            return
        if snapshot is None:
            return
        for field, value in snapshot.state.items():
            setattr(self, field, value)
        self._normalize_legacy_provider_labels()
        self._state_revision = snapshot.revision
        self._reconcile_interrupted_sync_jobs()
        self._reconcile_document_statistics()
        self._restore_wiki_repository()
        self._restore_runtime_indexes()
        self.persist_state()

    def reset(self) -> None:
        with self._lock:
            self._timed_out_jobs.clear()
            self.knowledge_bases: dict[str, dict[str, Any]] = {}
            self.documents: dict[str, dict[str, Any]] = {}
            self.jobs: dict[str, dict[str, Any]] = {}
            self.threads: dict[str, dict[str, Any]] = {}
            self.messages: dict[str, list[dict[str, Any]]] = {}
            self.wiki_pages: dict[str, dict[str, Any]] = {}
            self.settings = self._default_settings()
            self.storage = storage_from_environment(get_settings())
            self.retriever = HybridRetriever()
            self.memory = MemoryService()
            self.graph_repository = self._graph_repository()
            self.wiki_repository = InMemoryWikiRepository()
            self.wiki_graph_compiler = WikiGraphCompiler(
                self.graph_repository, self.wiki_repository
            )
            self.asset_keys: dict[str, str] = {}

    def _graph_repository(self):
        runtime = get_settings()
        if runtime.env == "test" or runtime.neo4j_password is None:
            return InMemoryGraphRepository()
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            runtime.neo4j_uri,
            auth=(runtime.neo4j_user, runtime.neo4j_password.get_secret_value()),
            connection_timeout=5.0,
            connection_acquisition_timeout=10.0,
            max_connection_lifetime=3600,
        )
        return Neo4jGraphRepository(driver)

    @staticmethod
    def _lineage_payload(lineage: EvidenceLineage) -> dict[str, Any]:
        return {
            "document_id": lineage.document_id,
            "page": lineage.page,
            "bbox": list(lineage.bbox),
            "chunk_id": lineage.chunk_id,
            "element_id": lineage.element_id,
        }

    @staticmethod
    def _lineage_from_payload(payload: dict[str, Any]) -> EvidenceLineage:
        return EvidenceLineage(
            document_id=str(payload["document_id"]),
            page=int(payload["page"]),
            bbox=tuple(float(value) for value in payload["bbox"]),
            chunk_id=str(payload["chunk_id"]),
            element_id=(
                str(payload["element_id"])
                if payload.get("element_id") is not None
                else None
            ),
        )

    def _snapshot_wiki_repository(self, knowledge_base_id: str) -> None:
        pages = self.wiki_repository.list_pages(knowledge_base_id)
        self.wiki_pages[knowledge_base_id] = {
            "version": self.wiki_repository.current_version(knowledge_base_id),
            "pages": {
                page.id: {
                    "id": page.id,
                    "knowledge_base_id": page.knowledge_base_id,
                    "entity_id": page.entity_id,
                    "slug": page.slug,
                    "title": page.title,
                    "summary": page.summary,
                    "sections": deepcopy(page.sections),
                    "conclusions": [
                        {
                            "id": conclusion.id,
                            "text": conclusion.text,
                            "kind": conclusion.kind,
                            "evidence": [
                                self._lineage_payload(item)
                                for item in conclusion.evidence
                            ],
                            "predicate": conclusion.predicate,
                            "related_entity_id": conclusion.related_entity_id,
                        }
                        for conclusion in page.conclusions
                    ],
                    "related_page_ids": list(page.related_page_ids),
                    "graph_version": page.graph_version,
                    "revision": page.revision,
                    "status": page.status,
                    "locked_fields": sorted(page.locked_fields),
                    "updated_at": page.updated_at,
                }
                for page in pages
            },
        }

    def _restore_wiki_repository(self) -> None:
        repository = InMemoryWikiRepository()
        for knowledge_base_id, raw_bundle in self.wiki_pages.items():
            raw_pages = raw_bundle.get("pages", {})
            pages: dict[str, WikiPageRecord] = {}
            for page_id, payload in raw_pages.items():
                conclusions = tuple(
                    WikiConclusion(
                        id=str(item["id"]),
                        text=str(item["text"]),
                        kind=str(item["kind"]),
                        evidence=tuple(
                            self._lineage_from_payload(lineage)
                            for lineage in item.get("evidence", [])
                        ),
                        predicate=(
                            str(item["predicate"])
                            if item.get("predicate") is not None
                            else None
                        ),
                        related_entity_id=(
                            str(item["related_entity_id"])
                            if item.get("related_entity_id") is not None
                            else None
                        ),
                    )
                    for item in payload.get("conclusions", [])
                )
                pages[str(page_id)] = WikiPageRecord(
                    id=str(payload["id"]),
                    knowledge_base_id=str(payload["knowledge_base_id"]),
                    entity_id=str(payload["entity_id"]),
                    slug=str(payload["slug"]),
                    title=str(payload["title"]),
                    summary=str(payload.get("summary", "")),
                    sections={
                        str(key): str(value)
                        for key, value in payload.get("sections", {}).items()
                    },
                    conclusions=conclusions,
                    related_page_ids=tuple(
                        str(value) for value in payload.get("related_page_ids", [])
                    ),
                    graph_version=int(payload["graph_version"]),
                    revision=int(payload["revision"]),
                    status=str(payload["status"]),
                    locked_fields=frozenset(
                        str(value) for value in payload.get("locked_fields", [])
                    ),
                    updated_at=payload["updated_at"],
                )
            repository._pages[str(knowledge_base_id)] = pages
            repository._versions[str(knowledge_base_id)] = int(
                raw_bundle.get("version", 0)
            )
            for page in pages.values():
                repository._history[(str(knowledge_base_id), page.id)] = [
                    deepcopy(page)
                ]
        self.wiki_repository = repository
        self.wiki_graph_compiler = WikiGraphCompiler(
            self.graph_repository, self.wiki_repository
        )

    def _restore_runtime_indexes(self) -> None:
        """Rehydrate published retrieval indexes and immutable answer assets.

        Retrieval should use only the newest successful compilation of each
        document.  Rich answers, however, may persist references to assets from
        any earlier compilation, so every successful compiler manifest must be
        restored into ``asset_keys``.
        """

        latest_by_document: dict[str, dict[str, Any]] = {}
        successful_jobs: list[dict[str, Any]] = []
        for job in self.jobs.values():
            document_id = str(job.get("document_id") or "")
            if (
                not document_id
                or job.get("status") != "succeeded"
                or not isinstance(job.get("result"), dict)
            ):
                continue
            successful_jobs.append(job)
            previous = latest_by_document.get(document_id)
            if previous is None or (
                job.get("updated_at") or job.get("created_at")
            ) > (previous.get("updated_at") or previous.get("created_at")):
                latest_by_document[document_id] = job

        for job in sorted(
            successful_jobs,
            key=lambda item: item.get("updated_at") or item.get("created_at"),
        ):
            compiler_manifest_key = job["result"].get("manifest_key")
            if not compiler_manifest_key:
                continue
            try:
                compiler_manifest = get_json(
                    self.storage, str(compiler_manifest_key)
                )
            except Exception:
                continue
            for asset in compiler_manifest.get("assets", []):
                if asset.get("id") and asset.get("object_key"):
                    self.asset_keys[str(asset["id"])] = str(
                        asset["object_key"]
                    )

        for job in latest_by_document.values():
            result = job["result"]
            key = result.get("index_manifest_key")
            if not key:
                continue
            # Skip stale indexes for documents that no longer exist (deleted or
            # recompiled under a new id).  Loading them would make the hybrid
            # retriever return old copies and answers carry duplicate citations,
            # half of which cannot resolve to a region.
            document_id = job.get("document_id")
            if document_id and document_id not in self.documents:
                continue
            try:
                embedder = self._embedding_provider()
                loaded = DocumentIndexer(
                    storage=self.storage, embedder=embedder
                ).load_retriever(
                    str(key),
                    embedder=(
                        embedder if result.get("index_mode") == "hybrid" else None
                    ),
                )
                self.retriever.upsert(loaded._chunks.values())
                if result.get("index_mode") == "hybrid":
                    self.retriever.embedder = embedder
            except Exception:
                # Persisted metadata remains visible; a corrupt/missing index
                # is surfaced by retrieval instead of fabricating evidence.
                continue

    def _reconcile_interrupted_sync_jobs(self) -> None:
        """Close jobs that cannot still be running after an API restart.

        In the local synchronous delivery mode the request process owns the
        whole compile.  A persisted ``queued`` or ``running`` job therefore
        represents an interrupted request, not work that a separate worker can
        resume.  Recording that explicitly keeps the task center truthful and
        lets the user retry from the document card.
        """

        if os.getenv("APP_COMPILE_MODE", "deferred").lower() != "sync":
            return
        timestamp = now()
        for job in self.jobs.values():
            if job.get("status") not in {"queued", "running"}:
                continue
            job.update(
                {
                    "status": "failed",
                    "stage": "interrupted",
                    "error_code": "process_restarted",
                    "error_message": "服务重启中断了同步任务，请重试",
                    "retryable": True,
                    "updated_at": timestamp,
                }
            )
            document_id = job.get("document_id")
            document = self.documents.get(str(document_id))
            if document is not None and document.get("status") in {
                "queued",
                "compiling",
            }:
                document["status"] = "failed"
                document["updated_at"] = timestamp

    def _reconcile_document_statistics(self) -> None:
        """Backfill public document counters from the published job manifest."""

        for document_id, document in self.documents.items():
            successful = [
                job
                for job in self.jobs.values()
                if job.get("document_id") == document_id
                and job.get("status") == "succeeded"
                and isinstance(job.get("result"), dict)
            ]
            if not successful:
                continue
            latest = max(
                successful,
                key=lambda item: item.get("updated_at") or item.get("created_at"),
            )
            result = latest["result"]
            for field in ("page_count", "element_count", "asset_count", "chunk_count"):
                value = int(result.get(field) or 0)
                if value:
                    document[field] = value

    def persist_state(self) -> None:
        if self._persistence is None:
            return
        # Route handlers may mutate different collections concurrently.  The
        # persisted snapshot and its compare-and-swap revision are one logical
        # critical section; serializing only the repository adapter still lets
        # two callers read the same in-memory revision and race.  When the CAS
        # revision does collide (e.g. two local job threads), re-read the latest
        # persisted revision and retry with bounded backoff instead of failing
        # the whole write.
        with self._lock:
            for attempt in range(_STATE_PERSIST_RETRIES):
                try:
                    snapshot = self._persistence.save(
                        self, expected_revision=self._state_revision
                    )
                    self._state_revision = snapshot.revision
                    return
                except ConcurrentStateUpdateError:
                    if attempt == _STATE_PERSIST_RETRIES - 1:
                        raise
                    latest = self._persistence.load()
                    if latest is not None:
                        self._state_revision = latest.revision
                    time.sleep(0.05 * (attempt + 1))

    def _normalize_legacy_provider_labels(self) -> None:
        """One-time relabel of persisted provider entries after the Kimi K3 ->
        DeepSeek rename, so a running install does not show mixed old names."""
        providers = self.settings.get("providers")
        if not isinstance(providers, list):
            return
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            if entry.get("provider") == "moonshot":
                entry["provider"] = "deepseek"
            if entry.get("model") == "kimi-k3":
                entry["model"] = "deepseek-v4-flash"

    def _default_settings(self) -> dict[str, Any]:
        return {
            "providers": [
                {
                    "id": "embedding-primary",
                    "category": "embedding",
                    "provider": "aliyun-bailian",
                    "model": "text-embedding-v4",
                    "enabled": True,
                    "configured": False,
                    "health": "unknown",
                    "credential_ref": None,
                    "detail": "索引维度固定为 1024；模型切换需要重建索引",
                },
                {
                    "id": "embedding-openai",
                    "category": "embedding",
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "enabled": False,
                    "configured": False,
                    "health": "disabled",
                    "credential_ref": None,
                    "detail": "当前仅保留配置槽位",
                },
                {
                    "id": "ocr-primary",
                    "category": "ocr",
                    "provider": "paddleocr",
                    "model": "PaddleOCR-VL-1.6",
                    "enabled": True,
                    "configured": False,
                    "health": "unknown",
                    "credential_ref": None,
                    "detail": "Dspark 服务地址由后端受保护环境配置",
                },
                {
                    "id": "vision-chat",
                    "category": "vision",
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "enabled": True,
                    "configured": False,
                    "health": "unknown",
                    "credential_ref": None,
                    "detail": "凭证引用只在服务端解析",
                },
                {
                    "id": "answer-primary",
                    "category": "chat",
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "enabled": True,
                    "configured": False,
                    "health": "unknown",
                    "credential_ref": None,
                    "detail": "最终问答与多模态语义理解",
                },
            ],
            "rag": {
                "dense_top_k": 12,
                "lexical_top_k": 12,
                "rerank_top_k": 8,
                "graph_hops": 2,
                "max_tool_calls": 8,
                "citation_required": True,
            },
            "compiler": {
                "render_dpi": 144,
                "preserve_original": True,
                "require_bbox": True,
                "publish_on_partial_failure": False,
            },
            "agent_framework": {
                "name": "deepagents",
                "mode": "bounded_grounding_validator",
                "available": False,
                "code": "not_probed",
                "versions": {},
                "detail": "运行状态由后端动态检测",
            },
            "updated_at": now(),
        }

    def create(self, collection: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            timestamp = now()
            item = {
                "id": str(uuid4()),
                "created_at": timestamp,
                "updated_at": timestamp,
                **deepcopy(data),
            }
            getattr(self, collection)[item["id"]] = item
            result = deepcopy(item)
        self.persist_state()
        return result

    def get(self, collection: str, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = getattr(self, collection).get(item_id)
            return deepcopy(item) if item else None

    def list(self, collection: str) -> list[dict[str, Any]]:
        with self._lock:
            items = getattr(self, collection).values()
            return sorted(
                (deepcopy(item) for item in items),
                key=lambda item: item["created_at"],
                reverse=True,
            )

    def mark_job_timed_out(self, job_id: str) -> None:
        """Record that a job exceeded its execution budget.

        The abandoned worker thread may still be running; ``update`` uses this
        to reject writes that would flip the job back to running/succeeded.
        """
        with self._lock:
            self._timed_out_jobs.add(job_id)

    def update(
        self, collection: str, item_id: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        with self._lock:
            item = getattr(self, collection).get(item_id)
            if item is None:
                return None
            if (
                collection == "jobs"
                and item_id in self._timed_out_jobs
                and changes.get("status") in {"running", "succeeded", "partial"}
            ):
                # A timed-out job's abandoned worker thread must not be able to
                # resurrect it after the runner already recorded the failure.
                return deepcopy(item)
            item.update(deepcopy(changes))
            item["updated_at"] = now()
            result = deepcopy(item)
        self.persist_state()
        return result

    def delete(self, collection: str, item_id: str) -> bool:
        with self._lock:
            removed = getattr(self, collection).pop(item_id, None) is not None
        if removed:
            self.persist_state()
        return removed

    def remove_document_cascade(self, document_id: str) -> bool:
        """Delete a document and cascade through the full pipeline.

        Drops the document's retrieval chunks, removes its knowledge-graph
        contribution (Neo4j nodes/edges), regenerates the LLM Wiki from the
        updated graph (orphaned pages disappear), purges compiled artifacts from
        object storage, and finally deletes the record.  The record delete is
        authoritative; downstream cleanup is best-effort so a storage/back-end
        hiccup never leaves an orphaned document in the workspace.
        """
        document = self.get("documents", document_id)
        if document is None:
            return False
        knowledge_base_id = str(document["knowledge_base_id"])
        try:
            self.retriever.remove_document(document_id)
        except Exception:
            pass
        try:
            self.graph_repository.remove_document(knowledge_base_id, document_id)
            self.wiki_graph_compiler.rebuild_wiki(knowledge_base_id)
            self._snapshot_wiki_repository(knowledge_base_id)
        except Exception:
            pass
        if int(document.get("active_version") or 0) >= 1:
            try:
                self.storage.delete_prefix(self.compiled_prefix(document))
            except Exception:
                pass
        self.delete("documents", document_id)
        return True

    def remove_knowledge_base_cascade(self, knowledge_base_id: str) -> bool:
        """Delete a knowledge base and cascade through the full pipeline.

        Removes every document's retrieval chunks and compiled artifacts,
        drops the whole knowledge graph (Neo4j), clears the LLM Wiki, removes
        the KB's jobs, and finally deletes the records.
        """
        knowledge_base = self.get("knowledge_bases", knowledge_base_id)
        if knowledge_base is None:
            return False
        for document in [
            item
            for item in self.list("documents")
            if item["knowledge_base_id"] == knowledge_base_id
        ]:
            try:
                self.retriever.remove_document(document["id"])
            except Exception:
                pass
            if int(document.get("active_version") or 0) >= 1:
                try:
                    self.storage.delete_prefix(self.compiled_prefix(document))
                except Exception:
                    pass
            self.delete("documents", document["id"])
        for job_id in [
            job_id
            for job_id, job in self.jobs.items()
            if job.get("knowledge_base_id") == knowledge_base_id
        ]:
            self.delete("jobs", job_id)
        try:
            self.graph_repository.remove_knowledge_base(knowledge_base_id)
        except Exception:
            pass
        try:
            self.wiki_repository.remove_knowledge_base(knowledge_base_id)
        except Exception:
            pass
        self.wiki_pages.pop(knowledge_base_id, None)
        self.delete("knowledge_bases", knowledge_base_id)
        return True

    def reconcile_graph_and_wiki(self, knowledge_base_id: str) -> dict[str, Any]:
        """Purge graph contributions from documents that no longer exist and
        rebuild the LLM Wiki from the cleaned graph.

        Deletes performed before the cascade logic was in place can leave stale
        graph edges and wiki evidence pointing at removed documents.  This
        reconciles the live graph/wiki against the current document set.
        """
        valid_document_ids = {
            item["id"]
            for item in self.list("documents")
            if item.get("knowledge_base_id") == knowledge_base_id
        }
        graph = self.graph_repository.snapshot(knowledge_base_id)
        stale_document_ids: set[str] = set()
        for edge in graph.edges:
            for lineage in edge.evidence:
                if lineage.document_id not in valid_document_ids:
                    stale_document_ids.add(lineage.document_id)
        # Also purge retrieval chunks belonging to documents that no longer
        # exist (e.g. a document recompiled under a new id).  Without this, the
        # hybrid retriever keeps returning the old copy and answers carry
        # duplicate citations, half of which cannot resolve to a region.
        for chunk_document_id in self.retriever.document_ids():
            if chunk_document_id not in valid_document_ids:
                try:
                    self.retriever.remove_document(chunk_document_id)
                except Exception:
                    pass
                stale_document_ids.add(chunk_document_id)
        for document_id in stale_document_ids:
            try:
                self.graph_repository.remove_document(
                    knowledge_base_id, document_id
                )
            except Exception:
                pass
        try:
            self.wiki_graph_compiler.rebuild_wiki(knowledge_base_id)
            self._snapshot_wiki_repository(knowledge_base_id)
        except Exception:
            pass
        self.persist_state()
        return {"purged_documents": sorted(stale_document_ids)}

    def compilation_repository(self) -> "StoreCompilationRepository":
        return StoreCompilationRepository(self)

    def compiled_prefix(self, document: dict[str, Any]) -> str:
        version = int(document.get("active_version") or 0)
        if version < 1:
            raise RepositoryError("document has no compiled version")
        return f"compiled/{document['id']}/v{version}"

    def compiled_payloads(
        self, document: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        prefix = self.compiled_prefix(document)
        manifest = get_json(self.storage, f"{prefix}/manifest.json")
        elements = get_json(self.storage, str(manifest["elements_key"]))
        chunks = get_json(self.storage, str(manifest["chunks_key"]))
        return manifest, elements, chunks

    @staticmethod
    def _embedding_provider():
        from app.services.embeddings import (
            BailianEmbeddingProvider,
            OpenAIEmbeddingProvider,
        )

        preference = os.getenv("APP_EMBEDDING_PROVIDER", "").strip().lower()
        if preference == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddingProvider.from_env()
        if os.getenv("DASHSCOPE_API_KEY"):
            return BailianEmbeddingProvider.from_env()
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddingProvider.from_env()
        return None

    def _run_enrichment(
        self, *, document: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        import asyncio
        import hashlib

        from app.services.enrichment import (
            ENRICHMENT_VERSION,
            PAGE_ENRICHMENT_VERSION,
            SEMANTIC_ARTIFACT_VERSION,
            DeepSeekElementEnricher,
            semantic_artifact_key,
        )
        from app.services.providers import DeepSeekProvider

        manifest, elements_payload, _ = self.compiled_payloads(document)
        eligible = int(manifest.get("asset_count") or 0)
        page_count = int(manifest.get("page_count") or 0)
        if not os.getenv("MOONSHOT_API_KEY"):
            return {
                "fully_enriched": False,
                "enriched_count": 0,
                "page_enriched_count": 0,
                "pending_enrichment": eligible + page_count,
                "enrichment_mode": "deepseek_not_configured",
            }
        normalized: list[IngestionElement] = []
        for raw in elements_payload.get("elements", []):
            bbox = BoundingBox(
                **{key: float(value) for key, value in raw["bbox"].items()}
            )
            normalized.append(
                IngestionElement(
                    id=str(raw["id"]),
                    page=int(raw["page"]),
                    order=int(raw.get("order") or 1),
                    kind=str(raw["kind"]),
                    label=str(raw.get("label") or raw["kind"]),
                    content=str(raw.get("content") or ""),
                    bbox=bbox,
                    bbox_normalized=tuple(
                        float(value) for value in raw["bbox_normalized"]
                    ),
                    polygon=[
                        tuple(float(value) for value in point)
                        for point in raw.get("polygon", [])
                    ],
                    polygon_normalized=[
                        tuple(float(value) for value in point)
                        for point in raw.get("polygon_normalized", [])
                    ],
                    confidence=(
                        float(raw["confidence"])
                        if raw.get("confidence") is not None
                        else None
                    ),
                    asset_id=(
                        str(raw["asset_id"]) if raw.get("asset_id") else None
                    ),
                    asset_key=(
                        str(raw["asset_key"]) if raw.get("asset_key") else None
                    ),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
        async def enrich_all():
            enricher = DeepSeekElementEnricher(
                storage=self.storage, provider=DeepSeekProvider.from_env()
            )
            return await asyncio.gather(
                enricher.enrich_elements(
                    document_id=str(document["id"]),
                    document_version=int(result["version"]),
                    elements=normalized,
                ),
                enricher.enrich_pages(
                    document_id=str(document["id"]),
                    document_version=int(result["version"]),
                    pages=list(elements_payload.get("pages") or []),
                    elements=normalized,
                ),
            )

        try:
            batch, page_batch = asyncio.run(enrich_all())
        except Exception:
            # Enrichment is best-effort: a provider/LLM failure degrades to
            # "pending" rather than failing the whole document compilation.
            from app.services.enrichment import (
                EnrichmentBatchResult,
                PageEnrichmentBatchResult,
            )

            batch = EnrichmentBatchResult(
                document_id=str(document["id"]),
                document_version=int(result["version"]),
                eligible_count=0,
                enriched_count=0,
                pending_count=0,
                fully_enriched=False,
                elements=[],
            )
            page_batch = PageEnrichmentBatchResult(
                document_id=str(document["id"]),
                document_version=int(result["version"]),
                eligible_count=0,
                enriched_count=0,
                pending_count=0,
                fully_enriched=False,
                pages=[],
            )
        artifact = {
            "schema_version": 2,
            "semantic_artifact_version": SEMANTIC_ARTIFACT_VERSION,
            "document_id": str(document["id"]),
            "document_version": int(result["version"]),
            "source_sha256": str(manifest["source_sha256"]),
            "trusted_layout_contract": (
                manifest.get("trusted_layout", {}).get("contract")
                or "paddle-v3-layout-lineage-v1"
            ),
            "page_enrichment_version": PAGE_ENRICHMENT_VERSION,
            "element_enrichment_version": ENRICHMENT_VERSION,
            "pages": [item.to_dict() for item in page_batch.pages],
            "elements": [item.to_dict() for item in batch.elements],
        }
        encoded = json.dumps(
            artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        artifact["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
        artifact_key = semantic_artifact_key(
            str(document["id"]), int(result["version"])
        )
        put_json(self.storage, artifact_key, artifact)
        return {
            "fully_enriched": batch.fully_enriched and page_batch.fully_enriched,
            "enriched_count": batch.enriched_count,
            "page_enriched_count": page_batch.enriched_count,
            "pending_enrichment": batch.pending_count + page_batch.pending_count,
            "enrichment_mode": "paddle-v3+deepseek-vision",
            "semantic_artifact_key": artifact_key,
            "semantic_artifact_sha256": artifact["artifact_sha256"],
        }

    def compile_document_now(
        self,
        document_id: str,
        job_id: str,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        if not settings.ocr_base_url:
            self.compilation_repository().mark_failed(
                document_id,
                stage="ocr",
                code="ocr_not_configured",
                message="PaddleOCR 服务地址未配置，编译未执行",
                retryable=False,
                job_id=job_id,
            )
            raise CompilationError(
                "PaddleOCR 服务地址未配置，编译未执行",
                stage="ocr",
                code="ocr_not_configured",
            )
        document = self.get("documents", document_id)
        if document is None:
            raise RepositoryError(f"document {document_id!r} does not exist")
        version = int(document.get("active_version") or 0) + 1
        ocr = PaddleOCRHttpClient(
            settings.ocr_base_url,
            endpoint=os.getenv("APP_OCR_PARSE_ENDPOINT", "/v1/layout-parsing"),
            health_endpoint=os.getenv("APP_OCR_HEALTH_ENDPOINT", "/health"),
            timeout_seconds=float(os.getenv("APP_OCR_TIMEOUT_SECONDS", "180")),
        )
        compiler = DocumentCompiler(
            storage=self.storage,
            renderer=PyMuPDFPageRenderer(
                dpi=int(os.getenv("APP_COMPILER_RENDER_DPI", "144"))
            ),
            ocr=ocr,
            repository=self.compilation_repository(),
            render_dpi=int(os.getenv("APP_COMPILER_RENDER_DPI", "144")),
            chunk_target_characters=int(
                os.getenv("APP_COMPILER_CHUNK_CHARACTERS", "1800")
            ),
        )
        try:
            result = compiler.compile(
                document_id,
                version,
                job_id=job_id,
                progress_callback=progress_callback,
            )
            compiled = result.to_dict()
            self.update(
                "jobs",
                job_id,
                {
                    "status": "running",
                    "stage": "multimodal_enrichment",
                    "progress": 76,
                },
            )
            current_document = self.get("documents", document_id) or document
            try:
                enrichment = self._run_enrichment(
                    document=current_document, result=compiled
                )
            except Exception:
                manifest, _, _ = self.compiled_payloads(current_document)
                enrichment = {
                    "fully_enriched": False,
                    "enriched_count": 0,
                    "pending_enrichment": int(manifest.get("asset_count") or 0),
                    "enrichment_mode": "provider_unavailable",
                }

            embedder = self._embedding_provider()
            self.update(
                "jobs",
                job_id,
                {"status": "running", "stage": "embedding_index", "progress": 88},
            )
            indexer = DocumentIndexer(storage=self.storage, embedder=embedder)
            index_result = __import__("asyncio").run(
                indexer.index(
                    result.manifest_key,
                    document_title=str(document["title"]),
                )
            )
            loaded = indexer.load_retriever(
                index_result.manifest_key,
                embedder=embedder if index_result.mode == "hybrid" else None,
            )
            # Merge the verified payload into the process-wide KB retriever.
            # HybridRetriever intentionally keeps its storage private; this is
            # the local composition root and performs only a read/copy.
            self.retriever.upsert(loaded._chunks.values())
            if index_result.mode == "hybrid":
                self.retriever.embedder = embedder
            indexed = len(loaded._chunks)
            manifest, _, _ = self.compiled_payloads(current_document)
            for asset in manifest.get("assets", []):
                if asset.get("id") and asset.get("object_key"):
                    self.asset_keys[str(asset["id"])] = str(asset["object_key"])
            payload = {
                **compiled,
                **enrichment,
                "retrieval_chunk_count": indexed,
                "index_manifest_key": index_result.manifest_key,
                "index_mode": index_result.mode,
                "index_degraded_reason": index_result.degraded_reason,
            }
            self.update(
                "jobs",
                job_id,
                {
                    "status": "succeeded",
                    "stage": "completed",
                    "progress": 100,
                    "result": payload,
                },
            )
            return payload
        finally:
            ocr.close()

    async def retrieve(
        self,
        *,
        query: str,
        knowledge_base_id: str,
        rag: RagSettings | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalHit]:
        hits = await self.retriever.retrieve(
            query,
            knowledge_base_id=knowledge_base_id,
            settings=rag,
        )
        if document_ids:
            allowed = set(document_ids)
            hits = [hit for hit in hits if hit.chunk.document_id in allowed]
        return hits

    def normalized_document(
        self, document: dict[str, Any]
    ) -> tuple[list[NormalizedChunk], list[GraphNormalizedElement], dict[str, Any]]:
        manifest, elements_payload, chunks_payload = self.compiled_payloads(document)
        raw_elements = {
            str(item["id"]): item for item in elements_payload.get("elements", [])
        }
        chunks: list[NormalizedChunk] = []
        elements: list[GraphNormalizedElement] = []
        for item in chunks_payload.get("chunks", []):
            element_ids = tuple(
                str(value)
                for value in item.get("element_ids", [])
                if str(value) in raw_elements
            )
            element_items = [raw_elements[value] for value in element_ids]
            if element_items:
                boxes = [entry["bbox_normalized"] for entry in element_items]
                bbox = (
                    max(0.0, min(float(box[0]) for box in boxes)),
                    max(0.0, min(float(box[1]) for box in boxes)),
                    min(1.0, max(float(box[2]) for box in boxes)),
                    min(1.0, max(float(box[3]) for box in boxes)),
                )
            else:
                bbox = (0.000001, 0.000001, 0.999999, 0.999999)
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                bbox = (0.000001, 0.000001, 0.999999, 0.999999)
            chunk_id = str(item["id"])
            chunks.append(
                NormalizedChunk(
                    id=chunk_id,
                    document_id=str(document["id"]),
                    page=int(item.get("page_start") or 1),
                    bbox=bbox,
                    text=str(item.get("markdown") or "未识别内容"),
                    element_ids=element_ids,
                )
            )
            for raw in element_items:
                text = str(raw.get("content") or raw.get("label") or "").strip()
                if not text:
                    text = f"{raw.get('kind', 'element')} 元素"
                raw_bbox = tuple(
                    float(value) for value in raw.get("bbox_normalized", bbox)
                )
                if (
                    len(raw_bbox) != 4
                    or raw_bbox[0] >= raw_bbox[2]
                    or raw_bbox[1] >= raw_bbox[3]
                ):
                    raw_bbox = bbox
                elements.append(
                    GraphNormalizedElement(
                        id=str(raw["id"]),
                        document_id=str(document["id"]),
                        chunk_id=chunk_id,
                        page=int(raw.get("page") or item.get("page_start") or 1),
                        bbox=raw_bbox,
                        kind=str(raw.get("kind") or "text"),
                        text=text,
                    )
                )
        return chunks, elements, manifest

    async def build_graph_and_wiki(
        self,
        knowledge_base_id: str,
        document_ids: list[str] | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, int]:
        from app.services.graph_extraction import merge_graph_extractions
        from app.services.providers import structured_provider

        # Graph/Wiki extraction is a strict structured task without vision.
        # Route it through the structured model (glm-4-flash) rather than the
        # vision chat model, which may not follow the JSON schema reliably.
        provider = structured_provider()
        candidates = [
            item
            for item in self.list("documents")
            if item["knowledge_base_id"] == knowledge_base_id
            and item.get("status") == "ready"
            and (not document_ids or item["id"] in set(document_ids))
        ]
        if not candidates:
            raise ValueError("知识库中没有已编译、可构建图谱的文档")
        result = None
        extracted_any = False
        extraction_targets = [
            document
            for document in candidates
            if not self.graph_repository.has_document_contribution(
                knowledge_base_id, str(document["id"])
            )
        ]
        # Count LLM batches up front so progress can advance per batch instead
        # of appearing frozen at 82% for a whole large document.
        total_batches = 0
        for document in extraction_targets:
            try:
                chunks, _, _ = self.normalized_document(document)
                total_batches += (len(chunks) + 7) // 8
            except Exception:
                pass
        processed_batches = 0
        processed_targets = 0
        for document in reversed(candidates):
            # Skip LLM re-extraction for documents whose graph contribution
            # already exists; only newly compiled documents need extraction.
            # The wiki is still regenerated from the merged graph afterwards.
            # This keeps rebuilds after a new upload fast instead of re-running
            # the full LLM extraction for every ready document.
            if self.graph_repository.has_document_contribution(
                knowledge_base_id, str(document["id"])
            ):
                continue
            extracted_any = True
            chunks, elements, _ = self.normalized_document(document)
            if not chunks:
                continue
            extractions: list[dict[str, Any]] = []
            for start in range(0, len(chunks), 8):
                batch_chunks = chunks[start : start + 8]
                chunk_ids = {item.id for item in batch_chunks}
                batch_elements = [
                    item for item in elements if item.chunk_id in chunk_ids
                ][:64]
                source = {
                    "document_id": document["id"],
                    "title": document["title"],
                    "batch": {
                        "number": start // 8 + 1,
                        "total": (len(chunks) + 7) // 8,
                    },
                    "chunks": [
                        {
                            "id": item.id,
                            "page": item.page,
                            "text": item.text[:1600],
                            "element_ids": list(item.element_ids),
                        }
                        for item in batch_chunks
                    ],
                    "elements": [
                        {
                            "id": item.id,
                            "chunk_id": item.chunk_id,
                            "page": item.page,
                            "kind": item.kind,
                            "text": item.text[:400],
                        }
                        for item in batch_elements
                    ],
                }
                raw = await provider.complete_structured(
                    system_prompt=(
                        "你是 PDF 知识抽取器。只从当前批次 chunks/elements "
                        "提取实体、主张与关系。所有 evidence_refs 必须逐字使用"
                        "输入中的 id，且每个实体、主张和关系都至少引用一条真实"
                        "证据 id（evidence_refs 不得为空）；document_id 必须逐字"
                        "返回输入中的 document_id；kind/predicate 使用大写英文"
                        "下划线标识，禁止补造证据。当前批次最多返回 10 个实体、"
                        "12 条主张和 12 条关系；宁缺毋滥。"
                    ),
                    user_text=json.dumps(source, ensure_ascii=False),
                    schema_name="grounded_pdf_graph_extraction",
                    schema=extraction_json_schema(),
                    reasoning_effort="low",
                )
                # Lineage is fixed caller-side: this extraction was requested
                # for exactly this document, so a mis-echoed id is corrected.
                raw["document_id"] = str(document["id"])
                # Free chat models occasionally invent evidence ids.  Sanitize
                # evidence to the ids actually present in this batch and drop
                # items without any verifiable evidence.
                valid_ids = {item.id for item in batch_chunks} | {
                    item.id for item in batch_elements
                }
                for group in ("entities", "claims", "relations"):
                    kept: list[dict[str, Any]] = []
                    for item in raw.get(group, []):
                        if not isinstance(item, dict):
                            continue
                        refs = item.get("evidence_refs")
                        if not isinstance(refs, list):
                            continue
                        filtered = [r for r in refs if str(r) in valid_ids]
                        if not filtered:
                            continue
                        item["evidence_refs"] = filtered
                        kept.append(item)
                    raw[group] = kept
                extractions.append(raw)
                processed_batches += 1
                if progress_callback is not None and total_batches:
                    percent = 82 + round(
                        processed_batches / total_batches * 18
                    )
                    progress_callback(min(percent, 100), "graph_wiki")
            extraction = merge_graph_extractions(
                str(document["id"]), extractions
            )
            # Serialize the graph/wiki commit and re-read versions inside the
            # lock so concurrent jobs cannot race on the versioned commits.
            with self._graph_wiki_lock:
                expected_graph_version = self.graph_repository.snapshot(
                    knowledge_base_id
                ).version
                expected_wiki_version = self.wiki_repository.current_version(
                    knowledge_base_id
                )
                result = self.wiki_graph_compiler.compile_document(
                    knowledge_base_id=knowledge_base_id,
                    document_id=str(document["id"]),
                    document_title=str(document["title"]),
                    chunks=chunks,
                    elements=elements,
                    extraction=extraction,
                    expected_graph_version=expected_graph_version,
                    expected_wiki_version=expected_wiki_version,
                )
            processed_targets += 1
            if progress_callback is not None and extraction_targets:
                percent = 82 + round(
                    processed_targets / len(extraction_targets) * 18
                )
                progress_callback(min(percent, 100), "graph_wiki")
        if not extracted_any:
            # Every candidate already has a graph contribution: still regenerate
            # the LLM Wiki from the current merged graph so orphaned pages are
            # dropped and the wiki stays in sync with the live graph.
            self.wiki_graph_compiler.rebuild_wiki(knowledge_base_id)
            self._snapshot_wiki_repository(knowledge_base_id)
            self.persist_state()
            graph = self.graph_repository.snapshot(knowledge_base_id)
            return {
                "graph_version": graph.version,
                "wiki_version": self.wiki_repository.current_version(
                    knowledge_base_id
                ),
                "document_count": len(candidates),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "wiki_page_count": len(
                    self.wiki_repository.list_pages(knowledge_base_id)
                ),
            }
        if result is None:
            raise ValueError("文档没有可用于图谱构建的结构化片段")
        self._snapshot_wiki_repository(knowledge_base_id)
        self.persist_state()
        graph = self.graph_repository.snapshot(knowledge_base_id)
        return {
            "graph_version": graph.version,
            "wiki_version": self.wiki_repository.current_version(knowledge_base_id),
            "document_count": len(candidates),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "wiki_page_count": len(
                self.wiki_repository.list_pages(knowledge_base_id)
            ),
        }


class StoreCompilationRepository(CompilationRepository):
    """Compilation status adapter for the prototype metadata repository."""

    def __init__(self, target: MemoryStore) -> None:
        self.store = target

    def get_document(self, document_id: str) -> DocumentSource:
        document = self.store.get("documents", document_id)
        if document is None:
            raise RepositoryError(f"document {document_id!r} does not exist")
        if not document.get("object_key"):
            raise RepositoryError(f"document {document_id!r} has no source object")
        if not document.get("sha256"):
            raise RepositoryError(f"document {document_id!r} has no sha256")
        return DocumentSource(
            id=str(document["id"]),
            knowledge_base_id=str(document["knowledge_base_id"]),
            filename=str(document["filename"]),
            title=str(document["title"]),
            object_key=str(document["object_key"]),
            sha256=str(document["sha256"]),
            active_version=int(document.get("active_version") or 0),
        )

    def mark_stage(
        self,
        document_id: str,
        *,
        stage: str,
        progress: int,
        job_id: str | None = None,
    ) -> None:
        statuses = {
            "loading_source": "parsing",
            "rendering": "parsing",
            "ocr": "parsing",
            "normalizing": "enriching",
            "materializing_assets": "enriching",
            "writing_markdown": "enriching",
            "chunking": "indexing",
            "writing_manifest": "indexing",
        }
        self.store.update(
            "documents",
            document_id,
            {"status": statuses.get(stage, "parsing"), "error_message": None},
        )
        if job_id:
            self.store.update(
                "jobs",
                job_id,
                {
                    "status": "running",
                    "stage": stage,
                    "progress": max(0, min(99, progress)),
                    "error_code": None,
                    "error_message": None,
                    "retryable": False,
                },
            )

    def mark_succeeded(
        self,
        document_id: str,
        *,
        version: int,
        result: dict[str, Any],
        job_id: str | None = None,
    ) -> None:
        self.store.update(
            "documents",
            document_id,
            {
                "status": "ready",
                "active_version": version,
                "page_count": int(result.get("page_count") or 0) or None,
                "element_count": int(result.get("element_count") or 0),
                "asset_count": int(result.get("asset_count") or 0),
                "chunk_count": int(result.get("chunk_count") or 0),
                "error_message": None,
            },
        )
        if job_id:
            self.store.update(
                "jobs",
                job_id,
                {
                    "status": "succeeded",
                    "stage": "completed",
                    "progress": 100,
                    "error_code": None,
                    "error_message": None,
                    "retryable": False,
                    "result": result,
                },
            )

    def mark_failed(
        self,
        document_id: str,
        *,
        stage: str,
        code: str,
        message: str,
        retryable: bool = False,
        job_id: str | None = None,
    ) -> None:
        self.store.update(
            "documents",
            document_id,
            {"status": "failed", "error_message": message},
        )
        if job_id:
            self.store.update(
                "jobs",
                job_id,
                {
                    "status": "failed",
                    "stage": stage,
                    "error_code": code,
                    "error_message": message,
                    "retryable": retryable,
                },
            )


store = MemoryStore()
