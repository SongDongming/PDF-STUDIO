from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.assets import LocalObjectStorage, put_json
from app.services.retrieval import HybridRetriever, RetrievalChunk
from app.services.state_persistence import (
    ConcurrentStateUpdateError,
    SqlAlchemyStatePersistence,
)
from app.store import MemoryStore


def sample_state() -> dict:
    timestamp = datetime(2026, 7, 30, 8, 30, tzinfo=timezone.utc)
    return {
        "knowledge_bases": {
            "kb-1": {
                "id": "kb-1",
                "name": "多模态文档库",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        },
        "documents": {
            "doc-1": {
                "id": "doc-1",
                "knowledge_base_id": "kb-1",
                "title": "图文手册",
                "active_version": 3,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        },
        "jobs": {
            "job-1": {
                "id": "job-1",
                "status": "succeeded",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        },
        "threads": {
            "thread-1": {
                "id": "thread-1",
                "knowledge_base_id": "kb-1",
                "title": "检索会话",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
        },
        "messages": {
            "thread-1": [
                {
                    "id": "message-1",
                    "thread_id": "thread-1",
                    "role": "user",
                    "blocks": [{"type": "text", "markdown": "解释图中的流程"}],
                    "created_at": timestamp,
                }
            ]
        },
        "wiki_pages": {},
        "settings": {
            "providers": [
                {
                    "id": "vision-chat",
                    "provider": "moonshot",
                    "configured": True,
                    "credential_ref": "env:MOONSHOT_API_KEY",
                }
            ],
            "rag": {"dense_top_k": 12},
            "updated_at": timestamp,
        },
    }


def test_sqlalchemy_sqlite_recovers_state_and_increments_revision(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'state.db'}"
    first_process = SqlAlchemyStatePersistence(database_url, namespace="test")
    first = first_process.save(sample_state())
    changed = sample_state()
    changed["threads"]["thread-1"]["title"] = "第二轮会话"
    second = first_process.save(changed, expected_revision=first.revision)

    restarted = SqlAlchemyStatePersistence(database_url, namespace="test")
    loaded = restarted.load()

    assert loaded is not None
    assert second.revision == loaded.revision == 2
    assert loaded.state["threads"]["thread-1"]["title"] == "第二轮会话"
    with pytest.raises(ConcurrentStateUpdateError):
        restarted.save(changed, expected_revision=1)


def test_sqlalchemy_compare_and_swap_rejects_competing_writer(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'concurrent.db'}"
    first_worker = SqlAlchemyStatePersistence(database_url, namespace="test")
    second_worker = SqlAlchemyStatePersistence(database_url, namespace="test")
    created = first_worker.save(sample_state(), expected_revision=0)

    first_worker.save(sample_state(), expected_revision=created.revision)
    with pytest.raises(ConcurrentStateUpdateError):
        second_worker.save(sample_state(), expected_revision=created.revision)


def test_secret_fields_are_removed_but_reference_metadata_is_preserved(
    tmp_path,
) -> None:
    state = sample_state()
    provider = state["settings"]["providers"][0]
    provider["client_secret"] = "sensitive-value"
    provider["access_token"] = "another-sensitive-value"
    state["jobs"]["job-1"]["authorization"] = "protected-header"
    persistence = SqlAlchemyStatePersistence(
        f"sqlite+pysqlite:///{tmp_path / 'secrets.db'}", namespace="test"
    )

    saved = persistence.save(state)
    loaded_provider = saved.state["settings"]["providers"][0]

    assert "client_secret" not in loaded_provider
    assert "access_token" not in loaded_provider
    assert loaded_provider["configured"] is True
    assert loaded_provider["credential_ref"] == "env:MOONSHOT_API_KEY"


def test_runtime_indexes_and_asset_keys_are_restored_from_latest_success(
    tmp_path, monkeypatch
) -> None:
    runtime = MemoryStore()
    runtime.storage = LocalObjectStorage(tmp_path / "objects")
    put_json(
        runtime.storage,
        "compiled/doc-1/v1/manifest.json",
        {
            "assets": [
                {
                    "id": "asset-v1",
                    "object_key": "compiled/doc-1/v1/assets/asset-v1.png",
                }
            ]
        },
    )
    put_json(
        runtime.storage,
        "compiled/doc-1/v2/manifest.json",
        {
            "assets": [
                {
                    "id": "asset-v2",
                    "object_key": "compiled/doc-1/v2/assets/asset-v2.png",
                }
            ]
        },
    )
    earlier = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)
    later = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)
    # The document must still exist: the index restore skips documents that
    # were deleted or recompiled under a new id so the retriever never serves
    # stale copies.
    runtime.documents = {
        "doc-1": {
            "id": "doc-1",
            "knowledge_base_id": "kb-1",
            "title": "持久化手册",
            "status": "ready",
            "active_version": 2,
            "created_at": earlier,
            "updated_at": earlier,
        }
    }
    runtime.jobs = {
        "job-old": {
            "document_id": "doc-1",
            "status": "succeeded",
            "updated_at": earlier,
            "result": {
                "index_manifest_key": "indexes/doc-1/v1.json",
                "manifest_key": "compiled/doc-1/v1/manifest.json",
                "index_mode": "hybrid",
            },
        },
        "job-new": {
            "document_id": "doc-1",
            "status": "succeeded",
            "updated_at": later,
            "result": {
                "index_manifest_key": "indexes/doc-1/v2.json",
                "manifest_key": "compiled/doc-1/v2/manifest.json",
                "index_mode": "hybrid",
            },
        },
    }
    restored_chunk = RetrievalChunk(
        id="chunk-v2",
        knowledge_base_id="kb-1",
        document_id="doc-1",
        document_title="持久化手册",
        page=2,
        text="SqliteSaver 跨进程保留记忆",
        embedding=[1.0, 0.0],
    )
    loaded_keys: list[str] = []

    class FakeIndexer:
        def __init__(self, **_):
            pass

        def load_retriever(self, key, *, embedder):
            loaded_keys.append(key)
            return HybridRetriever([restored_chunk], embedder=embedder)

    monkeypatch.setattr("app.store.DocumentIndexer", FakeIndexer)
    monkeypatch.setattr(runtime, "_embedding_provider", lambda: object())

    runtime._restore_runtime_indexes()

    assert loaded_keys == ["indexes/doc-1/v2.json"]
    assert "chunk-v2" in runtime.retriever._chunks
    assert runtime.asset_keys == {
        "asset-v1": "compiled/doc-1/v1/assets/asset-v1.png",
        "asset-v2": "compiled/doc-1/v2/assets/asset-v2.png",
    }


def test_wiki_repository_is_restored_from_durable_snapshot() -> None:
    runtime = MemoryStore()
    updated_at = datetime(2026, 7, 30, 10, 0, tzinfo=timezone.utc)
    runtime.wiki_pages = {
        "kb-1": {
            "version": 3,
            "pages": {
                "wiki:agentic-rag": {
                    "id": "wiki:agentic-rag",
                    "knowledge_base_id": "kb-1",
                    "entity_id": "entity:agentic-rag",
                    "slug": "agentic-rag",
                    "title": "Agentic RAG",
                    "summary": "检索由模型自主决策。",
                    "sections": {"关键结论": "检索器作为工具。"},
                    "conclusions": [
                        {
                            "id": "conclusion-1",
                            "text": "检索器被包装为 Agent 工具。",
                            "kind": "claim",
                            "evidence": [
                                {
                                    "document_id": "doc-1",
                                    "page": 2,
                                    "bbox": [0.1, 0.2, 0.8, 0.7],
                                    "chunk_id": "chunk-1",
                                    "element_id": "element-1",
                                }
                            ],
                            "predicate": None,
                            "related_entity_id": None,
                        }
                    ],
                    "related_page_ids": [],
                    "graph_version": 3,
                    "revision": 1,
                    "status": "published",
                    "locked_fields": [],
                    "updated_at": updated_at,
                }
            },
        }
    }

    runtime._restore_wiki_repository()

    page = runtime.wiki_repository.get_page("kb-1", "agentic-rag")
    assert page is not None
    assert page.title == "Agentic RAG"
    assert page.conclusions[0].evidence[0].element_id == "element-1"
    assert runtime.wiki_repository.current_version("kb-1") == 3
