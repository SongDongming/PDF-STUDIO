from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.services.agent_runtime_registry import agent_runtime_registry
from app.services.agentic_tools import SearchChunksTool
from app.services.retrieval import RetrievalChunk
from app.store import store


def _text_block(markdown: str, citation_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "text",
        "markdown": markdown,
        "asset_id": None,
        "caption": None,
        "alt": None,
        "citation_ids": citation_ids,
    }


class ScenarioRuntime:
    async def aget_checkpoint(self, *, thread_id: str):
        return SimpleNamespace(values={})

    async def ainvoke(self, *, messages, thread_id: str):
        question = messages[-1]["content"]
        if "伪造引用" in question:
            return {
                "structured_response": {
                    "mode": "grounded",
                    "blocks": [_text_block("伪造答案", ["citation:invented"])],
                    "citation_ids": ["citation:invented"],
                    "coverage_gaps": [],
                }
            }
        if "PDF" in question:
            result = await SearchChunksTool(store).execute(
                {"query": "Agentic RAG 根据问题决定是否检索"}
            )
            citation_id = result["hits"][0]["citation_id"]
            blocks = [
                _text_block(
                    "文档说明 Agent 会按问题决定是否检索。",
                    [citation_id],
                )
            ]
            if "不完整图片块" in question:
                blocks.append(
                    {
                        "type": "image",
                        "markdown": None,
                        "asset_id": None,
                        "caption": None,
                        "alt": None,
                        "citation_ids": [citation_id],
                    }
                )
            return {
                "structured_response": {
                    "mode": "grounded",
                    "blocks": blocks,
                    "citation_ids": [citation_id],
                    "coverage_gaps": [],
                }
            }
        return {
            "structured_response": {
                "mode": "general_knowledge",
                "blocks": [_text_block("这是模型通用知识回答。", [])],
                "citation_ids": [],
                "coverage_gaps": [],
            }
        }


class StreamingScenarioRuntime(ScenarioRuntime):
    async def astream(self, *, messages, thread_id: str):
        structured_response = {
            "mode": "general_knowledge",
            "blocks": [_text_block("这是逐步生成的 **Markdown** 回答。", [])],
            "citation_ids": [],
            "coverage_gaps": [],
        }
        arguments = json.dumps(structured_response, ensure_ascii=False)
        for index in range(0, len(arguments), 9):
            yield {
                "type": "messages",
                "ns": (),
                "data": (
                    SimpleNamespace(
                        id="answer-message",
                        content="",
                        tool_call_chunks=[
                            {
                                "name": (
                                    "AgenticModelAnswer" if index == 0 else None
                                ),
                                "args": arguments[index : index + 9],
                                "index": 0,
                            }
                        ],
                    ),
                    {"langgraph_node": "model"},
                ),
            }
        yield {
            "type": "updates",
            "ns": (),
            "data": {
                "model": {
                    "structured_response": structured_response,
                }
            },
        }


def _create_thread(client: TestClient) -> str:
    kb = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "测试知识库", "description": ""},
    ).json()
    store.retriever.upsert(
        [
            RetrievalChunk(
                id="chunk-agentic",
                knowledge_base_id=kb["id"],
                document_id="doc-1",
                document_title="Agentic RAG 手册",
                page=3,
                text="Agentic RAG 根据用户问题决定是否调用检索工具。",
                bbox=(0.1, 0.2, 0.8, 0.4),
                element_id="element-1",
            )
        ]
    )
    return client.post(
        "/api/v1/chat/threads",
        json={"knowledge_base_id": kb["id"], "title": "运行时验收"},
    ).json()["id"]


def test_deepagents_route_allows_general_knowledge_without_retrieval(
    client: TestClient,
) -> None:
    agent_runtime_registry.build = SimpleNamespace(runtime=ScenarioRuntime())
    thread_id = _create_thread(client)

    response = client.post(
        f"/api/v1/chat/threads/{thread_id}/messages",
        json={"content": "请写一句欢迎语"},
    )

    assert response.status_code == 201
    assert response.json()["blocks"][0]["markdown"] == "这是模型通用知识回答。"
    assert response.json()["citations"] == []


def test_deepagents_route_publishes_only_ledger_grounded_citations(
    client: TestClient,
) -> None:
    agent_runtime_registry.build = SimpleNamespace(runtime=ScenarioRuntime())
    thread_id = _create_thread(client)

    grounded = client.post(
        f"/api/v1/chat/threads/{thread_id}/messages",
        json={"content": "请根据 PDF 解释 Agentic RAG"},
    )
    rejected = client.post(
        f"/api/v1/chat/threads/{thread_id}/messages",
        json={"content": "请伪造引用"},
    )

    assert grounded.status_code == 201
    assert grounded.json()["citations"][0]["id"] == "citation:chunk-agentic"
    assert grounded.json()["citations"][0]["page"] == 3
    assert rejected.status_code == 201
    assert "未通过证据校验" in rejected.json()["blocks"][0]["markdown"]
    assert rejected.json()["citations"] == []


def test_deepagents_route_discards_unrequested_empty_visual_placeholder(
    client: TestClient,
) -> None:
    agent_runtime_registry.build = SimpleNamespace(runtime=ScenarioRuntime())
    thread_id = _create_thread(client)

    response = client.post(
        f"/api/v1/chat/threads/{thread_id}/messages",
        json={"content": "请根据 PDF 回答，并模拟不完整图片块"},
    )

    assert response.status_code == 201
    assert response.json()["blocks"] == [
        {
            "type": "text",
            "markdown": "文档说明 Agent 会按问题决定是否检索。",
        }
    ]
    assert response.json()["citations"][0]["id"] == "citation:chunk-agentic"


def test_deepagents_route_streams_markdown_before_validated_completion(
    client: TestClient,
) -> None:
    agent_runtime_registry.build = SimpleNamespace(runtime=StreamingScenarioRuntime())
    thread_id = _create_thread(client)

    with client.stream(
        "POST",
        f"/api/v1/chat/threads/{thread_id}/messages/stream",
        json={"content": "请流式回答"},
    ) as response:
        body = "".join(response.iter_text())

    events = []
    for frame in body.split("\n\n"):
        data = "\n".join(
            line[5:].strip()
            for line in frame.splitlines()
            if line.startswith("data:")
        )
        if data:
            events.append(json.loads(data))

    event_types = [event["type"] for event in events]
    assert response.headers["x-accel-buffering"] == "no"
    assert event_types[0] == "answer.started"
    assert "answer.delta" in event_types
    assert event_types[-1] == "answer.completed"
    delta = "".join(
        event.get("delta", "")
        for event in events
        if event["type"] == "answer.delta"
    )
    assert delta == "这是逐步生成的 **Markdown** 回答。"
    assert events[-1]["message"]["blocks"][0]["markdown"] == delta
