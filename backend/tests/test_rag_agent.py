import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from app.schemas import Citation, RagSettings
from app.services.agent_runtime import (
    AgentDecision,
    AgentLimitError,
    AgentRuntime,
    DeepSeekAgentPlanner,
    ToolCall,
    hybrid_search_tool,
)
from app.services.embeddings import (
    BailianEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from app.services.memory import (
    ConversationMessage,
    MemoryIsolationError,
    MemoryService,
)
from app.services.providers import (
    DeepSeekProvider,
    GroundedEvidence,
    GroundingValidationError,
    ModelAnswer,
    ProviderUnavailableError,
    ValidatedAnswer,
    VisionInput,
    validate_grounded_answer,
)
from app.services.retrieval import HybridRetriever, RetrievalChunk


def run(coroutine):
    return asyncio.run(coroutine)


def response(status_code: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://provider.test"),
    )


class QueueHttpClient:
    def __init__(
        self,
        post_responses: Sequence[httpx.Response] = (),
        get_response: httpx.Response | None = None,
    ) -> None:
        self.post_responses = list(post_responses)
        self.get_response = get_response or response(200, {"data": []})
        self.posts: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.posts.append({"url": url, **kwargs})
        return self.post_responses.pop(0)

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.get_response


def test_bailian_uses_native_query_document_contract_and_fixed_dimensions() -> None:
    client = QueueHttpClient(
        [
            response(
                200,
                {
                    "output": {
                        "embeddings": [
                            {"text_index": 0, "embedding": [1, 0, 0]}
                        ]
                    }
                },
            ),
            response(
                200,
                {
                    "output": {
                        "embeddings": [
                            {"text_index": 1, "embedding": [0, 1, 0]},
                            {"text_index": 0, "embedding": [1, 0, 0]},
                        ]
                    }
                },
            ),
        ]
    )
    provider = BailianEmbeddingProvider(
        api_key="test-only",
        base_url="https://dashscope.test",
        dimensions=3,
        client=client,
    )

    assert run(provider.embed_query("图谱是什么")) == [1.0, 0.0, 0.0]
    assert run(provider.embed_documents(["第一段", "第二段"])) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert client.posts[0]["json"]["parameters"] == {
        "text_type": "query",
        "dimension": 3,
        "output_type": "dense",
    }
    assert client.posts[1]["json"]["parameters"]["text_type"] == "document"
    assert provider.index_signature == "aliyun-bailian:text-embedding-v4:3"


def test_openai_embedding_health_is_sanitized_when_quota_is_unavailable() -> None:
    quota_response = {
        "error": {
            "code": "insufficient_quota",
            "message": "secret account detail",
        }
    }
    client = QueueHttpClient(
        [
            response(429, quota_response),
            response(429, quota_response),
        ]
    )
    provider = OpenAIEmbeddingProvider(
        api_key="must-not-appear",
        dimensions=3,
        client=client,
    )

    health = run(provider.health())
    assert health.healthy is False
    assert health.detail == "模型服务暂不可用"
    assert "must-not-appear" not in health.model_dump_json()
    with pytest.raises(ProviderUnavailableError) as caught:
        run(provider.embed_query("测试"))
    assert "secret account detail" not in str(caught.value)


def evidence_fixture() -> GroundedEvidence:
    return GroundedEvidence(
        citation=Citation(
            id="citation:chunk-1",
            document_id="doc-1",
            document_title="Agentic RAG 图文手册",
            page=8,
            bbox=(0.1, 0.2, 0.8, 0.7),
            element_id="figure-2",
            excerpt="图 2 展示检索、工具调用与证据校验。",
            score=0.95,
        ),
        text="图 2 展示检索、工具调用与证据校验。",
        asset_ids=["asset-figure-2"],
    )


def valid_model_answer() -> dict[str, Any]:
    return {
        "blocks": [
            {
                "type": "text",
                "markdown": "系统先检索，再执行受限工具调用。",
                "asset_id": None,
                "caption": None,
                "alt": None,
                "citation_ids": ["citation:chunk-1"],
            },
            {
                "type": "image",
                "markdown": None,
                "asset_id": "asset-figure-2",
                "caption": "Agentic RAG 流程",
                "alt": "检索、工具调用和证据校验流程图",
                "citation_ids": ["citation:chunk-1"],
            },
        ],
        "citation_ids": ["citation:chunk-1"],
    }


def test_deepseek_uses_strict_json_multimodal_contract() -> None:
    client = QueueHttpClient(
        [
            response(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    valid_model_answer(), ensure_ascii=False
                                )
                            }
                        }
                    ]
                },
            )
        ]
    )
    provider = DeepSeekProvider(
        api_key="test-only",
        base_url="https://moonshot.test/v1",
        client=client,
    )
    answer = run(
        provider.answer(
            question="图中流程是什么？",
            evidence=[evidence_fixture()],
            images=[
                VisionInput(
                    url="data:image/png;base64,ZmFrZQ==",
                    detail="high",
                )
            ],
        )
    )

    assert [block["type"] for block in answer.blocks] == ["text", "image"]
    assert answer.blocks[1]["asset_id"] == "asset-figure-2"
    assert answer.citations[0].page == 8
    payload = client.posts[0]["json"]
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["response_format"]["type"] == "json_object"
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert payload["messages"][-1]["content"][1]["type"] == "image_url"
    system_prompt = payload["messages"][0]["content"]
    assert "必须严格输出符合以下 JSON Schema" in system_prompt

    with pytest.raises(ValueError):
        VisionInput(url="https://public.example/image.png")


def test_deepseek_can_answer_from_general_knowledge_without_citations() -> None:
    client = QueueHttpClient(
        [
            response(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "blocks": [
                                            {
                                                "type": "text",
                                                "markdown": "巴黎是法国的首都。",
                                                "asset_id": None,
                                                "caption": None,
                                                "alt": None,
                                                "citation_ids": [],
                                            }
                                        ],
                                        "citation_ids": [],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                },
            )
        ]
    )
    provider = DeepSeekProvider(
        api_key="test-only",
        base_url="https://moonshot.test/v1",
        client=client,
    )

    answer = run(
        provider.answer(
            question="法国的首都是什么？",
            evidence=[],
            citation_required=False,
        )
    )

    assert answer.blocks == [{"type": "text", "markdown": "巴黎是法国的首都。"}]
    assert answer.citations == []
    payload = client.posts[0]["json"]
    assert "通用问答分支" in payload["messages"][0]["content"]
    assert (
        "不得声称答案来自 PDF"
        in payload["messages"][-1]["content"][0]["text"]
    )


def test_deepseek_retries_rate_limit_using_retry_after() -> None:
    limited = httpx.Response(
        429,
        headers={"Retry-After": "0"},
        json={"error": {"code": "rate_limit"}},
        request=httpx.Request("POST", "https://provider.test"),
    )
    accepted = response(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "blocks": [
                                    {
                                        "type": "text",
                                        "markdown": "重试后成功。",
                                        "asset_id": None,
                                        "caption": None,
                                        "alt": None,
                                        "citation_ids": [],
                                    }
                                ],
                                "citation_ids": [],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        },
    )
    client = QueueHttpClient([limited, accepted])
    provider = DeepSeekProvider(
        api_key="test-only",
        base_url="https://moonshot.test/v1",
        client=client,
    )

    answer = run(
        provider.answer(
            question="不用检索",
            evidence=[],
            citation_required=False,
        )
    )

    assert answer.blocks[0]["markdown"] == "重试后成功。"
    assert len(client.posts) == 2


def test_grounding_validator_rejects_hallucinated_assets_and_citations() -> None:
    hallucinated_asset = valid_model_answer()
    hallucinated_asset["blocks"][1]["asset_id"] = "invented"
    with pytest.raises(GroundingValidationError, match="unavailable asset"):
        validate_grounded_answer(hallucinated_asset, [evidence_fixture()])

    hallucinated_citation = valid_model_answer()
    hallucinated_citation["citation_ids"] = ["citation:invented"]
    with pytest.raises(GroundingValidationError, match="unavailable citation"):
        validate_grounded_answer(hallucinated_citation, [evidence_fixture()])


class FakeEmbedder:
    provider_name = "fake"
    model = "fake"
    dimensions = 2

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def retrieval_fixture() -> HybridRetriever:
    return HybridRetriever(
        [
            RetrievalChunk(
                id="both",
                knowledge_base_id="kb-1",
                document_id="doc-1",
                document_title="图文手册",
                page=1,
                text="知识图谱检索与 Agentic RAG",
                embedding=[1.0, 0.0],
                asset_ids=["asset-1"],
            ),
            RetrievalChunk(
                id="lexical",
                knowledge_base_id="kb-1",
                document_id="doc-1",
                document_title="图文手册",
                page=2,
                text="知识图谱帮助发现实体关系",
                embedding=[0.0, 1.0],
            ),
            RetrievalChunk(
                id="other-kb",
                knowledge_base_id="kb-2",
                document_id="doc-2",
                document_title="隔离文档",
                page=1,
                text="知识图谱检索",
                embedding=[1.0, 0.0],
            ),
        ],
        embedder=FakeEmbedder(),
    )


def test_hybrid_retrieval_combines_bm25_dense_rrf_and_enforces_kb_scope() -> None:
    hits = run(
        retrieval_fixture().retrieve(
            "知识图谱检索",
            knowledge_base_id="kb-1",
            settings=RagSettings(
                dense_top_k=3,
                lexical_top_k=3,
                rerank_top_k=3,
            ),
        )
    )

    assert hits[0].chunk.id == "both"
    assert hits[0].lexical_rank is not None
    assert hits[0].dense_rank == 1
    assert hits[0].score == 1
    assert {hit.chunk.id for hit in hits} <= {"both", "lexical"}
    assert hits[0].as_evidence().citation.id == "citation:both"


def test_thread_and_long_term_memory_are_user_isolated_and_bounded() -> None:
    memory = MemoryService(max_messages_per_thread=2)
    run(
        memory.ensure_thread(
            thread_id="thread-1", user_id="user-a", knowledge_base_id="kb-1"
        )
    )
    for content in ["一", "二", "三"]:
        run(
            memory.append(
                thread_id="thread-1",
                user_id="user-a",
                knowledge_base_id="kb-1",
                message=ConversationMessage(role="user", content=content),
            )
        )
    history = run(
        memory.history(
            thread_id="thread-1",
            user_id="user-a",
            knowledge_base_id="kb-1",
        )
    )
    assert [item.content for item in history] == ["二", "三"]
    with pytest.raises(MemoryIsolationError):
        run(
            memory.history(
                thread_id="thread-1",
                user_id="user-b",
                knowledge_base_id="kb-1",
            )
        )


class ScriptedPlanner:
    def __init__(self, decisions: Sequence[AgentDecision]) -> None:
        self.decisions = list(decisions)

    async def plan(self, **_: Any) -> AgentDecision:
        return self.decisions.pop(0)


def test_deepseek_planner_receives_exact_knowledge_base_scope() -> None:
    class CapturingProvider:
        def __init__(self) -> None:
            self.user_text = ""

        async def complete_structured(self, **kwargs: Any) -> dict[str, Any]:
            self.user_text = kwargs["user_text"]
            return {
                "action": "tools",
                "tool_calls": [
                    {
                        "id": "search-1",
                        "name": "hybrid_search",
                        "arguments": {
                            "query": "Agentic RAG",
                            "knowledge_base_id": "kb-live-uuid",
                        },
                    }
                ],
                "answer_hint": None,
            }

    provider = CapturingProvider()
    planner = DeepSeekAgentPlanner(provider)  # type: ignore[arg-type]

    decision = run(
        planner.plan(
            question="检索 case-7",
            knowledge_base_id="kb-live-uuid",
            history=[],
            long_term_memory=[],
            observations=[],
            tools=[],
            remaining_tool_calls=3,
        )
    )

    assert json.loads(provider.user_text)["knowledge_base_id"] == "kb-live-uuid"
    assert (
        decision.tool_calls[0].arguments["knowledge_base_id"]
        == "kb-live-uuid"
    )


class FakeAnswerProvider:
    async def answer(
        self,
        *,
        evidence: Sequence[GroundedEvidence],
        citation_required: bool,
        **_: Any,
    ) -> ValidatedAnswer:
        source = evidence[0]
        asset_id = source.asset_ids[0]
        citation_id = source.citation.id
        return validate_grounded_answer(
            ModelAnswer.model_validate(
                {
                    "blocks": [
                        {
                            "type": "text",
                            "markdown": "系统先检索，再执行受限工具调用。",
                            "asset_id": None,
                            "caption": None,
                            "alt": None,
                            "citation_ids": [citation_id],
                        },
                        {
                            "type": "image",
                            "markdown": None,
                            "asset_id": asset_id,
                            "caption": "Agentic RAG 流程",
                            "alt": "检索、工具调用和证据校验流程图",
                            "citation_ids": [citation_id],
                        },
                    ],
                    "citation_ids": [citation_id],
                }
            ),
            evidence,
            citation_required=citation_required,
        )


class GeneralKnowledgeAnswerProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def answer(
        self,
        *,
        evidence: Sequence[GroundedEvidence],
        citation_required: bool,
        **_: Any,
    ) -> ValidatedAnswer:
        self.calls.append(
            {
                "evidence_count": len(evidence),
                "citation_required": citation_required,
            }
        )
        return validate_grounded_answer(
            ModelAnswer.model_validate(
                {
                    "blocks": [
                        {
                            "type": "text",
                            "markdown": "这是模型通用知识生成的回答。",
                            "asset_id": None,
                            "caption": None,
                            "alt": None,
                            "citation_ids": [],
                        }
                    ],
                    "citation_ids": [],
                }
            ),
            evidence,
            citation_required=citation_required,
        )


def test_agent_runtime_can_skip_retrieval_for_general_question() -> None:
    provider = GeneralKnowledgeAnswerProvider()
    runtime = AgentRuntime(
        planner=ScriptedPlanner(
            [AgentDecision(action="answer", tool_calls=[], answer_hint=None)]
        ),
        answer_provider=provider,  # type: ignore[arg-type]
        memory=MemoryService(),
        tools=[hybrid_search_tool(retrieval_fixture())],
        max_tool_calls=2,
        max_rounds=2,
    )

    result = run(
        runtime.run(
            question="帮我写一句欢迎语",
            thread_id="thread-general",
            user_id="user-1",
            knowledge_base_id="kb-1",
        )
    )

    assert result.tool_calls_used == 0
    assert result.evidence_count == 0
    assert result.answer.citations == []
    assert "Agent 判断本轮无需检索知识库" in result.answer.blocks[0]["markdown"]
    assert provider.calls == [{"evidence_count": 0, "citation_required": False}]


def test_agent_runtime_falls_back_to_model_knowledge_after_empty_retrieval() -> None:
    provider = GeneralKnowledgeAnswerProvider()
    runtime = AgentRuntime(
        planner=ScriptedPlanner(
            [
                AgentDecision(
                    action="tools",
                    tool_calls=[
                        ToolCall(
                            id="search-empty",
                            name="hybrid_search",
                            arguments={
                                "query": "知识库不存在的主题",
                                "knowledge_base_id": "kb-1",
                            },
                        )
                    ],
                    answer_hint=None,
                ),
                AgentDecision(action="answer", tool_calls=[], answer_hint=None),
            ]
        ),
        answer_provider=provider,  # type: ignore[arg-type]
        memory=MemoryService(),
        tools=[hybrid_search_tool(HybridRetriever([]))],
        max_tool_calls=2,
        max_rounds=2,
    )

    result = run(
        runtime.run(
            question="解释一个知识库里没有的概念",
            thread_id="thread-empty-retrieval",
            user_id="user-1",
            knowledge_base_id="kb-1",
        )
    )

    assert result.tool_calls_used == 1
    assert result.evidence_count == 0
    assert result.answer.citations == []
    assert "当前知识库没有召回到相关证据" in result.answer.blocks[0]["markdown"]
    assert provider.calls == [{"evidence_count": 0, "citation_required": False}]


def test_agent_runtime_can_discard_irrelevant_retrieval_results() -> None:
    provider = GeneralKnowledgeAnswerProvider()
    runtime = AgentRuntime(
        planner=ScriptedPlanner(
            [
                AgentDecision(
                    action="tools",
                    tool_calls=[
                        ToolCall(
                            id="search-irrelevant",
                            name="hybrid_search",
                            arguments={
                                "query": "与知识库无关的问题",
                                "knowledge_base_id": "kb-1",
                            },
                        )
                    ],
                    answer_hint=None,
                ),
                AgentDecision(
                    action="answer",
                    tool_calls=[],
                    answer_hint="general_knowledge",
                ),
            ]
        ),
        answer_provider=provider,  # type: ignore[arg-type]
        memory=MemoryService(),
        tools=[hybrid_search_tool(retrieval_fixture())],
        max_tool_calls=2,
        max_rounds=2,
    )

    result = run(
        runtime.run(
            question="苹果公司的市值是多少？",
            thread_id="thread-irrelevant-retrieval",
            user_id="user-1",
            knowledge_base_id="kb-1",
        )
    )

    assert result.tool_calls_used == 1
    assert result.evidence_count == 0
    assert result.answer.citations == []
    assert "当前知识库没有召回到相关证据" in result.answer.blocks[0]["markdown"]
    assert provider.calls == [{"evidence_count": 0, "citation_required": False}]


def test_agent_runtime_executes_allowlisted_tool_and_persists_grounded_answer() -> None:
    planner = ScriptedPlanner(
        [
            AgentDecision(
                action="tools",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="hybrid_search",
                        arguments={
                            "query": "知识图谱检索",
                            "knowledge_base_id": "kb-1",
                        },
                    )
                ],
                answer_hint=None,
            ),
            AgentDecision(action="answer", tool_calls=[], answer_hint="已经足够"),
        ]
    )
    memory = MemoryService()
    runtime = AgentRuntime(
        planner=planner,
        answer_provider=FakeAnswerProvider(),  # type: ignore[arg-type]
        memory=memory,
        tools=[hybrid_search_tool(retrieval_fixture())],
        max_tool_calls=2,
        max_rounds=2,
    )
    result = run(
        runtime.run(
            question="图中流程是什么？",
            thread_id="thread-1",
            user_id="user-1",
            knowledge_base_id="kb-1",
        )
    )

    assert result.tool_calls_used == 1
    assert result.evidence_count >= 1
    assert result.answer.blocks[1]["type"] == "image"
    assert result.answer.citations[0].id == "citation:both"
    history = run(
        memory.history(
            thread_id="thread-1",
            user_id="user-1",
            knowledge_base_id="kb-1",
        )
    )
    assert [message.role for message in history] == ["user", "assistant"]


def test_agent_runtime_projects_rich_assistant_history_to_provider_text() -> None:
    planner = ScriptedPlanner(
        [
            AgentDecision(
                action="tools",
                tool_calls=[
                    ToolCall(
                        id="call-first",
                        name="hybrid_search",
                        arguments={
                            "query": "知识图谱检索",
                            "knowledge_base_id": "kb-1",
                        },
                    )
                ],
                answer_hint=None,
            ),
            AgentDecision(action="answer", tool_calls=[], answer_hint=None),
            AgentDecision(
                action="tools",
                tool_calls=[
                    ToolCall(
                        id="call-followup",
                        name="hybrid_search",
                        arguments={
                            "query": "继续说明",
                            "knowledge_base_id": "kb-1",
                        },
                    )
                ],
                answer_hint=None,
            ),
            AgentDecision(action="answer", tool_calls=[], answer_hint=None),
        ]
    )

    class CapturingProvider(FakeAnswerProvider):
        def __init__(self) -> None:
            self.histories: list[Sequence[Mapping[str, Any]]] = []

        async def answer(self, *, history=(), **kwargs: Any) -> ValidatedAnswer:
            self.histories.append(history)
            return await super().answer(**kwargs)

    provider = CapturingProvider()
    runtime = AgentRuntime(
        planner=planner,
        answer_provider=provider,  # type: ignore[arg-type]
        memory=MemoryService(),
        tools=[hybrid_search_tool(retrieval_fixture())],
        max_tool_calls=2,
        max_rounds=2,
    )
    for question in ("第一轮", "继续说明"):
        run(
            runtime.run(
                question=question,
                thread_id="thread-history",
                user_id="user-1",
                knowledge_base_id="kb-1",
            )
        )

    assert provider.histories[0] == []
    second_history = provider.histories[1]
    assert [item["role"] for item in second_history] == ["user", "assistant"]
    assert all(isinstance(item["content"], str) for item in second_history)
    assert "系统先检索" in second_history[1]["content"]
    assert "Agentic RAG 流程" in second_history[1]["content"]


def test_agent_runtime_hard_stops_at_tool_budget() -> None:
    planner = ScriptedPlanner(
        [
            AgentDecision(
                action="tools",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="hybrid_search",
                        arguments={
                            "query": "第一次",
                            "knowledge_base_id": "kb-1",
                        },
                    ),
                    ToolCall(
                        id="call-2",
                        name="hybrid_search",
                        arguments={
                            "query": "第二次",
                            "knowledge_base_id": "kb-1",
                        },
                    ),
                ],
                answer_hint=None,
            )
        ]
    )
    runtime = AgentRuntime(
        planner=planner,
        answer_provider=FakeAnswerProvider(),  # type: ignore[arg-type]
        memory=MemoryService(),
        tools=[hybrid_search_tool(retrieval_fixture())],
        max_tool_calls=1,
        max_rounds=1,
    )

    with pytest.raises(AgentLimitError, match="max_tool_calls"):
        run(
            runtime.run(
                question="无限搜索",
                thread_id="thread-budget",
                user_id="user-1",
                knowledge_base_id="kb-1",
            )
        )
