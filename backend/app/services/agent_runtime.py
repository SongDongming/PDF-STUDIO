"""Bounded Agentic RAG runtime.

Deep Agents/LangGraph can supply the planner and persistent backends; the
limits, tool validation, grounding and public result contract remain enforced
here so framework behavior cannot bypass product safety.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas import RagSettings
from app.services.memory import ConversationMessage, MemoryService
from app.services.providers import (
    GroundedEvidence,
    DeepSeekProvider,
    ModelAnswer,
    ValidatedAnswer,
)
from app.services.retrieval import HybridRetriever, RetrievalHit


class AgentLimitError(RuntimeError):
    """Raised when a planner exceeds an explicit execution budget."""


def _model_history_content(content: Any) -> str:
    """Project internal rich-message state into provider-compatible text."""

    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        blocks = content.get("blocks")
        if isinstance(blocks, list):
            parts: list[str] = []
            for block in blocks:
                if not isinstance(block, Mapping):
                    continue
                markdown = block.get("markdown")
                caption = block.get("caption")
                if isinstance(markdown, str) and markdown.strip():
                    parts.append(markdown.strip())
                elif isinstance(caption, str) and caption.strip():
                    parts.append(f"[多模态素材：{caption.strip()}]")
            if parts:
                return "\n\n".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any]


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["tools", "answer"]
    tool_calls: list[ToolCall]
    answer_hint: str | None


class ToolObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    tool_name: str
    ok: bool
    data: Any = None
    error: str | None = None


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data: Any
    evidence: list[GroundedEvidence] = Field(default_factory=list)


class AgentPlanner(Protocol):
    async def plan(
        self,
        *,
        question: str,
        knowledge_base_id: str,
        history: Sequence[ConversationMessage],
        long_term_memory: Sequence[Mapping[str, Any]],
        observations: Sequence[ToolObservation],
        tools: Sequence[Mapping[str, Any]],
        remaining_tool_calls: int,
    ) -> AgentDecision: ...


ToolHandler = Callable[[BaseModel], Awaitable[ToolExecutionResult]]


class AgentTool:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: ToolHandler,
    ) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model
        self.handler = handler

    def specification(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }

    async def execute(self, arguments: Mapping[str, Any]) -> ToolExecutionResult:
        validated = self.input_model.model_validate(arguments)
        return await self.handler(validated)


class AgentRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: ValidatedAnswer
    observations: list[ToolObservation]
    tool_calls_used: int = Field(ge=0)
    evidence_count: int = Field(ge=0)


class DeepSeekAgentPlanner:
    """Strict-JSON planner backed by the same DeepSeek provider."""

    def __init__(self, provider: DeepSeekProvider) -> None:
        self.provider = provider

    async def plan(
        self,
        *,
        question: str,
        knowledge_base_id: str,
        history: Sequence[ConversationMessage],
        long_term_memory: Sequence[Mapping[str, Any]],
        observations: Sequence[ToolObservation],
        tools: Sequence[Mapping[str, Any]],
        remaining_tool_calls: int,
    ) -> AgentDecision:
        schema = AgentDecision.model_json_schema()
        payload = {
            "question": question,
            "knowledge_base_id": knowledge_base_id,
            "recent_history": [
                {"role": item.role, "content": item.content} for item in history
            ],
            "long_term_memory": list(long_term_memory),
            "observations": [item.model_dump(mode="json") for item in observations],
            "tools": list(tools),
            "remaining_tool_calls": remaining_tool_calls,
        }
        raw = await self.provider.complete_structured(
            system_prompt=(
                "你是 Agentic RAG 规划器，由你判断本轮是否需要知识库检索。"
                "只有用户明确要求依据 PDF、当前知识库、来源核验、图表/表格/公式，"
                "或问题依赖文档私有内容时，才调用检索工具。寒暄、写作、翻译、"
                "计算、头脑风暴和可由模型通用知识直接回答的问题，action=answer。"
                "工具返回空结果时不要重复搜索，action=answer，让回答模型使用通用知识"
                "并明确没有知识库证据。若检索结果与问题不相关，也 action=answer，"
                "且 answer_hint 必须精确填写 general_knowledge；"
                "直接用通用知识回答时同样填写 general_knowledge。"
                "只要已有任意检索结果能支撑问题的一部分，就必须填写 grounded，"
                "基于现有证据回答并明确未覆盖部分；尤其当用户点名 PDF/case、要求"
                "图表或可回溯引用时，不得因为证据不完整而退回通用知识。"
                "调用 hybrid_search 时，arguments.knowledge_base_id 必须逐字复制"
                "输入载荷中的 knowledge_base_id，不得猜测、缩写或改写。"
                "使用检索证据回答时 answer_hint 填写 grounded。"
                "调用工具时 answer_hint=null。已有足够证据或预算耗尽时也 action=answer；"
                "不得虚构工具名。"
            ),
            user_text=json.dumps(payload, ensure_ascii=False),
            schema_name="agentic_rag_decision",
            schema=schema,
            reasoning_effort="low",
        )
        return AgentDecision.model_validate(raw)


class HybridSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    knowledge_base_id: str = Field(min_length=1)


def hybrid_search_tool(
    retriever: HybridRetriever,
    *,
    settings: RagSettings | None = None,
) -> AgentTool:
    async def search(payload: BaseModel) -> ToolExecutionResult:
        query = HybridSearchInput.model_validate(payload.model_dump())
        hits = await retriever.retrieve(
            query.query,
            knowledge_base_id=query.knowledge_base_id,
            settings=settings,
        )
        return ToolExecutionResult(
            data=[
                {
                    "chunk_id": hit.chunk.id,
                    "document_title": hit.chunk.document_title,
                    "page": hit.chunk.page,
                    "text": hit.chunk.text,
                    "asset_ids": hit.chunk.asset_ids,
                    "score": hit.score,
                }
                for hit in hits
            ],
            evidence=[hit.as_evidence() for hit in hits],
        )

    return AgentTool(
        name="hybrid_search",
        description="在指定知识库执行 BM25 + dense + RRF 混合检索",
        input_model=HybridSearchInput,
        handler=search,
    )


class AgentRuntime:
    def __init__(
        self,
        *,
        planner: AgentPlanner,
        answer_provider: DeepSeekProvider,
        memory: MemoryService,
        tools: Sequence[AgentTool],
        max_tool_calls: int = 8,
        max_rounds: int = 4,
        tool_timeout_seconds: float = 30.0,
        total_timeout_seconds: float = 300.0,
        citation_required: bool = True,
    ) -> None:
        if max_tool_calls < 1 or max_rounds < 1:
            raise ValueError("agent limits must be positive")
        self.planner = planner
        self.answer_provider = answer_provider
        self.memory = memory
        self.tools = {tool.name: tool for tool in tools}
        self.max_tool_calls = max_tool_calls
        self.max_rounds = max_rounds
        self.tool_timeout_seconds = tool_timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.citation_required = citation_required

    async def run(
        self,
        *,
        question: str,
        thread_id: str,
        user_id: str,
        knowledge_base_id: str,
        memory_namespace: str = "preferences",
    ) -> AgentRuntimeResult:
        if not question.strip():
            raise ValueError("question must be non-empty")

        await self.memory.ensure_thread(
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        history = await self.memory.history(
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        long_term_records = await self.memory.recall(
            user_id=user_id, namespace=memory_namespace
        )
        long_term = [
            {"key": record.key, "value": record.value} for record in long_term_records
        ]
        await self.memory.append(
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            message=ConversationMessage(role="user", content=question),
        )

        observations: list[ToolObservation] = []
        evidence_by_id: dict[str, GroundedEvidence] = {}
        seen_calls: set[str] = set()
        calls_used = 0
        force_general_knowledge = False

        async with asyncio.timeout(self.total_timeout_seconds):
            for _round in range(self.max_rounds):
                decision = await self.planner.plan(
                    question=question,
                    knowledge_base_id=knowledge_base_id,
                    history=history,
                    long_term_memory=long_term,
                    observations=observations,
                    tools=[tool.specification() for tool in self.tools.values()],
                    remaining_tool_calls=self.max_tool_calls - calls_used,
                )
                if decision.action == "answer":
                    force_general_knowledge = (
                        decision.answer_hint == "general_knowledge"
                    )
                    break
                if not decision.tool_calls:
                    raise AgentLimitError(
                        "planner selected tools without providing a tool call"
                    )

                for call in decision.tool_calls:
                    if calls_used >= self.max_tool_calls:
                        raise AgentLimitError("agent exceeded max_tool_calls")
                    calls_used += 1
                    fingerprint = json.dumps(
                        [call.name, call.arguments],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if fingerprint in seen_calls:
                        observations.append(
                            ToolObservation(
                                call_id=call.id,
                                tool_name=call.name,
                                ok=False,
                                error="重复工具调用已拒绝",
                            )
                        )
                        continue
                    seen_calls.add(fingerprint)

                    tool = self.tools.get(call.name)
                    if tool is None:
                        observations.append(
                            ToolObservation(
                                call_id=call.id,
                                tool_name=call.name,
                                ok=False,
                                error="工具不在允许列表中",
                            )
                        )
                        continue
                    try:
                        async with asyncio.timeout(self.tool_timeout_seconds):
                            result = await tool.execute(call.arguments)
                        observations.append(
                            ToolObservation(
                                call_id=call.id,
                                tool_name=call.name,
                                ok=True,
                                data=result.data,
                            )
                        )
                        for item in result.evidence:
                            evidence_by_id[item.citation.id] = item
                    except (ValidationError, ValueError, TimeoutError) as exc:
                        observations.append(
                            ToolObservation(
                                call_id=call.id,
                                tool_name=call.name,
                                ok=False,
                                error=type(exc).__name__,
                            )
                        )
            else:
                # Tool use is optional in Agentic RAG. If the planner used its
                # bounded rounds without finding evidence, answer from model
                # knowledge instead of treating an empty retrieval as failure.
                pass

            evidence = (
                []
                if force_general_knowledge
                else list(evidence_by_id.values())
            )
            answer = await self.answer_provider.answer(
                question=question,
                evidence=evidence,
                history=[
                    {
                        "role": item.role,
                        "content": _model_history_content(item.content),
                    }
                    for item in history
                    if item.role in {"user", "assistant", "system"}
                ],
                citation_required=self.citation_required and bool(evidence),
            )
            if not evidence:
                if calls_used:
                    disclosure = (
                        "> **通用知识回答**：当前知识库没有召回到相关证据，"
                        "以下内容来自模型的通用知识，不包含 PDF 引用。"
                    )
                else:
                    disclosure = (
                        "> **通用知识回答**：Agent 判断本轮无需检索知识库，"
                        "以下内容来自模型的通用知识，不包含 PDF 引用。"
                    )
                blocks = [dict(block) for block in answer.blocks]
                first_text = next(
                    (block for block in blocks if block.get("type") == "text"),
                    None,
                )
                if first_text is None:
                    blocks.insert(
                        0,
                        {"type": "text", "markdown": disclosure},
                    )
                else:
                    first_text["markdown"] = (
                        f"{disclosure}\n\n{first_text['markdown']}"
                    )
                answer = answer.model_copy(update={"blocks": blocks, "citations": []})

        await self.memory.append(
            thread_id=thread_id,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            message=ConversationMessage(
                role="assistant",
                content={
                    "blocks": answer.blocks,
                    "citations": [
                        citation.model_dump(mode="json")
                        for citation in answer.citations
                    ],
                },
            ),
        )
        return AgentRuntimeResult(
            answer=answer,
            observations=observations,
            tool_calls_used=calls_used,
            evidence_count=len(evidence),
        )
