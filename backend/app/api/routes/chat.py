import json
import logging
import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any, Mapping
from uuid import uuid4

from fastapi import APIRouter, Response, status
from fastapi.responses import StreamingResponse

from app.api.routes._common import require_item
from app.schemas import (
    MessageCreate,
    MessageView,
    RagSettings,
    TextBlock,
    ThreadCreate,
    ThreadUpdate,
    ThreadView,
    utc_now,
)
from app.services.agent_runtime import (
    AgentRuntime,
    AgentTool,
    DeepSeekAgentPlanner,
    hybrid_search_tool,
)
from app.services.memory import ConversationMessage
from app.services.agentic_tools import AgentRequestScope, bind_agent_request
from app.services.agent_runtime_registry import agent_runtime_registry
from app.services.deepagents_runtime import AgenticModelAnswer
from app.services.providers import (
    DeepSeekProvider,
    GroundingValidationError,
    ModelAnswer,
    ProviderConfigurationError,
    ProviderError,
    structured_provider,
    validate_grounded_answer,
)
from app.config import get_settings
from app.store import store

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger("uvicorn.error")


def _safe_upstream_error(response: Any) -> tuple[str, str]:
    if response is None:
        return "none", "none"
    try:
        payload = response.json()
    except Exception:
        return "unknown", "unavailable"
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if not isinstance(error, Mapping):
        return "unknown", "unavailable"
    code = str(error.get("code") or error.get("type") or "unknown")[:80]
    message = str(error.get("message") or "unavailable")[:300]
    return code, message


def _discard_unrequested_incomplete_visual_blocks(
    answer: AgenticModelAnswer,
    *,
    tool_calls: list[dict[str, Any]],
) -> AgenticModelAnswer:
    """Ignore empty visual placeholders when no visual inspection occurred.

    The model may occasionally emit an ``image`` block with nullable placeholder
    fields after an otherwise valid text-only retrieval answer.  Such a block
    cannot be published and is not evidence.  It is safe to remove only when
    the agent never called ``inspect_visual`` and at least one valid text block
    remains; invented or inspected asset references still reach the hard
    grounding validator and fail closed.
    """

    inspected_visual = any(
        str(item.get("tool") or "") == "inspect_visual" for item in tool_calls
    )
    clean_blocks = []
    removed_count = 0
    for block in answer.blocks:
        incomplete_visual = (
            block.type != "text"
            and (block.markdown is not None or not block.asset_id or not block.alt)
        )
        if incomplete_visual and not inspected_visual:
            removed_count += 1
            continue
        clean_blocks.append(block)
    if not removed_count:
        return answer
    if not clean_blocks or not any(block.type == "text" for block in clean_blocks):
        raise GroundingValidationError(
            "incomplete visual blocks cannot replace the answer"
        )
    logger.info(
        "discarded unrequested incomplete visual placeholders count=%s",
        removed_count,
    )
    return answer.model_copy(update={"blocks": clean_blocks})


@router.get("/threads", response_model=list[ThreadView], operation_id="listThreads")
def list_threads(knowledge_base_id: str | None = None) -> list[ThreadView]:
    items = store.list("threads")
    if knowledge_base_id:
        items = [item for item in items if item["knowledge_base_id"] == knowledge_base_id]
    items.sort(key=lambda item: item["updated_at"], reverse=True)
    return [ThreadView(**item) for item in items]


@router.post(
    "/threads",
    response_model=ThreadView,
    status_code=status.HTTP_201_CREATED,
    operation_id="createThread",
)
async def create_thread(payload: ThreadCreate) -> ThreadView:
    require_item("knowledge_bases", payload.knowledge_base_id, "知识库")
    item = store.create(
        "threads",
        {
            **payload.model_dump(),
            "status": "active",
        },
    )
    store.messages[item["id"]] = []
    store.persist_state()
    await store.memory.ensure_thread(
        thread_id=item["id"],
        user_id="local-user",
        knowledge_base_id=payload.knowledge_base_id,
    )
    return ThreadView(**item)


@router.get(
    "/threads/{thread_id}", response_model=ThreadView, operation_id="getThread"
)
def get_thread(thread_id: str) -> ThreadView:
    return ThreadView(**require_item("threads", thread_id, "会话"))


@router.patch(
    "/threads/{thread_id}", response_model=ThreadView, operation_id="updateThread"
)
def update_thread(thread_id: str, payload: ThreadUpdate) -> ThreadView:
    require_item("threads", thread_id, "会话")
    item = store.update("threads", thread_id, payload.model_dump(exclude_none=True))
    return ThreadView(**item)


@router.delete(
    "/threads/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deleteThread",
)
async def delete_thread(thread_id: str) -> Response:
    require_item("threads", thread_id, "会话")
    runtime = agent_runtime_registry.runtime
    if runtime is not None:
        await runtime.adelete_thread(thread_id=thread_id)
    store.delete("threads", thread_id)
    store.messages.pop(thread_id, None)
    store.persist_state()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/threads/{thread_id}/messages",
    response_model=list[MessageView],
    operation_id="listMessages",
)
def list_messages(thread_id: str) -> list[MessageView]:
    require_item("threads", thread_id, "会话")
    return [MessageView(**message) for message in store.messages.get(thread_id, [])]


def append_message(thread_id: str, role: str, blocks: list, citations: list) -> dict:
    message = {
        "id": str(uuid4()),
        "thread_id": thread_id,
        "role": role,
        "blocks": blocks,
        "citations": citations,
        "created_at": utc_now(),
    }
    store.messages.setdefault(thread_id, []).append(message)
    store.persist_state()
    return message


def _refusal(thread_id: str, message: str) -> dict:
    return append_message(
        thread_id,
        "assistant",
        [TextBlock(markdown=message).model_dump()],
        [],
    )


async def _hydrate_thread_memory(thread_id: str, knowledge_base_id: str) -> None:
    """Restore provider-facing conversation history from the durable snapshot."""

    await store.memory.ensure_thread(
        thread_id=thread_id,
        user_id="local-user",
        knowledge_base_id=knowledge_base_id,
    )
    existing = await store.memory.history(
        thread_id=thread_id,
        user_id="local-user",
        knowledge_base_id=knowledge_base_id,
    )
    if existing:
        return
    for message in store.messages.get(thread_id, []):
        role = message.get("role")
        if role not in {"user", "assistant", "system"}:
            continue
        blocks = message.get("blocks") or []
        if role == "user":
            content = "\n\n".join(
                str(block.get("markdown") or "")
                for block in blocks
                if isinstance(block, Mapping)
            ).strip()
        else:
            content = {
                "blocks": blocks,
                "citations": message.get("citations") or [],
            }
        if not content:
            continue
        await store.memory.append(
            thread_id=thread_id,
            user_id="local-user",
            knowledge_base_id=knowledge_base_id,
            message=ConversationMessage(role=role, content=content),
        )


async def _answer_with_legacy_runtime(
    *, thread: dict[str, Any], thread_id: str, content: str, rag: RagSettings
) -> dict:
    await _hydrate_thread_memory(thread_id, thread["knowledge_base_id"])
    rag = RagSettings(**store.settings["rag"])
    try:
        # The chat model (deepseek-v4-flash) handles both planning and the
        # grounded answer; its structured-output reliability improved markedly
        # after the answer prompt was simplified in DeepSeekProvider.answer.
        chat_provider = DeepSeekProvider.from_env()
        answer_provider = chat_provider
    except ProviderConfigurationError:
        return _refusal(
            thread_id,
            "已经召回到 PDF 线索，但模型服务端凭证尚未配置，"
            "本次不会用模板答案冒充模型回答。请在设置页完成服务端连接后重试。",
        )
    search_tool: AgentTool = hybrid_search_tool(store.retriever, settings=rag)
    runtime = AgentRuntime(
        planner=DeepSeekAgentPlanner(chat_provider),
        answer_provider=answer_provider,
        memory=store.memory,
        tools=[search_tool],
        max_tool_calls=rag.max_tool_calls,
        citation_required=rag.citation_required,
    )
    try:
        result = await runtime.run(
            question=content,
            thread_id=thread_id,
            user_id="local-user",
            knowledge_base_id=thread["knowledge_base_id"],
        )
    except (ProviderError, TimeoutError, ValueError) as exc:
        cause = exc.__cause__
        response = getattr(cause, "response", None)
        logger.warning(
            "grounded answer pipeline stopped thread=%s error_type=%s "
            "cause_type=%s upstream_status=%s",
            thread_id,
            type(exc).__name__,
            type(cause).__name__ if cause is not None else "none",
            getattr(response, "status_code", "none"),
        )
        return _refusal(
            thread_id,
            "本轮问答模型当前不可用，或输出未通过所需校验。"
            "系统没有用模板答案替代，稍后可从当前上下文重试。",
        )
    if (
        result.evidence_count > 0
        and rag.citation_required
        and not result.answer.citations
    ):
        return _refusal(
            thread_id,
            "本轮使用了知识库证据，但回答没有生成可回溯引用，"
            "为避免伪造来源，本次回答已拒绝发布。",
        )
    assistant = append_message(
        thread_id,
        "assistant",
        result.answer.blocks,
        [
            citation.model_dump(mode="json")
            for citation in result.answer.citations
        ],
    )
    return assistant


def _message_content(message: Mapping[str, Any]) -> str:
    blocks = message.get("blocks") or []
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
    return "\n\n".join(parts)


def _validate_deepagent_output(
    raw_answer: Any,
    *,
    tool_calls: list[dict[str, Any]],
    evidence_items: list[Any],
    rag: RagSettings,
) -> tuple[AgenticModelAnswer, ModelAnswer]:
    answer = AgenticModelAnswer.model_validate(raw_answer)
    answer = _discard_unrequested_incomplete_visual_blocks(
        answer,
        tool_calls=tool_calls,
    )
    if answer.mode == "general_knowledge":
        if answer.citation_ids or any(
            block.type != "text" or block.citation_ids
            for block in answer.blocks
        ):
            raise GroundingValidationError(
                "general-knowledge answer contains grounded references"
            )
        citation_required = False
    else:
        if not evidence_items:
            raise GroundingValidationError(
                "grounded answer has no authoritative retrieval evidence"
            )
        citation_required = rag.citation_required
    validated = validate_grounded_answer(
        ModelAnswer(
            blocks=answer.blocks,
            citation_ids=answer.citation_ids,
        ),
        evidence_items,
        citation_required=citation_required,
    )
    return answer, validated


async def _deep_agent_messages(
    *, thread_id: str, content: str
) -> list[dict[str, str]]:
    runtime = agent_runtime_registry.runtime
    assert runtime is not None
    checkpoint = await runtime.aget_checkpoint(thread_id=thread_id)
    values = getattr(checkpoint, "values", None)
    if values is None and isinstance(checkpoint, Mapping):
        values = checkpoint.get("values")
    if isinstance(values, Mapping) and values.get("messages"):
        return [{"role": "user", "content": content}]
    messages: list[dict[str, str]] = []
    # The current user message is already in the durable application log.
    # Seed a fresh LangGraph checkpoint with the bounded existing conversation.
    for item in store.messages.get(thread_id, [])[-20:]:
        role = str(item.get("role") or "")
        text = _message_content(item)
        if role in {"user", "assistant", "system"} and text:
            messages.append({"role": role, "content": text})
    return messages or [{"role": "user", "content": content}]


async def _answer_with_deepagents(
    *, thread: dict[str, Any], thread_id: str, content: str, rag: RagSettings
) -> dict:
    if agent_runtime_registry.runtime is None:
        build = agent_runtime_registry.refresh(store)
        if build.runtime is None:
            logger.warning(
                "deep agents unavailable thread=%s code=%s",
                thread_id,
                build.status.code,
            )
            return _refusal(
                thread_id,
                "Deep Agents 运行时当前不可用，系统没有静默退回旧 RAG。"
                "请检查模型凭证和 Agent 依赖，或显式切换 legacy 回滚开关。",
            )
    runtime = agent_runtime_registry.runtime
    assert runtime is not None
    scope = AgentRequestScope(
        thread_id=thread_id,
        user_id="local-user",
        knowledge_base_id=str(thread["knowledge_base_id"]),
        rag=rag,
    )
    try:
        messages = await _deep_agent_messages(thread_id=thread_id, content=content)
        with bind_agent_request(scope) as ledger:
            result = await asyncio.wait_for(
                runtime.ainvoke(messages=messages, thread_id=thread_id),
                timeout=300,
            )
            raw_answer = (
                result.get("structured_response")
                if isinstance(result, Mapping)
                else None
            )
            answer, validated = _validate_deepagent_output(
                raw_answer,
                tool_calls=ledger.tool_calls,
                evidence_items=ledger.items,
                rag=rag,
            )
            logger.info(
                "deep agents completed thread=%s mode=%s tools=%s "
                "evidence_count=%s citation_count=%s",
                thread_id,
                answer.mode,
                ",".join(
                    str(item.get("tool") or "unknown")
                    for item in ledger.tool_calls
                )
                or "none",
                len(ledger.items),
                len(validated.citations),
            )
    except Exception as exc:
        cause = exc.__cause__
        response = getattr(cause, "response", None) or getattr(exc, "response", None)
        upstream_code, upstream_message = _safe_upstream_error(response)
        local_message = (
            str(exc)[:300]
            if isinstance(exc, (RuntimeError, ValueError, PermissionError))
            else "unavailable"
        )
        logger.warning(
            "deep agents pipeline stopped thread=%s error_type=%s "
            "cause_type=%s upstream_status=%s upstream_code=%s "
            "upstream_message=%s local_message=%s",
            thread_id,
            type(exc).__name__,
            type(cause).__name__ if cause is not None else "none",
            getattr(response, "status_code", "none"),
            upstream_code,
            upstream_message,
            local_message,
        )
        return _refusal(
            thread_id,
            "本轮 Deep Agents 执行失败，或回答未通过证据校验。"
            "系统没有用模板答案或旧 RAG 冒充成功，可从当前会话重试。",
        )
    return append_message(
        thread_id,
        "assistant",
        validated.blocks,
        [citation.model_dump(mode="json") for citation in validated.citations],
    )


async def answer_question(
    thread_id: str, content: str
) -> tuple[dict, dict]:
    thread = require_item("threads", thread_id, "会话")
    user = append_message(
        thread_id, "user", [TextBlock(markdown=content).model_dump()], []
    )
    rag = RagSettings(**store.settings["rag"])
    runtime_mode = get_settings().agent_runtime
    if runtime_mode == "legacy":
        assistant = await _answer_with_legacy_runtime(
            thread=thread,
            thread_id=thread_id,
            content=content,
            rag=rag,
        )
    else:
        assistant = await _answer_with_deepagents(
            thread=thread,
            thread_id=thread_id,
            content=content,
            rag=rag,
        )
    return user, assistant


@router.post(
    "/threads/{thread_id}/messages",
    response_model=MessageView,
    status_code=status.HTTP_201_CREATED,
    operation_id="createMessage",
)
async def create_message(thread_id: str, payload: MessageCreate) -> MessageView:
    _, assistant = await answer_question(thread_id, payload.content)
    return MessageView(**assistant)


def _sse(event_type: str, **payload: Any) -> str:
    event = {"type": event_type, **payload}
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    )


def _decode_partial_json_string(value: str, start: int) -> tuple[str, bool]:
    """Decode the available prefix of a JSON string without inventing bytes."""

    output: list[str] = []
    index = start
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while index < len(value):
        character = value[index]
        if character == '"':
            return "".join(output), True
        if character != "\\":
            output.append(character)
            index += 1
            continue
        if index + 1 >= len(value):
            break
        escape = value[index + 1]
        if escape != "u":
            decoded = escapes.get(escape)
            if decoded is None:
                break
            output.append(decoded)
            index += 2
            continue
        if index + 6 > len(value):
            break
        digits = value[index + 2 : index + 6]
        if not all(character in "0123456789abcdefABCDEF" for character in digits):
            break
        codepoint = int(digits, 16)
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 12 > len(value) or value[index + 6 : index + 8] != "\\u":
                break
            low_digits = value[index + 8 : index + 12]
            if not all(
                character in "0123456789abcdefABCDEF"
                for character in low_digits
            ):
                break
            low = int(low_digits, 16)
            if not 0xDC00 <= low <= 0xDFFF:
                break
            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            index += 12
        else:
            index += 6
        output.append(chr(codepoint))
    return "".join(output), False


_MARKDOWN_FIELD = re.compile(r'"markdown"\s*:\s*"')


def _markdown_preview(value: str) -> str:
    fields: list[str] = []
    for match in _MARKDOWN_FIELD.finditer(value):
        text, _ = _decode_partial_json_string(value, match.end())
        fields.append(text)
    return "\n\n".join(fields)


def _tool_status(name: str) -> str | None:
    statuses = {
        "search_chunks": "正在检索 PDF 文字、表格与图像片段",
        "search_graph": "正在沿知识图谱查找关联证据",
        "inspect_visual": "正在核对命中页面的视觉元素",
        "fetch_evidence": "正在读取召回线索的原始证据",
        "fetch_chunk": "正在读取命中的文档片段",
        "fetch_asset": "正在读取命中的多模态素材",
        "read_wiki_page": "正在读取关联的 LLM Wiki 页面",
        "search_wiki": "正在检索 LLM Wiki",
        "follow_graph_path": "正在沿知识关系进行多跳检索",
        "check_evidence_sufficiency": "正在检查证据是否足以回答",
        "validate_citations": "正在校验回答引用",
    }
    return statuses.get(name)


def _is_answer_tool(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized == "agenticmodelanswer"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and block.get("type") in {
            "text",
            "text_delta",
            "output_text",
        }:
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class _MarkdownStreamProjector:
    """Project structured-output token chunks into provisional Markdown."""

    def __init__(self) -> None:
        self._answer_tool_keys: set[str] = set()
        self._tool_buffers: dict[str, str] = {}
        self._content_buffers: dict[str, str] = {}
        self._emitted = ""

    def feed(self, part: Mapping[str, Any]) -> tuple[list[str], str]:
        data = part.get("data")
        if (
            part.get("type") != "messages"
            or not isinstance(data, (tuple, list))
            or len(data) != 2
        ):
            return [], ""
        message, metadata = data
        metadata = metadata if isinstance(metadata, Mapping) else {}
        message_id = str(
            getattr(message, "id", None)
            or metadata.get("run_id")
            or metadata.get("langgraph_step")
            or "model"
        )
        statuses: list[str] = []
        tool_chunks = getattr(message, "tool_call_chunks", None) or []
        for tool_chunk in tool_chunks:
            if not isinstance(tool_chunk, Mapping):
                continue
            index = tool_chunk.get("index", 0)
            key = f"{message_id}:{index}"
            name = tool_chunk.get("name")
            if isinstance(name, str) and name:
                status = _tool_status(name)
                if status:
                    statuses.append(status)
                if _is_answer_tool(name):
                    self._answer_tool_keys.add(key)
                    statuses.append("证据准备完成，正在组织回答")
            arguments = tool_chunk.get("args")
            if key in self._answer_tool_keys and arguments:
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                self._tool_buffers[key] = (
                    self._tool_buffers.get(key, "") + arguments
                )

        content = _message_text(getattr(message, "content", ""))
        if content:
            existing = self._content_buffers.get(message_id, "")
            candidate = existing + content
            if existing or candidate.lstrip().startswith(("{", "[")):
                self._content_buffers[message_id] = candidate

        previews = [
            preview
            for buffer in (
                *self._tool_buffers.values(),
                *self._content_buffers.values(),
            )
            if (preview := _markdown_preview(buffer))
        ]
        if not previews:
            return statuses, ""
        preview = max(previews, key=len)
        if not preview.startswith(self._emitted):
            return statuses, ""
        delta = preview[len(self._emitted) :]
        self._emitted = preview
        return statuses, delta


def _find_structured_response(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    if value.get("structured_response") is not None:
        return value["structured_response"]
    for nested in value.values():
        found = _find_structured_response(nested)
        if found is not None:
            return found
    return None


async def _stream_deepagents_answer(
    *,
    thread: dict[str, Any],
    thread_id: str,
    content: str,
    rag: RagSettings,
) -> AsyncIterator[dict[str, Any]]:
    if agent_runtime_registry.runtime is None:
        build = agent_runtime_registry.refresh(store)
        if build.runtime is None:
            assistant = _refusal(
                thread_id,
                "Deep Agents 运行时当前不可用，系统没有静默退回旧 RAG。"
                "请检查模型凭证和 Agent 依赖，或显式切换 legacy 回滚开关。",
            )
            yield {
                "type": "answer.completed",
                "message": MessageView(**assistant).model_dump(mode="json"),
            }
            return

    runtime = agent_runtime_registry.runtime
    assert runtime is not None
    scope = AgentRequestScope(
        thread_id=thread_id,
        user_id="local-user",
        knowledge_base_id=str(thread["knowledge_base_id"]),
        rag=rag,
    )
    projector = _MarkdownStreamProjector()
    last_status = ""
    raw_answer: Any = None
    try:
        messages = await _deep_agent_messages(thread_id=thread_id, content=content)
        with bind_agent_request(scope) as ledger:
            async with asyncio.timeout(300):
                async for part in runtime.astream(
                    messages=messages,
                    thread_id=thread_id,
                ):
                    if not isinstance(part, Mapping):
                        continue
                    found = _find_structured_response(
                        part.get("data")
                        if part.get("type") == "updates"
                        else None
                    )
                    if found is not None:
                        raw_answer = found
                    statuses, delta = projector.feed(part)
                    for status_text in statuses:
                        if status_text != last_status:
                            last_status = status_text
                            yield {
                                "type": "agent.status",
                                "status": status_text,
                            }
                    if delta:
                        yield {"type": "answer.delta", "delta": delta}

            if raw_answer is None:
                checkpoint = await runtime.aget_checkpoint(thread_id=thread_id)
                values = getattr(checkpoint, "values", None)
                if values is None and isinstance(checkpoint, Mapping):
                    values = checkpoint.get("values")
                raw_answer = _find_structured_response(values)

            answer, validated = _validate_deepagent_output(
                raw_answer,
                tool_calls=ledger.tool_calls,
                evidence_items=ledger.items,
                rag=rag,
            )
            assistant = append_message(
                thread_id,
                "assistant",
                validated.blocks,
                [
                    citation.model_dump(mode="json")
                    for citation in validated.citations
                ],
            )
            logger.info(
                "deep agents streamed thread=%s mode=%s tools=%s "
                "evidence_count=%s citation_count=%s",
                thread_id,
                answer.mode,
                ",".join(
                    str(item.get("tool") or "unknown")
                    for item in ledger.tool_calls
                )
                or "none",
                len(ledger.items),
                len(validated.citations),
            )
    except Exception as exc:
        cause = exc.__cause__
        response = getattr(cause, "response", None) or getattr(exc, "response", None)
        upstream_code, upstream_message = _safe_upstream_error(response)
        logger.warning(
            "deep agents stream stopped thread=%s error_type=%s "
            "cause_type=%s upstream_status=%s upstream_code=%s "
            "upstream_message=%s",
            thread_id,
            type(exc).__name__,
            type(cause).__name__ if cause is not None else "none",
            getattr(response, "status_code", "none"),
            upstream_code,
            upstream_message,
        )
        assistant = _refusal(
            thread_id,
            "本轮 Deep Agents 执行失败，或回答未通过证据校验。"
            "系统已丢弃未完成的流式草稿，可从当前会话重试。",
        )
        yield {
            "type": "answer.failed",
            "message": MessageView(**assistant).model_dump(mode="json"),
        }
        return

    yield {
        "type": "answer.completed",
        "message": MessageView(**assistant).model_dump(mode="json"),
    }


async def event_stream(
    *,
    thread: dict[str, Any],
    thread_id: str,
    content: str,
) -> AsyncIterator[str]:
    append_message(
        thread_id,
        "user",
        [TextBlock(markdown=content).model_dump()],
        [],
    )
    yield _sse(
        "answer.started",
        status="正在分析问题并决定是否调用知识库",
    )
    rag = RagSettings(**store.settings["rag"])
    if get_settings().agent_runtime == "legacy":
        yield _sse("agent.status", status="正在通过兼容运行时组织回答")
        assistant = await _answer_with_legacy_runtime(
            thread=thread,
            thread_id=thread_id,
            content=content,
            rag=rag,
        )
        yield _sse(
            "answer.completed",
            message=MessageView(**assistant).model_dump(mode="json"),
        )
        return

    async for event in _stream_deepagents_answer(
        thread=thread,
        thread_id=thread_id,
        content=content,
        rag=rag,
    ):
        event_type = str(event.pop("type"))
        yield _sse(event_type, **event)


@router.post(
    "/threads/{thread_id}/messages/stream",
    response_class=StreamingResponse,
    operation_id="streamMessage",
)
async def stream_message(
    thread_id: str, payload: MessageCreate
) -> StreamingResponse:
    thread = require_item("threads", thread_id, "会话")
    return StreamingResponse(
        event_stream(
            thread=thread,
            thread_id=thread_id,
            content=payload.content,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
