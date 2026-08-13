"""Optional Deep Agents/LangGraph runtime adapter.

The rest of the application can import this module without installing the
optional agent stack.  Construction returns a diagnostic status when packages,
supported versions, or the Moonshot credential are unavailable; it never
silently falls back to a mock agent.

Deep Agents injects middleware-owned tools in addition to caller supplied
tools.  The adapter therefore enforces the product tool allowlist twice:

* only allowlisted caller tools are accepted at construction time;
* middleware filters model-visible tools and rejects any out-of-policy tool
  call at execution time.

Filesystem operations are denied as a second defence.  The application RAG
tools, rather than a general filesystem backend, are the only supported data
access path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from app.services.providers import ModelAnswerBlock


PRODUCT_TOOL_ALLOWLIST = frozenset(
    {
        "search_chunks",
        "search_graph",
        "inspect_visual",
        "fetch_evidence",
        "hybrid_search",
        "graph_search",
        "search_wiki",
        "read_wiki_page",
        "fetch_chunk",
        "fetch_asset",
        "open_pdf_region",
        "follow_graph_path",
        "check_evidence_sufficiency",
        "validate_citations",
    }
)

_SUPPORTED_VERSION_PREFIXES = {
    "deepagents": "0.7.",
    "langchain": "1.3.",
    "langgraph": "1.2.",
    "langchain-openai": "1.4.",
}
class ToolPolicyError(ValueError):
    """Raised when a caller attempts to expose an out-of-policy tool."""


class ToolNotAllowedError(PermissionError):
    """Raised if a model attempts to execute a hidden or injected tool."""


@dataclass(frozen=True)
class DeepAgentsStatus:
    available: bool
    code: Literal[
        "available",
        "missing_credential",
        "missing_dependency",
        "incompatible_dependency",
        "initialization_failed",
    ]
    detail: str
    versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeepAgentsBuildResult:
    status: DeepAgentsStatus
    runtime: DeepAgentsRuntime | None = None


@dataclass(frozen=True)
class _Dependencies:
    create_deep_agent: Any
    chat_openai: Any
    structured_tool: Any
    in_memory_saver: Any
    command: Any
    agent_middleware: Any
    filesystem_permission: Any
    versions: Mapping[str, str]


class _DependencyError(RuntimeError):
    def __init__(self, code: str, detail: str, versions: Mapping[str, str] | None = None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.versions = versions or {}


def _load_dependencies() -> _Dependencies:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for distribution, supported_prefix in _SUPPORTED_VERSION_PREFIXES.items():
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            missing.append(distribution)
            continue
        versions[distribution] = installed
        if not installed.startswith(supported_prefix):
            raise _DependencyError(
                "incompatible_dependency",
                (
                    f"{distribution} {installed} is outside the supported "
                    f"{supported_prefix}x line"
                ),
                versions,
            )
    if missing:
        raise _DependencyError(
            "missing_dependency",
            f"optional agent dependencies are not installed: {', '.join(missing)}",
            versions,
        )

    try:
        deepagents = import_module("deepagents")
        langchain_openai = import_module("langchain_openai")
        langchain_tools = import_module("langchain_core.tools")
        checkpoint_memory = import_module("langgraph.checkpoint.memory")
        langgraph_types = import_module("langgraph.types")
        middleware_types = import_module("langchain.agents.middleware.types")
        filesystem_middleware = import_module("deepagents.middleware.filesystem")
    except (ImportError, ModuleNotFoundError) as exc:
        module_name = getattr(exc, "name", None) or "unknown module"
        raise _DependencyError(
            "missing_dependency",
            f"optional agent module could not be imported: {module_name}",
            versions,
        ) from None

    return _Dependencies(
        create_deep_agent=deepagents.create_deep_agent,
        chat_openai=langchain_openai.ChatOpenAI,
        structured_tool=langchain_tools.StructuredTool,
        in_memory_saver=checkpoint_memory.InMemorySaver,
        command=langgraph_types.Command,
        agent_middleware=middleware_types.AgentMiddleware,
        filesystem_permission=filesystem_middleware.FilesystemPermission,
        versions=versions,
    )


def _tool_name(tool: Any) -> str:
    if isinstance(tool, Mapping):
        name = tool.get("name")
    else:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if not isinstance(name, str) or not name.strip():
        raise ToolPolicyError("every Deep Agents tool must have a non-empty name")
    return name.strip()


def _validate_tool_policy(
    tools: Sequence[Any], allowed_tool_names: frozenset[str]
) -> None:
    if not allowed_tool_names:
        raise ToolPolicyError("the Deep Agents tool allowlist must be explicit and non-empty")
    unsupported_allowlist = allowed_tool_names - PRODUCT_TOOL_ALLOWLIST
    if unsupported_allowlist:
        raise ToolPolicyError(
            "unsupported names in tool allowlist: "
            + ", ".join(sorted(unsupported_allowlist))
        )

    names = [_tool_name(tool) for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ToolPolicyError("duplicate tool names: " + ", ".join(duplicates))
    blocked = sorted(set(names) - allowed_tool_names)
    if blocked:
        raise ToolPolicyError("tools are not allowlisted: " + ", ".join(blocked))


def _normalise_tool_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, Mapping):
        return dict(result)
    return result


def _as_langchain_tool(tool: Any, structured_tool: Any) -> Any:
    """Convert the application's AgentTool protocol to StructuredTool.

    Existing LangChain tools, callables, and provider tool dictionaries are
    passed through unchanged.  A duck-typed application tool has ``name``,
    ``description``, ``input_model`` and async ``execute`` attributes.
    """

    required = ("name", "description", "input_model", "execute")
    if not all(hasattr(tool, attr) for attr in required):
        return tool

    async def execute(**kwargs: Any) -> Any:
        return _normalise_tool_result(await tool.execute(kwargs))

    return structured_tool.from_function(
        coroutine=execute,
        name=tool.name,
        description=tool.description,
        args_schema=tool.input_model,
        response_format=getattr(tool, "response_format", "content"),
    )


def _build_allowlist_middleware(
    agent_middleware: type[Any], allowed_tool_names: frozenset[str]
) -> Any:
    class ToolAllowlistMiddleware(agent_middleware):
        def _filter_request(self, request: Any) -> Any:
            filtered = [
                tool for tool in request.tools if _tool_name(tool) in allowed_tool_names
            ]
            return request.override(tools=filtered)

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            return handler(self._filter_request(request))

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            return await handler(self._filter_request(request))

        @staticmethod
        def _assert_allowed(request: Any) -> None:
            name = request.tool_call.get("name")
            if name not in allowed_tool_names:
                raise ToolNotAllowedError(
                    f"Deep Agents attempted a non-allowlisted tool: {name or '<missing>'}"
                )

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            self._assert_allowed(request)
            return handler(request)

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            self._assert_allowed(request)
            return await handler(request)

    return ToolAllowlistMiddleware()


class DeepAgentsRuntime:
    """Thin async facade over a compiled Deep Agents LangGraph."""

    def __init__(
        self,
        *,
        graph: Any,
        command_type: Any,
        status: DeepAgentsStatus,
        interrupt_on: Mapping[str, Any],
        allowed_tool_names: frozenset[str],
    ) -> None:
        self._graph = graph
        self._command_type = command_type
        self.status = status
        self.interrupt_on = dict(interrupt_on)
        self.allowed_tool_names = allowed_tool_names

    @staticmethod
    def thread_config(thread_id: str) -> dict[str, Any]:
        if not thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        return {"configurable": {"thread_id": thread_id}}

    async def ainvoke(
        self,
        *,
        messages: Sequence[Any],
        thread_id: str,
    ) -> Any:
        if not messages:
            raise ValueError("messages must be non-empty")
        return await self._graph.ainvoke(
            {"messages": list(messages)},
            config=self.thread_config(thread_id),
        )

    async def astream(
        self,
        *,
        messages: Sequence[Any],
        thread_id: str,
    ) -> Any:
        """Yield the graph's native v2 token and state-update stream."""

        if not messages:
            raise ValueError("messages must be non-empty")
        async for part in self._graph.astream(
            {"messages": list(messages)},
            config=self.thread_config(thread_id),
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            yield part

    async def aget_checkpoint(self, *, thread_id: str) -> Any:
        return await self._graph.aget_state(self.thread_config(thread_id))

    async def adelete_thread(self, *, thread_id: str) -> None:
        checkpointer = getattr(self._graph, "checkpointer", None)
        delete = getattr(checkpointer, "adelete_thread", None)
        if delete is not None:
            await delete(thread_id)


class AgenticModelAnswer(BaseModel):
    """Structured final answer produced by the Deep Agents control loop."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["grounded", "partial", "general_knowledge"]
    blocks: list[ModelAnswerBlock] = Field(min_length=1)
    citation_ids: list[str]
    coverage_gaps: list[str]

    @model_validator(mode="after")
    def validate_mode_contract(self) -> "AgenticModelAnswer":
        if self.mode == "general_knowledge":
            if self.citation_ids or any(
                block.type != "text" or block.citation_ids
                for block in self.blocks
            ):
                raise ValueError(
                    "general_knowledge mode cannot contain PDF citations or assets"
                )
        if self.mode == "partial" and not any(
            value.strip() for value in self.coverage_gaps
        ):
            raise ValueError("partial mode requires at least one coverage gap")
        if len(set(self.citation_ids)) != len(self.citation_ids):
            raise ValueError("citation_ids must not contain duplicates")
        return self


def create_deepagents_runtime(
    *,
    tools: Sequence[Any],
    api_key: str | SecretStr | None,
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com/v1",
    allowed_tool_names: frozenset[str] = PRODUCT_TOOL_ALLOWLIST,
    interrupt_on: Mapping[str, Any] | None = None,
    checkpointer: Any = None,
    store: Any = None,
    response_format: Any = AgenticModelAnswer,
    system_prompt: str = (
        "你是企业级多模态 PDF 知识库的 Deep Agent。你必须根据问题决定是否检索，"
        "不是每轮都调用工具：通用解释、写作、翻译、计算、寒暄等直接使用模型知识，"
        "mode=general_knowledge，且不得生成 PDF 引用或素材块。用户要求依据 PDF、"
        "知识库、指定文档、页码、图表、表格、公式或私有内容时调用 search_chunks；"
        "跨文档比较、因果、依赖和多跳关系可同时调用 search_graph；只有需要核对"
        "视觉细节时才对已召回 asset_id 调用 inspect_visual。检索无结果时停止搜索，"
        "可以 mode=general_knowledge 回答并说明未使用知识库；部分证据可回答时使用"
        " mode=partial 并填写 coverage_gaps，但 partial 只能回答证据覆盖部分，"
        "不得在同一回答中掺入无引用的通用知识推断。grounded/partial 的 citation_id 和"
        " asset_id 必须逐字来自工具结果。图片、表格和公式块要穿插在引用它的正文之间，"
        "不能统一堆到答案末尾；没有成功调用 inspect_visual 时只能生成 text 块，严禁"
        "生成 asset_id 为空的图片占位块。每轮最多调用一次 search_chunks 和一次 search_graph；"
        "拿到结果后直接作答，不要反复改写查询。回答以解决当前问题为准，避免扩展成长篇教程。"
    ),
    dependency_loader: Any = _load_dependencies,
) -> DeepAgentsBuildResult:
    """Build the real runtime or return a sanitized unavailable result."""

    raw_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    if not raw_key:
        return DeepAgentsBuildResult(
            status=DeepAgentsStatus(
                available=False,
                code="missing_credential",
                detail="Moonshot API credential is not configured",
            )
        )

    _validate_tool_policy(tools, allowed_tool_names)
    requested_interrupts = dict(interrupt_on or {})
    unsupported_interrupts = sorted(set(requested_interrupts) - allowed_tool_names)
    if unsupported_interrupts:
        raise ToolPolicyError(
            "interrupt rules reference non-allowlisted tools: "
            + ", ".join(unsupported_interrupts)
        )

    try:
        dependencies = dependency_loader()
    except _DependencyError as exc:
        return DeepAgentsBuildResult(
            status=DeepAgentsStatus(
                available=False,
                code=exc.code,  # type: ignore[arg-type]
                detail=exc.detail,
                versions=exc.versions,
            )
        )

    try:
        model_client = dependencies.chat_openai(
            model=model,
            api_key=SecretStr(raw_key),
            base_url=base_url.rstrip("/"),
            # The DeepSeek chat endpoint requires temperature=1.
            temperature=1,
            max_retries=4,
            timeout=180,
            max_completion_tokens=4096,
            reasoning_effort="low",
            use_responses_api=False,
        )
        converted_tools = [
            _as_langchain_tool(tool, dependencies.structured_tool) for tool in tools
        ]
        middleware = _build_allowlist_middleware(
            dependencies.agent_middleware, allowed_tool_names
        )
        effective_checkpointer = (
            checkpointer if checkpointer is not None else dependencies.in_memory_saver()
        )
        graph = dependencies.create_deep_agent(
            model=model_client,
            tools=converted_tools,
            system_prompt=system_prompt,
            middleware=[middleware],
            subagents=[],
            skills=None,
            memory=None,
            permissions=[
                dependencies.filesystem_permission(
                    operations=["read", "write"],
                    paths=["/**"],
                    mode="deny",
                )
            ],
            interrupt_on=requested_interrupts or None,
            checkpointer=effective_checkpointer,
            store=store,
            response_format=response_format,
            name="multimodal_pdf_rag",
        )
    except Exception as exc:
        return DeepAgentsBuildResult(
            status=DeepAgentsStatus(
                available=False,
                code="initialization_failed",
                detail=f"Deep Agents initialization failed ({type(exc).__name__})",
                versions=dependencies.versions,
            )
        )

    status = DeepAgentsStatus(
        available=True,
        code="available",
        detail="Deep Agents runtime is available",
        versions=dependencies.versions,
    )
    return DeepAgentsBuildResult(
        status=status,
        runtime=DeepAgentsRuntime(
            graph=graph,
            command_type=dependencies.command,
            status=status,
            interrupt_on=requested_interrupts,
            allowed_tool_names=allowed_tool_names,
        ),
    )


def create_deepagents_runtime_from_env(
    *, tools: Sequence[Any], **kwargs: Any
) -> DeepAgentsBuildResult:
    """Environment composition helper without logging or returning the secret."""

    return create_deepagents_runtime(
        tools=tools,
        api_key=os.environ.get("MOONSHOT_API_KEY"),
        model=os.environ.get("MOONSHOT_CHAT_MODEL", "deepseek-v4-flash"),
        base_url=os.environ.get("MOONSHOT_BASE_URL", "https://api.deepseek.com/v1"),
        **kwargs,
    )
