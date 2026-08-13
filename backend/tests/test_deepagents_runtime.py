from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from app.services import deepagents_runtime as runtime_module
from app.services.deepagents_runtime import (
    DeepAgentsRuntime,
    ToolNotAllowedError,
    ToolPolicyError,
    create_deepagents_runtime,
)


class SearchInput(BaseModel):
    query: str


class SearchResult(BaseModel):
    hits: list[str]


class FakeApplicationTool:
    name = "hybrid_search"
    description = "search"
    input_model = SearchInput

    async def execute(self, arguments: dict[str, Any]) -> SearchResult:
        return SearchResult(hits=[arguments["query"]])


class FakeStructuredTool:
    calls: list[dict[str, Any]] = []

    @classmethod
    def from_function(cls, **kwargs: Any) -> dict[str, Any]:
        cls.calls.append(kwargs)
        return {"name": kwargs["name"], "coroutine": kwargs["coroutine"]}


class FakeChatOpenAI:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class FakeCommand:
    def __init__(self, *, resume: Any) -> None:
        self.resume = resume


class FakeAgentMiddleware:
    pass


class FakeFilesystemPermission:
    def __init__(self, **kwargs: Any) -> None:
        self.operations = kwargs["operations"]
        self.paths = kwargs["paths"]
        self.mode = kwargs["mode"]


class FakeGraph:
    def __init__(self) -> None:
        self.invocations: list[tuple[Any, dict[str, Any]]] = []
        self.streams: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
        self.updates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, payload: Any, *, config: dict[str, Any]) -> dict[str, Any]:
        self.invocations.append((payload, config))
        return {"ok": True}

    async def astream(
        self,
        payload: Any,
        *,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self.streams.append((payload, config, kwargs))
        yield {"type": "updates", "ns": (), "data": {"model": {"ok": True}}}

    async def aget_state(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"config": config, "interrupts": [{"id": "approval-1"}]}

    async def aget_state_history(
        self, config: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        for index in range(3):
            yield {"index": index, "config": config}

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.updates.append((config, values, kwargs))
        return {"updated": True}


@dataclass
class FakeDependencies:
    create_deep_agent: Any
    chat_openai: Any = FakeChatOpenAI
    structured_tool: Any = FakeStructuredTool
    in_memory_saver: Any = object
    command: Any = FakeCommand
    agent_middleware: Any = FakeAgentMiddleware
    filesystem_permission: Any = FakeFilesystemPermission
    versions: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.versions is None:
            self.versions = {
                "deepagents": "0.7.0",
                "langchain": "1.3.14",
                "langgraph": "1.2.10",
                "langchain-openai": "1.4.1",
            }


def test_missing_credential_is_diagnostic_and_does_not_load_packages() -> None:
    loaded = False

    def loader() -> Any:
        nonlocal loaded
        loaded = True
        raise AssertionError("must not load optional packages without a credential")

    result = create_deepagents_runtime(
        tools=[],
        api_key=None,
        dependency_loader=loader,
    )

    assert result.runtime is None
    assert result.status.code == "missing_credential"
    assert result.status.available is False
    assert loaded is False


def test_missing_packages_return_unavailable_without_mocking_success() -> None:
    def loader() -> Any:
        raise runtime_module._DependencyError(
            "missing_dependency", "optional packages are absent"
        )

    result = create_deepagents_runtime(
        tools=[],
        api_key="test-only-key",
        dependency_loader=loader,
    )

    assert result.runtime is None
    assert result.status.code == "missing_dependency"
    assert result.status.detail == "optional packages are absent"
    assert "test-only-key" not in repr(result)


def test_builds_deepseek_deep_agent_with_checkpoint_interrupt_and_allowlist() -> None:
    FakeChatOpenAI.calls.clear()
    FakeStructuredTool.calls.clear()
    graph = FakeGraph()
    create_calls: list[dict[str, Any]] = []
    checkpointer = object()
    store = object()

    def fake_create(**kwargs: Any) -> FakeGraph:
        create_calls.append(kwargs)
        return graph

    dependencies = FakeDependencies(create_deep_agent=fake_create)
    result = create_deepagents_runtime(
        tools=[FakeApplicationTool()],
        api_key="test-only-key",
        base_url="https://api.moonshot.cn/v1/",
        interrupt_on={"hybrid_search": True},
        checkpointer=checkpointer,
        store=store,
        dependency_loader=lambda: dependencies,
    )

    assert result.status.available is True
    assert result.runtime is not None
    assert FakeChatOpenAI.calls[0]["model"] == "deepseek-v4-flash"
    assert FakeChatOpenAI.calls[0]["temperature"] == 1
    assert FakeChatOpenAI.calls[0]["base_url"] == "https://api.moonshot.cn/v1"
    assert FakeChatOpenAI.calls[0]["api_key"].get_secret_value() == "test-only-key"
    assert create_calls[0]["checkpointer"] is checkpointer
    assert create_calls[0]["store"] is store
    assert create_calls[0]["interrupt_on"] == {"hybrid_search": True}
    assert create_calls[0]["tools"][0]["name"] == "hybrid_search"
    assert create_calls[0]["permissions"][0].mode == "deny"
    assert create_calls[0]["subagents"] == []
    assert result.runtime.interrupt_on == {"hybrid_search": True}
    assert "test-only-key" not in repr(result.status)


def test_runtime_invocation_and_checkpoint_interfaces() -> None:
    graph = FakeGraph()
    status = runtime_module.DeepAgentsStatus(
        available=True,
        code="available",
        detail="ready",
    )
    runtime = DeepAgentsRuntime(
        graph=graph,
        command_type=FakeCommand,
        status=status,
        interrupt_on={"hybrid_search": True},
        allowed_tool_names=frozenset({"hybrid_search"}),
    )

    async def exercise() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        await runtime.ainvoke(
            messages=[{"role": "user", "content": "问题"}],
            thread_id="thread-1",
        )
        streamed = [
            part
            async for part in runtime.astream(
                messages=[{"role": "user", "content": "流式问题"}],
                thread_id="thread-1",
            )
        ]
        checkpoint = await runtime.aget_checkpoint(thread_id="thread-1")
        return checkpoint, streamed

    checkpoint, streamed = asyncio.run(exercise())

    assert graph.invocations[0][1] == {
        "configurable": {"thread_id": "thread-1"}
    }
    assert graph.streams[0] == (
        {"messages": [{"role": "user", "content": "流式问题"}]},
        {"configurable": {"thread_id": "thread-1"}},
        {"stream_mode": ["messages", "updates"], "version": "v2"},
    )
    assert streamed[0]["type"] == "updates"
    assert checkpoint["interrupts"][0]["id"] == "approval-1"


def test_allowlist_filters_visible_tools_and_blocks_injected_execution() -> None:
    middleware = runtime_module._build_allowlist_middleware(
        FakeAgentMiddleware, frozenset({"hybrid_search"})
    )

    class Request:
        def __init__(self, tools: list[Any]) -> None:
            self.tools = tools

        def override(self, *, tools: list[Any]) -> Request:
            return Request(tools)

    filtered = middleware.wrap_model_call(
        Request([{"name": "hybrid_search"}, {"name": "write_file"}]),
        lambda request: request.tools,
    )
    assert filtered == [{"name": "hybrid_search"}]

    blocked_request = type(
        "ToolRequest",
        (),
        {"tool_call": {"name": "write_file"}},
    )()
    with pytest.raises(ToolNotAllowedError, match="write_file"):
        middleware.wrap_tool_call(blocked_request, lambda request: request)


def test_rejects_tools_and_interrupt_rules_outside_product_policy() -> None:
    class DangerousTool:
        name = "shell"

    with pytest.raises(ToolPolicyError, match="not allowlisted"):
        create_deepagents_runtime(
            tools=[DangerousTool()],
            api_key="test-only-key",
            dependency_loader=lambda: None,
        )

    with pytest.raises(ToolPolicyError, match="interrupt rules"):
        create_deepagents_runtime(
            tools=[],
            api_key="test-only-key",
            interrupt_on={"write_file": True},
            dependency_loader=lambda: None,
        )
