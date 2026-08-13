"""Process-local registry for the compiled Deep Agents graph and its resources."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.agentic_tools import build_agentic_tools
from app.services.deepagents_runtime import (
    DeepAgentsBuildResult,
    DeepAgentsRuntime,
    create_deepagents_runtime_from_env,
)
from app.services.langgraph_persistence import LangGraphPersistence


@dataclass(slots=True)
class AgentRuntimeRegistry:
    persistence: LangGraphPersistence | None = None
    build: DeepAgentsBuildResult | None = None

    @property
    def runtime(self) -> DeepAgentsRuntime | None:
        return self.build.runtime if self.build is not None else None

    def configure(self, persistence: LangGraphPersistence, store: object) -> None:
        self.persistence = persistence
        self.refresh(store)

    def refresh(self, store: object) -> DeepAgentsBuildResult:
        tools = build_agentic_tools(store)
        self.build = create_deepagents_runtime_from_env(
            tools=tools,
            allowed_tool_names=frozenset(tool.name for tool in tools),
            checkpointer=(
                self.persistence.checkpointer if self.persistence is not None else None
            ),
            store=self.persistence.store if self.persistence is not None else None,
        )
        return self.build

    def clear(self) -> None:
        self.persistence = None
        self.build = None


agent_runtime_registry = AgentRuntimeRegistry()
