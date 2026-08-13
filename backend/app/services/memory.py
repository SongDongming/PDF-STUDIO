"""Thread-scoped conversation memory and namespaced long-term memory."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryIsolationError(RuntimeError):
    """Raised when a caller attempts to cross a memory ownership boundary."""


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system", "tool"]
    content: Any
    created_at: datetime = Field(default_factory=_utc_now)


class ThreadMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    user_id: str
    knowledge_base_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)


class LongTermMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    namespace: str
    key: str
    value: dict[str, Any]
    updated_at: datetime = Field(default_factory=_utc_now)


class MemoryService:
    """Concurrency-safe reference store.

    The service deliberately requires ``user_id`` on every access.  PostgreSQL
    or LangGraph-backed adapters can preserve this contract without changing
    the agent runtime.
    """

    def __init__(self, *, max_messages_per_thread: int = 100) -> None:
        if max_messages_per_thread < 2:
            raise ValueError("max_messages_per_thread must be at least 2")
        self.max_messages_per_thread = max_messages_per_thread
        self._threads: dict[str, ThreadMemory] = {}
        self._long_term: dict[tuple[str, str, str], LongTermMemory] = {}
        self._lock = asyncio.Lock()

    async def ensure_thread(
        self, *, thread_id: str, user_id: str, knowledge_base_id: str
    ) -> ThreadMemory:
        async with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                thread = ThreadMemory(
                    thread_id=thread_id,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                self._threads[thread_id] = thread
            self._assert_owner(thread, user_id, knowledge_base_id)
            return thread.model_copy(deep=True)

    @staticmethod
    def _assert_owner(
        thread: ThreadMemory, user_id: str, knowledge_base_id: str | None = None
    ) -> None:
        if thread.user_id != user_id:
            raise MemoryIsolationError("thread does not belong to this user")
        if (
            knowledge_base_id is not None
            and thread.knowledge_base_id != knowledge_base_id
        ):
            raise MemoryIsolationError("thread does not belong to this knowledge base")

    async def append(
        self,
        *,
        thread_id: str,
        user_id: str,
        knowledge_base_id: str,
        message: ConversationMessage,
    ) -> None:
        async with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                thread = ThreadMemory(
                    thread_id=thread_id,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                )
                self._threads[thread_id] = thread
            self._assert_owner(thread, user_id, knowledge_base_id)
            thread.messages.append(message)
            if len(thread.messages) > self.max_messages_per_thread:
                thread.messages = thread.messages[-self.max_messages_per_thread :]

    async def history(
        self,
        *,
        thread_id: str,
        user_id: str,
        knowledge_base_id: str,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        if limit < 1:
            raise ValueError("history limit must be positive")
        async with self._lock:
            thread = self._threads.get(thread_id)
            if thread is None:
                return []
            self._assert_owner(thread, user_id, knowledge_base_id)
            return [
                message.model_copy(deep=True) for message in thread.messages[-limit:]
            ]

    async def recall(
        self,
        *,
        user_id: str,
        namespace: str,
        keys: Sequence[str] | None = None,
    ) -> list[LongTermMemory]:
        key_filter = set(keys) if keys is not None else None
        async with self._lock:
            records = [
                record.model_copy(deep=True)
                for (owner, record_namespace, key), record in self._long_term.items()
                if owner == user_id
                and record_namespace == namespace
                and (key_filter is None or key in key_filter)
            ]
        return sorted(records, key=lambda item: (item.key, item.updated_at))
