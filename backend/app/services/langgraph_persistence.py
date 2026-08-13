"""Application-lifespan LangGraph checkpoint and long-term store resources."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True, slots=True)
class LangGraphPersistence:
    checkpointer: Any
    store: Any
    durable: bool
    backend: str
    detail: str


def _postgres_dsn(database_url: str) -> str | None:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {
        "postgres",
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
    }:
        return None
    return urlunsplit(
        ("postgresql", parsed.netloc, parsed.path, parsed.query, parsed.fragment)
    )


@asynccontextmanager
async def open_langgraph_persistence(
    database_url: str,
) -> AsyncIterator[LangGraphPersistence]:
    """Use Postgres in deployed environments and an explicit dev fallback."""

    requested = os.getenv("APP_LANGGRAPH_DATABASE_URL") or database_url
    dsn = _postgres_dsn(requested)
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("app.services.deepagents_runtime", "AgenticModelAnswer"),
            ("app.services.providers", "ModelAnswerBlock"),
        ]
    )
    if dsn is None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        yield LangGraphPersistence(
            checkpointer=InMemorySaver(serde=serializer),
            store=InMemoryStore(),
            durable=False,
            backend="memory",
            detail=(
                "APP_DATABASE_URL/APP_LANGGRAPH_DATABASE_URL is not PostgreSQL; "
                "using process-local LangGraph persistence"
            ),
        )
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres import AsyncPostgresStore

    async with (
        AsyncPostgresSaver.from_conn_string(dsn, serde=serializer) as checkpointer,
        AsyncPostgresStore.from_conn_string(dsn) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        yield LangGraphPersistence(
            checkpointer=checkpointer,
            store=store,
            durable=True,
            backend="postgresql",
            detail="LangGraph checkpoints and cross-thread memory use PostgreSQL",
        )
