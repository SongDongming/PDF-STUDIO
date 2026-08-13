import contextlib
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    from app.services.agent_runtime_registry import agent_runtime_registry
    from app.services.langgraph_persistence import open_langgraph_persistence
    from app.store import store

    logger = logging.getLogger("uvicorn.error")
    runtime = get_settings()
    stack = contextlib.AsyncExitStack()
    persistence = None
    try:
        persistence = await stack.enter_async_context(
            open_langgraph_persistence(runtime.database_url)
        )
    except Exception:
        # A Postgres outage at boot must not crash the API.  Fall back to the
        # in-memory checkpointer and keep serving (checkpoints are lost on
        # restart, which health/settings surfaces make visible).
        logger.exception(
            "LangGraph persistence unavailable; using in-memory checkpointer"
        )
        persistence = None
    try:
        agent_runtime_registry.configure(persistence, store)
        try:
            yield
        finally:
            agent_runtime_registry.clear()
    finally:
        await stack.aclose()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="多模态 PDF 知识库、问答、Wiki 与知识图谱 API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Last-Event-ID"],
)
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "pdfwiki-api", "docs": "/docs", "health": f"{settings.api_prefix}/health"}
