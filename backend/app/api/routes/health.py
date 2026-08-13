import os
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter

from app import __version__
from app.config import get_settings
from app.schemas import HealthResponse
from app.store import store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, operation_id="getHealth")
def health() -> HealthResponse:
    settings = get_settings()
    try:
        deepagents_version = version("deepagents")
    except PackageNotFoundError:
        deepagents_version = "missing"
    return HealthResponse(
        status="ok",
        service="pdfwiki-api",
        version=__version__,
        environment=settings.env,
        dependencies={
            "metadata_repository": (
                type(store._persistence).__name__
                if store._persistence is not None
                else "in_memory"
            ),
            "database": "configured" if settings.database_url else "not_configured",
            "object_storage": type(store.storage).__name__,
            "graph": type(store.graph_repository).__name__,
            "ocr": "configured" if settings.ocr_base_url else "not_configured",
            "embedding": (
                "configured"
                if os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
                else "not_configured"
            ),
            "deepseek_chat": (
                "configured" if os.getenv("MOONSHOT_API_KEY") else "not_configured"
            ),
            "agent_framework": f"deepagents:{deepagents_version}",
        },
    )
