from fastapi import APIRouter

from app.api.routes import (
    chat,
    documents,
    graph,
    health,
    jobs,
    knowledge_bases,
    retrieval,
    settings,
    wiki,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(knowledge_bases.router)
api_router.include_router(documents.router)
api_router.include_router(jobs.router)
api_router.include_router(retrieval.router)
api_router.include_router(chat.router)
api_router.include_router(wiki.router)
api_router.include_router(graph.router)
api_router.include_router(settings.router)
