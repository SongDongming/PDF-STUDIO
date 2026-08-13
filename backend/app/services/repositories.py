from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class DocumentSource:
    id: str
    knowledge_base_id: str
    filename: str
    title: str
    object_key: str
    sha256: str
    active_version: int


class CompilationRepository(Protocol):
    def get_document(self, document_id: str) -> DocumentSource: ...

    def mark_stage(
        self,
        document_id: str,
        *,
        stage: str,
        progress: int,
        job_id: str | None = None,
    ) -> None: ...

    def mark_succeeded(
        self,
        document_id: str,
        *,
        version: int,
        result: dict[str, Any],
        job_id: str | None = None,
    ) -> None: ...

    def mark_failed(
        self,
        document_id: str,
        *,
        stage: str,
        code: str,
        message: str,
        retryable: bool = False,
        job_id: str | None = None,
    ) -> None: ...


class RepositoryError(RuntimeError):
    pass
