"""Versioned LLM Wiki persistence with explicit manual-lock semantics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Protocol

from app.services.graph_repository import EvidenceLineage


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class WikiConclusion:
    id: str
    text: str
    kind: str
    evidence: tuple[EvidenceLineage, ...]
    predicate: str | None = None
    related_entity_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.text.strip() or not self.kind.strip():
            raise ValueError("wiki conclusions require id, kind, and text")
        if not self.evidence:
            raise ValueError("wiki conclusions require PDF-grounded evidence")


@dataclass(frozen=True, slots=True)
class WikiPageDraft:
    id: str
    knowledge_base_id: str
    entity_id: str
    slug: str
    title: str
    summary: str
    sections: dict[str, str]
    conclusions: tuple[WikiConclusion, ...]
    related_page_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.id,
                self.knowledge_base_id,
                self.entity_id,
                self.slug,
                self.title,
            )
        ):
            raise ValueError("wiki page identity, slug, and title must be non-empty")
        if not self.conclusions:
            raise ValueError("a generated wiki page requires grounded conclusions")
        if any(not key.strip() for key in self.sections):
            raise ValueError("wiki section names must be non-empty")


@dataclass(frozen=True, slots=True)
class WikiPageRecord:
    id: str
    knowledge_base_id: str
    entity_id: str
    slug: str
    title: str
    summary: str
    sections: dict[str, str]
    conclusions: tuple[WikiConclusion, ...]
    related_page_ids: tuple[str, ...]
    graph_version: int
    revision: int
    status: str
    locked_fields: frozenset[str] = frozenset()
    updated_at: datetime = field(default_factory=_now)

    @property
    def markdown(self) -> str:
        body = [f"# {self.title}", "", self.summary.strip()]
        for name, content in self.sections.items():
            if content.strip():
                body.extend(("", f"## {name}", "", content.strip()))
        return "\n".join(body).strip() + "\n"


class WikiVersionConflict(RuntimeError):
    pass


class WikiRepository(Protocol):
    def current_version(self, knowledge_base_id: str) -> int: ...

    def list_pages(self, knowledge_base_id: str) -> tuple[WikiPageRecord, ...]: ...

    def get_page(
        self, knowledge_base_id: str, page_id_or_slug: str
    ) -> WikiPageRecord | None: ...

    def commit_generated(
        self,
        knowledge_base_id: str,
        graph_version: int,
        pages: tuple[WikiPageDraft, ...],
        *,
        expected_version: int | None = None,
    ) -> tuple[WikiPageRecord, ...]: ...

    def remove_knowledge_base(self, knowledge_base_id: str) -> None: ...


class InMemoryWikiRepository:
    """Reference repository for incremental generation and human ownership."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._versions: dict[str, int] = {}
        self._pages: dict[str, dict[str, WikiPageRecord]] = {}
        self._history: dict[
            tuple[str, str], list[WikiPageRecord]
        ] = {}

    def current_version(self, knowledge_base_id: str) -> int:
        with self._lock:
            return self._versions.get(knowledge_base_id, 0)

    def list_pages(self, knowledge_base_id: str) -> tuple[WikiPageRecord, ...]:
        with self._lock:
            pages = self._pages.get(knowledge_base_id, {})
            return tuple(deepcopy(pages[key]) for key in sorted(pages))

    def get_page(
        self, knowledge_base_id: str, page_id_or_slug: str
    ) -> WikiPageRecord | None:
        with self._lock:
            pages = self._pages.get(knowledge_base_id, {})
            direct = pages.get(page_id_or_slug)
            if direct is not None:
                return deepcopy(direct)
            result = next(
                (page for page in pages.values() if page.slug == page_id_or_slug), None
            )
            return deepcopy(result)

    def commit_generated(
        self,
        knowledge_base_id: str,
        graph_version: int,
        pages: tuple[WikiPageDraft, ...],
        *,
        expected_version: int | None = None,
    ) -> tuple[WikiPageRecord, ...]:
        if graph_version < 1:
            raise ValueError("wiki publication requires a positive graph version")
        if any(page.knowledge_base_id != knowledge_base_id for page in pages):
            raise ValueError("wiki draft belongs to another knowledge base")
        if len({page.id for page in pages}) != len(pages):
            raise ValueError("wiki publication contains duplicate page IDs")
        with self._lock:
            current_version = self._versions.get(knowledge_base_id, 0)
            if expected_version is not None and current_version != expected_version:
                raise WikiVersionConflict(
                    f"expected wiki version {expected_version}, found {current_version}"
                )
            existing_pages = self._pages.setdefault(knowledge_base_id, {})
            next_pages: dict[str, WikiPageRecord] = {}
            for draft in pages:
                existing = existing_pages.get(draft.id)
                record = self._merge(existing, draft, graph_version)
                next_pages[record.id] = record
                self._append_history(record)
            # Locked pages remain visible even if all machine evidence disappears;
            # generated-only pages are removed on the next graph publication.
            for page_id, existing in existing_pages.items():
                if page_id not in next_pages and existing.locked_fields:
                    retained = replace(
                        existing,
                        graph_version=graph_version,
                        status="locked",
                        revision=existing.revision + 1,
                        updated_at=_now(),
                    )
                    next_pages[page_id] = retained
                    self._append_history(retained)
            self._pages[knowledge_base_id] = next_pages
            self._versions[knowledge_base_id] = current_version + 1
            return tuple(deepcopy(next_pages[key]) for key in sorted(next_pages))

    def remove_knowledge_base(self, knowledge_base_id: str) -> None:
        with self._lock:
            self._pages.pop(knowledge_base_id, None)
            self._versions.pop(knowledge_base_id, None)
            self._history = {
                key: value
                for key, value in self._history.items()
                if key[0] != knowledge_base_id
            }

    def _merge(
        self,
        existing: WikiPageRecord | None,
        draft: WikiPageDraft,
        graph_version: int,
    ) -> WikiPageRecord:
        if existing is None:
            return WikiPageRecord(
                id=draft.id,
                knowledge_base_id=draft.knowledge_base_id,
                entity_id=draft.entity_id,
                slug=draft.slug,
                title=draft.title,
                summary=draft.summary,
                sections=deepcopy(draft.sections),
                conclusions=deepcopy(draft.conclusions),
                related_page_ids=draft.related_page_ids,
                graph_version=graph_version,
                revision=1,
                status="published",
            )
        locks = existing.locked_fields
        sections: dict[str, str] = {}
        for name in dict.fromkeys((*existing.sections, *draft.sections)):
            lock_key = f"section:{name}"
            if lock_key in locks and name in existing.sections:
                sections[name] = existing.sections[name]
            elif name in draft.sections:
                sections[name] = draft.sections[name]
        return WikiPageRecord(
            id=existing.id,
            knowledge_base_id=existing.knowledge_base_id,
            entity_id=existing.entity_id,
            slug=existing.slug,
            title=existing.title if "title" in locks else draft.title,
            summary=existing.summary if "summary" in locks else draft.summary,
            sections=sections,
            conclusions=(
                existing.conclusions
                if "conclusions" in locks
                else deepcopy(draft.conclusions)
            ),
            related_page_ids=(
                existing.related_page_ids
                if "related_page_ids" in locks
                else draft.related_page_ids
            ),
            graph_version=graph_version,
            revision=existing.revision + 1,
            status="locked" if locks else "published",
            locked_fields=locks,
            updated_at=_now(),
        )

    def _append_history(self, record: WikiPageRecord) -> None:
        key = (record.knowledge_base_id, record.id)
        self._history.setdefault(key, []).append(deepcopy(record))
