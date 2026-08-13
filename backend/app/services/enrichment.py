"""LLM semantic enrichment for cropped multimodal PDF elements.

PaddleOCR remains the authority for element identity and geometry.  The LLM
may describe an element, but an enrichment is publishable only after the echoed
lineage matches the trusted ingestion record exactly.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.assets import ObjectStorage, get_json, put_json
from app.services.ingestion import MULTIMODAL_KINDS, NormalizedElement
from app.services.providers import (
    DeepSeekProvider,
    ProviderUnavailableError,
    VisionInput,
)

ENRICHMENT_VERSION = "deepseek-element-v1"
PAGE_ENRICHMENT_VERSION = "deepseek-page-v1"
SEMANTIC_ARTIFACT_VERSION = "multimodal-semantic-v2"
PENDING_MESSAGE = "Paddle 原始内容待增强"


class EnrichmentError(RuntimeError):
    """A sanitized enrichment failure suitable for a durable job record."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        element_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.element_id = element_id


class EnrichmentBatchError(EnrichmentError):
    """Raised when degraded results are disabled and any element fails."""


class StructuredSemantics(BaseModel):
    """Uniform strict shape that works for figures, tables, and formulae."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    rows: list[str]
    columns: list[str]
    relations: list[str]


class DeepSeekElementOutput(BaseModel):
    """Strict provider output including lineage that must be echoed unchanged."""

    model_config = ConfigDict(extra="forbid")

    element_id: str
    document_id: str
    document_version: int
    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    description: str = Field(min_length=1)
    search_text: str = Field(min_length=1)
    structure: StructuredSemantics
    semantic_tags: list[str]


class DeepSeekPageOutput(BaseModel):
    """Strict page-level semantics without permission to rewrite geometry."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_version: int
    page: int
    summary: str = Field(min_length=1)
    search_text: str = Field(min_length=1)
    section_intent: str = Field(min_length=1)
    element_relations: list[str]
    semantic_tags: list[str]


class StructuredProvider(Protocol):
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
        images: Sequence[VisionInput] = (),
        history: Sequence[Mapping[str, Any]] = (),
        reasoning_effort: Literal["low", "high", "max"] = "high",
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class EnrichmentConfig:
    max_concurrency: int = 3
    allow_degraded: bool = True
    image_detail: Literal["auto", "low", "high"] = "high"
    maximum_asset_bytes: int = 12 * 1024 * 1024
    maximum_page_enrichments: int = 6

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        if self.maximum_asset_bytes < 1:
            raise ValueError("maximum_asset_bytes must be at least one")
        if self.maximum_page_enrichments < 1:
            raise ValueError("maximum_page_enrichments must be at least one")


@dataclass(frozen=True, slots=True)
class ElementEnrichment:
    element_id: str
    document_id: str
    document_version: int
    page: int
    bbox: tuple[float, float, float, float]
    status: Literal["fully_enriched", "pending_enrichment"]
    publishable: bool
    description: str
    search_text: str
    structure: dict[str, Any]
    semantic_tags: list[str]
    cache_key: str | None
    idempotent_replay: bool = False
    error_code: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EnrichmentBatchResult:
    document_id: str
    document_version: int
    eligible_count: int
    enriched_count: int
    pending_count: int
    fully_enriched: bool
    elements: list[ElementEnrichment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PageEnrichment:
    document_id: str
    document_version: int
    page: int
    status: Literal["fully_enriched", "pending_enrichment"]
    publishable: bool
    summary: str
    search_text: str
    section_intent: str
    element_relations: list[str]
    semantic_tags: list[str]
    cache_key: str | None
    idempotent_replay: bool = False
    error_code: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PageEnrichmentBatchResult:
    document_id: str
    document_version: int
    eligible_count: int
    enriched_count: int
    pending_count: int
    fully_enriched: bool
    pages: list[PageEnrichment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_bbox(element: NormalizedElement) -> tuple[float, float, float, float]:
    return tuple(round(float(value), 6) for value in element.bbox_normalized)  # type: ignore[return-value]


def _lineage(
    document_id: str, document_version: int, element: NormalizedElement
) -> dict[str, Any]:
    return {
        "element_id": element.id,
        "document_id": document_id,
        "document_version": document_version,
        "page": element.page,
        "bbox": list(_canonical_bbox(element)),
    }


def _cache_key(document_id: str, document_version: int, element_id: str) -> str:
    return (
        f"enrichment/{document_id}/v{document_version}/"
        f"{ENRICHMENT_VERSION}/{element_id}.json"
    )


def semantic_artifact_key(document_id: str, document_version: int) -> str:
    return (
        f"compiled/{document_id}/v{document_version}/"
        f"{SEMANTIC_ARTIFACT_VERSION}.json"
    )


def _page_cache_key(document_id: str, document_version: int, page: int) -> str:
    return (
        f"enrichment/{document_id}/v{document_version}/"
        f"{PAGE_ENRICHMENT_VERSION}/page-{page:04d}.json"
    )


def _fingerprint(
    *,
    lineage: Mapping[str, Any],
    element: NormalizedElement,
    asset: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(ENRICHMENT_VERSION.encode("utf-8"))
    digest.update(
        json.dumps(
            dict(lineage),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(element.kind.encode("utf-8"))
    digest.update(element.content.encode("utf-8"))
    digest.update(asset)
    return digest.hexdigest()


def _data_url(payload: bytes, content_type: str = "image/png") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _rate_limited(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            return current.response.status_code == 429
        current = current.__cause__ or current.__context__
    return False


def _validate_lineage(
    output: DeepSeekElementOutput,
    *,
    trusted: Mapping[str, Any],
    element_id: str,
) -> None:
    echoed = {
        "element_id": output.element_id,
        "document_id": output.document_id,
        "document_version": output.document_version,
        "page": output.page,
        "bbox": [round(float(value), 6) for value in output.bbox],
    }
    if echoed != dict(trusted):
        raise EnrichmentError(
            "LLM output changed trusted Paddle element lineage",
            code="lineage_mismatch",
            retryable=False,
            element_id=element_id,
        )


def _enriched_result(
    output: DeepSeekElementOutput,
    *,
    cache_key: str,
    replay: bool,
) -> ElementEnrichment:
    return ElementEnrichment(
        element_id=output.element_id,
        document_id=output.document_id,
        document_version=output.document_version,
        page=output.page,
        bbox=tuple(round(float(value), 6) for value in output.bbox),  # type: ignore[arg-type]
        status="fully_enriched",
        publishable=True,
        description=output.description.strip(),
        search_text=output.search_text.strip(),
        structure=output.structure.model_dump(),
        semantic_tags=list(dict.fromkeys(tag.strip() for tag in output.semantic_tags if tag.strip())),
        cache_key=cache_key,
        idempotent_replay=replay,
    )


def _pending_result(
    *,
    document_id: str,
    document_version: int,
    element: NormalizedElement,
    error: EnrichmentError,
) -> ElementEnrichment:
    original = element.content.strip()
    return ElementEnrichment(
        element_id=element.id,
        document_id=document_id,
        document_version=document_version,
        page=element.page,
        bbox=_canonical_bbox(element),
        status="pending_enrichment",
        publishable=False,
        description=PENDING_MESSAGE,
        search_text=original or PENDING_MESSAGE,
        structure={
            "summary": original,
            "rows": [],
            "columns": [],
            "relations": [],
        },
        semantic_tags=[],
        cache_key=None,
        error_code=error.code,
        retryable=error.retryable,
    )


class DeepSeekElementEnricher:
    """Concurrency-limited, cache-backed semantic element enrichment."""

    def __init__(
        self,
        *,
        storage: ObjectStorage,
        provider: DeepSeekProvider | StructuredProvider,
        config: EnrichmentConfig | None = None,
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.config = config or EnrichmentConfig()

    async def enrich_elements(
        self,
        *,
        document_id: str,
        document_version: int,
        elements: Sequence[NormalizedElement],
    ) -> EnrichmentBatchResult:
        eligible = [
            element for element in elements if element.kind in MULTIMODAL_KINDS
        ]
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def run(element: NormalizedElement) -> ElementEnrichment:
            async with semaphore:
                try:
                    return await self._enrich_one(
                        document_id=document_id,
                        document_version=document_version,
                        element=element,
                    )
                except EnrichmentError as exc:
                    if not self.config.allow_degraded:
                        raise EnrichmentBatchError(
                            str(exc),
                            code=exc.code,
                            retryable=exc.retryable,
                            element_id=element.id,
                        ) from exc
                    return _pending_result(
                        document_id=document_id,
                        document_version=document_version,
                        element=element,
                        error=exc,
                    )

        results = await asyncio.gather(*(run(element) for element in eligible))
        enriched_count = sum(item.publishable for item in results)
        pending_count = len(results) - enriched_count
        return EnrichmentBatchResult(
            document_id=document_id,
            document_version=document_version,
            eligible_count=len(eligible),
            enriched_count=enriched_count,
            pending_count=pending_count,
            fully_enriched=pending_count == 0,
            elements=results,
        )

    async def enrich_pages(
        self,
        *,
        document_id: str,
        document_version: int,
        pages: Sequence[Mapping[str, Any]],
        elements: Sequence[NormalizedElement],
    ) -> PageEnrichmentBatchResult:
        """Understand whole-page visual context while preserving Paddle lineage."""

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        by_page: dict[int, list[NormalizedElement]] = {}
        for element in elements:
            by_page.setdefault(element.page, []).append(element)
        # Element-level LLM vision covers every materialized figure/table/formula.
        # Whole-page vision is reserved for the visually richest pages where
        # cross-element context adds value, keeping compilation cost bounded.
        candidate_pages = [
            page
            for page in pages
            if any(
                element.kind in MULTIMODAL_KINDS
                for element in by_page.get(int(page.get("page") or 0), [])
            )
        ]
        candidate_pages.sort(
            key=lambda page: (
                -sum(
                    element.kind in MULTIMODAL_KINDS
                    for element in by_page.get(int(page.get("page") or 0), [])
                ),
                int(page.get("page") or 0),
            )
        )
        candidate_pages = candidate_pages[: self.config.maximum_page_enrichments]

        async def run(page_record: Mapping[str, Any]) -> PageEnrichment:
            async with semaphore:
                page_number = int(page_record.get("page") or 0)
                try:
                    return await self._enrich_page(
                        document_id=document_id,
                        document_version=document_version,
                        page_record=page_record,
                        elements=by_page.get(page_number, []),
                    )
                except EnrichmentError as exc:
                    if not self.config.allow_degraded:
                        raise EnrichmentBatchError(
                            str(exc),
                            code=exc.code,
                            retryable=exc.retryable,
                            element_id=f"page:{page_number}",
                        ) from exc
                    return PageEnrichment(
                        document_id=document_id,
                        document_version=document_version,
                        page=page_number,
                        status="pending_enrichment",
                        publishable=False,
                        summary=PENDING_MESSAGE,
                        search_text=" ".join(
                            item.content.strip()
                            for item in by_page.get(page_number, [])
                            if item.content.strip()
                        )[:6000]
                        or PENDING_MESSAGE,
                        section_intent="待增强",
                        element_relations=[],
                        semantic_tags=[],
                        cache_key=None,
                        error_code=exc.code,
                        retryable=exc.retryable,
                    )

        results = await asyncio.gather(*(run(page) for page in candidate_pages))
        enriched_count = sum(item.publishable for item in results)
        return PageEnrichmentBatchResult(
            document_id=document_id,
            document_version=document_version,
            eligible_count=len(results),
            enriched_count=enriched_count,
            pending_count=len(results) - enriched_count,
            fully_enriched=enriched_count == len(results),
            pages=results,
        )

    async def _enrich_page(
        self,
        *,
        document_id: str,
        document_version: int,
        page_record: Mapping[str, Any],
        elements: Sequence[NormalizedElement],
    ) -> PageEnrichment:
        page = int(page_record.get("page") or 0)
        image_key = str(page_record.get("image_key") or "")
        if page < 1 or not image_key:
            raise EnrichmentError(
                "compiled page has no trusted page number or image",
                code="invalid_page_lineage",
                retryable=False,
                element_id=f"page:{page}",
            )
        try:
            image = self.storage.get_bytes(image_key)
        except Exception as exc:
            raise EnrichmentError(
                "compiled page image is unavailable",
                code="page_image_unavailable",
                retryable=True,
                element_id=f"page:{page}",
            ) from exc
        if not image or len(image) > self.config.maximum_asset_bytes:
            raise EnrichmentError(
                "compiled page image is empty or exceeds the provider limit",
                code="page_image_invalid",
                retryable=False,
                element_id=f"page:{page}",
            )

        trusted = {
            "document_id": document_id,
            "document_version": document_version,
            "page": page,
        }
        paddle_context = [
            {
                "element_id": item.id,
                "kind": item.kind,
                "bbox": list(_canonical_bbox(item)),
                "content": item.content[:1200],
            }
            for item in elements
        ]
        digest = hashlib.sha256()
        digest.update(PAGE_ENRICHMENT_VERSION.encode())
        digest.update(json.dumps(trusted, sort_keys=True).encode())
        digest.update(json.dumps(paddle_context, ensure_ascii=False, sort_keys=True).encode())
        digest.update(image)
        fingerprint = digest.hexdigest()
        cache_key = _page_cache_key(document_id, document_version, page)
        cached = self._read_page_cache(
            cache_key=cache_key,
            fingerprint=fingerprint,
            trusted=trusted,
        )
        if cached is not None:
            return self._page_result(cached, cache_key=cache_key, replay=True)

        try:
            raw = await self.provider.complete_structured(
                system_prompt=(
                    "你负责理解 PaddleOCR 已完成结构定位的 PDF 整页。"
                    "只能补充页面主题、检索文本和元素之间可见关系，不能改写"
                    " document_id、document_version、page，也不能臆造页面外信息。"
                ),
                user_text=(
                    "请结合整页图像和 Paddle 元素理解页面。受信任 lineage 必须原样返回：\n"
                    + json.dumps(
                        {**trusted, "paddle_elements": paddle_context},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                schema_name="pdf_page_semantic_enrichment",
                schema=DeepSeekPageOutput.model_json_schema(),
                images=[VisionInput(url=_data_url(image), detail=self.config.image_detail)],
                reasoning_effort="low",
            )
            output = DeepSeekPageOutput.model_validate(raw)
        except ValidationError as exc:
            raise EnrichmentError(
                "LLM page output does not match the enrichment schema",
                code="invalid_provider_output",
                retryable=True,
                element_id=f"page:{page}",
            ) from exc
        except Exception as exc:
            limited = _rate_limited(exc)
            raise EnrichmentError(
                "LLM rate limit reached"
                if limited
                else "LLM page enrichment unavailable",
                code="rate_limited" if limited else "provider_unavailable",
                retryable=True,
                element_id=f"page:{page}",
            ) from exc
        echoed = {
            "document_id": output.document_id,
            "document_version": output.document_version,
            "page": output.page,
        }
        if echoed != trusted:
            raise EnrichmentError(
                "LLM output changed trusted Paddle page lineage",
                code="lineage_mismatch",
                retryable=False,
                element_id=f"page:{page}",
            )
        put_json(
            self.storage,
            cache_key,
            {
                "enrichment_version": PAGE_ENRICHMENT_VERSION,
                "input_fingerprint": fingerprint,
                "output": output.model_dump(),
            },
        )
        return self._page_result(output, cache_key=cache_key, replay=False)

    @staticmethod
    def _page_result(
        output: DeepSeekPageOutput, *, cache_key: str, replay: bool
    ) -> PageEnrichment:
        return PageEnrichment(
            document_id=output.document_id,
            document_version=output.document_version,
            page=output.page,
            status="fully_enriched",
            publishable=True,
            summary=output.summary.strip(),
            search_text=output.search_text.strip(),
            section_intent=output.section_intent.strip(),
            element_relations=list(dict.fromkeys(output.element_relations)),
            semantic_tags=list(dict.fromkeys(output.semantic_tags)),
            cache_key=cache_key,
            idempotent_replay=replay,
        )

    def _read_page_cache(
        self,
        *,
        cache_key: str,
        fingerprint: str,
        trusted: Mapping[str, Any],
    ) -> DeepSeekPageOutput | None:
        if not self.storage.exists(cache_key):
            return None
        try:
            payload = get_json(self.storage, cache_key)
            if (
                payload.get("enrichment_version") != PAGE_ENRICHMENT_VERSION
                or payload.get("input_fingerprint") != fingerprint
            ):
                return None
            output = DeepSeekPageOutput.model_validate(payload.get("output"))
            if {
                "document_id": output.document_id,
                "document_version": output.document_version,
                "page": output.page,
            } != dict(trusted):
                return None
            return output
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
            return None

    async def _enrich_one(
        self,
        *,
        document_id: str,
        document_version: int,
        element: NormalizedElement,
    ) -> ElementEnrichment:
        if not element.asset_key:
            raise EnrichmentError(
                "multimodal element has no materialized crop",
                code="asset_not_materialized",
                retryable=False,
                element_id=element.id,
            )
        try:
            asset = self.storage.get_bytes(element.asset_key)
        except Exception as exc:
            raise EnrichmentError(
                "multimodal crop is unavailable in object storage",
                code="asset_unavailable",
                retryable=True,
                element_id=element.id,
            ) from exc
        if not asset:
            raise EnrichmentError(
                "multimodal crop is empty",
                code="asset_empty",
                retryable=False,
                element_id=element.id,
            )
        if len(asset) > self.config.maximum_asset_bytes:
            raise EnrichmentError(
                "multimodal crop exceeds the configured provider limit",
                code="asset_too_large",
                retryable=False,
                element_id=element.id,
            )

        trusted = _lineage(document_id, document_version, element)
        fingerprint = _fingerprint(
            lineage=trusted, element=element, asset=asset
        )
        cache_key = _cache_key(document_id, document_version, element.id)
        cached = self._read_cache(
            cache_key=cache_key,
            fingerprint=fingerprint,
            trusted=trusted,
            element_id=element.id,
        )
        if cached is not None:
            return _enriched_result(cached, cache_key=cache_key, replay=True)

        prompt_payload = {
            **trusted,
            "kind": element.kind,
            "paddle_content": element.content,
        }
        try:
            raw = await self.provider.complete_structured(
                system_prompt=(
                    "你负责增强 PaddleOCR 已定位的 PDF 多模态元素。"
                    "只描述裁图中可见信息，不猜测；严格原样回传 element_id、document_id、"
                    "document_version、page、bbox。description 用于阅读，search_text 用于"
                    "语义检索；structure 分别填写摘要、表格行列或图中关系；semantic_tags"
                    "只给图中可证实的短标签。"
                ),
                user_text=(
                    "请增强下列受信任元素。lineage 字段必须逐字逐数值原样返回：\n"
                    + json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
                ),
                schema_name="pdf_element_semantic_enrichment",
                schema=DeepSeekElementOutput.model_json_schema(),
                images=[
                    VisionInput(
                        url=_data_url(asset),
                        detail=self.config.image_detail,
                    )
                ],
                # The task is bounded visual transcription against trusted
                # lineage, not open-ended reasoning.  Low effort materially
                # reduces batch latency while the strict schema and lineage
                # validator retain the publication safety boundary.
                reasoning_effort="low",
            )
        except ProviderUnavailableError as exc:
            limited = _rate_limited(exc)
            raise EnrichmentError(
                "LLM rate limit reached" if limited else "LLM enrichment unavailable",
                code="rate_limited" if limited else "provider_unavailable",
                retryable=True,
                element_id=element.id,
            ) from exc
        except (httpx.HTTPError, TimeoutError) as exc:
            limited = _rate_limited(exc)
            raise EnrichmentError(
                "LLM rate limit reached" if limited else "LLM enrichment unavailable",
                code="rate_limited" if limited else "provider_unavailable",
                retryable=True,
                element_id=element.id,
            ) from exc
        except Exception as exc:
            raise EnrichmentError(
                "LLM enrichment failed",
                code="provider_failure",
                retryable=True,
                element_id=element.id,
            ) from exc

        try:
            output = DeepSeekElementOutput.model_validate(raw)
        except ValidationError as exc:
            raise EnrichmentError(
                "LLM output does not match the enrichment schema",
                code="invalid_provider_output",
                retryable=True,
                element_id=element.id,
            ) from exc
        _validate_lineage(output, trusted=trusted, element_id=element.id)
        put_json(
            self.storage,
            cache_key,
            {
                "enrichment_version": ENRICHMENT_VERSION,
                "input_fingerprint": fingerprint,
                "output": output.model_dump(),
            },
        )
        return _enriched_result(output, cache_key=cache_key, replay=False)

    def _read_cache(
        self,
        *,
        cache_key: str,
        fingerprint: str,
        trusted: Mapping[str, Any],
        element_id: str,
    ) -> DeepSeekElementOutput | None:
        if not self.storage.exists(cache_key):
            return None
        try:
            payload = get_json(self.storage, cache_key)
            if (
                not isinstance(payload, dict)
                or payload.get("enrichment_version") != ENRICHMENT_VERSION
                or payload.get("input_fingerprint") != fingerprint
            ):
                return None
            output = DeepSeekElementOutput.model_validate(payload.get("output"))
            _validate_lineage(output, trusted=trusted, element_id=element_id)
            return output
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
            # A malformed/stale cache is a miss.  It is never treated as a
            # successful enrichment and will be safely replaced after the
            # provider returns a validated result.
            return None
