from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from app.services.assets import (
    ObjectStorage,
    get_json,
    image_is_visually_blank,
    put_json,
)
from app.services.ocr_client import OCRClient, OCRServiceError
from app.services.repositories import CompilationRepository

PIPELINE_VERSION = "multimodal-pdf-v2"
TRUSTED_LAYOUT_CONTRACT = "paddle-v3-layout-lineage-v1"
MULTIMODAL_KINDS = {"figure", "table", "formula"}


class CompilationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        code: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RenderedPage:
    page_number: int
    width: int
    height: int
    image: bytes
    content_type: str = "image/png"


class PageRenderer(Protocol):
    def render(self, pdf: bytes) -> list[RenderedPage]: ...


class PyMuPDFPageRenderer:
    """Render immutable PDF bytes while preserving a stable pixel coordinate space."""

    def __init__(self, dpi: int = 144) -> None:
        self.dpi = dpi

    def render(self, pdf: bytes) -> list[RenderedPage]:
        try:
            import fitz
        except ImportError as exc:
            raise CompilationError(
                "PyMuPDF is required for PDF page rendering",
                stage="rendering",
                code="renderer_dependency_missing",
            ) from exc
        try:
            document = fitz.open(stream=pdf, filetype="pdf")
        except Exception as exc:
            raise CompilationError(
                "source object is not a readable PDF",
                stage="rendering",
                code="invalid_pdf",
            ) from exc
        pages: list[RenderedPage] = []
        try:
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
                pages.append(
                    RenderedPage(
                        page_number=index + 1,
                        width=pixmap.width,
                        height=pixmap.height,
                        image=pixmap.tobytes("png"),
                    )
                )
        finally:
            document.close()
        if not pages:
            raise CompilationError(
                "PDF contains no pages",
                stage="rendering",
                code="empty_pdf",
            )
        return pages

    @staticmethod
    def render_page(
        pdf: bytes, *, page_number: int, dpi: int
    ) -> RenderedPage:
        """Render one page at an alternate DPI for a page-local OCR fallback."""

        try:
            import fitz
        except ImportError as exc:
            raise CompilationError(
                "PyMuPDF is required for PDF page rendering",
                stage="rendering",
                code="renderer_dependency_missing",
            ) from exc
        try:
            document = fitz.open(stream=pdf, filetype="pdf")
            if page_number < 1 or page_number > document.page_count:
                raise ValueError("page number is outside the PDF")
            pixmap = document[page_number - 1].get_pixmap(dpi=dpi, alpha=False)
            return RenderedPage(
                page_number=page_number,
                width=pixmap.width,
                height=pixmap.height,
                image=pixmap.tobytes("png"),
            )
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(
                "unable to render the OCR fallback page",
                stage="rendering",
                code="renderer_fallback_failed",
            ) from exc
        finally:
            if "document" in locals():
                document.close()

    @staticmethod
    def extract_text_layer_layout(
        pdf: bytes,
        *,
        page_number: int,
        rendered_width: int,
        rendered_height: int,
    ) -> dict[str, Any]:
        """Recover grounded boxes from a PDF text layer after a VLM parse failure."""

        try:
            import fitz
        except ImportError as exc:
            raise OCRServiceError(
                "PyMuPDF is required for the PDF text-layer fallback",
                code="ocr_fallback_dependency_missing",
                retryable=False,
            ) from exc
        try:
            document = fitz.open(stream=pdf, filetype="pdf")
            page = document[page_number - 1]
            page_dict = page.get_text("dict")
            scale_x = rendered_width / float(page.rect.width)
            scale_y = rendered_height / float(page.rect.height)
            elements: list[dict[str, Any]] = []
            for order, block in enumerate(page_dict.get("blocks", []), start=1):
                bbox = block.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                scaled_bbox = [
                    float(bbox[0]) * scale_x,
                    float(bbox[1]) * scale_y,
                    float(bbox[2]) * scale_x,
                    float(bbox[3]) * scale_y,
                ]
                if block.get("type") == 1 and isinstance(block.get("image"), bytes):
                    elements.append(
                        {
                            "type": "figure",
                            "text": "PDF 内嵌图像",
                            "bbox": scaled_bbox,
                            "order": order,
                            "image_base64": base64.b64encode(block["image"]).decode(),
                            "recognizer": "pdf_text_layer_fallback",
                        }
                    )
                    continue
                lines: list[str] = []
                for line in block.get("lines", []):
                    text = "".join(
                        str(span.get("text", "")) for span in line.get("spans", [])
                    ).rstrip()
                    if text:
                        lines.append(text)
                content = "\n".join(lines).strip()
                if content:
                    elements.append(
                        {
                            "type": "text",
                            "text": content,
                            "bbox": scaled_bbox,
                            "order": order,
                            "recognizer": "pdf_text_layer_fallback",
                        }
                    )
            if not elements:
                raise OCRServiceError(
                    "PaddleOCR failed and the PDF page has no usable text layer",
                    code="ocr_no_text_layer_fallback",
                    retryable=False,
                )
            return {
                "page_number": page_number,
                "width": rendered_width,
                "height": rendered_height,
                "elements": elements,
                "recognizer": "pdf_text_layer_fallback",
            }
        except OCRServiceError:
            raise
        except Exception as exc:
            raise OCRServiceError(
                "unable to extract the PDF text-layer fallback",
                code="ocr_text_layer_fallback_failed",
                retryable=False,
            ) from exc
        finally:
            if "document" in locals():
                document.close()


def _downsample_page(page: RenderedPage, *, ratio: float = 5 / 6) -> RenderedPage:
    """Return a lower-resolution page while preserving its visual coordinate space.

    PaddleOCR-VL occasionally emits an invalid structured sequence for a dense
    page at the default 144 DPI even though the same page succeeds at 120 DPI.
    The fallback remains page-local and the returned dimensions become the
    canonical dimensions for both OCR boxes and the comparison viewer.
    """

    if not 0 < ratio < 1:
        raise ValueError("downsample ratio must be between zero and one")
    try:
        from PIL import Image
    except ImportError as exc:
        raise OCRServiceError(
            "Pillow is required for the OCR resolution fallback",
            code="ocr_fallback_dependency_missing",
            retryable=False,
        ) from exc
    try:
        source = Image.open(io.BytesIO(page.image)).convert("RGB")
        width = max(1, round(source.width * ratio))
        height = max(1, round(source.height * ratio))
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        resized.save(output, format="PNG")
    except Exception as exc:
        raise OCRServiceError(
            "unable to downsample a page for the OCR fallback",
            code="ocr_fallback_failed",
            retryable=False,
        ) from exc
    return RenderedPage(
        page_number=page.page_number,
        width=width,
        height=height,
        image=output.getvalue(),
    )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("page dimensions must be positive")
        return (
            round(self.x0 / width, 6),
            round(self.y0 / height, 6),
            round(self.x1 / width, 6),
            round(self.y1 / height, 6),
        )

    def clamped(self, width: int, height: int) -> "BoundingBox":
        x0 = min(max(self.x0, 0.0), float(width))
        x1 = min(max(self.x1, 0.0), float(width))
        y0 = min(max(self.y0, 0.0), float(height))
        y1 = min(max(self.y1, 0.0), float(height))
        return BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


@dataclass(slots=True)
class NormalizedElement:
    id: str
    page: int
    order: int
    kind: str
    label: str
    content: str
    bbox: BoundingBox
    bbox_normalized: tuple[float, float, float, float]
    polygon: list[tuple[float, float]]
    polygon_normalized: list[tuple[float, float]]
    confidence: float | None
    asset_id: str | None = None
    asset_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _embedded_asset: bytes | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_embedded_asset", None)
        payload["bbox"] = asdict(self.bbox)
        return payload


@dataclass(frozen=True, slots=True)
class StructuredChunk:
    id: str
    document_id: str
    version: int
    ordinal: int
    markdown: str
    page_start: int
    page_end: int
    element_ids: list[str]
    asset_ids: list[str]
    heading_path: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompilationResult:
    document_id: str
    version: int
    source_sha256: str
    page_count: int
    element_count: int
    asset_count: int
    chunk_count: int
    manifest_key: str
    markdown_key: str
    elements_key: str
    chunks_key: str
    idempotent_replay: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def _canonical_json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _kind_for(label: str) -> str:
    value = label.strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in value for token in ("image", "figure", "chart", "diagram")):
        return "figure"
    if "table" in value:
        return "table"
    if any(token in value for token in ("formula", "equation")):
        return "formula"
    if any(token in value for token in ("title", "heading", "header")):
        return "heading"
    if any(token in value for token in ("list", "reference")):
        return "list"
    return "text"


def _bbox_from(raw: Any, width: int, height: int) -> BoundingBox:
    values: list[float]
    if isinstance(raw, dict):
        aliases = [
            ("x0", "y0", "x1", "y1"),
            ("left", "top", "right", "bottom"),
            ("x", "y", "width", "height"),
        ]
        for names in aliases:
            if all(name in raw for name in names):
                values = [float(raw[name]) for name in names]
                if names == ("x", "y", "width", "height"):
                    values = [
                        values[0],
                        values[1],
                        values[0] + values[2],
                        values[1] + values[3],
                    ]
                break
        else:
            raise ValueError("bbox object has no supported coordinate fields")
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        values = [float(item) for item in raw[:4]]
    else:
        raise ValueError("bbox must contain four coordinates")

    if max(abs(item) for item in values) <= 1.0:
        values = [
            values[0] * width,
            values[1] * height,
            values[2] * width,
            values[3] * height,
        ]
    return BoundingBox(*values).clamped(width, height)


def _polygon_from(
    raw: Any, bbox: BoundingBox, width: int, height: int
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    points: list[tuple[float, float]] = []
    if isinstance(raw, (list, tuple)):
        if raw and all(isinstance(item, (int, float)) for item in raw):
            iterator = iter(raw)
            points = [(float(x), float(y)) for x, y in zip(iterator, iterator)]
        else:
            for item in raw:
                if isinstance(item, dict) and {"x", "y"} <= item.keys():
                    points.append((float(item["x"]), float(item["y"])))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    points.append((float(item[0]), float(item[1])))
    if not points:
        points = [
            (bbox.x0, bbox.y0),
            (bbox.x1, bbox.y0),
            (bbox.x1, bbox.y1),
            (bbox.x0, bbox.y1),
        ]
    if max(max(abs(x), abs(y)) for x, y in points) <= 1.0:
        points = [(x * width, y * height) for x, y in points]
    clamped = [
        (min(max(x, 0.0), width), min(max(y, 0.0), height)) for x, y in points
    ]
    normalized = [
        (round(x / width, 6), round(y / height, 6)) for x, y in clamped
    ]
    return clamped, normalized


def _decode_embedded_asset(block: dict[str, Any], resources: dict[str, Any]) -> bytes | None:
    for key in ("image_base64", "asset_base64", "crop_base64"):
        encoded = block.get(key)
        if isinstance(encoded, str):
            encoded = encoded.partition(",")[2] if encoded.startswith("data:") else encoded
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error):
                return None
    resource_key = block.get("image_path") or block.get("asset_path")
    encoded = resources.get(resource_key) if resource_key else None
    if isinstance(encoded, str):
        encoded = encoded.partition(",")[2] if encoded.startswith("data:") else encoded
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            return None
    return None


def _extract_blocks(raw: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "parsing_res_list",
        "parsing_results",
        "elements",
        "blocks",
        "layout",
    ):
        blocks = raw.get(key)
        if isinstance(blocks, list):
            return [item for item in blocks if isinstance(item, dict)]
    return []


def normalize_page_elements(
    *,
    document_id: str,
    page: RenderedPage,
    raw: dict[str, Any],
) -> list[NormalizedElement]:
    resources = raw.get("images") if isinstance(raw.get("images"), dict) else {}
    elements: list[NormalizedElement] = []
    for fallback_order, block in enumerate(_extract_blocks(raw), start=1):
        label = str(
            block.get("block_label")
            or block.get("label")
            or block.get("type")
            or "text"
        )
        content = str(
            block.get("block_content")
            or block.get("content")
            or block.get("text")
            or block.get("markdown")
            or ""
        ).strip()
        bbox_raw = (
            block.get("block_bbox")
            or block.get("bbox")
            or block.get("box")
            or block.get("coordinate")
        )
        if bbox_raw is None:
            continue
        try:
            bbox = _bbox_from(bbox_raw, page.width, page.height)
        except (TypeError, ValueError):
            continue
        polygon, polygon_normalized = _polygon_from(
            block.get("polygon") or block.get("poly"), bbox, page.width, page.height
        )
        order = int(block.get("block_order") or block.get("order") or fallback_order)
        kind = _kind_for(label)
        confidence_raw = (
            block.get("confidence")
            if block.get("confidence") is not None
            else block.get("score")
        )
        try:
            confidence = (
                min(max(float(confidence_raw), 0.0), 1.0)
                if confidence_raw is not None
                else None
            )
        except (TypeError, ValueError):
            confidence = None
        element_id = _stable_id(
            "el",
            document_id,
            page.page_number,
            order,
            label,
            *bbox.normalized(page.width, page.height),
            content,
        )
        known = {
            "block_label",
            "label",
            "type",
            "block_content",
            "content",
            "text",
            "markdown",
            "block_bbox",
            "bbox",
            "box",
            "coordinate",
            "polygon",
            "poly",
            "block_order",
            "order",
            "confidence",
            "score",
            "image_base64",
            "asset_base64",
            "crop_base64",
        }
        elements.append(
            NormalizedElement(
                id=element_id,
                page=page.page_number,
                order=order,
                kind=kind,
                label=label,
                content=content,
                bbox=bbox,
                bbox_normalized=bbox.normalized(page.width, page.height),
                polygon=polygon,
                polygon_normalized=polygon_normalized,
                confidence=confidence,
                metadata={key: value for key, value in block.items() if key not in known},
                _embedded_asset=_decode_embedded_asset(block, resources),
            )
        )
    return sorted(elements, key=lambda item: (item.order, item.bbox.y0, item.bbox.x0))


def _crop_page(page: RenderedPage, bbox: BoundingBox) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CompilationError(
            "Pillow is required to crop multimodal PDF elements",
            stage="materializing_assets",
            code="cropper_dependency_missing",
        ) from exc
    try:
        image = Image.open(io.BytesIO(page.image))
        x0, y0, x1, y1 = (
            int(bbox.x0),
            int(bbox.y0),
            max(int(bbox.x1), int(bbox.x0) + 1),
            max(int(bbox.y1), int(bbox.y0) + 1),
        )
        crop = image.crop((x0, y0, x1, y1))
        output = io.BytesIO()
        crop.save(output, format="PNG")
        return output.getvalue()
    except Exception as exc:
        raise CompilationError(
            "unable to crop a multimodal element from the rendered page",
            stage="materializing_assets",
            code="asset_crop_failed",
        ) from exc


def _asset_tag(element: NormalizedElement) -> str:
    assert element.asset_id
    bbox = ",".join(f"{value:.6f}" for value in element.bbox_normalized)
    asset_type = "image" if element.kind == "figure" else element.kind
    return (
        f'<asset id="{element.asset_id}" type="{asset_type}" '
        f'page="{element.page}" bbox="{bbox}">'
    )


def element_markdown(element: NormalizedElement) -> str:
    if element.kind == "heading":
        return f"## {element.content or '未命名章节'}"
    if element.kind == "formula":
        content = element.content.strip("$ \n")
        return f"{_asset_tag(element)}\n\n$$\n{content}\n$$"
    if element.kind == "table":
        return f"{_asset_tag(element)}\n\n{element.content or '表格'}"
    if element.kind == "figure":
        return f"{_asset_tag(element)}\n\n{element.content or '图片'}"
    return element.content


def build_rich_markdown(
    title: str, pages: list[RenderedPage], elements: list[NormalizedElement]
) -> str:
    by_page: dict[int, list[NormalizedElement]] = {}
    for element in elements:
        by_page.setdefault(element.page, []).append(element)
    lines = [f"# {title}", ""]
    for page in pages:
        lines.extend([f'<!-- page:{page.page_number} -->', ""])
        for element in by_page.get(page.page_number, []):
            markdown = element_markdown(element)
            if markdown:
                lines.extend([markdown, ""])
    return "\n".join(lines).strip() + "\n"


def build_structured_chunks(
    *,
    document_id: str,
    version: int,
    elements: list[NormalizedElement],
    target_characters: int,
) -> list[StructuredChunk]:
    chunks: list[StructuredChunk] = []
    current: list[NormalizedElement] = []
    current_size = 0
    heading_path: list[str] = []
    current_headings: list[str] = []

    def flush() -> None:
        nonlocal current, current_size, current_headings
        if not current:
            return
        markdown = "\n\n".join(
            part for element in current if (part := element_markdown(element))
        )
        if not markdown:
            current = []
            current_size = 0
            return
        ordinal = len(chunks) + 1
        element_ids = [element.id for element in current]
        asset_ids = [
            element.asset_id for element in current if element.asset_id is not None
        ]
        chunks.append(
            StructuredChunk(
                id=_stable_id(
                    "ch",
                    document_id,
                    version,
                    ordinal,
                    *element_ids,
                    markdown,
                ),
                document_id=document_id,
                version=version,
                ordinal=ordinal,
                markdown=markdown,
                page_start=min(element.page for element in current),
                page_end=max(element.page for element in current),
                element_ids=element_ids,
                asset_ids=asset_ids,
                heading_path=list(current_headings),
            )
        )
        current = []
        current_size = 0
        current_headings = list(heading_path)

    for element in elements:
        markdown = element_markdown(element)
        if element.kind == "heading":
            flush()
            if element.content:
                heading_path[:] = [element.content]
            current_headings = list(heading_path)
        projected = current_size + len(markdown) + (2 if current else 0)
        if current and projected > target_characters:
            flush()
        current.append(element)
        current_size += len(markdown) + (2 if current_size else 0)
        if element.kind in MULTIMODAL_KINDS:
            # Assets stay atomic and are never split away from their semantic text.
            flush()
    flush()
    return chunks


class DocumentCompiler:
    def __init__(
        self,
        *,
        storage: ObjectStorage,
        renderer: PageRenderer,
        ocr: OCRClient,
        repository: CompilationRepository,
        render_dpi: int = 144,
        chunk_target_characters: int = 1800,
    ) -> None:
        self.storage = storage
        self.renderer = renderer
        self.ocr = ocr
        self.repository = repository
        self.render_dpi = render_dpi
        self.chunk_target_characters = chunk_target_characters

    def compile(
        self,
        document_id: str,
        version: int,
        *,
        job_id: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> CompilationResult:
        stage = "loading_source"

        def update(next_stage: str, progress: int) -> None:
            nonlocal stage
            stage = next_stage
            self.repository.mark_stage(
                document_id,
                stage=next_stage,
                progress=progress,
                job_id=job_id,
            )
            if progress_callback:
                progress_callback(next_stage, progress)

        try:
            source = self.repository.get_document(document_id)
            prefix = f"compiled/{source.id}/v{version}"
            manifest_key = f"{prefix}/manifest.json"
            compiler_config = {
                "pipeline_version": PIPELINE_VERSION,
                "trusted_layout_contract": TRUSTED_LAYOUT_CONTRACT,
                "render_dpi": self.render_dpi,
                "chunk_target_characters": self.chunk_target_characters,
            }
            compiler_hash = _canonical_json_hash(compiler_config)
            if self.storage.exists(manifest_key):
                prior = get_json(self.storage, manifest_key)
                if (
                    prior.get("source_sha256") == source.sha256
                    and prior.get("compiler_config_hash") == compiler_hash
                ):
                    result = self._result_from_manifest(
                        prior, manifest_key, idempotent_replay=True
                    )
                    self.repository.mark_succeeded(
                        document_id,
                        version=version,
                        result=result.to_dict(),
                        job_id=job_id,
                    )
                    return result
                raise CompilationError(
                    "the requested version already contains different source or compiler settings",
                    stage="loading_source",
                    code="version_conflict",
                )

            update("loading_source", 2)
            try:
                pdf = self.storage.get_bytes(source.object_key)
            except Exception as exc:
                raise CompilationError(
                    "source PDF is not available in object storage",
                    stage=stage,
                    code="source_not_found",
                ) from exc
            actual_sha = hashlib.sha256(pdf).hexdigest()
            if actual_sha != source.sha256:
                raise CompilationError(
                    "source PDF checksum does not match document metadata",
                    stage=stage,
                    code="source_hash_mismatch",
                )

            update("rendering", 8)
            pages = self.renderer.render(pdf)
            elements: list[NormalizedElement] = []
            for index, page in enumerate(pages, start=1):
                page_key = f"{prefix}/pages/page-{page.page_number:04d}.png"
                self.storage.put_bytes(page_key, page.image, page.content_type)
                update(
                    "ocr",
                    10 + int((index - 1) / max(len(pages), 1) * 42),
                )
                try:
                    raw = self.ocr.analyze_page(
                        page_number=page.page_number,
                        image=page.image,
                        filename=f"page-{page.page_number:04d}.png",
                    )
                except OCRServiceError as exc:
                    if not exc.retryable:
                        # A single rejected/unparseable page should not sink the
                        # whole document: recover the PDF text layer instead.
                        if isinstance(self.renderer, PyMuPDFPageRenderer):
                            try:
                                raw = self.renderer.extract_text_layer_layout(
                                    pdf,
                                    page_number=page.page_number,
                                    rendered_width=page.width,
                                    rendered_height=page.height,
                                )
                            except OCRServiceError as text_layer_exc:
                                raise CompilationError(
                                    str(text_layer_exc),
                                    stage="ocr",
                                    code=text_layer_exc.code,
                                    retryable=text_layer_exc.retryable,
                                ) from text_layer_exc
                        else:
                            raise CompilationError(
                                str(exc),
                                stage="ocr",
                                code=exc.code,
                                retryable=False,
                            ) from exc
                    elif isinstance(self.renderer, PyMuPDFPageRenderer):
                        fallback_page = self.renderer.render_page(
                            pdf,
                            page_number=page.page_number,
                            dpi=max(72, round(self.render_dpi * 5 / 6)),
                        )
                    else:
                        fallback_page = _downsample_page(page)
                    try:
                        raw = self.ocr.analyze_page(
                            page_number=fallback_page.page_number,
                            image=fallback_page.image,
                            filename=f"page-{fallback_page.page_number:04d}-120dpi.png",
                        )
                    except OCRServiceError as fallback_exc:
                        if (
                            fallback_exc.retryable
                            and isinstance(self.renderer, PyMuPDFPageRenderer)
                        ):
                            try:
                                raw = self.renderer.extract_text_layer_layout(
                                    pdf,
                                    page_number=fallback_page.page_number,
                                    rendered_width=fallback_page.width,
                                    rendered_height=fallback_page.height,
                                )
                            except OCRServiceError as text_layer_exc:
                                raise CompilationError(
                                    str(text_layer_exc),
                                    stage="ocr",
                                    code=text_layer_exc.code,
                                    retryable=text_layer_exc.retryable,
                                ) from text_layer_exc
                        else:
                            raise CompilationError(
                                str(fallback_exc),
                                stage="ocr",
                                code=fallback_exc.code,
                                retryable=fallback_exc.retryable,
                            ) from fallback_exc
                    pages[index - 1] = fallback_page
                    page = fallback_page
                    self.storage.put_bytes(page_key, page.image, page.content_type)
                update(
                    "normalizing",
                    12 + int(index / max(len(pages), 1) * 45),
                )
                elements.extend(
                    normalize_page_elements(
                        document_id=source.id,
                        page=page,
                        raw=raw,
                    )
                )

            update("materializing_assets", 62)
            page_map = {page.page_number: page for page in pages}
            asset_records: list[dict[str, Any]] = []
            for element in elements:
                if element.kind not in MULTIMODAL_KINDS:
                    continue
                asset_id = _stable_id(
                    "asset", source.id, version, element.id, element.kind
                )
                payload = element._embedded_asset
                if payload is None or image_is_visually_blank(payload):
                    payload = _crop_page(page_map[element.page], element.bbox)
                asset_key = f"{prefix}/assets/{asset_id}.png"
                self.storage.put_bytes(asset_key, payload, "image/png")
                element.asset_id = asset_id
                element.asset_key = asset_key
                lineage = {
                    "id": asset_id,
                    "document_id": source.id,
                    "version": version,
                    "page": element.page,
                    "element_id": element.id,
                    "kind": element.kind,
                    "source_page_key": (
                        f"{prefix}/pages/page-{element.page:04d}.png"
                    ),
                    "object_key": asset_key,
                    "bbox": asdict(element.bbox),
                    "bbox_normalized": element.bbox_normalized,
                    "polygon": element.polygon,
                    "polygon_normalized": element.polygon_normalized,
                    "content_sha256": hashlib.sha256(payload).hexdigest(),
                    "content_type": "image/png",
                }
                put_json(
                    self.storage,
                    f"{prefix}/assets/{asset_id}.json",
                    lineage,
                )
                asset_records.append(lineage)

            update("writing_markdown", 72)
            markdown = build_rich_markdown(source.title, pages, elements)
            markdown_key = f"{prefix}/document.md"
            self.storage.put_bytes(
                markdown_key, markdown.encode("utf-8"), "text/markdown; charset=utf-8"
            )

            elements_key = f"{prefix}/elements.json"
            put_json(
                self.storage,
                elements_key,
                {
                    "document_id": source.id,
                    "version": version,
                    "pages": [
                        {
                            "page": page.page_number,
                            "width": page.width,
                            "height": page.height,
                            "image_key": (
                                f"{prefix}/pages/page-{page.page_number:04d}.png"
                            ),
                        }
                        for page in pages
                    ],
                    "elements": [element.to_dict() for element in elements],
                },
            )

            update("chunking", 82)
            chunks = build_structured_chunks(
                document_id=source.id,
                version=version,
                elements=elements,
                target_characters=self.chunk_target_characters,
            )
            chunks_key = f"{prefix}/chunks.json"
            put_json(
                self.storage,
                chunks_key,
                {
                    "document_id": source.id,
                    "version": version,
                    "chunks": [chunk.to_dict() for chunk in chunks],
                },
            )

            update("writing_manifest", 94)
            manifest = {
                "schema_version": 2,
                "pipeline_version": PIPELINE_VERSION,
                "trusted_layout": {
                    "contract": TRUSTED_LAYOUT_CONTRACT,
                    "authority": "paddleocr",
                    "immutable_fields": [
                        "document_id",
                        "version",
                        "page",
                        "element_id",
                        "bbox",
                        "bbox_normalized",
                        "reading_order",
                    ],
                },
                "compiler_config": compiler_config,
                "compiler_config_hash": compiler_hash,
                "document_id": source.id,
                "knowledge_base_id": source.knowledge_base_id,
                "version": version,
                "source": {
                    "filename": source.filename,
                    "object_key": source.object_key,
                },
                "source_sha256": source.sha256,
                "page_count": len(pages),
                "element_count": len(elements),
                "asset_count": len(asset_records),
                "chunk_count": len(chunks),
                "markdown_key": markdown_key,
                "elements_key": elements_key,
                "chunks_key": chunks_key,
                "assets": asset_records,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            # Manifest is written last and is the only completion signal.
            put_json(self.storage, manifest_key, manifest)
            result = self._result_from_manifest(manifest, manifest_key)
            self.repository.mark_succeeded(
                document_id,
                version=version,
                result=result.to_dict(),
                job_id=job_id,
            )
            return result
        except CompilationError as exc:
            self.repository.mark_failed(
                document_id,
                stage=exc.stage,
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
                job_id=job_id,
            )
            raise
        except Exception as exc:
            wrapped = CompilationError(
                "unexpected document compilation failure",
                stage=stage,
                code="compiler_internal_error",
            )
            self.repository.mark_failed(
                document_id,
                stage=wrapped.stage,
                code=wrapped.code,
                message=str(wrapped),
                retryable=False,
                job_id=job_id,
            )
            raise wrapped from exc

    @staticmethod
    def _result_from_manifest(
        manifest: dict[str, Any],
        manifest_key: str,
        *,
        idempotent_replay: bool = False,
    ) -> CompilationResult:
        return CompilationResult(
            document_id=str(manifest["document_id"]),
            version=int(manifest["version"]),
            source_sha256=str(manifest["source_sha256"]),
            page_count=int(manifest["page_count"]),
            element_count=int(manifest["element_count"]),
            asset_count=int(manifest["asset_count"]),
            chunk_count=int(manifest["chunk_count"]),
            manifest_key=manifest_key,
            markdown_key=str(manifest["markdown_key"]),
            elements_key=str(manifest["elements_key"]),
            chunks_key=str(manifest["chunks_key"]),
            idempotent_replay=idempotent_replay,
        )
