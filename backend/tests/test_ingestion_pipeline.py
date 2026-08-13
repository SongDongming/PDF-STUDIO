from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.services.assets import LocalObjectStorage, ObjectStorageError, get_json
from app.services.ingestion import (
    CompilationError,
    DocumentCompiler,
    PyMuPDFPageRenderer,
    RenderedPage,
)
from app.services.ocr_client import OCRServiceError, PaddleOCRHttpClient
from app.services.repositories import DocumentSource


PNG_FIXTURE = b"\x89PNG\r\n\x1a\nfixture-image-bytes"


@dataclass
class _MemoryRepository:
    """Small in-memory repository for compiler contract tests."""

    documents: dict[str, DocumentSource] = field(default_factory=dict)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_document(self, document_id: str) -> DocumentSource:
        try:
            return self.documents[document_id]
        except KeyError as exc:
            raise RuntimeError(f"document {document_id!r} does not exist") from exc

    def mark_stage(
        self,
        document_id: str,
        *,
        stage: str,
        progress: int,
        job_id: str | None = None,
    ) -> None:
        self.states[document_id] = {
            "status": "running",
            "stage": stage,
            "progress": progress,
            "job_id": job_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def mark_succeeded(
        self,
        document_id: str,
        *,
        version: int,
        result: dict[str, Any],
        job_id: str | None = None,
    ) -> None:
        source = self.get_document(document_id)
        self.documents[document_id] = DocumentSource(
            id=source.id,
            knowledge_base_id=source.knowledge_base_id,
            filename=source.filename,
            title=source.title,
            object_key=source.object_key,
            sha256=source.sha256,
            active_version=version,
        )
        self.states[document_id] = {
            "status": "succeeded",
            "stage": "completed",
            "progress": 100,
            "job_id": job_id,
            "result": result,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def mark_failed(
        self,
        document_id: str,
        *,
        stage: str,
        code: str,
        message: str,
        retryable: bool = False,
        job_id: str | None = None,
    ) -> None:
        self.states[document_id] = {
            "status": "failed",
            "stage": stage,
            "progress": self.states.get(document_id, {}).get("progress", 0),
            "job_id": job_id,
            "error_code": code,
            "error_message": message,
            "retryable": retryable,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


class FixtureRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, pdf: bytes) -> list[RenderedPage]:
        assert pdf.startswith(b"%PDF")
        self.calls += 1
        return [
            RenderedPage(1, 1000, 1400, PNG_FIXTURE),
            RenderedPage(2, 1200, 1600, PNG_FIXTURE + b"-page-2"),
        ]


class FixtureOCR:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def analyze_page(
        self, *, page_number: int, image: bytes, filename: str
    ) -> dict[str, Any]:
        self.calls.append(page_number)
        encoded = base64.b64encode(PNG_FIXTURE + bytes([page_number])).decode()
        if page_number == 1:
            return {
                "parsing_res_list": [
                    {
                        "block_label": "doc_title",
                        "block_content": "Agentic RAG",
                        "block_bbox": [80, 60, 900, 150],
                        "block_order": 1,
                        "confidence": 0.99,
                    },
                    {
                        "block_label": "text",
                        "block_content": "代理会根据证据充分度决定是否继续检索。",
                        "block_bbox": [80, 180, 900, 270],
                        "block_order": 2,
                        "confidence": 0.96,
                    },
                    {
                        "block_label": "figure",
                        "block_content": "检索与证据校验流程图",
                        "block_bbox": [100, 300, 500, 700],
                        "polygon": [
                            {"x": 100, "y": 300},
                            {"x": 500, "y": 300},
                            {"x": 500, "y": 700},
                            {"x": 100, "y": 700},
                        ],
                        "block_order": 3,
                        "image_base64": encoded,
                    },
                ]
            }
        return {
            "elements": [
                {
                    "type": "table",
                    "markdown": "| 阶段 | 作用 |\n|---|---|\n| 检索 | 找证据 |",
                    "bbox": {"x": 120, "y": 200, "width": 900, "height": 420},
                    "order": 1,
                    "asset_base64": encoded,
                },
                {
                    "type": "formula",
                    "text": "score = 0.7d + 0.3g",
                    "bbox": [0.1, 0.55, 0.9, 0.68],
                    "poly": [0.1, 0.55, 0.9, 0.55, 0.9, 0.68, 0.1, 0.68],
                    "order": 2,
                    "crop_base64": encoded,
                },
            ]
        }


@pytest.fixture
def compiler_fixture(tmp_path: Path) -> tuple[
    DocumentCompiler,
    LocalObjectStorage,
    _MemoryRepository,
    FixtureRenderer,
    FixtureOCR,
]:
    storage = LocalObjectStorage(tmp_path / "objects")
    pdf = b"%PDF-1.7\nmultimodal fixture\n%%EOF"
    source_key = "source/kb-1/agentic-rag.pdf"
    storage.put_bytes(source_key, pdf, "application/pdf")
    source = DocumentSource(
        id="doc-1",
        knowledge_base_id="kb-1",
        filename="agentic-rag.pdf",
        title="Agentic RAG 图文手册",
        object_key=source_key,
        sha256=hashlib.sha256(pdf).hexdigest(),
        active_version=0,
    )
    repository = _MemoryRepository(documents={source.id: source})
    renderer = FixtureRenderer()
    ocr = FixtureOCR()
    compiler = DocumentCompiler(
        storage=storage,
        renderer=renderer,
        ocr=ocr,
        repository=repository,
        chunk_target_characters=200,
    )
    return compiler, storage, repository, renderer, ocr


def test_compilation_preserves_coordinates_assets_and_chunk_lineage(
    compiler_fixture: tuple[
        DocumentCompiler,
        LocalObjectStorage,
        _MemoryRepository,
        FixtureRenderer,
        FixtureOCR,
    ],
) -> None:
    compiler, storage, repository, renderer, ocr = compiler_fixture
    result = compiler.compile("doc-1", 1, job_id="job-1")

    assert result.page_count == 2
    assert result.element_count == 5
    assert result.asset_count == 3
    assert result.chunk_count >= 3
    assert renderer.calls == 1
    assert ocr.calls == [1, 2]
    assert repository.states["doc-1"]["status"] == "succeeded"
    assert repository.documents["doc-1"].active_version == 1

    elements_payload = get_json(storage, result.elements_key)
    figure = next(
        element
        for element in elements_payload["elements"]
        if element["kind"] == "figure"
    )
    assert figure["bbox"] == {"x0": 100.0, "y0": 300.0, "x1": 500.0, "y1": 700.0}
    assert figure["bbox_normalized"] == [0.1, 0.214286, 0.5, 0.5]
    assert figure["polygon_normalized"][2] == [0.5, 0.5]

    formula = next(
        element
        for element in elements_payload["elements"]
        if element["kind"] == "formula"
    )
    assert formula["bbox"] == {
        "x0": 120.0,
        "y0": 880.0000000000001,
        "x1": 1080.0,
        "y1": 1088.0,
    }
    assert formula["bbox_normalized"] == [0.1, 0.55, 0.9, 0.68]

    manifest = get_json(storage, result.manifest_key)
    assert manifest["source"]["object_key"] == "source/kb-1/agentic-rag.pdf"
    figure_asset = next(
        asset for asset in manifest["assets"] if asset["kind"] == "figure"
    )
    assert figure_asset["document_id"] == "doc-1"
    assert figure_asset["element_id"] == figure["id"]
    assert figure_asset["page"] == 1
    assert figure_asset["source_page_key"].endswith("pages/page-0001.png")
    assert storage.exists(figure_asset["object_key"])
    asset_metadata = get_json(
        storage,
        figure_asset["object_key"].removesuffix(".png") + ".json",
    )
    assert asset_metadata["content_sha256"] == figure_asset["content_sha256"]

    markdown = storage.get_bytes(result.markdown_key).decode()
    assert '<asset id="asset_' in markdown
    assert 'type="image" page="1"' in markdown
    assert "| 检索 | 找证据 |" in markdown
    assert "$$\nscore = 0.7d + 0.3g\n$$" in markdown

    chunks = get_json(storage, result.chunks_key)["chunks"]
    asset_chunks = [chunk for chunk in chunks if chunk["asset_ids"]]
    assert {asset for chunk in asset_chunks for asset in chunk["asset_ids"]} == {
        asset["id"] for asset in manifest["assets"]
    }
    for chunk in asset_chunks:
        assert len(chunk["asset_ids"]) == 1
        assert chunk["element_ids"]
        assert chunk["page_start"] <= chunk["page_end"]


def test_manifest_is_an_idempotent_completion_signal(
    compiler_fixture: tuple[
        DocumentCompiler,
        LocalObjectStorage,
        _MemoryRepository,
        FixtureRenderer,
        FixtureOCR,
    ],
) -> None:
    compiler, _, _, renderer, ocr = compiler_fixture
    first = compiler.compile("doc-1", 7)
    replay = compiler.compile("doc-1", 7)

    assert first.manifest_key == replay.manifest_key
    assert replay.idempotent_replay is True
    assert renderer.calls == 1
    assert ocr.calls == [1, 2]


def test_retryable_ocr_failure_uses_page_local_resolution_fallback(
    tmp_path: Path,
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    pdf = b"%PDF-1.7\nresolution fallback fixture\n%%EOF"
    source_key = "source/kb-1/fallback.pdf"
    storage.put_bytes(source_key, pdf, "application/pdf")
    source = DocumentSource(
        id="doc-fallback",
        knowledge_base_id="kb-1",
        filename="fallback.pdf",
        title="OCR resolution fallback",
        object_key=source_key,
        sha256=hashlib.sha256(pdf).hexdigest(),
        active_version=0,
    )
    repository = _MemoryRepository(documents={source.id: source})

    class ValidImageRenderer:
        def render(self, pdf: bytes) -> list[RenderedPage]:
            from PIL import Image
            import io

            image = Image.new("RGB", (120, 180), "white")
            output = io.BytesIO()
            image.save(output, format="PNG")
            return [RenderedPage(1, 120, 180, output.getvalue())]

    class FailOnceOCR:
        def __init__(self) -> None:
            self.sizes: list[tuple[int, int]] = []

        def analyze_page(
            self, *, page_number: int, image: bytes, filename: str
        ) -> dict[str, Any]:
            from PIL import Image
            import io

            opened = Image.open(io.BytesIO(image))
            self.sizes.append(opened.size)
            if len(self.sizes) == 1:
                raise OCRServiceError("invalid structured output")
            return {
                "elements": [
                    {
                        "type": "text",
                        "text": "降采样后识别成功",
                        "bbox": [10, 10, 90, 40],
                        "order": 1,
                    }
                ]
            }

    ocr = FailOnceOCR()
    compiler = DocumentCompiler(
        storage=storage,
        renderer=ValidImageRenderer(),
        ocr=ocr,
        repository=repository,
    )

    result = compiler.compile("doc-fallback", 1)

    assert ocr.sizes == [(120, 180), (100, 150)]
    assert result.element_count == 1
    manifest = get_json(storage, result.manifest_key)
    elements_payload = get_json(storage, manifest["elements_key"])
    assert elements_payload["pages"][0]["width"] == 100
    assert elements_payload["pages"][0]["height"] == 150
    stored_page = storage.get_bytes(elements_payload["pages"][0]["image_key"])
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(stored_page)).size == (100, 150)


def test_pdf_text_layer_fallback_preserves_text_and_pixel_coordinates() -> None:
    import fitz

    document = fitz.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((30, 60), "SqliteSaver persistent memory")
    pdf = document.tobytes()
    document.close()

    payload = PyMuPDFPageRenderer.extract_text_layer_layout(
        pdf,
        page_number=1,
        rendered_width=600,
        rendered_height=800,
    )

    assert payload["recognizer"] == "pdf_text_layer_fallback"
    assert "SqliteSaver persistent memory" in payload["elements"][0]["text"]
    assert payload["elements"][0]["bbox"][0] >= 59
    assert payload["elements"][0]["recognizer"] == "pdf_text_layer_fallback"


def test_checksum_failure_has_explicit_stage_and_writes_no_manifest(
    compiler_fixture: tuple[
        DocumentCompiler,
        LocalObjectStorage,
        _MemoryRepository,
        FixtureRenderer,
        FixtureOCR,
    ],
) -> None:
    compiler, storage, repository, renderer, ocr = compiler_fixture
    source = repository.documents["doc-1"]
    repository.documents["doc-1"] = DocumentSource(
        id=source.id,
        knowledge_base_id=source.knowledge_base_id,
        filename=source.filename,
        title=source.title,
        object_key=source.object_key,
        sha256="0" * 64,
        active_version=source.active_version,
    )

    with pytest.raises(CompilationError) as caught:
        compiler.compile("doc-1", 1)

    assert caught.value.code == "source_hash_mismatch"
    assert caught.value.stage == "loading_source"
    assert repository.states["doc-1"]["error_code"] == "source_hash_mismatch"
    assert not storage.exists("compiled/doc-1/v1/manifest.json")
    assert renderer.calls == 0
    assert ocr.calls == []


def test_local_storage_rejects_path_traversal(tmp_path: Path) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    with pytest.raises(ObjectStorageError):
        storage.put_bytes("../outside.pdf", b"not allowed")


def test_paddle_http_client_unwraps_official_pipeline_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/layout-parsing"
        return httpx.Response(
            200,
            json={
                "result": {
                    "layoutParsingResults": [
                        {
                            "page_number": 1,
                            "prunedResult": {
                                "parsing_res_list": [
                                    {
                                        "block_label": "text",
                                        "block_content": "正文",
                                        "block_bbox": [1, 2, 3, 4],
                                    }
                                ]
                            },
                            "images": {"figure.png": base64.b64encode(PNG_FIXTURE).decode()},
                        }
                    ]
                }
            },
        )

    http = httpx.Client(
        base_url="http://127.0.0.1:18080",
        transport=httpx.MockTransport(handler),
    )
    client = PaddleOCRHttpClient("http://127.0.0.1:18080", client=http)
    payload = client.analyze_page(
        page_number=1,
        image=PNG_FIXTURE,
        filename="page-0001.png",
    )

    assert payload["page_number"] == 1
    assert payload["parsing_res_list"][0]["block_content"] == "正文"
    assert "figure.png" in payload["images"]
