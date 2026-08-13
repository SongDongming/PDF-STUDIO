from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image

from app.api.routes._common import require_item
from app.schemas import (
    AssetView,
    DocumentCreate,
    DocumentUpdate,
    DocumentView,
    ElementView,
    JobCreate,
    JobView,
    PageView,
    RegionView,
)
from app.services.assets import (
    ObjectStorageError,
    crop_image_region,
    get_json,
    image_content_type,
    image_is_visually_blank,
)
from app.services.indexing import DocumentIndexer
from app.services.job_execution import job_runner
from app.store import store

router = APIRouter(tags=["documents"])


def _compiled(document_id: str) -> tuple[dict, dict, dict, dict]:
    document = require_item("documents", document_id, "文档")
    if document.get("status") != "ready" or not document.get("active_version"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "document_not_ready", "message": "文档尚未完成编译入库"},
        )
    try:
        manifest, elements, chunks = store.compiled_payloads(document)
    except (ObjectStorageError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "compiled_artifact_unavailable",
                "message": "编译产物暂不可读取",
            },
        ) from exc
    return document, manifest, elements, chunks


def _element_payload(
    document_id: str, element_id: str
) -> tuple[dict, dict, dict, dict]:
    document, manifest, payload, chunks = _compiled(document_id)
    element = next(
        (
            item
            for item in payload.get("elements", [])
            if str(item.get("id")) == element_id
        ),
        None,
    )
    if element is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "页面元素不存在"},
        )
    return document, manifest, payload, element


def _as_element(
    document_id: str, version: int, item: dict
) -> ElementView:
    asset_id = item.get("asset_id")
    return ElementView(
        id=str(item["id"]),
        document_id=document_id,
        version=version,
        page=int(item["page"]),
        order=int(item.get("order") or 1),
        kind=str(item.get("kind") or "text"),
        label=str(item.get("label") or item.get("kind") or "text"),
        content=str(item.get("content") or ""),
        bbox={key: float(value) for key, value in item["bbox"].items()},
        bbox_normalized=tuple(float(value) for value in item["bbox_normalized"]),
        polygon_normalized=[
            tuple(float(value) for value in point)
            for point in item.get("polygon_normalized", [])
        ],
        confidence=(
            float(item["confidence"]) if item.get("confidence") is not None else None
        ),
        asset_id=str(asset_id) if asset_id else None,
        asset_url=f"/api/v1/assets/{asset_id}" if asset_id else None,
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deleteDocument",
)
def delete_document(document_id: str) -> Response:
    require_item("documents", document_id, "文档")
    store.remove_document_cascade(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=list[DocumentView],
    operation_id="listDocuments",
)
def list_documents(knowledge_base_id: str) -> list[DocumentView]:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    return [
        DocumentView(**item)
        for item in store.list("documents")
        if item["knowledge_base_id"] == knowledge_base_id
    ]


def persist_document(knowledge_base_id: str, payload: DocumentCreate) -> DocumentView:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    item = store.create(
        "documents",
        {
            **payload.model_dump(),
            "knowledge_base_id": knowledge_base_id,
            "title": payload.title or payload.filename.removesuffix(".pdf"),
            "status": "uploaded",
            "active_version": 0,
            "error_message": None,
        },
    )
    return DocumentView(**item)


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentView,
    status_code=status.HTTP_201_CREATED,
    operation_id="createDocument",
)
def create_document(
    knowledge_base_id: str, payload: DocumentCreate
) -> DocumentView:
    """Register source metadata already present in protected object storage."""
    return persist_document(knowledge_base_id, payload)


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents/upload",
    response_model=DocumentView,
    status_code=status.HTTP_201_CREATED,
    operation_id="uploadDocument",
)
async def upload_document(
    knowledge_base_id: str,
    file: UploadFile = File(..., description="PDF source file"),
    title: str | None = Form(default=None),
) -> DocumentView:
    require_item("knowledge_bases", knowledge_base_id, "知识库")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "unsupported_media_type", "message": "只支持 PDF 文件"},
        )
    payload = await file.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "empty_file", "message": "PDF 文件为空"},
        )
    if not payload.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_pdf_signature", "message": "文件不是有效的 PDF"},
        )
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    digest = hashlib.sha256(payload).hexdigest()
    object_key = (
        f"sources/{knowledge_base_id}/{uuid4()}/{filename.replace(' ', '-')}"
    )
    try:
        stored = store.storage.put_bytes(object_key, payload, "application/pdf")
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "storage_unavailable", "message": "PDF 保存失败"},
        ) from exc
    return persist_document(
        knowledge_base_id,
        DocumentCreate(
            filename=filename,
            title=title,
            sha256=digest,
            object_key=stored.key,
        ),
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentView,
    operation_id="getDocument",
)
def get_document(document_id: str) -> DocumentView:
    return DocumentView(**require_item("documents", document_id, "文档"))


@router.patch(
    "/documents/{document_id}",
    response_model=DocumentView,
    operation_id="updateDocument",
)
def update_document(document_id: str, payload: DocumentUpdate) -> DocumentView:
    require_item("documents", document_id, "文档")
    changes = payload.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "empty_update", "message": "至少提供一个需要更新的字段"},
        )
    item = store.update("documents", document_id, changes)
    return DocumentView(**item)


@router.get("/documents/{document_id}/pdf", operation_id="getDocumentPdf")
def get_document_pdf(document_id: str) -> Response:
    document = require_item("documents", document_id, "文档")
    object_key = document.get("object_key")
    if not object_key:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "source_not_registered", "message": "文档没有源 PDF 对象"},
        )
    try:
        payload = store.storage.get_bytes(str(object_key))
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "source_not_found", "message": "源 PDF 不存在"},
        ) from exc
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{Path(document["filename"]).name}"'
        },
    )


@router.post(
    "/documents/{document_id}/compile",
    response_model=JobView,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="compileDocument",
)
def compile_document(document_id: str) -> JobView:
    document = require_item("documents", document_id, "文档")
    payload = JobCreate(
        kind="compile_document",
        knowledge_base_id=document["knowledge_base_id"],
        document_id=document_id,
    )
    item = store.create(
        "jobs",
        {
            **payload.model_dump(),
            "status": "queued",
            "stage": "waiting",
            "progress": 0,
            "error_code": None,
            "error_message": None,
            "retryable": False,
            "attempt": 1,
            "retry_of": None,
            "superseded_by": None,
            "result": {"execution": "queued"},
        },
    )
    store.update("documents", document_id, {"status": "queued"})
    if os.getenv("APP_COMPILE_MODE", "deferred").lower() == "sync":
        job_runner.submit(str(item["id"]))
    return JobView(**item)


@router.post(
    "/documents/{document_id}/reindex",
    operation_id="reindexDocument",
)
async def reindex_document(document_id: str) -> dict:
    """Rebuild only the retrieval index from an immutable compiled version."""

    document, manifest, _, _ = _compiled(document_id)
    embedder = store._embedding_provider()
    indexer = DocumentIndexer(storage=store.storage, embedder=embedder)
    try:
        result = await indexer.index(
            str(manifest["manifest_key"])
            if manifest.get("manifest_key")
            else f"compiled/{document_id}/v{manifest['version']}/manifest.json",
            document_title=str(document["title"]),
        )
        loaded = indexer.load_retriever(
            result.manifest_key,
            embedder=embedder if result.mode == "hybrid" else None,
        )
        store.retriever.upsert(loaded._chunks.values())
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "compiled_artifact_unavailable",
                "message": "编译产物缺失，无法重建索引",
            },
        ) from exc
    if result.mode == "hybrid":
        store.retriever.embedder = embedder
    payload = result.to_dict()
    for job in store.jobs.values():
        if job.get("document_id") == document_id and job.get("status") == "succeeded":
            job.setdefault("result", {}).update(
                {
                    "index_manifest_key": result.manifest_key,
                    "index_mode": result.mode,
                    "index_degraded_reason": result.degraded_reason,
                }
            )
            break
    store.persist_state()
    return payload


@router.get(
    "/documents/{document_id}/pages",
    response_model=list[PageView],
    operation_id="listDocumentPages",
)
def list_pages(document_id: str) -> list[PageView]:
    _, manifest, payload, _ = _compiled(document_id)
    return [
        PageView(
            document_id=document_id,
            version=int(manifest["version"]),
            page=int(item["page"]),
            width=int(item["width"]),
            height=int(item["height"]),
            image_url=(
                f"/api/v1/documents/{document_id}/pages/{int(item['page'])}/image"
            ),
        )
        for item in payload.get("pages", [])
    ]


@router.get(
    "/documents/{document_id}/pages/{page}/image",
    operation_id="getDocumentPageImage",
)
def get_page_image(document_id: str, page: int) -> Response:
    _, _, payload, _ = _compiled(document_id)
    record = next(
        (item for item in payload.get("pages", []) if int(item["page"]) == page),
        None,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "PDF 页面不存在"},
        )
    try:
        payload = store.storage.get_bytes(str(record["image_key"]))
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "page_image_not_found", "message": "PDF 页面图像不存在"},
        ) from exc
    return Response(payload, media_type="image/png")


@router.get(
    "/documents/{document_id}/pages/{page}/elements",
    response_model=list[ElementView],
    operation_id="listPageElements",
)
def list_page_elements(document_id: str, page: int) -> list[ElementView]:
    _, manifest, payload, _ = _compiled(document_id)
    return [
        _as_element(document_id, int(manifest["version"]), item)
        for item in payload.get("elements", [])
        if int(item.get("page") or 0) == page
    ]


@router.get(
    "/documents/{document_id}/assets",
    response_model=list[AssetView],
    operation_id="listDocumentAssets",
)
def list_assets(document_id: str) -> list[AssetView]:
    _, manifest, _, _ = _compiled(document_id)
    return [
        AssetView(
            id=str(item["id"]),
            document_id=document_id,
            version=int(item["version"]),
            page=int(item["page"]),
            element_id=str(item["element_id"]),
            kind=item["kind"],
            bbox_normalized=tuple(
                float(value) for value in item["bbox_normalized"]
            ),
            content_type=str(item.get("content_type") or "image/png"),
            url=f"/api/v1/assets/{item['id']}",
        )
        for item in manifest.get("assets", [])
    ]


@router.get("/assets/{asset_id}", operation_id="getAsset")
def get_asset(asset_id: str) -> Response:
    object_key = store.asset_keys.get(asset_id)
    if object_key is None:
        for document in store.list("documents"):
            if document.get("status") != "ready":
                continue
            try:
                manifest, _, _ = store.compiled_payloads(document)
            except (ObjectStorageError, KeyError, ValueError):
                continue
            asset = next(
                (
                    item
                    for item in manifest.get("assets", [])
                    if str(item.get("id")) == asset_id
                ),
                None,
            )
            if asset:
                object_key = str(asset["object_key"])
                store.asset_keys[asset_id] = object_key
                break
    if object_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": "多模态素材不存在"},
        )
    try:
        payload = store.storage.get_bytes(object_key)
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": "多模态素材不存在或已丢失"},
        ) from exc
    if image_is_visually_blank(payload):
        metadata_key = f"{object_key.rsplit('.', 1)[0]}.json"
        try:
            metadata = get_json(store.storage, metadata_key)
            bbox = metadata["bbox"]
            source_page = store.storage.get_bytes(str(metadata["source_page_key"]))
            repaired = crop_image_region(
                source_page,
                (
                    float(bbox["x0"]),
                    float(bbox["y0"]),
                    float(bbox["x1"]),
                    float(bbox["y1"]),
                ),
            )
            if not image_is_visually_blank(repaired):
                payload = repaired
        except (KeyError, TypeError, ValueError, ObjectStorageError):
            pass
    return Response(payload, media_type=image_content_type(payload))


@router.get(
    "/documents/{document_id}/regions/{element_id}",
    response_model=RegionView,
    operation_id="getPdfRegion",
)
def get_region(document_id: str, element_id: str) -> RegionView:
    _, manifest, _, element = _element_payload(document_id, element_id)
    asset_id = element.get("asset_id")
    return RegionView(
        document_id=document_id,
        version=int(manifest["version"]),
        page=int(element["page"]),
        element_id=element_id,
        bbox_normalized=tuple(
            float(value) for value in element["bbox_normalized"]
        ),
        page_image_url=(
            f"/api/v1/documents/{document_id}/pages/{int(element['page'])}/image"
        ),
        asset_url=(
            f"/api/v1/assets/{asset_id}"
            if asset_id
            else f"/api/v1/documents/{document_id}/regions/{element_id}/image"
        ),
    )


@router.get(
    "/documents/{document_id}/regions/{element_id}/image",
    operation_id="getPdfRegionImage",
)
def get_region_image(document_id: str, element_id: str) -> StreamingResponse:
    _, _, payload, element = _element_payload(document_id, element_id)
    page = next(
        (
            item
            for item in payload.get("pages", [])
            if int(item["page"]) == int(element["page"])
        ),
        None,
    )
    if page is None:
        raise HTTPException(status_code=404, detail={"code": "not_found"})
    try:
        page_bytes = store.storage.get_bytes(str(page["image_key"]))
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "page_image_not_found", "message": "PDF 页面图像不存在"},
        ) from exc
    source = Image.open(io.BytesIO(page_bytes))
    box = element["bbox"]
    crop = source.crop(
        (
            max(0, int(float(box["x0"]))),
            max(0, int(float(box["y0"]))),
            min(source.width, max(1, int(float(box["x1"])))),
            min(source.height, max(1, int(float(box["y1"])))),
        )
    )
    output = io.BytesIO()
    crop.save(output, format="PNG")
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")
