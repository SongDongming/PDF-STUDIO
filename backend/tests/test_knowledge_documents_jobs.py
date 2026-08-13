from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.services.assets import (
    LocalObjectStorage,
    image_is_visually_blank,
    put_json,
)
from app.store import store


def create_kb(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "LangChain 课程手册", "description": "中文多模态测试库"},
    )
    assert response.status_code == 201
    return response.json()


def test_knowledge_base_document_and_job_lifecycle(client: TestClient) -> None:
    kb = create_kb(client)

    document = client.post(
        f"/api/v1/knowledge-bases/{kb['id']}/documents",
        json={
            "filename": "agentic-rag.pdf",
            "title": "Agentic RAG 图文手册",
            "page_count": 24,
        },
    )
    assert document.status_code == 201
    document_data = document.json()
    assert document_data["status"] == "uploaded"

    compile_response = client.post(
        f"/api/v1/documents/{document_data['id']}/compile"
    )
    assert compile_response.status_code == 202
    job = compile_response.json()
    assert job["kind"] == "compile_document"
    assert job["status"] == "queued"
    assert job["document_id"] == document_data["id"]

    canceled = client.post(f"/api/v1/jobs/{job['id']}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"

    rejected_retry = client.post(f"/api/v1/jobs/{job['id']}/retry")
    assert rejected_retry.status_code == 409

    store.update(
        "jobs",
        job["id"],
        {
            "status": "failed",
            "stage": "ocr",
            "error_code": "ocr_unavailable",
            "error_message": "temporary OCR outage",
            "retryable": True,
        },
    )
    retried = client.post(f"/api/v1/jobs/{job['id']}/retry")
    assert retried.status_code == 202
    assert retried.json()["result"]["retry_of"] == job["id"]
    assert retried.json()["attempt"] == 2


def test_document_metadata_can_be_renamed_without_recompiling(
    client: TestClient,
) -> None:
    kb = create_kb(client)
    document = client.post(
        f"/api/v1/knowledge-bases/{kb['id']}/documents",
        json={"filename": "case-7-agentic-rag.pdf", "title": "case-7-agentic-rag.pdf"},
    ).json()

    response = client.patch(
        f"/api/v1/documents/{document['id']}",
        json={
            "filename": "Agentic RAG 自主检索与回退.pdf",
            "title": "Agentic RAG 自主检索与回退",
        },
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "Agentic RAG 自主检索与回退.pdf"
    assert response.json()["title"] == "Agentic RAG 自主检索与回退"
    assert response.json()["active_version"] == document["active_version"]
    assert client.patch(f"/api/v1/documents/{document['id']}", json={}).status_code == 422


def test_job_list_marks_superseded_failure_as_history(client: TestClient) -> None:
    kb = create_kb(client)
    document = client.post(
        f"/api/v1/knowledge-bases/{kb['id']}/documents",
        json={"filename": "history.pdf", "title": "History"},
    ).json()
    failed = store.create(
        "jobs",
        {
            "kind": "compile_document",
            "knowledge_base_id": kb["id"],
            "document_id": document["id"],
            "status": "failed",
            "stage": "ocr",
            "progress": 10,
            "error_code": "ocr_unavailable",
            "error_message": "temporary",
            "retryable": True,
            "result": {},
        },
    )
    succeeded = store.create(
        "jobs",
        {
            "kind": "compile_document",
            "knowledge_base_id": kb["id"],
            "document_id": document["id"],
            "status": "succeeded",
            "stage": "completed",
            "progress": 100,
            "error_code": None,
            "error_message": None,
            "retryable": False,
            "result": {},
        },
    )

    jobs = {item["id"]: item for item in client.get("/api/v1/jobs").json()}

    assert jobs[failed["id"]]["is_current"] is False
    assert jobs[succeeded["id"]]["is_current"] is True
    resolved_retry = client.post(f"/api/v1/jobs/{failed['id']}/retry")
    assert resolved_retry.status_code == 409


def test_whole_library_runner_compiles_pending_documents_and_builds_wiki(
    client: TestClient, monkeypatch
) -> None:
    from app.services.job_execution import LocalJobRunner

    kb = create_kb(client)
    documents = [
        client.post(
            f"/api/v1/knowledge-bases/{kb['id']}/documents",
            json={"filename": f"case-{index}.pdf", "title": f"Case {index}"},
        ).json()
        for index in range(2)
    ]
    job = store.create(
        "jobs",
        {
            "kind": "rebuild_knowledge_base",
            "knowledge_base_id": kb["id"],
            "document_id": None,
            "status": "queued",
            "stage": "waiting",
            "progress": 0,
            "error_code": None,
            "error_message": None,
            "result": {},
        },
    )

    def compile_now(
        document_id: str,
        child_job_id: str,
        progress_callback=None,
    ) -> dict:
        store.update("documents", document_id, {"status": "ready"})
        store.update(
            "jobs",
            child_job_id,
            {"status": "succeeded", "stage": "completed", "progress": 100},
        )
        return {"document_id": document_id}

    async def build_graph_and_wiki(
        knowledge_base_id: str, progress_callback=None
    ) -> dict:
        assert knowledge_base_id == kb["id"]
        return {
            "graph_version": 3,
            "wiki_version": 4,
            "document_count": 2,
            "node_count": 8,
            "edge_count": 7,
            "wiki_page_count": 4,
        }

    monkeypatch.setattr(store, "compile_document_now", compile_now)
    monkeypatch.setattr(store, "build_graph_and_wiki", build_graph_and_wiki)

    LocalJobRunner._rebuild_knowledge_base(job)

    finished = store.get("jobs", job["id"])
    assert finished is not None
    assert finished["status"] == "succeeded"
    assert finished["progress"] == 100
    assert len(finished["result"]["child_job_ids"]) == len(documents)
    assert store.get("knowledge_bases", kb["id"])["published_version"] == 4


def test_pdf_upload_rejects_wrong_type_and_accepts_pdf(client: TestClient) -> None:
    kb = create_kb(client)
    wrong = client.post(
        f"/api/v1/knowledge-bases/{kb['id']}/documents/upload",
        files={"file": ("notes.txt", b"text", "text/plain")},
    )
    assert wrong.status_code == 415

    pdf = client.post(
        f"/api/v1/knowledge-bases/{kb['id']}/documents/upload",
        data={"title": "结构化输出手册"},
        files={"file": ("structured-output.pdf", b"%PDF-1.7 mock", "application/pdf")},
    )
    assert pdf.status_code == 201
    payload = pdf.json()
    assert payload["title"] == "结构化输出手册"
    assert len(payload["sha256"]) == 64


def test_unknown_parent_returns_structured_404(client: TestClient) -> None:
    response = client.get("/api/v1/knowledge-bases/missing/documents")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"


def test_blank_embedded_asset_falls_back_to_pdf_page_region(
    client: TestClient, tmp_path
) -> None:
    storage = LocalObjectStorage(tmp_path / "objects")
    store.storage = storage
    asset_id = "asset-blank"
    object_key = f"compiled/doc-1/v1/assets/{asset_id}.png"
    page_key = "compiled/doc-1/v1/pages/page-0001.png"

    blank = Image.new("RGB", (220, 90), "white")
    ImageDraw.Draw(blank).rectangle((0, 0, 219, 89), outline="black", width=2)
    blank_bytes = BytesIO()
    blank.save(blank_bytes, format="JPEG")
    assert image_is_visually_blank(blank_bytes.getvalue())

    page = Image.new("RGB", (260, 160), "white")
    drawing = ImageDraw.Draw(page)
    drawing.rectangle((20, 25, 240, 135), outline="black", width=3)
    drawing.line((45, 80, 110, 45, 180, 105, 225, 55), fill="black", width=5)
    page_bytes = BytesIO()
    page.save(page_bytes, format="PNG")

    storage.put_bytes(object_key, blank_bytes.getvalue(), "image/jpeg")
    storage.put_bytes(page_key, page_bytes.getvalue(), "image/png")
    put_json(
        storage,
        f"compiled/doc-1/v1/assets/{asset_id}.json",
        {
            "id": asset_id,
            "object_key": object_key,
            "source_page_key": page_key,
            "bbox": {"x0": 20, "y0": 25, "x1": 240, "y1": 135},
        },
    )
    store.asset_keys[asset_id] = object_key

    response = client.get(f"/api/v1/assets/{asset_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert not image_is_visually_blank(response.content)
    assert Image.open(BytesIO(response.content)).size == (220, 110)
