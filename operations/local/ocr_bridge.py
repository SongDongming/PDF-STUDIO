"""Local OCR bridge satisfying the PDF-Studio backend OCR contract.

The backend expects a PaddleOCR-VL pipeline behind a protected loopback tunnel
at ``http://127.0.0.1:18111`` exposing ``POST /v1/layout-parsing`` and
``GET /health`` (see ``app/services/ocr_client.py``).  This service implements
that exact HTTP contract locally and forwards each page to the official
asynchronous PaddleOCR-VL job API (paddleocr.aistudio-app.com), then returns
the page's ``layoutParsingResults`` element unchanged — the backend's
``_unwrap_payload`` already reads ``result.layoutParsingResults[i].prunedResult``.
No backend code is changed.

Configuration is read from environment variables, falling back to
``infra/ocr.env`` (a mode-600 file) when present:

    PADDLE_OCR_TOKEN       access token from the PaddleOCR website
    PADDLE_OCR_JOB_URL     jobs endpoint (defaults to the official URL)
    PADDLE_OCR_MODEL       model name (defaults to PaddleOCR-VL-1.6)
    PADDLE_POLL_INTERVAL   seconds between job polls (default 2)
    PADDLE_POLL_TIMEOUT    max seconds to wait per page (default 150)

Run from the repository root:

    backend/.venv/bin/python operations/local/ocr_bridge.py

Set ``OCR_BRIDGE_BACKEND=mock`` for an offline end-to-end pipeline check.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="local-ocr-bridge", version="0.3.0")

_SCRIPT_DIR = Path(__file__).resolve().parent
INFRA_ENV = _SCRIPT_DIR.parent.parent / "infra" / "ocr.env"


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


_load_dotenv(INFRA_ENV)

BACKEND = env("OCR_BRIDGE_BACKEND", "paddle")
PADDLE_OCR_TOKEN = env("PADDLE_OCR_TOKEN")
PADDLE_OCR_JOB_URL = env(
    "PADDLE_OCR_JOB_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
)
PADDLE_OCR_MODEL = env("PADDLE_OCR_MODEL", "PaddleOCR-VL-1.6")
PADDLE_OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}
PADDLE_POLL_INTERVAL = float(env("PADDLE_POLL_INTERVAL", "2"))
PADDLE_POLL_TIMEOUT = float(env("PADDLE_POLL_TIMEOUT", "150"))


class OcrBridgeError(RuntimeError):
    """Raised when the configured backend cannot satisfy a page request."""


async def _paddle_recognize(image: bytes) -> dict[str, Any]:
    """Run one async PaddleOCR job for a single page image and return the
    page's layoutParsingResults element (with prunedResult)."""
    if not PADDLE_OCR_TOKEN:
        raise OcrBridgeError("PADDLE_OCR_TOKEN is not configured")
    headers = {"Authorization": f"bearer {PADDLE_OCR_TOKEN}"}
    form = {
        "model": PADDLE_OCR_MODEL,
        "optionalPayload": json.dumps(PADDLE_OPTIONAL_PAYLOAD),
    }
    files = {"file": ("page.png", image, "image/png")}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                PADDLE_OCR_JOB_URL, headers=headers, data=form, files=files
            )
        except httpx.HTTPError as exc:
            raise OcrBridgeError(f"paddle ocr job submit failed: {exc}") from exc
        if response.status_code != 200:
            raise OcrBridgeError(
                f"paddle ocr job submit rejected ({response.status_code}): "
                f"{response.text[:200]}"
            )
        try:
            job_id = response.json()["data"]["jobId"]
        except (KeyError, TypeError, ValueError) as exc:
            raise OcrBridgeError("paddle ocr job submit returned no jobId") from exc

        deadline = time.monotonic() + PADDLE_POLL_TIMEOUT
        state = "pending"
        try:
            while time.monotonic() < deadline:
                result = await client.get(f"{PADDLE_OCR_JOB_URL}/{job_id}", headers=headers)
                result.raise_for_status()
                payload = result.json()["data"]
                state = payload.get("state")
                if state == "done":
                    break
                if state == "failed":
                    raise OcrBridgeError(
                        f"paddle ocr job failed: {payload.get('errorMsg')}"
                    )
                await asyncio.sleep(PADDLE_POLL_INTERVAL)
            else:
                raise OcrBridgeError(f"paddle ocr job timed out (state={state})")
        except httpx.HTTPError as exc:
            raise OcrBridgeError(f"paddle ocr job poll failed: {exc}") from exc

        try:
            jsonl_url = result.json()["data"]["resultUrl"]["jsonUrl"]
        except (KeyError, TypeError, ValueError) as exc:
            raise OcrBridgeError("paddle ocr job result has no jsonUrl") from exc
        try:
            jsonl = await client.get(jsonl_url)
            jsonl.raise_for_status()
        except httpx.HTTPError as exc:
            raise OcrBridgeError(f"paddle ocr result download failed: {exc}") from exc

    elements: list[dict[str, Any]] = []
    for line in jsonl.text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        elements.extend(
            record.get("result", {}).get("layoutParsingResults", [])
        )
    if not elements:
        raise OcrBridgeError("paddle ocr returned no layout parsing results")
    # A single page image produces exactly one result; take it.
    return elements[0]


def _mock_recognize(image: bytes) -> list[dict[str, Any]]:
    """Deterministic backend for end-to-end pipeline verification."""
    elements: list[dict[str, Any]] = []
    for index, text in enumerate(
        ("本地 OCR 桥接服务工作正常", "这是用于联调测试的模拟识别结果"), start=1
    ):
        y0 = (index - 1) * 120
        elements.append(
            {
                "block_label": "text",
                "block_content": text,
                "block_bbox": [40, y0, 1100, y0 + 80],
                "polygon": [[40, y0], [1100, y0], [1100, y0 + 80], [40, y0 + 80]],
                "block_order": index,
                "confidence": 0.99,
            }
        )
    return elements


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "backend": BACKEND}


@app.post("/v1/layout-parsing")
async def layout_parsing(
    page_number: int = Form(1),
    file: UploadFile = File(...),
) -> Any:
    image = await file.read()
    if not image:
        return JSONResponse(
            status_code=422, content={"detail": "empty page image"}
        )
    try:
        if BACKEND == "paddle":
            return await _paddle_recognize(image)
        if BACKEND == "mock":
            return {"page_number": page_number, "elements": _mock_recognize(image)}
        raise OcrBridgeError(f"unknown OCR_BRIDGE_BACKEND: {BACKEND}")
    except OcrBridgeError as exc:
        # Configuration/backend errors are non-transient to the ingestion retry
        # loop, mirroring the 4xx path in PaddleOCRHttpClient.
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(env("OCR_BRIDGE_PORT", "18111")), log_level="info")
