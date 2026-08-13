from __future__ import annotations

import time
from typing import Any, Protocol

import httpx


class OCRServiceError(ConnectionError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "ocr_unavailable",
        retryable: bool = True,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


class OCRClient(Protocol):
    def analyze_page(
        self,
        *,
        page_number: int,
        image: bytes,
        filename: str,
    ) -> dict[str, Any]: ...


class PaddleOCRHttpClient:
    """HTTP adapter for a PaddleOCR pipeline behind a protected SSH tunnel.

    The adapter deliberately sends one rendered page per request. That keeps
    retries page-local and preserves the page coordinate system used by the PDF
    comparison viewer.
    """

    def __init__(
        self,
        base_url: str,
        *,
        endpoint: str = "/v1/layout-parsing",
        health_endpoint: str = "/health",
        timeout_seconds: float = 180.0,
        max_attempts: int = 3,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("PaddleOCR base URL is required")
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.health_endpoint = health_endpoint
        self._owns_client = client is None
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.max_attempts = max_attempts
        self.client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            # The OCR endpoint is a protected loopback SSH tunnel. Corporate
            # HTTP(S)_PROXY settings must never intercept localhost traffic.
            trust_env=False,
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def health(self) -> dict[str, Any]:
        try:
            response = self.client.get(self.health_endpoint)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"status": "ok"}
        except (httpx.HTTPError, ValueError) as exc:
            raise OCRServiceError("PaddleOCR health check failed") from exc

    def analyze_page(
        self,
        *,
        page_number: int,
        image: bytes,
        filename: str,
    ) -> dict[str, Any]:
        response: httpx.Response | None = None
        last_transport_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.post(
                    self.endpoint,
                    data={
                        "page_number": str(page_number),
                        # The validated Dspark split runtime initializes layout +
                        # VLM only; orientation/unwarping models are intentionally
                        # absent to save memory and are therefore disabled.
                        "use_doc_orientation_classify": "false",
                        "use_doc_unwarping": "false",
                        "use_layout_detection": "true",
                    },
                    files={"file": (filename, image, "image/png")},
                )
                last_transport_error = None
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_transport_error = exc
                response = None
            transient = response is None or response.status_code >= 500 or response.status_code in {
                408,
                429,
            }
            if not transient:
                break
            if attempt < self.max_attempts:
                time.sleep(float(attempt))

        if response is None:
            raise OCRServiceError(
                "PaddleOCR request timed out or disconnected"
            ) from last_transport_error
        if response.status_code >= 500 or response.status_code in {408, 429}:
            raise OCRServiceError(
                "PaddleOCR service returned a transient error after retries",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise OCRServiceError(
                "PaddleOCR rejected the page",
                code="ocr_request_rejected",
                retryable=False,
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OCRServiceError(
                "PaddleOCR returned invalid JSON",
                code="ocr_invalid_response",
                retryable=False,
            ) from exc
        if not isinstance(payload, dict):
            raise OCRServiceError(
                "PaddleOCR response must be a JSON object",
                code="ocr_invalid_response",
                retryable=False,
            )
        return self._unwrap_payload(payload, page_number)

    @staticmethod
    def _unwrap_payload(payload: dict[str, Any], page_number: int) -> dict[str, Any]:
        """Accept the official pipeline wrapper and compact proxy responses."""

        current: Any = payload
        for key in ("data", "result"):
            candidate = current.get(key) if isinstance(current, dict) else None
            if isinstance(candidate, dict):
                current = candidate

        results = None
        if isinstance(current, dict):
            for key in ("layoutParsingResults", "results", "pages"):
                candidate = current.get(key)
                if isinstance(candidate, list):
                    results = candidate
                    break
        if results:
            selected = next(
                (
                    item
                    for item in results
                    if isinstance(item, dict)
                    and int(item.get("page_number", item.get("page", page_number)))
                    == page_number
                ),
                results[0],
            )
            current = selected

        if isinstance(current, dict) and isinstance(current.get("prunedResult"), dict):
            pruned = dict(current["prunedResult"])
            for key in ("markdown", "images", "input_path"):
                if key in current and key not in pruned:
                    pruned[key] = current[key]
            current = pruned

        if not isinstance(current, dict):
            raise OCRServiceError(
                "PaddleOCR page result is malformed",
                code="ocr_invalid_response",
                retryable=False,
            )
        current.setdefault("page_number", page_number)
        return current
