"""Real model-provider adapters and grounded-answer validation.

The public API never accepts provider secrets.  Adapters are constructed from
environment variables by the application composition root, or with injected
HTTP clients in tests.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import httpx
import jsonschema
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.schemas import AssetBlock, Citation, TextBlock


class ProviderError(RuntimeError):
    """Base class for sanitized provider failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a server-side provider credential is not configured."""


class ProviderUnavailableError(ProviderError):
    """Raised when a configured provider cannot complete a request."""


class GroundingValidationError(ProviderError):
    """Raised when model output references evidence that was not retrieved."""


# Repair attempts after a schema-validation failure, for endpoints that do not
# strictly enforce the json_schema response_format (e.g. Zhipu GLM).
MAX_SCHEMA_REPAIR_ATTEMPTS = 4
# Extra rolls for the grounded-answer semantic contract, which the raw JSON
# schema cannot capture.
MAX_ANSWER_RETRIES = 2


class AsyncHttpClient(Protocol):
    async def post(self, url: str, **kwargs: Any) -> httpx.Response: ...

    async def get(self, url: str, **kwargs: Any) -> httpx.Response: ...


class ProviderHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    configured: bool
    healthy: bool
    detail: str


class VisionInput(BaseModel):
    """A vision-compatible image reference.

    The chat endpoint accepts base64 data URLs and ``ms://`` object references;
    public remote image URLs are deliberately rejected so the backend controls
    every byte sent to the model.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    detail: Literal["auto", "low", "high"] = "auto"

    @field_validator("url")
    @classmethod
    def validate_image_reference(cls, value: str) -> str:
        if value.startswith("data:image/") or value.startswith("ms://"):
            return value
        raise ValueError("image must be a data:image URL or an ms:// object reference")


class GroundedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: Citation
    text: str
    asset_ids: list[str] = Field(default_factory=list)


class ModelAnswerBlock(BaseModel):
    """Uniform block shape used by the strict provider response schema.

    Nullable fields are still required.  This keeps the JSON schema compatible
    with strict structured output while the semantic validator below enforces
    the fields allowed for each discriminated block type.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image", "table", "formula"]
    markdown: str | None
    asset_id: str | None
    caption: str | None
    alt: str | None
    citation_ids: list[str]


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks: list[ModelAnswerBlock] = Field(min_length=1)
    citation_ids: list[str]


class ValidatedAnswer(BaseModel):
    """Answer after every citation and asset reference has been checked."""

    model_config = ConfigDict(extra="forbid")

    blocks: list[dict[str, Any]]
    citations: list[Citation]


def validate_grounded_answer(
    answer: ModelAnswer | Mapping[str, Any],
    evidence: Sequence[GroundedEvidence],
    *,
    citation_required: bool = True,
) -> ValidatedAnswer:
    """Validate and convert a model answer into the public block contract.

    The model is allowed to *select* retrieved citations and assets, never to
    define them.  Citation metadata is copied from trusted retrieval results.
    """

    parsed = answer if isinstance(answer, ModelAnswer) else ModelAnswer.model_validate(answer)
    citations_by_id = {item.citation.id: item.citation for item in evidence}
    assets_by_id = {
        asset_id: item.citation.id
        for item in evidence
        for asset_id in item.asset_ids
    }

    requested_citation_ids: list[str] = []
    for citation_id in parsed.citation_ids:
        if citation_id not in citations_by_id:
            raise GroundingValidationError(
                f"answer references unavailable citation: {citation_id}"
            )
        if citation_id not in requested_citation_ids:
            requested_citation_ids.append(citation_id)

    public_blocks: list[dict[str, Any]] = []
    cited_from_blocks: list[str] = []
    for block in parsed.blocks:
        for citation_id in block.citation_ids:
            if citation_id not in citations_by_id:
                raise GroundingValidationError(
                    f"block references unavailable citation: {citation_id}"
                )
            if citation_id not in cited_from_blocks:
                cited_from_blocks.append(citation_id)

        if block.type == "text":
            if not block.markdown or block.asset_id is not None:
                raise GroundingValidationError(
                    "text blocks require markdown and must not contain asset_id"
                )
            public_blocks.append(TextBlock(markdown=block.markdown).model_dump())
            continue

        if block.markdown is not None or not block.asset_id or not block.alt:
            raise GroundingValidationError(
                f"{block.type} blocks require asset_id and alt, without markdown"
            )
        if block.asset_id not in assets_by_id:
            raise GroundingValidationError(
                f"answer references unavailable asset: {block.asset_id}"
            )
        source_citation_id = assets_by_id[block.asset_id]
        if source_citation_id not in block.citation_ids:
            raise GroundingValidationError(
                f"asset {block.asset_id} must cite its retrieved source"
            )
        public_blocks.append(
            AssetBlock(
                type=block.type,
                asset_id=block.asset_id,
                caption=block.caption,
                alt=block.alt,
            ).model_dump()
        )

    selected_ids = requested_citation_ids.copy()
    for citation_id in cited_from_blocks:
        if citation_id not in selected_ids:
            selected_ids.append(citation_id)

    if citation_required and not selected_ids:
        raise GroundingValidationError("a grounded answer must include a citation")

    return ValidatedAnswer(
        blocks=public_blocks,
        citations=[citations_by_id[citation_id] for citation_id in selected_ids],
    )


def _parse_json_content(content_value: Any) -> dict[str, Any]:
    """Parse structured-output content, tolerating markdown code fences.

    Several OpenAI-compatible providers (e.g. Zhipu GLM) wrap the JSON payload
    in ```json fences even when a strict json_schema response format is
    requested.  Strip the fences before the strict parse.
    """

    if isinstance(content_value, dict):
        return content_value
    if not isinstance(content_value, str):
        raise TypeError("message content has an unsupported type")
    text = content_value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Loose endpoints (e.g. Zhipu GLM) sometimes wrap the JSON in prose
        # instead of code fences.  Extract the first balanced JSON object.
        start = text.find("{")
        if start == -1:
            raise
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(text[start : index + 1])
                    break
        else:
            raise
    if not isinstance(parsed, dict):
        raise TypeError("structured output is not an object")
    return parsed


class DeepSeekProvider:
    """DeepSeek adapter using the OpenAI-compatible chat endpoint.

    The credential is read from ``MOONSHOT_API_KEY``/``MOONSHOT_BASE_URL`` for
    backward compatibility with the environment and secret store this project
    already ships.
    """

    provider_name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-v4-flash",
        client: AsyncHttpClient | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        if not api_key:
            raise ProviderConfigurationError("DeepSeek API credential is not configured")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls, *, client: AsyncHttpClient | None = None) -> "DeepSeekProvider":
        key = os.environ.get("MOONSHOT_API_KEY", "")
        return cls(
            api_key=key,
            base_url=os.environ.get(
                "MOONSHOT_BASE_URL", "https://api.deepseek.com/v1"
            ),
            model=os.environ.get("MOONSHOT_CHAT_MODEL", "deepseek-v4-flash"),
            client=client,
            timeout_seconds=float(
                os.environ.get("MOONSHOT_TIMEOUT_SECONDS", "180")
            ),
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        # 429/502/503/504 are transient provider overloads and are retried with
        # bounded backoff; other 4xx errors are configuration problems and fail
        # immediately.
        retryable_statuses = {429, 502, 503, 504}
        try:
            for attempt in range(3):
                if self._client is not None:
                    response = await self._client.post(
                        f"{self.base_url}{path}",
                        headers=self._headers,
                        json=dict(payload),
                        timeout=self.timeout_seconds,
                    )
                else:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{self.base_url}{path}",
                            headers=self._headers,
                            json=dict(payload),
                            timeout=self.timeout_seconds,
                        )
                if response.status_code not in retryable_statuses or attempt == 2:
                    break
                retry_header = response.headers.get("retry-after", "").strip()
                try:
                    retry_after = float(retry_header)
                except ValueError:
                    retry_after = float(2 ** (attempt + 1))
                await asyncio.sleep(min(max(retry_after, 0.25), 30.0))
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("provider response is not an object")
            return body
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderUnavailableError(
                "DeepSeek request failed; inspect protected server logs"
            ) from exc

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
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image.url, "detail": image.detail},
            }
            for image in images
        )
        # History is passed through as complete message objects.  In particular,
        # assistant reasoning/tool-call fields must not be reconstructed.
        #
        # Some OpenAI-compatible endpoints (e.g. Zhipu GLM) accept the strict
        # json_schema response_format field but do not enforce it, so the
        # required structure is also spelled out in the system prompt.
        schema_hint = (
            "\n\n必须严格输出符合以下 JSON Schema 的对象，字段名、类型和层级不得改变：\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": system_prompt + schema_hint},
            *history,
            {"role": "user", "content": content},
        ]
        # Some endpoints (e.g. Zhipu GLM) accept the strict json_schema
        # response_format field but do not reliably enforce it.  Validate the
        # parsed object against the schema and, on mismatch, feed the validation
        # error back to the model for a bounded number of repair attempts.
        for attempt in range(MAX_SCHEMA_REPAIR_ATTEMPTS + 1):
            body = await self._post(
                "/chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "reasoning_effort": reasoning_effort,
                    # json_object is the widest-compatible response format:
                    # DeepSeek rejects json_schema (400 "unavailable now") while
                    # accepting json_object; Zhipu GLM tolerates both.  The exact
                    # schema is spelled out in the system prompt and enforced by
                    # the jsonschema repair loop below.
                    "response_format": {"type": "json_object"},
                },
            )
            try:
                content_value = body["choices"][0]["message"]["content"]
                parsed = _parse_json_content(content_value)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                if attempt == MAX_SCHEMA_REPAIR_ATTEMPTS:
                    raise ProviderUnavailableError(
                        "DeepSeek returned invalid structured output"
                    )
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "你上次输出的内容不是合法 JSON。请只输出一个符合 JSON "
                            "Schema 的 JSON 对象，不要任何解释、Markdown 或前后缀文本。"
                        ),
                    },
                ]
                continue
            try:
                jsonschema.validate(parsed, schema)
            except jsonschema.ValidationError as exc:
                if attempt == MAX_SCHEMA_REPAIR_ATTEMPTS:
                    raise ProviderUnavailableError(
                        "DeepSeek returned invalid structured output"
                    ) from exc
                location = "$" + "".join(f"[{part}]" for part in exc.absolute_path)
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            f"你上次输出的 JSON 校验失败：{exc.message}（位置 "
                            f"{location}）。请修复该字段后重新输出完整 JSON。"
                            "若涉及 evidence_refs，必须包含至少一条来自输入的真实 "
                            "id；字段名、类型和层级严格按 schema，不要任何解释或 "
                            "Markdown。"
                        ),
                    },
                ]
                continue
            return parsed
        raise ProviderUnavailableError("DeepSeek returned invalid structured output")

    async def answer(
        self,
        *,
        question: str,
        evidence: Sequence[GroundedEvidence],
        images: Sequence[VisionInput] = (),
        history: Sequence[Mapping[str, Any]] = (),
        citation_required: bool = True,
        reasoning_effort: Literal["low", "high", "max"] = "high",
    ) -> ValidatedAnswer:
        evidence_json = [
            {
                "citation_id": item.citation.id,
                "document": item.citation.document_title,
                "page": item.citation.page,
                "bbox": item.citation.bbox,
                "text": item.text,
                "asset_ids": item.asset_ids,
            }
            for item in evidence
        ]
        if evidence:
            user_text = (
                f"问题：{question}\n\n"
                "以下是唯一允许使用和引用的检索证据，请只从证据中引用：\n"
                f"{json.dumps(evidence_json, ensure_ascii=False)}"
            )
            system_prompt = (
                "你是多模态 PDF 知识库问答助手。回答必须忠实于证据。"
                "text 块填写 markdown；image/table/formula 块只能填写证据中存在的"
                " asset_id，并在其 citation_ids 中包含来源引用。"
            )
        else:
            user_text = (
                f"问题：{question}\n\n"
                "本轮没有使用知识库证据。请使用你的通用知识直接回答；"
                "只输出 text 块，使用清晰的 Markdown。所有 citation_ids 必须为空，"
                "不得声称答案来自 PDF 或当前知识库。对不确定或时效性强的内容明确说明。"
            )
            system_prompt = (
                "你是 Agentic RAG 系统中的通用问答分支。检索工具不是每轮必用；"
                "当前没有知识库证据，因此使用模型自身知识回答。不得生成素材块、"
                "PDF 引用、页码或虚构来源。"
            )
        # The strict schema is validated inside complete_structured; the
        # semantic block contract is enforced here.  Looser endpoints (e.g.
        # Zhipu GLM) occasionally violate the semantic rules, so a bounded
        # retry gives the model another roll before failing.
        for _attempt in range(MAX_ANSWER_RETRIES + 1):
            raw = await self.complete_structured(
                system_prompt=system_prompt,
                user_text=user_text,
                schema_name="grounded_multimodal_answer",
                schema=ModelAnswer.model_json_schema(),
                images=images,
                history=history,
                reasoning_effort=reasoning_effort,
            )
            try:
                answer = ModelAnswer.model_validate(raw)
            except ValidationError as exc:
                raise ProviderUnavailableError(
                    "DeepSeek answer does not match the required schema"
                ) from exc
            try:
                return validate_grounded_answer(
                    answer, evidence, citation_required=citation_required
                )
            except GroundingValidationError:
                if _attempt == MAX_ANSWER_RETRIES:
                    raise
                continue
        raise GroundingValidationError("answer validation failed")

    async def health(self) -> ProviderHealth:
        try:
            if self._client is not None:
                response = await self._client.get(
                    f"{self.base_url}/models",
                    headers=self._headers,
                    timeout=15.0,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/models",
                        headers=self._headers,
                        timeout=15.0,
                    )
            response.raise_for_status()
            return ProviderHealth(
                provider=self.provider_name,
                model=self.model,
                configured=True,
                healthy=True,
                detail="模型服务可访问",
            )
        except httpx.HTTPError:
            return ProviderHealth(
                provider=self.provider_name,
                model=self.model,
                configured=True,
                healthy=False,
                detail="模型服务暂不可用",
            )


def structured_provider() -> DeepSeekProvider:
    """Provider for strict structured tasks (graph/wiki, grounded answers).

    Uses ``MOONSHOT_STRUCTURED_MODEL`` when set (a separate text model for
    strict JSON tasks) and otherwise falls back to the chat model.
    """

    return DeepSeekProvider(
        api_key=os.environ.get("MOONSHOT_API_KEY", ""),
        base_url=os.environ.get("MOONSHOT_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.environ.get("MOONSHOT_STRUCTURED_MODEL")
        or os.environ.get("MOONSHOT_CHAT_MODEL", "deepseek-v4-flash"),
    )
