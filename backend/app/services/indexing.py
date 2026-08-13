"""Bridge immutable compilation artifacts into a versioned retrieval index.

The compiler manifest is the only accepted completion signal.  The index
payload is written first and its manifest last, so readers never observe a
partially materialized index.  Dense-provider outages have an explicit
``lexical-only`` representation; contract violations (dimensions or lineage)
fail closed instead of silently degrading.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from app.services.assets import ObjectStorage, put_json
from app.services.embeddings import EmbeddingProvider
from app.services.providers import ProviderUnavailableError
from app.services.retrieval import HybridRetriever, RetrievalChunk

INDEX_SCHEMA_VERSION = 2
INDEX_DIMENSIONS = 1024


class IndexingError(RuntimeError):
    """A sanitized, stage-aware indexing failure suitable for job reporting."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        stage: str = "indexing",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    document_id: str
    knowledge_base_id: str
    version: int
    mode: Literal["hybrid", "lexical-only"]
    index_signature: str
    dimensions: int | None
    chunk_count: int
    payload_key: str
    manifest_key: str
    degraded_reason: str | None = None
    idempotent_replay: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(storage: ObjectStorage, key: str, *, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(storage.get_bytes(key).decode("utf-8"))
    except Exception as exc:
        raise IndexingError(
            f"{kind} is missing or invalid",
            code=f"invalid_{kind.replace(' ', '_')}",
        ) from exc
    if not isinstance(payload, dict):
        raise IndexingError(
            f"{kind} must be a JSON object",
            code=f"invalid_{kind.replace(' ', '_')}",
        )
    return payload


def _provider_signature(provider: EmbeddingProvider) -> str:
    dimensions = getattr(provider, "dimensions", None)
    if dimensions != INDEX_DIMENSIONS:
        raise IndexingError(
            f"embedding provider must use exactly {INDEX_DIMENSIONS} dimensions",
            code="embedding_dimension_mismatch",
        )
    signature = getattr(provider, "index_signature", None)
    if signature is None:
        signature = (
            f"{provider.provider_name}:{provider.model}:{provider.dimensions}"
        )
    signature = str(signature).strip()
    if not signature or not signature.endswith(f":{INDEX_DIMENSIONS}"):
        raise IndexingError(
            "embedding index signature does not match the dimension contract",
            code="embedding_signature_mismatch",
        )
    return signature


def _bbox_tuple(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, dict) and {"x0", "y0", "x1", "y1"} <= raw.keys():
        values = (raw["x0"], raw["y0"], raw["x1"], raw["y1"])
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = tuple(raw)
    else:
        return None
    try:
        return tuple(float(value) for value in values)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _validate_vector(vector: Sequence[Any], *, chunk_id: str) -> list[float]:
    if len(vector) != INDEX_DIMENSIONS:
        raise IndexingError(
            f"embedding for chunk {chunk_id!r} violates the "
            f"{INDEX_DIMENSIONS}-dimension index contract",
            code="embedding_dimension_mismatch",
        )
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise IndexingError(
            f"embedding for chunk {chunk_id!r} contains a non-numeric value",
            code="invalid_embedding",
        ) from exc


class DocumentIndexer:
    """Materialize versioned :class:`RetrievalChunk` JSON from compiler output."""

    def __init__(
        self,
        *,
        storage: ObjectStorage,
        embedder: EmbeddingProvider | None,
        embedding_batch_size: int = 10,
    ) -> None:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self.storage = storage
        self.embedder = embedder
        self.embedding_batch_size = embedding_batch_size

    async def index(
        self,
        compiler_manifest_key: str,
        *,
        document_title: str | None = None,
    ) -> IndexBuildResult:
        """Build or idempotently reuse an index for one compiled document version."""

        compiler_manifest = _load_json_object(
            self.storage, compiler_manifest_key, kind="compiler manifest"
        )
        self._validate_compiler_manifest(compiler_manifest)

        chunks_payload = _load_json_object(
            self.storage, str(compiler_manifest["chunks_key"]), kind="chunks artifact"
        )
        elements_payload = _load_json_object(
            self.storage,
            str(compiler_manifest["elements_key"]),
            kind="elements artifact",
        )
        raw_chunks = chunks_payload.get("chunks")
        raw_elements = elements_payload.get("elements")
        raw_assets = compiler_manifest.get("assets", [])
        if not isinstance(raw_chunks, list) or not isinstance(raw_elements, list):
            raise IndexingError(
                "compiled chunks and elements must be JSON arrays",
                code="invalid_compilation_artifacts",
            )
        if not isinstance(raw_assets, list):
            raise IndexingError(
                "compiler asset lineage must be a JSON array",
                code="invalid_asset_lineage",
            )

        title = document_title or PurePosixPath(
            str(compiler_manifest["source"]["filename"])
        ).stem
        if not title.strip():
            raise IndexingError("document title must not be empty", code="invalid_title")

        elements_by_id = {
            str(item["id"]): item
            for item in raw_elements
            if isinstance(item, dict) and item.get("id")
        }
        assets_by_id = {
            str(item["id"]): item
            for item in raw_assets
            if isinstance(item, dict) and item.get("id")
        }
        semantic_overlay = self._load_semantic_overlay(compiler_manifest)
        compiler_manifest["_semantic_fingerprint"] = (
            str(semantic_overlay.get("artifact_sha256"))
            if semantic_overlay is not None
            else "none"
        )
        base_chunks = self._build_chunks(
            compiler_manifest=compiler_manifest,
            raw_chunks=raw_chunks,
            elements_by_id=elements_by_id,
            assets_by_id=assets_by_id,
            document_title=title,
        )
        if semantic_overlay is not None:
            base_chunks = self._apply_semantic_overlay(base_chunks, semantic_overlay)

        if self.embedder is None:
            return self._persist(
                compiler_manifest_key=compiler_manifest_key,
                compiler_manifest=compiler_manifest,
                chunks=base_chunks,
                document_title=title,
                mode="lexical-only",
                index_signature="lexical-only:v1",
                dimensions=None,
                degraded_reason="embedding_provider_not_configured",
            )

        signature = _provider_signature(self.embedder)
        reusable = self._load_completed_result(
            compiler_manifest_key=compiler_manifest_key,
            compiler_manifest=compiler_manifest,
            document_title=title,
            index_signature=signature,
            mode="hybrid",
            dimensions=INDEX_DIMENSIONS,
        )
        if reusable is not None:
            return reusable

        try:
            embedded_chunks = await self._embed_chunks(base_chunks)
        except ProviderUnavailableError:
            return self._persist(
                compiler_manifest_key=compiler_manifest_key,
                compiler_manifest=compiler_manifest,
                chunks=base_chunks,
                document_title=title,
                mode="lexical-only",
                index_signature="lexical-only:v1",
                dimensions=None,
                degraded_reason="embedding_provider_unavailable",
            )

        return self._persist(
            compiler_manifest_key=compiler_manifest_key,
            compiler_manifest=compiler_manifest,
            chunks=embedded_chunks,
            document_title=title,
            mode="hybrid",
            index_signature=signature,
            dimensions=INDEX_DIMENSIONS,
            degraded_reason=None,
        )

    def load_retriever(
        self,
        index_manifest_key: str,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> HybridRetriever:
        """Rebuild a retriever and verify payload integrity/signature first."""

        manifest = _load_json_object(
            self.storage, index_manifest_key, kind="index manifest"
        )
        payload_key = manifest.get("payload_key")
        if not isinstance(payload_key, str):
            raise IndexingError(
                "index manifest does not reference a payload",
                code="invalid_index_manifest",
            )
        raw_payload = self.storage.get_bytes(payload_key)
        if _sha256(raw_payload) != manifest.get("payload_sha256"):
            raise IndexingError(
                "index payload checksum does not match its manifest",
                code="index_payload_hash_mismatch",
            )
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
            chunks = [
                RetrievalChunk.model_validate(item) for item in payload["chunks"]
            ]
        except Exception as exc:
            raise IndexingError(
                "index payload is invalid",
                code="invalid_index_payload",
            ) from exc

        mode = manifest.get("mode")
        if mode == "hybrid":
            if embedder is None:
                raise IndexingError(
                    "the dense index requires its matching embedding provider",
                    code="embedding_provider_required",
                )
            signature = _provider_signature(embedder)
            if signature != manifest.get("index_signature"):
                raise IndexingError(
                    "query embedding provider does not match the stored index",
                    code="embedding_signature_mismatch",
                )
            query_embedder = embedder
        elif mode == "lexical-only":
            query_embedder = None
        else:
            raise IndexingError(
                "index manifest has an unsupported mode",
                code="invalid_index_manifest",
            )
        return HybridRetriever(chunks, embedder=query_embedder)

    async def _embed_chunks(
        self, chunks: list[RetrievalChunk]
    ) -> list[RetrievalChunk]:
        assert self.embedder is not None
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            raw_vectors = await self.embedder.embed_documents(
                [chunk.text for chunk in batch]
            )
            if len(raw_vectors) != len(batch):
                raise IndexingError(
                    "embedding response count does not match the document batch",
                    code="embedding_count_mismatch",
                )
            vectors.extend(
                _validate_vector(vector, chunk_id=chunk.id)
                for chunk, vector in zip(batch, raw_vectors, strict=True)
            )
        return [
            chunk.model_copy(update={"embedding": vector}, deep=True)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    @staticmethod
    def _validate_compiler_manifest(manifest: dict[str, Any]) -> None:
        required = {
            "document_id",
            "knowledge_base_id",
            "version",
            "source",
            "source_sha256",
            "compiler_config_hash",
            "chunks_key",
            "elements_key",
        }
        if not required <= manifest.keys():
            raise IndexingError(
                "compiler manifest is missing required fields",
                code="invalid_compiler_manifest",
            )
        if not isinstance(manifest["source"], dict) or not manifest["source"].get(
            "filename"
        ):
            raise IndexingError(
                "compiler manifest has no source filename",
                code="invalid_compiler_manifest",
            )

    def _load_semantic_overlay(
        self, compiler_manifest: dict[str, Any]
    ) -> dict[str, Any] | None:
        document_id = str(compiler_manifest["document_id"])
        version = int(compiler_manifest["version"])
        key = (
            f"compiled/{document_id}/v{version}/"
            "multimodal-semantic-v2.json"
        )
        if not self.storage.exists(key):
            return None
        overlay = _load_json_object(
            self.storage, key, kind="semantic enrichment artifact"
        )
        if (
            overlay.get("schema_version") != 2
            or overlay.get("semantic_artifact_version")
            != "multimodal-semantic-v2"
            or str(overlay.get("document_id")) != document_id
            or int(overlay.get("document_version") or 0) != version
            or str(overlay.get("source_sha256"))
            != str(compiler_manifest["source_sha256"])
        ):
            raise IndexingError(
                "semantic enrichment artifact does not match trusted compilation lineage",
                code="semantic_lineage_mismatch",
            )
        expected = str(overlay.get("artifact_sha256") or "")
        unsigned = dict(overlay)
        unsigned.pop("artifact_sha256", None)
        if not expected or _sha256(_canonical_bytes(unsigned)) != expected:
            raise IndexingError(
                "semantic enrichment artifact checksum mismatch",
                code="semantic_artifact_hash_mismatch",
            )
        if not isinstance(overlay.get("pages"), list) or not isinstance(
            overlay.get("elements"), list
        ):
            raise IndexingError(
                "semantic enrichment artifact has invalid page or element arrays",
                code="invalid_semantic_artifact",
            )
        return overlay

    @staticmethod
    def _apply_semantic_overlay(
        chunks: list[RetrievalChunk], overlay: dict[str, Any]
    ) -> list[RetrievalChunk]:
        pages = {
            int(item["page"]): item
            for item in overlay["pages"]
            if isinstance(item, dict)
            and item.get("publishable") is True
            and int(item.get("page") or 0) > 0
        }
        elements = {
            str(item["element_id"]): item
            for item in overlay["elements"]
            if isinstance(item, dict)
            and item.get("publishable") is True
            and item.get("element_id")
        }
        enriched: list[RetrievalChunk] = []
        for chunk in chunks:
            page_end = int(chunk.metadata.get("page_end") or chunk.page)
            semantic_parts: list[str] = []
            tags: list[str] = []
            relations: list[str] = []
            for page in range(chunk.page, page_end + 1):
                item = pages.get(page)
                if item is None:
                    continue
                semantic_parts.extend(
                    [
                        f"[DeepSeek整页视觉摘要] {str(item.get('summary') or '').strip()}",
                        f"[DeepSeek整页检索语义] {str(item.get('search_text') or '').strip()}",
                    ]
                )
                tags.extend(str(value) for value in item.get("semantic_tags", []))
                relations.extend(
                    str(value) for value in item.get("element_relations", [])
                )
            for element_id in chunk.metadata.get("element_ids", []):
                item = elements.get(str(element_id))
                if item is None:
                    continue
                semantic_parts.extend(
                    [
                        f"[DeepSeek元素视觉描述] {str(item.get('description') or '').strip()}",
                        f"[DeepSeek元素检索语义] {str(item.get('search_text') or '').strip()}",
                    ]
                )
                tags.extend(str(value) for value in item.get("semantic_tags", []))
                structure = item.get("structure")
                if isinstance(structure, dict):
                    relations.extend(
                        str(value) for value in structure.get("relations", [])
                    )
            semantic_parts = [
                value for value in dict.fromkeys(semantic_parts) if value.strip()
            ]
            if not semantic_parts:
                enriched.append(chunk)
                continue
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "semantic_enrichment_version": str(
                        overlay["semantic_artifact_version"]
                    ),
                    "semantic_artifact_sha256": str(overlay["artifact_sha256"]),
                    "semantic_tags": list(dict.fromkeys(value for value in tags if value)),
                    "visual_relations": list(
                        dict.fromkeys(value for value in relations if value)
                    ),
                }
            )
            enriched.append(
                chunk.model_copy(
                    update={
                        "text": chunk.text + "\n\n" + "\n".join(semantic_parts),
                        "metadata": metadata,
                    },
                    deep=True,
                )
            )
        return enriched

    @staticmethod
    def _build_chunks(
        *,
        compiler_manifest: dict[str, Any],
        raw_chunks: list[Any],
        elements_by_id: dict[str, dict[str, Any]],
        assets_by_id: dict[str, dict[str, Any]],
        document_title: str,
    ) -> list[RetrievalChunk]:
        built: list[RetrievalChunk] = []
        for raw in raw_chunks:
            if not isinstance(raw, dict):
                raise IndexingError(
                    "compiled chunk is not an object",
                    code="invalid_chunk",
                )
            chunk_id = str(raw.get("id") or "")
            markdown = str(raw.get("markdown") or "").strip()
            element_ids = [str(value) for value in raw.get("element_ids", [])]
            asset_ids = [str(value) for value in raw.get("asset_ids", [])]
            if not chunk_id or not markdown or not element_ids:
                raise IndexingError(
                    "compiled chunk lacks id, text, or element lineage",
                    code="invalid_chunk",
                )
            missing_elements = [
                element_id
                for element_id in element_ids
                if element_id not in elements_by_id
            ]
            missing_assets = [
                asset_id for asset_id in asset_ids if asset_id not in assets_by_id
            ]
            if missing_elements or missing_assets:
                raise IndexingError(
                    f"chunk {chunk_id!r} contains dangling element or asset lineage",
                    code="invalid_asset_lineage",
                )

            # Multimodal evidence is anchored to the exact source element.  Text
            # chunks use their first reading-order element as the citation anchor.
            asset_lineage = [assets_by_id[asset_id] for asset_id in asset_ids]
            if asset_lineage:
                anchor_element_id = str(asset_lineage[0]["element_id"])
                anchor_element = elements_by_id.get(anchor_element_id)
                if anchor_element is None or anchor_element_id not in element_ids:
                    raise IndexingError(
                        f"chunk {chunk_id!r} asset is detached from its source element",
                        code="invalid_asset_lineage",
                    )
                page = int(asset_lineage[0]["page"])
                bbox = _bbox_tuple(
                    asset_lineage[0].get("bbox_normalized")
                    or anchor_element.get("bbox_normalized")
                )
            else:
                anchor_element_id = element_ids[0]
                anchor_element = elements_by_id[anchor_element_id]
                page = int(anchor_element.get("page") or raw.get("page_start") or 0)
                bbox = _bbox_tuple(
                    anchor_element.get("bbox_normalized")
                    or anchor_element.get("bbox")
                )
            if page < 1 or bbox is None:
                raise IndexingError(
                    f"chunk {chunk_id!r} has no usable page/bbox anchor",
                    code="invalid_element_lineage",
                )

            metadata = {
                "version": int(compiler_manifest["version"]),
                "ordinal": int(raw.get("ordinal") or len(built) + 1),
                "page_end": int(raw.get("page_end") or page),
                "heading_path": list(raw.get("heading_path") or []),
                "element_ids": element_ids,
                "asset_lineage": [
                    {
                        "id": str(item["id"]),
                        "element_id": str(item["element_id"]),
                        "page": int(item["page"]),
                        "kind": str(item.get("kind") or "unknown"),
                        "object_key": str(item.get("object_key") or ""),
                        "source_page_key": str(item.get("source_page_key") or ""),
                        "bbox": item.get("bbox"),
                        "bbox_normalized": item.get("bbox_normalized"),
                    }
                    for item in asset_lineage
                ],
            }
            built.append(
                RetrievalChunk(
                    id=chunk_id,
                    knowledge_base_id=str(compiler_manifest["knowledge_base_id"]),
                    document_id=str(compiler_manifest["document_id"]),
                    document_title=document_title,
                    page=page,
                    text=markdown,
                    bbox=bbox,
                    element_id=anchor_element_id,
                    asset_ids=asset_ids,
                    metadata=metadata,
                )
            )
        if len(built) != int(compiler_manifest.get("chunk_count", len(built))):
            raise IndexingError(
                "compiled chunk count does not match the compiler manifest",
                code="chunk_count_mismatch",
            )
        return built

    @staticmethod
    def _identity(
        *,
        compiler_manifest_key: str,
        compiler_manifest: dict[str, Any],
        document_title: str,
        index_signature: str,
        mode: str,
        dimensions: int | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "compiler_manifest_key": compiler_manifest_key,
            "source_sha256": str(compiler_manifest["source_sha256"]),
            "compiler_config_hash": str(compiler_manifest["compiler_config_hash"]),
            "document_title": document_title,
            "semantic_fingerprint": str(
                compiler_manifest.get("_semantic_fingerprint") or "none"
            ),
            "index_signature": index_signature,
            "mode": mode,
            "dimensions": dimensions,
        }

    def _keys(
        self,
        *,
        compiler_manifest: dict[str, Any],
        identity: dict[str, Any],
    ) -> tuple[str, str]:
        variant = _sha256(_canonical_bytes(identity))[:16]
        prefix = (
            f"indexes/{compiler_manifest['knowledge_base_id']}/"
            f"{compiler_manifest['document_id']}/v{int(compiler_manifest['version'])}/"
            f"{variant}"
        )
        return f"{prefix}/chunks.json", f"{prefix}/manifest.json"

    def _load_completed_result(
        self,
        *,
        compiler_manifest_key: str,
        compiler_manifest: dict[str, Any],
        document_title: str,
        index_signature: str,
        mode: Literal["hybrid", "lexical-only"],
        dimensions: int | None,
    ) -> IndexBuildResult | None:
        identity = self._identity(
            compiler_manifest_key=compiler_manifest_key,
            compiler_manifest=compiler_manifest,
            document_title=document_title,
            index_signature=index_signature,
            mode=mode,
            dimensions=dimensions,
        )
        _, manifest_key = self._keys(
            compiler_manifest=compiler_manifest, identity=identity
        )
        if not self.storage.exists(manifest_key):
            return None
        manifest = _load_json_object(
            self.storage, manifest_key, kind="index manifest"
        )
        if manifest.get("identity") != identity:
            raise IndexingError(
                "existing index manifest conflicts with the requested identity",
                code="index_version_conflict",
            )
        payload_key = str(manifest.get("payload_key") or "")
        if not payload_key or not self.storage.exists(payload_key):
            raise IndexingError(
                "completed index manifest references a missing payload",
                code="missing_index_payload",
            )
        if _sha256(self.storage.get_bytes(payload_key)) != manifest.get(
            "payload_sha256"
        ):
            raise IndexingError(
                "completed index payload checksum mismatch",
                code="index_payload_hash_mismatch",
            )
        return self._result_from_manifest(
            manifest, manifest_key, idempotent_replay=True
        )

    def _persist(
        self,
        *,
        compiler_manifest_key: str,
        compiler_manifest: dict[str, Any],
        chunks: list[RetrievalChunk],
        document_title: str,
        mode: Literal["hybrid", "lexical-only"],
        index_signature: str,
        dimensions: int | None,
        degraded_reason: str | None,
    ) -> IndexBuildResult:
        reusable = self._load_completed_result(
            compiler_manifest_key=compiler_manifest_key,
            compiler_manifest=compiler_manifest,
            document_title=document_title,
            index_signature=index_signature,
            mode=mode,
            dimensions=dimensions,
        )
        if reusable is not None:
            return reusable

        identity = self._identity(
            compiler_manifest_key=compiler_manifest_key,
            compiler_manifest=compiler_manifest,
            document_title=document_title,
            index_signature=index_signature,
            mode=mode,
            dimensions=dimensions,
        )
        payload_key, manifest_key = self._keys(
            compiler_manifest=compiler_manifest, identity=identity
        )
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "identity": identity,
            "document_id": str(compiler_manifest["document_id"]),
            "knowledge_base_id": str(compiler_manifest["knowledge_base_id"]),
            "version": int(compiler_manifest["version"]),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        encoded = _canonical_bytes(payload)
        self.storage.put_bytes(
            payload_key, encoded, "application/json; charset=utf-8"
        )
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "identity": identity,
            "document_id": str(compiler_manifest["document_id"]),
            "knowledge_base_id": str(compiler_manifest["knowledge_base_id"]),
            "version": int(compiler_manifest["version"]),
            "mode": mode,
            "index_signature": index_signature,
            "dimensions": dimensions,
            "chunk_count": len(chunks),
            "payload_key": payload_key,
            "payload_sha256": _sha256(encoded),
            "degraded_reason": degraded_reason,
        }
        # Written last: this is the sole readable-index completion signal.
        put_json(self.storage, manifest_key, manifest)
        return self._result_from_manifest(manifest, manifest_key)

    @staticmethod
    def _result_from_manifest(
        manifest: dict[str, Any],
        manifest_key: str,
        *,
        idempotent_replay: bool = False,
    ) -> IndexBuildResult:
        return IndexBuildResult(
            document_id=str(manifest["document_id"]),
            knowledge_base_id=str(manifest["knowledge_base_id"]),
            version=int(manifest["version"]),
            mode=manifest["mode"],
            index_signature=str(manifest["index_signature"]),
            dimensions=(
                int(manifest["dimensions"])
                if manifest.get("dimensions") is not None
                else None
            ),
            chunk_count=int(manifest["chunk_count"]),
            payload_key=str(manifest["payload_key"]),
            manifest_key=manifest_key,
            degraded_reason=manifest.get("degraded_reason"),
            idempotent_replay=idempotent_replay,
        )
