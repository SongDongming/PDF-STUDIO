"""Hybrid lexical/dense retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import Citation, RagSettings
from app.services.embeddings import EmbeddingProvider
from app.services.providers import GroundedEvidence, ProviderUnavailableError

_LATIN_OR_NUMBER = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def lexical_tokens(text: str) -> list[str]:
    """Tokenize mixed Chinese/Latin technical text without a runtime dictionary.

    Chinese unigram + bigram tokens preserve recall for product names and short
    queries; Latin words and numbers are normalized as complete tokens.
    """

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = _LATIN_OR_NUMBER.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class RetrievalChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    knowledge_base_id: str
    document_id: str
    document_title: str
    page: int = Field(ge=1)
    text: str = Field(min_length=1)
    embedding: list[float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    element_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_evidence(self, score: float | None = None) -> GroundedEvidence:
        return GroundedEvidence(
            citation=Citation(
                id=f"citation:{self.id}",
                document_id=self.document_id,
                document_title=self.document_title,
                page=self.page,
                bbox=self.bbox,
                element_id=self.element_id,
                excerpt=self.text,
                score=score,
            ),
            text=self.text,
            asset_ids=self.asset_ids,
        )


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: RetrievalChunk
    score: float = Field(ge=0, le=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    lexical_score: float = Field(default=0, ge=0)
    dense_score: float | None = Field(default=None, ge=-1, le=1)

    def as_evidence(self) -> GroundedEvidence:
        return self.chunk.as_evidence(self.score)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("dense vectors must have the same non-zero dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class HybridRetriever:
    """In-process reference retriever.

    Production storage can implement the same ``retrieve``/``fetch_chunk``
    surface with a dedicated vector store.  This implementation is deterministic
    and complete enough for local development and contract tests.
    """

    def __init__(
        self,
        chunks: Iterable[RetrievalChunk] = (),
        *,
        embedder: EmbeddingProvider | None = None,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self._chunks: dict[str, RetrievalChunk] = {}
        self.upsert(chunks)

    def upsert(self, chunks: Iterable[RetrievalChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def remove_document(self, document_id: str) -> int:
        """Drop every chunk belonging to a document; returns count removed."""
        ids = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.document_id == document_id
        ]
        for chunk_id in ids:
            self._chunks.pop(chunk_id, None)
        return len(ids)

    def document_ids(self) -> set[str]:
        """Return the distinct document ids currently held in the index."""
        return {chunk.document_id for chunk in self._chunks.values()}

    def fetch_chunk(
        self, chunk_id: str, *, knowledge_base_id: str | None = None
    ) -> RetrievalChunk | None:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return None
        if knowledge_base_id and chunk.knowledge_base_id != knowledge_base_id:
            return None
        return chunk.model_copy(deep=True)

    def _lexical_ranking(
        self, query: str, chunks: Sequence[RetrievalChunk]
    ) -> list[tuple[RetrievalChunk, float]]:
        query_tokens = lexical_tokens(query)
        if not query_tokens or not chunks:
            return []

        document_tokens = [lexical_tokens(chunk.text) for chunk in chunks]
        average_length = sum(map(len, document_tokens)) / max(len(document_tokens), 1)
        average_length = max(average_length, 1.0)
        document_frequency: Counter[str] = Counter()
        for tokens in document_tokens:
            document_frequency.update(set(tokens))

        query_frequency = Counter(query_tokens)
        scored: list[tuple[RetrievalChunk, float]] = []
        total_documents = len(chunks)
        for chunk, tokens in zip(chunks, document_tokens, strict=True):
            term_frequency = Counter(tokens)
            score = 0.0
            for token, query_count in query_frequency.items():
                frequency = term_frequency[token]
                if frequency == 0:
                    continue
                df = document_frequency[token]
                inverse_document_frequency = math.log(
                    1 + (total_documents - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + self.bm25_k1 * (
                    1
                    - self.bm25_b
                    + self.bm25_b * len(tokens) / average_length
                )
                score += (
                    inverse_document_frequency
                    * (frequency * (self.bm25_k1 + 1) / denominator)
                    * query_count
                )
            if score > 0:
                scored.append((chunk, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].id))

    @staticmethod
    def _dense_ranking(
        query_vector: Sequence[float], chunks: Sequence[RetrievalChunk]
    ) -> list[tuple[RetrievalChunk, float]]:
        scored: list[tuple[RetrievalChunk, float]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            try:
                similarity = _cosine_similarity(query_vector, chunk.embedding)
            except ValueError:
                # An index-signature mismatch must not poison lexical fallback.
                continue
            scored.append((chunk, similarity))
        return sorted(scored, key=lambda item: (-item[1], item[0].id))

    async def retrieve(
        self,
        query: str,
        *,
        knowledge_base_id: str,
        settings: RagSettings | None = None,
    ) -> list[RetrievalHit]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        config = settings or RagSettings()
        chunks = [
            chunk
            for chunk in self._chunks.values()
            if chunk.knowledge_base_id == knowledge_base_id
        ]
        lexical = self._lexical_ranking(query, chunks)[: config.lexical_top_k]

        dense: list[tuple[RetrievalChunk, float]] = []
        if self.embedder is not None:
            try:
                query_vector = await self.embedder.embed_query(query)
                dense = self._dense_ranking(query_vector, chunks)[: config.dense_top_k]
            except ProviderUnavailableError:
                # Explicit degradation: BM25 continues to serve available
                # evidence while provider health reports the dense outage.
                dense = []

        lexical_rank = {
            chunk.id: (rank, score)
            for rank, (chunk, score) in enumerate(lexical, start=1)
        }
        dense_rank = {
            chunk.id: (rank, score)
            for rank, (chunk, score) in enumerate(dense, start=1)
        }
        candidates = set(lexical_rank) | set(dense_rank)
        fused: list[tuple[str, float]] = []
        for chunk_id in candidates:
            score = 0.0
            if chunk_id in lexical_rank:
                score += 1 / (self.rrf_k + lexical_rank[chunk_id][0])
            if chunk_id in dense_rank:
                score += 1 / (self.rrf_k + dense_rank[chunk_id][0])
            fused.append((chunk_id, score))
        fused.sort(key=lambda item: (-item[1], item[0]))

        top = fused[: config.rerank_top_k]
        max_score = top[0][1] if top else 1.0
        hits: list[RetrievalHit] = []
        for chunk_id, score in top:
            lex = lexical_rank.get(chunk_id)
            dense_item = dense_rank.get(chunk_id)
            hits.append(
                RetrievalHit(
                    chunk=self._chunks[chunk_id].model_copy(deep=True),
                    score=min(1.0, score / max_score),
                    lexical_rank=lex[0] if lex else None,
                    lexical_score=lex[1] if lex else 0.0,
                    dense_rank=dense_item[0] if dense_item else None,
                    dense_score=dense_item[1] if dense_item else None,
                )
            )
        return hits
