from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    dependencies: dict[str, str]


class KnowledgeBaseStatus(StrEnum):
    draft = "draft"
    compiling = "compiling"
    ready = "ready"
    partial = "partial"
    failed = "failed"
    archived = "archived"


class KnowledgeBaseCreate(ApiModel):
    name: NonEmpty
    description: str = ""


class KnowledgeBaseUpdate(ApiModel):
    name: NonEmpty | None = None
    description: str | None = None
    status: KnowledgeBaseStatus | None = None


class KnowledgeBaseView(ApiModel):
    id: str
    name: str
    description: str
    status: KnowledgeBaseStatus
    document_count: int = 0
    published_version: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentStatus(StrEnum):
    uploaded = "uploaded"
    queued = "queued"
    parsing = "parsing"
    enriching = "enriching"
    indexing = "indexing"
    ready = "ready"
    failed = "failed"
    archived = "archived"


class DocumentCreate(ApiModel):
    filename: NonEmpty
    title: NonEmpty | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    page_count: int | None = Field(default=None, ge=1)
    object_key: str | None = None


class DocumentUpdate(ApiModel):
    filename: NonEmpty | None = None
    title: NonEmpty | None = None


class DocumentView(ApiModel):
    id: str
    knowledge_base_id: str
    filename: str
    title: str
    status: DocumentStatus
    sha256: str | None = None
    object_key: str | None = Field(default=None, exclude=True)
    page_count: int | None = None
    element_count: int = 0
    asset_count: int = 0
    chunk_count: int = 0
    active_version: int = 0
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class PageView(ApiModel):
    document_id: str
    version: int
    page: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    image_url: str


class ElementView(ApiModel):
    id: str
    document_id: str
    version: int
    page: int = Field(ge=1)
    order: int = Field(ge=1)
    kind: str
    label: str
    content: str
    bbox: dict[str, float]
    bbox_normalized: tuple[float, float, float, float]
    polygon_normalized: list[tuple[float, float]] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    asset_id: str | None = None
    asset_url: str | None = None


class AssetView(ApiModel):
    id: str
    document_id: str
    version: int
    page: int = Field(ge=1)
    element_id: str
    kind: Literal["figure", "table", "formula"]
    bbox_normalized: tuple[float, float, float, float]
    content_type: str
    url: str


class RegionView(ApiModel):
    document_id: str
    version: int
    page: int = Field(ge=1)
    element_id: str
    bbox_normalized: tuple[float, float, float, float]
    page_image_url: str
    asset_url: str | None = None


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    partial = "partial"
    failed = "failed"
    canceled = "canceled"


class JobCreate(ApiModel):
    kind: Literal["compile_document", "rebuild_knowledge_base"]
    knowledge_base_id: str | None = None
    document_id: str | None = None


class JobView(ApiModel):
    id: str
    kind: str
    status: JobStatus
    stage: str
    progress: int = Field(ge=0, le=100)
    knowledge_base_id: str | None = None
    document_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    attempt: int = Field(default=1, ge=1)
    retry_of: str | None = None
    superseded_by: str | None = None
    is_current: bool = True
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class Citation(ApiModel):
    id: str
    document_id: str
    document_title: str
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None
    element_id: str | None = None
    excerpt: str
    score: float | None = Field(default=None, ge=0, le=1)


class RetrievalRequest(ApiModel):
    query: NonEmpty
    document_ids: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievalHitView(ApiModel):
    chunk_id: str
    document_id: str
    document_title: str
    page: int = Field(ge=1)
    text: str
    score: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float] | None = None
    element_id: str | None = None
    asset_ids: list[str] = Field(default_factory=list)
    citation: Citation


class RetrievalResponse(ApiModel):
    query: str
    knowledge_base_id: str
    mode: Literal["hybrid", "lexical_fallback", "empty"]
    hits: list[RetrievalHitView]


class TextBlock(ApiModel):
    type: Literal["text"] = "text"
    markdown: str


class AssetBlock(ApiModel):
    type: Literal["image", "table", "formula"]
    asset_id: str
    caption: str | None = None
    alt: str


AnswerBlock = Annotated[TextBlock | AssetBlock, Field(discriminator="type")]


class ThreadCreate(ApiModel):
    knowledge_base_id: str
    title: str = "新会话"
    scope: dict[str, Any] = Field(default_factory=dict)


class ThreadUpdate(ApiModel):
    title: NonEmpty | None = None
    status: Literal["active", "archived"] | None = None
    scope: dict[str, Any] | None = None


class ThreadView(ApiModel):
    id: str
    knowledge_base_id: str
    title: str
    status: Literal["active", "archived"]
    scope: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class MessageCreate(ApiModel):
    content: NonEmpty
    stream: bool = False


class MessageView(ApiModel):
    id: str
    thread_id: str
    role: Literal["user", "assistant", "system"]
    blocks: list[AnswerBlock]
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime


class WikiPageSummary(ApiModel):
    id: str
    knowledge_base_id: str
    slug: str
    title: str
    summary: str
    status: Literal["draft", "published", "locked"] = "published"
    updated_at: datetime


class WikiPage(WikiPageSummary):
    markdown: str
    citations: list[Citation] = Field(default_factory=list)
    related_page_ids: list[str] = Field(default_factory=list)


class GraphNode(ApiModel):
    id: str
    label: str
    kind: Literal["entity", "claim", "document", "page", "chunk", "asset", "wiki"]
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(ApiModel):
    id: str
    source: str
    target: str
    relation: str
    evidence_ids: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphView(ApiModel):
    knowledge_base_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    graph_version: int = 0
    total_node_count: int = 0
    total_edge_count: int = 0
    truncated: bool = False


class GraphBuildRequest(ApiModel):
    document_ids: list[str] = Field(default_factory=list)


class GraphBuildResponse(ApiModel):
    knowledge_base_id: str
    graph_version: int
    wiki_version: int
    document_count: int
    node_count: int
    edge_count: int
    wiki_page_count: int


class EvidenceLineageView(ApiModel):
    document_id: str
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    chunk_id: str
    element_id: str | None = None
    pdf_region_url: str


class ProviderStatus(ApiModel):
    id: str
    category: Literal["embedding", "ocr", "vision", "chat", "graph", "storage"]
    provider: str
    model: str | None = None
    enabled: bool
    configured: bool
    health: Literal["unknown", "healthy", "unavailable", "disabled"]
    credential_ref: str | None = None
    detail: str | None = None


class RagSettings(ApiModel):
    dense_top_k: int = Field(default=12, ge=1, le=100)
    lexical_top_k: int = Field(default=12, ge=1, le=100)
    rerank_top_k: int = Field(default=8, ge=1, le=50)
    graph_hops: int = Field(default=2, ge=0, le=5)
    max_tool_calls: int = Field(default=8, ge=1, le=30)
    citation_required: bool = True


class AgentFrameworkStatus(ApiModel):
    name: Literal["deepagents"] = "deepagents"
    mode: Literal["deepagents", "bounded_grounding_validator"]
    available: bool
    code: str
    versions: dict[str, str] = Field(default_factory=dict)
    detail: str


class SystemSettings(ApiModel):
    providers: list[ProviderStatus]
    rag: RagSettings
    compiler: dict[str, Any]
    agent_framework: AgentFrameworkStatus
    updated_at: datetime = Field(default_factory=utc_now)


class SettingsPatch(ApiModel):
    providers: list[ProviderStatus] | None = None
    rag: RagSettings | None = None
    compiler: dict[str, Any] | None = None


class ConnectionTestRequest(ApiModel):
    provider_id: str


class ProviderCredentialUpdate(ApiModel):
    provider_id: Literal[
        "embedding-primary",
        "embedding-openai",
        "vision-chat",
        "answer-primary",
    ]
    api_key: SecretStr


class ProviderCredentialStatus(ApiModel):
    provider_id: str
    configured: bool
    detail: str


class ConnectionTestResponse(ApiModel):
    provider_id: str
    reachable: bool
    health: Literal["healthy", "unavailable", "disabled"]
    detail: str
