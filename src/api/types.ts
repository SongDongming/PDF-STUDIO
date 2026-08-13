export type RuntimeMode = "connecting" | "live" | "offline";

export interface HealthResponse {
  status: "ok" | "degraded";
  service: string;
  version: string;
  environment: string;
  dependencies: Record<string, string>;
}

export type KnowledgeBaseStatus =
  | "draft"
  | "compiling"
  | "ready"
  | "partial"
  | "failed"
  | "archived";

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  status: KnowledgeBaseStatus;
  document_count: number;
  published_version: number;
  created_at: string;
  updated_at: string;
}

export type DocumentStatus =
  | "uploaded"
  | "queued"
  | "parsing"
  | "enriching"
  | "indexing"
  | "ready"
  | "failed"
  | "archived";

export interface ApiDocument {
  id: string;
  knowledge_base_id: string;
  filename: string;
  title: string;
  status: DocumentStatus;
  sha256: string | null;
  page_count: number | null;
  element_count: number;
  asset_count: number;
  chunk_count: number;
  active_version: number;
  created_at: string;
  updated_at: string;
  error_message: string | null;
}

export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "partial"
  | "failed"
  | "canceled";

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  stage: string;
  progress: number;
  knowledge_base_id: string | null;
  document_id: string | null;
  error_code: string | null;
  error_message: string | null;
  retryable: boolean;
  attempt: number;
  retry_of: string | null;
  superseded_by: string | null;
  is_current: boolean;
  result: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface Citation {
  id: string;
  document_id: string;
  document_title: string;
  page: number;
  bbox: [number, number, number, number] | null;
  element_id: string | null;
  excerpt: string;
  score: number | null;
}

export type AnswerBlock =
  | { type: "text"; markdown: string }
  | {
      type: "image" | "table" | "formula";
      asset_id: string;
      caption: string | null;
      alt: string;
    };

export interface Thread {
  id: string;
  knowledge_base_id: string;
  title: string;
  status: "active" | "archived";
  scope: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  thread_id: string;
  role: "user" | "assistant" | "system";
  blocks: AnswerBlock[];
  citations: Citation[];
  created_at: string;
}

export interface WikiPageSummary {
  id: string;
  knowledge_base_id: string;
  slug: string;
  title: string;
  summary: string;
  status: "draft" | "published" | "locked";
  updated_at: string;
}

export interface WikiPage extends WikiPageSummary {
  markdown: string;
  citations: Citation[];
  related_page_ids: string[];
}

export interface GraphNode {
  id: string;
  label: string;
  kind: "entity" | "claim" | "document" | "page" | "chunk" | "asset" | "wiki";
  properties: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  evidence_ids: string[];
  properties: Record<string, unknown>;
}

export interface GraphView {
  knowledge_base_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  graph_version: number;
  total_node_count: number;
  total_edge_count: number;
  truncated: boolean;
}

export interface EvidenceLineage {
  document_id: string;
  page: number;
  bbox: [number, number, number, number];
  chunk_id: string;
  element_id: string | null;
  pdf_region_url: string;
}

export interface ProviderStatus {
  id: string;
  category: "embedding" | "ocr" | "vision" | "chat" | "graph" | "storage";
  provider: string;
  model: string | null;
  enabled: boolean;
  configured: boolean;
  health: "unknown" | "healthy" | "unavailable" | "disabled";
  credential_ref: string | null;
  detail: string | null;
}

export interface RagSettings {
  dense_top_k: number;
  lexical_top_k: number;
  rerank_top_k: number;
  graph_hops: number;
  max_tool_calls: number;
  citation_required: boolean;
}

export interface SystemSettings {
  providers: ProviderStatus[];
  rag: RagSettings;
  compiler: Record<string, unknown>;
  updated_at: string;
}

export interface ConnectionTest {
  provider_id: string;
  reachable: boolean;
  health: "healthy" | "unavailable" | "disabled";
  detail: string;
}

export interface ProviderCredentialStatus {
  provider_id: string;
  configured: boolean;
  detail: string;
}
