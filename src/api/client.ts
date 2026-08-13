import type {
  ApiDocument,
  ConnectionTest,
  GraphView,
  EvidenceLineage,
  HealthResponse,
  Job,
  KnowledgeBase,
  Message,
  ProviderCredentialStatus,
  SystemSettings,
  Thread,
  WikiPage,
  WikiPageSummary,
} from "./types";

const host = typeof window === "undefined" ? "127.0.0.1" : window.location.hostname;
export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  `http://${host}:18800/api/v1`;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData) && init?.body !== undefined) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      signal: init?.signal ?? AbortSignal.timeout(12_000),
    });
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : "无法连接后端服务",
      0,
    );
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    const message =
      (detail as { detail?: { message?: string } } | null)?.detail?.message ??
      `请求失败（HTTP ${response.status}）`;
    throw new ApiError(message, response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<HealthResponse>("/health", { signal }),

  listKnowledgeBases: () => request<KnowledgeBase[]>("/knowledge-bases"),
  createKnowledgeBase: (name: string, description = "") =>
    request<KnowledgeBase>("/knowledge-bases", {
      method: "POST",
      body: JSON.stringify({ name, description }),
    }),
  compileKnowledgeBase: (knowledgeBaseId: string) =>
    request<Job>(`/knowledge-bases/${knowledgeBaseId}/compile`, { method: "POST" }),
  deleteDocument: (documentId: string) =>
    request<void>(`/documents/${documentId}`, { method: "DELETE" }),

  listDocuments: (knowledgeBaseId: string) =>
    request<ApiDocument[]>(`/knowledge-bases/${knowledgeBaseId}/documents`),
  uploadDocument: (knowledgeBaseId: string, file: File, title?: string) => {
    const body = new FormData();
    body.append("file", file);
    if (title) body.append("title", title);
    return request<ApiDocument>(
      `/knowledge-bases/${knowledgeBaseId}/documents/upload`,
      { method: "POST", body },
    );
  },
  compileDocument: (documentId: string) =>
    request<Job>(`/documents/${documentId}/compile`, { method: "POST" }),

  listJobs: () => request<Job[]>("/jobs"),
  retryJob: (jobId: string) =>
    request<Job>(`/jobs/${jobId}/retry`, { method: "POST" }),
  deleteJob: (jobId: string) =>
    request<void>(`/jobs/${jobId}`, { method: "DELETE" }),

  listThreads: (knowledgeBaseId?: string) =>
    request<Thread[]>(
      `/chat/threads${knowledgeBaseId ? `?knowledge_base_id=${encodeURIComponent(knowledgeBaseId)}` : ""}`,
    ),
  createThread: (knowledgeBaseId: string, title = "新会话") =>
    request<Thread>("/chat/threads", {
      method: "POST",
      body: JSON.stringify({ knowledge_base_id: knowledgeBaseId, title, scope: {} }),
    }),
  updateThread: (threadId: string, changes: { title?: string; status?: "active" | "archived" }) =>
    request<Thread>(`/chat/threads/${threadId}`, {
      method: "PATCH",
      body: JSON.stringify(changes),
    }),
  listMessages: (threadId: string) =>
    request<Message[]>(`/chat/threads/${threadId}/messages`),
  streamMessage: async (
    threadId: string,
    content: string,
    onEvent: (event: {
      type: string;
      message?: Message;
      status?: string;
      delta?: string;
    }) => void,
    signal?: AbortSignal,
  ) => {
    // A hung stream must not block future asks forever; the caller resets its
    // in-flight guard only after this resolves or rejects.
    const controller = new AbortController();
    const streamTimeout = window.setTimeout(() => controller.abort(), 120_000);
    const forwardAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener("abort", forwardAbort, { once: true });
    }
    const cleanup = () => {
      window.clearTimeout(streamTimeout);
      signal?.removeEventListener("abort", forwardAbort);
    };
    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}/chat/threads/${threadId}/messages/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ content, stream: true }),
        signal: controller.signal,
      });
    } catch (error) {
      cleanup();
      if (controller.signal.aborted) {
        throw new ApiError("流式请求超时或已取消", 0);
      }
      throw new ApiError(error instanceof Error ? error.message : "流式连接失败", 0);
    }
    if (!response.ok || !response.body) {
      cleanup();
      throw new ApiError(`流式请求失败（HTTP ${response.status}）`, response.status);
    }
    try {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const data = frame
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("\n");
          if (!data) continue;
          try {
            onEvent(JSON.parse(data) as {
              type: string;
              message?: Message;
              status?: string;
              delta?: string;
            });
          } catch {
            // Skip a malformed SSE frame instead of killing the whole stream.
          }
        }
        if (done) break;
      }
    } catch (error) {
      if (controller.signal.aborted) {
        throw new ApiError("流式请求超时或已取消", 0);
      }
      throw error;
    } finally {
      cleanup();
    }
  },

  listWikiPages: (knowledgeBaseId: string) =>
    request<WikiPageSummary[]>(`/knowledge-bases/${knowledgeBaseId}/wiki/pages`),
  getWikiPage: (knowledgeBaseId: string, slug: string) =>
    request<WikiPage>(`/knowledge-bases/${knowledgeBaseId}/wiki/pages/${encodeURIComponent(slug)}`),
  getGraph: (
    knowledgeBaseId: string,
    hops = 2,
    query?: string,
    nodeId?: string,
    limit = 80,
  ) => {
    const params = new URLSearchParams({
      hops: String(hops),
      limit: String(limit),
    });
    if (query?.trim()) params.set("query", query.trim());
    if (nodeId) params.set("node_id", nodeId);
    return request<GraphView>(
      `/knowledge-bases/${knowledgeBaseId}/graph?${params.toString()}`,
    );
  },
  getGraphEvidence: (knowledgeBaseId: string, nodeId: string) =>
    request<EvidenceLineage[]>(
      `/knowledge-bases/${knowledgeBaseId}/graph/nodes/${encodeURIComponent(nodeId)}/evidence`,
    ),

  getSettings: () => request<SystemSettings>("/settings"),
  updateSettings: (patch: Partial<SystemSettings>) =>
    request<SystemSettings>("/settings", {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  testConnection: (providerId: string) =>
    request<ConnectionTest>("/settings/connection-tests", {
      method: "POST",
      body: JSON.stringify({ provider_id: providerId }),
    }),
  updateCredential: (providerId: string, apiKey: string) =>
    request<ProviderCredentialStatus>("/settings/credentials", {
      method: "POST",
      body: JSON.stringify({ provider_id: providerId, api_key: apiKey }),
    }),
};
