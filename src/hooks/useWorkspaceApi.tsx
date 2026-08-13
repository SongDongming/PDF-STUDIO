import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, API_BASE_URL, ApiError } from "../api/client";
import type {
  ApiDocument,
  GraphView,
  HealthResponse,
  Job,
  KnowledgeBase,
  RuntimeMode,
  SystemSettings,
  Thread,
  WikiPage,
  WikiPageSummary,
} from "../api/types";

interface WorkspaceApiContextValue {
  apiBaseUrl: string;
  mode: RuntimeMode;
  health: HealthResponse | null;
  error: string;
  knowledgeBases: KnowledgeBase[];
  activeKnowledgeBase: KnowledgeBase | null;
  documents: ApiDocument[];
  jobs: Job[];
  threads: Thread[];
  settings: SystemSettings | null;
  wikiPages: WikiPageSummary[];
  graph: GraphView | null;
  refresh: () => Promise<void>;
  uploadDocuments: (
    files: File[],
  ) => Promise<{ documents: ApiDocument[]; job: Job }>;
  compileDocument: (documentId: string) => Promise<Job>;
  compileKnowledgeBase: () => Promise<Job>;
  deleteDocument: (documentId: string) => Promise<void>;
  retryJob: (jobId: string) => Promise<Job>;
  deleteJob: (jobId: string) => Promise<void>;
  createThread: (title?: string) => Promise<Thread>;
  renameThread: (threadId: string, title: string) => Promise<Thread>;
  archiveThread: (threadId: string) => Promise<Thread>;
  loadMessages: typeof api.listMessages;
  loadWikiPage: (slug: string) => Promise<WikiPage>;
  saveSettings: (patch: Partial<SystemSettings>) => Promise<SystemSettings>;
  testConnection: typeof api.testConnection;
  updateCredential: typeof api.updateCredential;
}

const WorkspaceApiContext = createContext<WorkspaceApiContextValue | null>(null);

function errorMessage(error: unknown) {
  if (error instanceof ApiError && error.status === 0) return "后端未连接";
  return error instanceof Error ? error.message : "后端请求失败";
}

export function WorkspaceApiProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<RuntimeMode>("connecting");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState("");
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [activeKnowledgeBase, setActiveKnowledgeBase] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<ApiDocument[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [wikiPages, setWikiPages] = useState<WikiPageSummary[]>([]);
  const [graph, setGraph] = useState<GraphView | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    // React StrictMode intentionally runs an extra setup/cleanup cycle in
    // development.  Resetting the guard in setup prevents that probe from
    // permanently suppressing the first successful API refresh.
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const nextHealth = await api.health();
      let bases = await api.listKnowledgeBases();
      if (!bases.length) {
        bases = [
          await api.createKnowledgeBase(
            "多模态 PDF 研究库",
            "PaddleOCR、DeepSeek 与 Agentic RAG 的可追溯文档知识库",
          ),
        ];
      }
      // Preserve the currently active knowledge base when it still exists;
      // otherwise fall back to the first one.  Without this, any refresh after
      // a document delete jumps to bases[0], making the graph look unchanged.
      const currentId = activeKnowledgeBase?.id;
      const active = bases.find((item) => item.id === currentId) ?? bases[0];
      const [nextDocuments, nextJobs, nextThreads, nextSettings, nextWiki, nextGraph] =
        await Promise.all([
          api.listDocuments(active.id),
          api.listJobs(),
          api.listThreads(active.id),
          api.getSettings(),
          api.listWikiPages(active.id),
          api.getGraph(active.id),
        ]);
      if (!mounted.current) return;
      setHealth(nextHealth);
      setKnowledgeBases(bases);
      setActiveKnowledgeBase(active);
      setDocuments(nextDocuments);
      setJobs(nextJobs);
      setThreads(nextThreads.filter((thread) => thread.status === "active"));
      setSettings(nextSettings);
      setWikiPages(nextWiki);
      setGraph(nextGraph);
      setMode("live");
      setError("");
    } catch (nextError) {
      if (!mounted.current) return;
      setMode("offline");
      setError(errorMessage(nextError));
    }
  }, [activeKnowledgeBase]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (mode !== "live") return;
    const hasActiveJob = jobs.some((job) => job.status === "queued" || job.status === "running");
    const timer = window.setInterval(() => {
      const base = activeKnowledgeBase;
      if (!base) return;
      void api
        .listKnowledgeBases()
        .then(async (bases) => {
          // Re-list first so a deleted knowledge base resolves to a surviving
          // one (or clears the workspace when none remain) before fetching its
          // documents/wiki/graph.  Using the old base id would 404 on getGraph
          // and silently leave stale graph/wiki data on screen.
          const nextActive =
            bases.find((item) => item.id === base.id) ?? bases[0] ?? null;
          if (!nextActive) {
            if (!mounted.current) return;
            setKnowledgeBases([]);
            setActiveKnowledgeBase(null);
            setDocuments([]);
            setWikiPages([]);
            setGraph(null);
            return;
          }
          const [nextDocuments, nextJobs, nextWiki, nextGraph] = await Promise.all([
            api.listDocuments(nextActive.id),
            api.listJobs(),
            api.listWikiPages(nextActive.id),
            api.getGraph(nextActive.id),
          ]);
          if (!mounted.current) return;
          setKnowledgeBases(bases);
          setActiveKnowledgeBase(nextActive);
          setDocuments(nextDocuments);
          setJobs(nextJobs);
          setWikiPages(nextWiki);
          setGraph(nextGraph);
        })
        .catch(() => {
          if (!mounted.current) return;
          setMode("offline");
          setError("后端连接中断，正在等待自动重连…");
        });
    }, hasActiveJob ? 2_500 : 10_000);
    return () => window.clearInterval(timer);
  }, [activeKnowledgeBase, jobs, mode]);

  // Reconnection: while offline, keep probing /health so a recovered backend is
  // picked up without a full page reload.
  useEffect(() => {
    if (mode !== "offline") return;
    const timer = window.setInterval(() => {
      void api
        .health()
        .then(() => {
          if (mounted.current) void refresh();
        })
        .catch(() => undefined);
    }, 8_000);
    return () => window.clearInterval(timer);
  }, [mode, refresh]);

  const requireKnowledgeBase = () => {
    if (!activeKnowledgeBase) throw new ApiError("尚无可用知识库", 409);
    return activeKnowledgeBase;
  };

  const uploadDocuments = useCallback(async (files: File[]) => {
    const base = requireKnowledgeBase();
    const created: ApiDocument[] = [];
    const failures: string[] = [];
    for (const file of files) {
      try {
        created.push(await api.uploadDocument(base.id, file));
      } catch {
        failures.push(file.name);
      }
    }
    // Commit successful uploads immediately so a partial failure does not lose
    // them; then report the failures explicitly.
    setDocuments((current) => [...created, ...current]);
    if (!created.length) {
      throw new ApiError("所有文件上传失败，请检查文件格式或后端状态", 0);
    }
    const job = await api.compileKnowledgeBase(base.id);
    setJobs((current) => [job, ...current]);
    setActiveKnowledgeBase((current) =>
      current?.id === base.id ? { ...current, status: "compiling" } : current,
    );
    if (failures.length) {
      throw new ApiError(
        `有 ${failures.length} 个文件上传失败：${failures.join("、")}。已上传成功的文档已进入编译队列。`,
        0,
      );
    }
    return { documents: created, job };
  }, [activeKnowledgeBase]);

  const compileDocument = useCallback(async (documentId: string) => {
    const job = await api.compileDocument(documentId);
    setJobs((current) => [job, ...current]);
    setDocuments((current) =>
      current.map((document) =>
        document.id === documentId ? { ...document, status: "queued" } : document,
      ),
    );
    return job;
  }, []);

  const compileKnowledgeBase = useCallback(async () => {
    const base = requireKnowledgeBase();
    const job = await api.compileKnowledgeBase(base.id);
    setJobs((current) => [job, ...current]);
    return job;
  }, [activeKnowledgeBase]);

  const deleteJob = useCallback(async (jobId: string) => {
    await api.deleteJob(jobId);
    setJobs((current) => current.filter((item) => item.id !== jobId));
  }, []);

  const deleteDocument = useCallback(async (documentId: string) => {
    await api.deleteDocument(documentId);
    setDocuments((current) =>
      current.filter((item) => item.id !== documentId),
    );
    // Cascade: the delete also rewrites the knowledge graph, LLM Wiki and
    // index, so re-pull the whole workspace immediately instead of waiting for
    // the next poll.
    await refresh();
  }, [refresh]);

  const retryJob = useCallback(async (jobId: string) => {
    const job = await api.retryJob(jobId);
    setJobs((current) => [job, ...current]);
    return job;
  }, []);

  const createThread = useCallback(async (title = "新会话") => {
    const base = requireKnowledgeBase();
    const thread = await api.createThread(base.id, title);
    setThreads((current) => [thread, ...current]);
    return thread;
  }, [activeKnowledgeBase]);

  const renameThread = useCallback(async (threadId: string, title: string) => {
    const thread = await api.updateThread(threadId, { title });
    setThreads((current) =>
      current.map((item) => (item.id === threadId ? thread : item)),
    );
    return thread;
  }, []);

  const archiveThread = useCallback(async (threadId: string) => {
    const thread = await api.updateThread(threadId, { status: "archived" });
    setThreads((current) => current.filter((item) => item.id !== threadId));
    return thread;
  }, []);

  const loadWikiPage = useCallback(async (slug: string) => {
    const base = requireKnowledgeBase();
    return api.getWikiPage(base.id, slug);
  }, [activeKnowledgeBase]);

  const saveSettings = useCallback(async (patch: Partial<SystemSettings>) => {
    const next = await api.updateSettings(patch);
    setSettings(next);
    return next;
  }, []);

  const value = useMemo<WorkspaceApiContextValue>(() => ({
    apiBaseUrl: API_BASE_URL,
    mode,
    health,
    error,
    knowledgeBases,
    activeKnowledgeBase,
    documents,
    jobs,
    threads,
    settings,
    wikiPages,
    graph,
    refresh,
    uploadDocuments,
    compileDocument,
    compileKnowledgeBase,
    deleteDocument,
    retryJob,
    deleteJob,
    createThread,
    renameThread,
    archiveThread,
    loadMessages: api.listMessages,
    loadWikiPage,
    saveSettings,
    testConnection: api.testConnection,
    updateCredential: api.updateCredential,
  }), [
    mode, health, error, knowledgeBases, activeKnowledgeBase, documents, jobs,
    threads, settings, wikiPages, graph, refresh, uploadDocuments,
    compileDocument, compileKnowledgeBase, deleteDocument,
    retryJob, deleteJob, createThread, renameThread, archiveThread, loadWikiPage,
    saveSettings,
  ]);

  return <WorkspaceApiContext.Provider value={value}>{children}</WorkspaceApiContext.Provider>;
}

export function useWorkspaceApi() {
  const value = useContext(WorkspaceApiContext);
  if (!value) throw new Error("useWorkspaceApi 必须在 WorkspaceApiProvider 内使用");
  return value;
}
