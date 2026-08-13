import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api/client";
import type { Citation, Message } from "./api/types";
import { ChatPanel } from "./components/ChatPanel";
import { DocumentSidebar } from "./components/DocumentSidebar";
import { EvidenceDrawer } from "./components/EvidenceRail";
import { Icon } from "./components/Icons";
import { PdfViewer } from "./components/PdfViewer";
import { ProductHeader } from "./components/ProductHeader";
import { ProductNav, type ProductRoute } from "./components/ProductNav";
import { SessionHeader } from "./components/SessionHeader";
import { KnowledgeBasePage } from "./pages/KnowledgeBasePage";
import { KnowledgeGraphPage } from "./pages/KnowledgeGraphPage";
import { ArchitecturePage } from "./pages/ArchitecturePage";
import { SettingsPage } from "./pages/SettingsPage";
import { TasksPage } from "./pages/TasksPage";
import { WikiPage } from "./pages/WikiPage";
import {
  useWorkspaceApi,
  WorkspaceApiProvider,
} from "./hooks/useWorkspaceApi";
import type {
  ChatTurn,
  Evidence,
  PdfDocument,
  PdfViewerTarget,
  Session,
  SourceRef,
} from "./types";

function messageMarkdown(message: Message) {
  return message.blocks
    .filter((block) => block.type === "text")
    .map((block) => ("markdown" in block ? block.markdown : ""))
    .join("\n\n");
}

function messageAnswer(message: Message) {
  const text = messageMarkdown(message);
  return text || "回答已完成。";
}

function messageAssets(message: Message) {
  return message.blocks
    .filter((block) => block.type !== "text")
    .map((block) => ({
      type: block.type,
      assetId: block.asset_id,
      caption: block.caption,
      alt: block.alt,
    }));
}

function messageBlocks(message: Message) {
  return message.blocks.map((block) =>
    block.type === "text"
      ? { type: "text" as const, markdown: block.markdown }
      : {
          type: block.type,
          assetId: block.asset_id,
          caption: block.caption,
          alt: block.alt,
        },
  );
}

function citationKind(citation: Citation) {
  if (citation.element_id?.includes("table")) return ["表格", "table"] as const;
  if (
    citation.element_id?.includes("figure") ||
    citation.element_id?.includes("image")
  )
    return ["图表", "figure"] as const;
  if (citation.element_id?.includes("formula")) return ["公式", "diagram"] as const;
  return ["原文", "excerpt"] as const;
}

const productRoutes: ProductRoute[] = [
  "chat",
  "architecture",
  "knowledge",
  "wiki",
  "graph",
  "tasks",
  "settings",
];

function AppWorkspace() {
  const workspace = useWorkspaceApi();
  const visibleDocuments = useMemo<PdfDocument[]>(
    () =>
      workspace.documents.map((document) => ({
        id: document.id,
        filename: document.filename,
        pages: document.page_count ?? 1,
      })),
    [workspace.documents],
  );
  const [activeDocumentId, setActiveDocumentId] = useState("");
  const [liveEvidences, setLiveEvidences] = useState<Record<string, Evidence[]>>({});
  const [liveSources, setLiveSources] = useState<Record<string, SourceRef[]>>({});
  const [evidenceTurnId, setEvidenceTurnId] = useState<string | null>(null);
  const [sessionItems, setSessionItems] = useState<Session[]>([]);
  const [activeSessionIndex, setActiveSessionIndex] = useState(0);
  const [threads, setThreads] = useState<Record<string, ChatTurn[]>>({});
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState("");
  const requestInFlightRef = useRef<string | null>(null);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [pdfTarget, setPdfTarget] = useState<PdfViewerTarget | null>(null);
  const [route, setRoute] = useState<ProductRoute>(() => {
    const hashRoute = window.location.hash.replace("#/", "") as ProductRoute;
    return productRoutes.includes(hashRoute) ? hashRoute : "chat";
  });
  const [navCompact, setNavCompact] = useState(false);

  useEffect(() => {
    if (workspace.mode !== "live") return;
    const apiSessions: Session[] = workspace.threads.map((thread) => {
      const preview = thread.scope.preview;
      return {
        id: thread.id,
        title: thread.title,
        question: typeof preview === "string" ? preview : "",
      };
    });
    setSessionItems(apiSessions);
    setActiveSessionIndex((current) =>
      Math.min(current, Math.max(0, apiSessions.length - 1)),
    );
    setThreads((current) => {
      const next: Record<string, ChatTurn[]> = {};
      for (const session of apiSessions) next[session.id] = current[session.id] ?? [];
      return next;
    });
  }, [workspace.mode, workspace.threads]);

  const activeSession = sessionItems[activeSessionIndex];

  useEffect(() => {
    setEvidenceTurnId(null);
  }, [activeSession?.id]);

  useEffect(() => {
    if (workspace.mode !== "live" || !activeSession) return;
    // Live mode must only load messages for real threads; demo session ids
    // (adaptive-rag etc.) would 404 against the backend.
    if (!workspace.threads.some((thread) => thread.id === activeSession.id)) return;
    let cancelled = false;
    void workspace
      .loadMessages(activeSession.id)
      .then((messages) => {
        if (cancelled) return;
        const restored: ChatTurn[] = [];
        const evidenceByTurn: Record<string, Evidence[]> = {};
        const sourcesByTurn: Record<string, SourceRef[]> = {};
        let pendingQuestion = "";
        const accents: Evidence["accent"][] = [
          "blue",
          "green",
          "violet",
          "orange",
          "rose",
        ];
        const tones: SourceRef["tone"][] = ["blue", "green", "violet"];
        for (const message of messages) {
          if (message.role === "user") {
            pendingQuestion = messageMarkdown(message);
            continue;
          }
          if (message.role !== "assistant") continue;
          restored.push({
            id: message.id,
            question: pendingQuestion || "继续分析",
            answer: messageAnswer(message),
            createdAt: new Date(message.created_at).toLocaleTimeString("zh-CN", {
              hour: "2-digit",
              minute: "2-digit",
              hour12: false,
            }),
            rich: message.blocks.some((block) => block.type !== "text"),
            blocks: messageBlocks(message),
            assets: messageAssets(message),
          });
          const citations = message.citations.slice(0, 5);
          evidenceByTurn[message.id] = citations.map((citation, index) => {
            const [kindLabel, kind] = citationKind(citation);
            return {
              id: citation.id,
              order: index + 1,
              kind,
              kindLabel,
              relevance: Math.round((citation.score ?? 0) * 100),
              page: citation.page,
              filename: citation.document_title,
              summary: citation.excerpt,
              documentId: citation.document_id,
              elementId: citation.element_id,
              accent: accents[index % accents.length],
            };
          });
          sourcesByTurn[message.id] = citations.map((citation, index) => ({
            id: `source-${citation.id}`,
            evidenceId: citation.id,
            page: citation.page,
            filename: citation.document_title,
            tone: tones[index % tones.length],
          }));
          pendingQuestion = "";
        }
        setThreads((current) => ({
          ...current,
          [activeSession.id]: restored,
        }));
        setLiveEvidences((current) => ({
          ...current,
          ...evidenceByTurn,
        }));
        setLiveSources((current) => ({
          ...current,
          ...sourcesByTurn,
        }));
        setSessionError("");
      })
      .catch(() => {
        if (!cancelled) setSessionError("历史消息加载失败，可稍后重试。");
      });
    return () => {
      cancelled = true;
    };
  }, [activeSession?.id, workspace.loadMessages, workspace.mode]);

  const activeTurns = useMemo(
    () => (activeSession ? threads[activeSession.id] ?? [] : []),
    [activeSession, threads],
  );
  const evidenceTurn = activeTurns.find((turn) => turn.id === evidenceTurnId);
  const visibleEvidences = evidenceTurnId
    ? liveEvidences[evidenceTurnId] ?? []
    : [];
  const visibleSourcesByTurn = liveSources;

  useEffect(() => {
    const syncRoute = () => {
      const hashRoute = window.location.hash.replace("#/", "") as ProductRoute;
      if (productRoutes.includes(hashRoute)) setRoute(hashRoute);
    };
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  const openDocument = (
    document: PdfDocument,
    page = 1,
    elementId?: string | null,
    relevance?: number,
  ) => {
    setActiveDocumentId(document.id);
    setPdfTarget({
      documentId: document.id,
      filename: document.filename,
      page,
      totalPages: document.pages,
      elementId,
      relevance,
    });
  };

  const openDocumentByFilename = (
    filename: string,
    page: number,
    elementId?: string | null,
    relevance?: number,
  ) => {
    const matchedDocument =
      visibleDocuments.find((document) => document.filename === filename) ??
      visibleDocuments.find((document) => filename.includes(document.filename.replace(".pdf", ""))) ??
      visibleDocuments[0];
    if (matchedDocument) openDocument(matchedDocument, page, elementId, relevance);
  };

  const handleEvidenceOpen = (evidence: Evidence) => {
    const matchedDocument = visibleDocuments.find(
      (document) => document.id === evidence.documentId,
    );
    if (matchedDocument) {
      openDocument(
        matchedDocument,
        evidence.page,
        evidence.elementId,
        evidence.relevance,
      );
      return;
    }
    openDocumentByFilename(
      evidence.filename,
      evidence.page,
      evidence.elementId,
      evidence.relevance,
    );
  };

  const handleNewChat = async () => {
    try {
      const thread = await workspace.createThread();
      const newSession: Session = { id: thread.id, title: thread.title, question: "" };
      setSessionItems((current) => [newSession, ...current]);
      setThreads((current) => ({ ...current, [thread.id]: [] }));
      setActiveSessionIndex(0);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "未知错误";
      window.alert(`创建会话失败：${detail}`);
    }
  };

  const handleRenameSession = async () => {
    if (!activeSession) return;
    const title = window.prompt("重命名当前会话", activeSession.title)?.trim();
    if (!title || title === activeSession.title) return;
    if (workspace.mode === "live" && !activeSession.id.startsWith("session-")) {
      try {
        await workspace.renameThread(activeSession.id, title);
      } catch {
        setSessionError("重命名会话失败，请重试。");
        return;
      }
    }
    setSessionItems((current) =>
      current.map((session) =>
        session.id === activeSession.id ? { ...session, title } : session,
      ),
    );
  };

  const handleArchiveSession = async () => {
    if (!activeSession || !window.confirm(`归档会话“${activeSession.title}”？`)) return;
    if (workspace.mode === "live" && !activeSession.id.startsWith("session-")) {
      try {
        await workspace.archiveThread(activeSession.id);
      } catch {
        setSessionError("归档会话失败，请重试。");
        return;
      }
    }
    setSessionItems((current) =>
      current.filter((session) => session.id !== activeSession.id),
    );
    setActiveSessionIndex((current) => Math.max(0, current - 1));
  };

  const handleAsk = async (question: string) => {
    if (!activeSession || requestInFlightRef.current) return;
    const sessionId = activeSession.id;
    const isLiveRequest =
      workspace.mode === "live" && !sessionId.startsWith("session-");
    if (!isLiveRequest) {
      const failureMessage =
        "无法提问：后端未连接或会话创建失败。请确认 API 服务已启动后重试。";
      setThreads((current) => ({
        ...current,
        [sessionId]: [
          ...(current[sessionId] ?? []),
          {
            id: `${sessionId}-${Date.now()}`,
            question,
            answer: failureMessage,
            createdAt: new Date().toLocaleTimeString("zh-CN", {
              hour: "2-digit",
              minute: "2-digit",
              hour12: false,
            }),
            rich: false,
            pending: false,
            blocks: [{ type: "text", markdown: failureMessage }],
          },
        ],
      }));
      return;
    }
    requestInFlightRef.current = sessionId;
    setPendingSessionId(sessionId);
    const existingTurns = threads[sessionId] ?? [];
    const nextTurn: ChatTurn = {
      id: `${sessionId}-${Date.now()}`,
      question,
      answer: "",
      createdAt: new Date().toLocaleTimeString("zh-CN", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }),
      rich: existingTurns.length === 0,
      pending: true,
      startedAt: Date.now(),
      blocks: [],
    };

    setThreads((current) => ({
      ...current,
      [sessionId]: [...(current[sessionId] ?? []), nextTurn],
    }));
    setLiveEvidences((current) => ({ ...current, [nextTurn.id]: [] }));
    setLiveSources((current) => ({ ...current, [nextTurn.id]: [] }));

    if (!activeSession.question) {
      const nextTitle =
        question.length > 12 ? `${question.slice(0, 12)}…` : question;
      setSessionItems((current) =>
        current.map((session) =>
          session.id === sessionId
            ? { ...session, title: nextTitle, question }
            : session,
        ),
      );
    }

    if (!isLiveRequest) return;
    let completed = false;
    try {
      await api.streamMessage(sessionId, question, (event) => {
        if (event.type === "answer.started" || event.type === "agent.status") {
          if (!event.status) return;
          setThreads((current) => ({
            ...current,
            [sessionId]: (current[sessionId] ?? []).map((turn) =>
              turn.id === nextTurn.id
                ? { ...turn, statusText: event.status }
                : turn,
            ),
          }));
          return;
        }
        if (event.type === "answer.delta") {
          if (!event.delta) return;
          setThreads((current) => ({
            ...current,
            [sessionId]: (current[sessionId] ?? []).map((turn) => {
              if (turn.id !== nextTurn.id) return turn;
              const answer = `${turn.answer}${event.delta}`;
              return {
                ...turn,
                answer,
                blocks: [{ type: "text", markdown: answer }],
              };
            }),
          }));
          return;
        }
        if (
          (event.type !== "answer.completed" && event.type !== "answer.failed")
          || !event.message
        ) {
          return;
        }
        const finalMessage = event.message;
        completed = true;
        const accents: Evidence["accent"][] = ["blue", "green", "violet", "orange", "rose"];
        const tones: SourceRef["tone"][] = ["blue", "green", "violet"];
        const citations = finalMessage.citations.slice(0, 5);
        const nextEvidences: Evidence[] = citations.map((citation, index) => ({
          id: citation.id,
          order: index + 1,
          kind: "excerpt",
          kindLabel: citation.element_id?.includes("table")
            ? "表格"
            : citation.element_id?.includes("figure") || citation.element_id?.includes("image")
              ? "图表"
              : citation.element_id?.includes("formula")
                ? "公式"
                : "原文",
          relevance: Math.round((citation.score ?? 0) * 100),
          page: citation.page,
          filename: citation.document_title,
          summary: citation.excerpt,
          documentId: citation.document_id,
          elementId: citation.element_id,
          accent: accents[index % accents.length],
        }));
        const nextSources: SourceRef[] = citations.map((citation, index) => ({
          id: `source-${citation.id}`,
          evidenceId: citation.id,
          page: citation.page,
          filename: citation.document_title,
          tone: tones[index % tones.length],
        }));
        setLiveEvidences((current) => ({
          ...current,
          [nextTurn.id]: nextEvidences,
        }));
        setLiveSources((current) => ({
          ...current,
          [nextTurn.id]: nextSources,
        }));
        setThreads((current) => ({
          ...current,
          [sessionId]: (current[sessionId] ?? []).map((turn) =>
            turn.id === nextTurn.id
              ? {
                  ...turn,
                  pending: false,
                  statusText: undefined,
                  answer: messageAnswer(finalMessage),
                  rich: finalMessage.blocks.some(
                    (block) => block.type !== "text",
                  ),
                  blocks: messageBlocks(finalMessage),
                  assets: messageAssets(finalMessage),
                }
              : turn,
          ),
        }));
      });
      if (!completed) {
        throw new Error("回答流已结束，但未收到完成事件");
      }
    } catch (error) {
      const errorMessage = `真实问答请求失败：${error instanceof Error ? error.message : "未知错误"}。本条未使用演示答案替代。`;
      setThreads((current) => ({
        ...current,
        [sessionId]: (current[sessionId] ?? []).map((turn) =>
          turn.id === nextTurn.id
            ? {
                ...turn,
                pending: false,
                answer: errorMessage,
                blocks: [{ type: "text", markdown: errorMessage }],
              }
            : turn,
        ),
      }));
    } finally {
      requestInFlightRef.current = null;
      setPendingSessionId(null);
    }
  };

  const shellClassName = [
    "app-shell",
    "is-evidence-modal-layout",
    leftCollapsed ? "is-left-collapsed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const renderWorkspace = () => {
    if (route === "architecture") return <ArchitecturePage />;
    if (route === "knowledge") return <KnowledgeBasePage />;
    if (route === "wiki")
      return (
        <WikiPage
          onOpenPdf={(documentId, page, elementId) => {
            const document = visibleDocuments.find((item) => item.id === documentId);
            if (document) openDocument(document, page, elementId);
          }}
        />
      );
    if (route === "graph")
      return (
        <KnowledgeGraphPage
          onOpenPdf={(documentId, page, elementId) => {
            const document = visibleDocuments.find((item) => item.id === documentId);
            if (document) openDocument(document, page, elementId);
          }}
        />
      );
    if (route === "tasks") return <TasksPage />;
    if (route === "settings") return <SettingsPage />;

    return (
      <main className={shellClassName}>
        <DocumentSidebar
          documents={visibleDocuments}
          activeId={activeDocumentId}
          onSelect={setActiveDocumentId}
          onOpen={openDocument}
          onUpload={async (files) => {
            const { documents: uploaded } = await workspace.uploadDocuments(files);
            if (uploaded[0]) setActiveDocumentId(uploaded[0].id);
          }}
          collapsed={leftCollapsed}
          onToggle={() => setLeftCollapsed((value) => !value)}
        />

        <section className="panel chat-panel" aria-label="多模态问答会话">
          <SessionHeader
            sessions={sessionItems}
            activeIndex={activeSessionIndex}
            onChange={setActiveSessionIndex}
            onNewChat={handleNewChat}
            onRename={() => void handleRenameSession()}
            onArchive={() => void handleArchiveSession()}
          />
          {sessionError && (
            <div className="session-error-banner" role="status">
              <Icon name="info" size={15} />
              <span>{sessionError}</span>
              <button onClick={() => setSessionError("")} aria-label="关闭提示">
                ×
              </button>
            </div>
          )}
          <ChatPanel
            turns={activeTurns}
            sourcesByTurn={visibleSourcesByTurn}
            busy={pendingSessionId !== null}
            onEvidenceOpen={setEvidenceTurnId}
            onAsk={handleAsk}
          />
        </section>
      </main>
    );
  };

  return (
    <>
      <div className={`product-shell ${navCompact ? "is-nav-compact" : ""}`}>
        <ProductHeader />
        <ProductNav
          active={route}
          onChange={(nextRoute) => {
            setRoute(nextRoute);
            window.location.hash = `/${nextRoute}`;
          }}
          compact={navCompact}
          onToggle={() => setNavCompact((value) => !value)}
          serviceMode={workspace.mode}
        />
        <div className={`product-content ${route === "chat" ? "is-chat" : ""}`}>
          {renderWorkspace()}
        </div>
      </div>
      {workspace.mode !== "live" && workspace.error && (
        <div className="offline-banner" role="status">
          <Icon name="info" size={16} />
          <span>
            {workspace.error}
            {workspace.mode === "connecting"
              ? "，正在连接后端…"
              : "，会自动检测后端恢复并重连。"}
          </span>
        </div>
      )}
      {evidenceTurnId && (
        <EvidenceDrawer
          evidences={visibleEvidences}
          question={evidenceTurn?.question ?? "当前回答"}
          onOpen={handleEvidenceOpen}
          onClose={() => setEvidenceTurnId(null)}
        />
      )}
      <PdfViewer target={pdfTarget} onClose={() => setPdfTarget(null)} />
    </>
  );
}

export default function App() {
  return (
    <WorkspaceApiProvider>
      <AppWorkspace />
    </WorkspaceApiProvider>
  );
}
