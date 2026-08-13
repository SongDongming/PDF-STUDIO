import {
  Fragment,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatTurn, SourceRef } from "../types";
import { API_BASE_URL } from "../api/client";
import { Icon } from "./Icons";

function MarkdownContent({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  return (
    <div className={`chat-markdown ${className ?? ""}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

interface ChatPanelProps {
  turns: ChatTurn[];
  sourcesByTurn: Record<string, SourceRef[]>;
  busy: boolean;
  onEvidenceOpen: (turnId: string) => void;
  onAsk: (question: string) => void;
}

function MessageIdentity({
  kind,
  label,
  time,
}: {
  kind: "user" | "assistant";
  label: string;
  time: string;
}) {
  return (
    <div className={`message-identity ${kind}`}>
      <span className="identity-mark">
        <Icon name={kind === "user" ? "user" : "sparkles"} size={17} />
      </span>
      <strong>{label}</strong>
      <span className="message-identity-meta">
        <time>{time}</time>
      </span>
    </div>
  );
}

function TurnEvidenceSummary({
  count,
  onOpen,
}: {
  count: number;
  onOpen: () => void;
}) {
  if (!count) {
    return (
      <div className="turn-evidence-summary is-empty">
        <span><Icon name="sparkles" size={16} /></span>
        <div>
          <strong>本轮未调用知识库</strong>
          <small>Agent 使用模型自身知识完成了这条回复</small>
        </div>
      </div>
    );
  }
  return (
    <button className="turn-evidence-summary" type="button" onClick={onOpen}>
      <span><Icon name="sparkles" size={17} /></span>
      <div>
        <strong>本轮线索</strong>
        <small>这条 Agent 回复使用了 {count} 条检索依据</small>
      </div>
      <em>{count} 条</em>
      <Icon name="arrowRight" size={17} />
    </button>
  );
}

function PendingAnswer({
  startedAt,
  statusText,
  hasContent,
}: {
  startedAt: number;
  statusText?: string;
  hasContent: boolean;
}) {
  const [elapsedSeconds, setElapsedSeconds] = useState(() =>
    Math.max(0, Math.floor((Date.now() - startedAt) / 1000)),
  );

  useEffect(() => {
    const updateElapsed = () =>
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1_000);
    return () => window.clearInterval(timer);
  }, [startedAt]);

  const minutes = Math.floor(elapsedSeconds / 60);
  const seconds = String(elapsedSeconds % 60).padStart(2, "0");

  return (
    <div className="agent-waiting" role="status" aria-live="polite">
      <span className="agent-waiting-mark" aria-hidden="true">
        <Icon name="sparkles" size={20} />
        <i />
      </span>
      <div>
        <strong>{hasContent ? "Deep Agents 正在生成回答" : "Deep Agents 正在分析"}</strong>
        <small>{statusText ?? "正在判断是否需要调用知识库，并组织本轮回答"}</small>
      </div>
      <time>{minutes}:{seconds}</time>
    </div>
  );
}

function LiveAnswer({
  turn,
}: {
  turn: ChatTurn;
}) {
  const blocks =
    turn.blocks && turn.blocks.length > 0
      ? turn.blocks
      : [
          { type: "text" as const, markdown: turn.answer },
          ...(turn.assets ?? []).map((asset) => ({ ...asset })),
        ];
  let visualIndex = 0;

  return (
    <>
      <div className="live-answer-flow">
        {blocks.map((block, index) => {
          if (block.type === "text") {
            return (
              <div
                className="answer-copy live-answer-copy"
                key={`${turn.id}-text-${index}`}
              >
                <MarkdownContent content={block.markdown} />
              </div>
            );
          }

          visualIndex += 1;
          const label =
            block.type === "image"
              ? `图 ${visualIndex}`
              : block.type === "table"
                ? `表 ${visualIndex}`
                : `公式 ${visualIndex}`;
          return (
            <figure
              className={`live-answer-asset live-answer-asset-${block.type}`}
              key={`${turn.id}-${block.assetId}-${index}`}
            >
              <img
                src={`${API_BASE_URL}/assets/${encodeURIComponent(block.assetId)}`}
                alt={block.alt}
                loading="lazy"
              />
              <figcaption>
                <strong>{label}</strong>
                <span>{block.caption ?? block.alt}</span>
              </figcaption>
            </figure>
          );
        })}
      </div>
    </>
  );
}

export function ChatPanel({
  turns,
  sourcesByTurn,
  busy,
  onEvidenceOpen,
  onAsk,
}: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const previousTurnCount = useRef(turns.length);

  useEffect(() => {
    if (turns.length === previousTurnCount.current + 1) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
    previousTurnCount.current = turns.length;
  }, [turns.length]);

  const submitDraft = () => {
    const nextQuestion = draft.trim();
    if (!nextQuestion || busy) return;
    onAsk(nextQuestion);
    setDraft("");
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submitDraft();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitDraft();
    }
  };

  return (
    <>
      <div className="chat-scroll">
        {turns.length === 0 ? (
          <section className="empty-conversation">
            <span>
              <Icon name="sparkles" size={24} />
            </span>
            <h2>开始一次新的文档研究</h2>
            <p>从下方提问，回答会同时检索文字、图表、表格和页面区域。</p>
          </section>
        ) : (
          turns.map((turn) => {
            const turnSources = sourcesByTurn[turn.id] ?? [];
            return (
              <Fragment key={turn.id}>
                <section className="message-card user-message">
                  <MessageIdentity
                    kind="user"
                    label="你"
                    time={turn.createdAt}
                  />
                  <MarkdownContent
                    content={turn.question}
                    className="user-question"
                  />
                </section>

                <section
                  className={`message-card assistant-message${turn.pending ? " is-pending" : ""}`}
                >
                  <MessageIdentity
                    kind="assistant"
                    label="研究助手"
                    time={turn.createdAt}
                  />
                  {turn.pending ? (
                    <>
                      <PendingAnswer
                        startedAt={turn.startedAt ?? Date.now()}
                        statusText={turn.statusText}
                        hasContent={Boolean(turn.answer)}
                      />
                      {turn.answer && (
                        <div className="streaming-answer">
                          <LiveAnswer turn={turn} />
                        </div>
                      )}
                    </>
                  ) : (
                    <LiveAnswer turn={turn} />
                  )}
                  {!turn.pending && (
                    <TurnEvidenceSummary
                      count={turnSources.length}
                      onOpen={() => onEvidenceOpen(turn.id)}
                    />
                  )}
                </section>
              </Fragment>
            );
          })
        )}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          rows={1}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={busy ? "请等待当前回答完成…" : "向这些文档继续提问"}
          aria-label="输入问题"
          disabled={busy}
        />
        <span className={`composer-hint${busy ? " is-busy" : ""}`}>
          {busy ? "回答生成中，输入已锁定" : "Enter 发送 · Shift + Enter 换行"}
        </span>
        <button
          type="submit"
          className="send-button"
          disabled={busy || !draft.trim()}
          aria-label="发送问题"
        >
          {busy ? <span className="send-button-loader" /> : <Icon name="send" size={22} />}
        </button>
      </form>
    </>
  );
}
