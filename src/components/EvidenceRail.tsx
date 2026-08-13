import { useEffect, useState } from "react";
import { API_BASE_URL } from "../api/client";
import type { Evidence, EvidenceKind } from "../types";
import { Icon, type IconName } from "./Icons";

interface EvidenceDrawerProps {
  evidences: Evidence[];
  question: string;
  onOpen: (evidence: Evidence) => void;
  onClose: () => void;
}

function EvidenceKindIcon({ kind }: { kind: EvidenceKind }) {
  const names: Record<EvidenceKind, IconName> = {
    figure: "image",
    table: "table",
    excerpt: "quote",
    diagram: "image",
  };

  return <Icon name={names[kind]} size={16} />;
}

function EvidenceThumbnail({ evidence }: { evidence: Evidence }) {
  const [failed, setFailed] = useState(false);
  if (evidence.documentId && evidence.elementId && !failed) {
    return (
      <div className="evidence-thumbnail live-region-thumbnail">
        <img
          src={`${API_BASE_URL}/documents/${encodeURIComponent(
            evidence.documentId,
          )}/regions/${encodeURIComponent(evidence.elementId)}/image`}
          alt={`${evidence.filename} 第 ${evidence.page} 页命中区域`}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      </div>
    );
  }
  return (
    <div className="evidence-thumbnail is-placeholder">
      <Icon name="file" size={28} />
    </div>
  );
}

export function EvidenceDrawer({
  evidences,
  question,
  onOpen,
  onClose,
}: EvidenceDrawerProps) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !document.querySelector(".pdf-viewer")) onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="evidence-drawer-layer">
      <button
        className="evidence-drawer-scrim"
        type="button"
        onClick={onClose}
        aria-label="关闭本轮线索"
      />
      <aside
        className="evidence-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="本轮检索线索"
      >
        <header className="evidence-drawer-header">
          <span className="evidence-drawer-mark">
            <Icon name="sparkles" size={20} />
          </span>
          <div>
            <div className="evidence-drawer-title">
              <h2>本轮线索</h2>
              <strong>{evidences.length} 条</strong>
            </div>
            <p title={question}>{question}</p>
            <small>仅属于这条 Agent 回复，按相关性从高到低排列</small>
          </div>
          <button type="button" onClick={onClose} aria-label="关闭本轮线索">×</button>
        </header>

        <div className="evidence-drawer-body">
          {evidences.length === 0 && (
            <div className="evidence-drawer-empty" role="status">
              <span>
                <Icon name="sparkles" size={24} />
              </span>
              <strong>这条回复未调用知识库</strong>
              <p>Agent 判断无需检索，使用模型自身知识完成了回答，因此没有参考线索。</p>
            </div>
          )}
          <div className="evidence-drawer-list">
            {evidences.map((evidence) => (
              <button
                key={evidence.id}
                className={`evidence-card accent-${evidence.accent}`}
                type="button"
                onClick={() => onOpen(evidence)}
                aria-label={`打开 ${evidence.filename} 第 ${evidence.page} 页线索`}
              >
                <div className="evidence-card-head">
                  <span className="evidence-order">{evidence.order}</span>
                  <span className="evidence-kind">
                    <EvidenceKindIcon kind={evidence.kind} />
                    {evidence.kindLabel}
                  </span>
                  <strong className="relevance-score">
                    <small>相关性</small>
                    {evidence.relevance}%
                  </strong>
                </div>

                <div className="evidence-card-body">
                  <EvidenceThumbnail evidence={evidence} />
                  <p>{evidence.summary}</p>
                </div>

                <span className="evidence-card-foot">
                  <span>
                    <Icon name="file" size={15} />
                    第 {evidence.page} 页
                  </span>
                  <small title={evidence.filename}>{evidence.filename}</small>
                  <Icon name="arrowRight" size={15} />
                </span>
              </button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
