import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL } from "../api/client";
import type { PdfViewerTarget } from "../types";
import { Icon } from "./Icons";

interface PageElement {
  id: string;
  page: number;
  order: number;
  kind: string;
  label: string;
  bbox_normalized: [number, number, number, number];
}

interface PdfViewerProps {
  target: PdfViewerTarget | null;
  onClose: () => void;
}

function pageImageUrl(documentId: string, page: number) {
  return `${API_BASE_URL}/documents/${encodeURIComponent(documentId)}/pages/${page}/image`;
}

function elementClass(kind: string) {
  if (kind === "figure" || kind === "image" || kind === "chart") return "figure";
  if (kind === "formula") return "formula";
  if (kind === "table") return "table";
  return "text";
}

function RealPdfPage({
  documentId,
  page,
  elements,
  analyzed,
  activeElementId,
  activeElementRef,
  onAspectChange,
}: {
  documentId: string;
  page: number;
  elements: PageElement[];
  analyzed?: boolean;
  activeElementId?: string | null;
  activeElementRef?: (node: HTMLSpanElement | null) => void;
  onAspectChange?: (aspect: number) => void;
}) {
  const [failed, setFailed] = useState(false);
  return (
    <div className={`real-pdf-page ${analyzed ? "is-analyzed" : ""}`}>
      {failed ? (
        <div className="real-pdf-error">
          <Icon name="info" size={22} />
          <strong>页面图暂不可用</strong>
          <span>文档可能仍在编译，可稍后重试。</span>
        </div>
      ) : (
        <img
          src={pageImageUrl(documentId, page)}
          alt={`PDF 第 ${page} 页`}
          onLoad={(event) => {
            const image = event.currentTarget;
            if (image.naturalWidth > 0 && image.naturalHeight > 0) {
              onAspectChange?.(image.naturalWidth / image.naturalHeight);
            }
          }}
          onError={() => setFailed(true)}
        />
      )}
      {analyzed &&
        elements.map((element) => {
          const [x1, y1, x2, y2] = element.bbox_normalized;
          return (
            <span
              key={element.id}
              ref={activeElementId === element.id ? activeElementRef : undefined}
              className={`real-element-box is-${elementClass(element.kind)} ${
                activeElementId === element.id ? "is-target" : ""
              }`}
              style={{
                left: `${x1 * 100}%`,
                top: `${y1 * 100}%`,
                width: `${(x2 - x1) * 100}%`,
                height: `${(y2 - y1) * 100}%`,
              }}
              title={`${element.label} · ${element.id}`}
            >
              <i>{element.label}</i>
            </span>
          );
        })}
    </div>
  );
}

function FocusedEvidenceRegion({
  target,
  label,
  regionRef,
}: {
  target: PdfViewerTarget;
  label: string;
  regionRef?: (node: HTMLDivElement | null) => void;
}) {
  const [failed, setFailed] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  if (!target.elementId || failed) {
    return (
      <div className="focused-region-error">
        <Icon name="info" size={22} />
        <strong>{target.elementId ? "局部截图暂不可用" : "未指定局部区域"}</strong>
        <span>
          {target.elementId
            ? "该素材可能已随文档重编译更新，右侧暂以当前页元素识别结果作为对照。"
            : "右侧暂以当前页元素识别结果作为对照。"}
        </span>
      </div>
    );
  }

  const imageUrl = `${API_BASE_URL}/documents/${encodeURIComponent(
    target.documentId,
  )}/regions/${encodeURIComponent(target.elementId)}/image?r=${retryKey}`;

  return (
    <figure className="focused-evidence-region">
      <div className="focused-region-canvas is-linked" ref={regionRef}>
        <img
          src={imageUrl}
          alt={`${target.filename} 第 ${target.page} 页本轮召回区域`}
          onError={() => {
            // One cache-busting retry for transient hiccups; a stale element id
            // (document recompiled) will keep failing and fall back gracefully.
            if (retryKey === 0) setRetryKey(1);
            else setFailed(true);
          }}
        />
      </div>
      <figcaption>
        <strong>{label}</strong>
        <small>来自第 {target.page} 页 · 按原始元素坐标裁切放大</small>
      </figcaption>
    </figure>
  );
}

export function PdfViewer({ target, onClose }: PdfViewerProps) {
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(100);
  const [pageAspect, setPageAspect] = useState(210 / 297);
  const [fitPageWidth, setFitPageWidth] = useState(420);
  const [elements, setElements] = useState<PageElement[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const compareStageRef = useRef<HTMLDivElement>(null);
  const sourceElementRef = useRef<HTMLSpanElement | null>(null);
  const targetRegionRef = useRef<HTMLDivElement | null>(null);
  const activeThumbnailRef = useRef<HTMLButtonElement | null>(null);
  const sourcePageViewportRef = useRef<HTMLDivElement | null>(null);
  const [connectorLines, setConnectorLines] = useState<{
    sourceX: number;
    sourceTop: number;
    sourceBottom: number;
    targetX: number;
    targetTop: number;
    targetBottom: number;
  } | null>(null);

  useEffect(() => {
    if (!target) return;
    setPage(target.page);
    setZoom(100);
    setPageAspect(210 / 297);
  }, [target]);

  useEffect(() => {
    const viewport = sourcePageViewportRef.current;
    if (!viewport || !target) return;

    const updateFitWidth = () => {
      const style = getComputedStyle(viewport);
      const horizontalPadding =
        Number.parseFloat(style.paddingLeft) + Number.parseFloat(style.paddingRight);
      const verticalPadding =
        Number.parseFloat(style.paddingTop) + Number.parseFloat(style.paddingBottom);
      const availableWidth = viewport.clientWidth - horizontalPadding - 4;
      const availableHeight = viewport.clientHeight - verticalPadding - 4;
      const nextWidth = Math.min(
        560,
        availableWidth,
        availableHeight * pageAspect,
      );
      setFitPageWidth(Math.max(240, Math.floor(nextWidth)));
    };

    updateFitWidth();
    const observer = new ResizeObserver(updateFitWidth);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [pageAspect, target]);

  useEffect(() => {
    if (!target) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [target, onClose]);

  useEffect(() => {
    if (!target) return;
    const controller = new AbortController();
    setLoading(true);
    setLoadError("");
    void fetch(
      `${API_BASE_URL}/documents/${encodeURIComponent(target.documentId)}/pages/${page}/elements`,
      { signal: controller.signal },
    )
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<PageElement[]>;
      })
      .then(setElements)
      .catch((error) => {
        if (controller.signal.aborted) return;
        setElements([]);
        setLoadError(
          `元素识别结果加载失败：${error instanceof Error ? error.message : "未知错误"}`,
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [page, target]);

  const documentPages = useMemo(() => {
    if (!target) return [];
    return Array.from({ length: target.totalPages }, (_, index) => index + 1);
  }, [target]);

  useEffect(() => {
    if (!target) return;
    const frame = requestAnimationFrame(() => {
      activeThumbnailRef.current?.scrollIntoView({
        block: "center",
        inline: "nearest",
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [page, target]);

  const counts = useMemo(
    () =>
      elements.reduce<Record<string, number>>((result, element) => {
        const kind = elementClass(element.kind);
        result[kind] = (result[kind] ?? 0) + 1;
        return result;
      }, {}),
    [elements],
  );
  const activeElement = elements.find((element) => element.id === target?.elementId);

  useEffect(() => {
    const stage = compareStageRef.current;
    if (!stage || page !== target?.page || !activeElement) {
      setConnectorLines(null);
      return;
    }

    let frame = 0;
    const updateLines = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const source = sourceElementRef.current;
        const destination = targetRegionRef.current;
        if (!source || !destination) {
          setConnectorLines(null);
          return;
        }
        const stageRect = stage.getBoundingClientRect();
        const sourceRect = source.getBoundingClientRect();
        const destinationRect = destination.getBoundingClientRect();
        const clampY = (value: number) =>
          Math.max(45, Math.min(stageRect.height - 8, value - stageRect.top));
        setConnectorLines({
          sourceX: sourceRect.right - stageRect.left,
          sourceTop: clampY(sourceRect.top + 2),
          sourceBottom: clampY(sourceRect.bottom - 2),
          targetX: destinationRect.left - stageRect.left,
          targetTop: clampY(destinationRect.top + 18),
          targetBottom: clampY(destinationRect.bottom - 18),
        });
      });
    };

    updateLines();
    const observer = new ResizeObserver(updateLines);
    observer.observe(stage);
    if (sourceElementRef.current) observer.observe(sourceElementRef.current);
    if (targetRegionRef.current) observer.observe(targetRegionRef.current);
    stage.addEventListener("scroll", updateLines, true);
    window.addEventListener("resize", updateLines);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      stage.removeEventListener("scroll", updateLines, true);
      window.removeEventListener("resize", updateLines);
    };
  }, [activeElement, page, target?.page, zoom]);

  if (!target) return null;

  const changePage = (nextPage: number) => {
    setPage(Math.min(target.totalPages, Math.max(1, nextPage)));
  };
  const pageWidth = Math.round(fitPageWidth * (zoom / 100));

  return (
    <div className="pdf-overlay" onMouseDown={onClose}>
      <section
        className="pdf-viewer"
        role="dialog"
        aria-modal="true"
        aria-label={`查看 ${target.filename}`}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="pdf-viewer-header">
          <div className="pdf-viewer-title">
            <span className="viewer-file-mark">PDF</span>
            <span>
              <strong>{target.filename}</strong>
              <small>真实原页 · 本轮检索命中内容对照</small>
            </span>
          </div>

          <div className="pdf-viewer-controls">
            <button onClick={() => changePage(page - 1)} disabled={page === 1} aria-label="上一页">
              <Icon name="arrowLeft" size={18} />
            </button>
            <label className="page-control">
              <span className="sr-only">当前页码</span>
              <input
                type="number"
                min={1}
                max={target.totalPages}
                value={page}
                onChange={(event) => changePage(Number(event.target.value))}
              />
              <span>/ {target.totalPages}</span>
            </label>
            <button
              onClick={() => changePage(page + 1)}
              disabled={page === target.totalPages}
              aria-label="下一页"
            >
              <Icon name="arrowRight" size={18} />
            </button>
            <span className="viewer-divider" />
            <button onClick={() => setZoom((value) => Math.max(60, value - 10))} aria-label="缩小">−</button>
            <strong className="zoom-value">{zoom}%</strong>
            <button onClick={() => setZoom((value) => Math.min(140, value + 10))} aria-label="放大">+</button>
          </div>

          <button className="pdf-close" onClick={onClose} aria-label="关闭 PDF 查看器">×</button>
        </header>

        <div className="pdf-viewer-body">
          <aside className="pdf-thumbnails" aria-label="PDF 全部页面">
            <h3>
              <span>全部页面</span>
              <small>{target.totalPages} 页</small>
            </h3>
            {documentPages.map((thumbnailPage) => (
              <button
                key={thumbnailPage}
                ref={thumbnailPage === page ? activeThumbnailRef : undefined}
                className={thumbnailPage === page ? "is-active" : ""}
                onClick={() => changePage(thumbnailPage)}
                aria-current={thumbnailPage === page ? "page" : undefined}
                aria-label={`查看第 ${thumbnailPage} 页`}
              >
                <span className="real-thumbnail-paper">
                  <img
                    src={pageImageUrl(target.documentId, thumbnailPage)}
                    alt=""
                    loading="lazy"
                  />
                </span>
                <strong>第 {thumbnailPage} 页</strong>
              </button>
            ))}
          </aside>

          <div className="pdf-compare-stage" ref={compareStageRef}>
            <section className="pdf-compare-column">
              <header>
                <span>原始 PDF</span>
                <small>{page === target.page && activeElement ? "红框为本轮命中位置" : `第 ${page} 页 · 页面渲染`}</small>
              </header>
              <div className="compare-page-scroll" ref={sourcePageViewportRef}>
                <div className="compare-page-scale" style={{ width: `${pageWidth}px` }}>
                  <RealPdfPage
                    documentId={target.documentId}
                    page={page}
                    elements={
                      page === target.page && activeElement ? [activeElement] : []
                    }
                    analyzed={page === target.page && Boolean(activeElement)}
                    activeElementId={target.elementId}
                    activeElementRef={(node) => {
                      sourceElementRef.current = node;
                    }}
                    onAspectChange={setPageAspect}
                  />
                </div>
              </div>
            </section>
            <section className="pdf-compare-column analyzed-column">
              <header>
                <span>{loading ? "正在加载线索…" : "本轮召回区域"}</span>
                <small>局部放大 · 原始比例</small>
              </header>
              <div className="compare-page-scroll">
                {page === target.page && target.elementId ? (
                  <FocusedEvidenceRegion
                    target={target}
                    label={activeElement?.label ?? "本轮检索命中内容"}
                    regionRef={(node) => {
                      targetRegionRef.current = node;
                    }}
                  />
                ) : (
                  <div className="compare-page-scale" style={{ width: `${pageWidth}px` }}>
                    <RealPdfPage
                      documentId={target.documentId}
                      page={page}
                      elements={elements}
                      analyzed
                      activeElementId={target.elementId}
                    />
                  </div>
                )}
              </div>
            </section>
            {connectorLines && (
              <svg className="pdf-evidence-connectors" aria-hidden="true">
                <defs>
                  <filter id="connector-glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="2" result="blur" />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                <line
                  x1={connectorLines.sourceX}
                  y1={connectorLines.sourceTop}
                  x2={connectorLines.targetX}
                  y2={connectorLines.targetTop}
                />
                <line
                  x1={connectorLines.sourceX}
                  y1={connectorLines.sourceBottom}
                  x2={connectorLines.targetX}
                  y2={connectorLines.targetBottom}
                />
                <circle cx={connectorLines.sourceX} cy={connectorLines.sourceTop} r="3" />
                <circle cx={connectorLines.sourceX} cy={connectorLines.sourceBottom} r="3" />
              </svg>
            )}
          </div>
        </div>

        <footer className="pdf-inspector-bar">
          <span className="match-badge">{target.elementId ? "当前命中" : "页面元素"}</span>
          <div>
            <strong>{activeElement?.label ?? (loadError || `第 ${page} 页识别结果`)}</strong>
            <small>
              正文 {counts.text ?? 0} · 图表 {counts.figure ?? 0} · 表格 {counts.table ?? 0} ·
              公式 {counts.formula ?? 0}
            </small>
          </div>
          <span className="inspector-score">
            <small>线索相关性</small>
            <strong>{target.relevance === undefined ? "—" : `${target.relevance}%`}</strong>
          </span>
          <button className="use-in-chat" onClick={onClose}>返回回答</button>
        </footer>
      </section>
    </div>
  );
}
