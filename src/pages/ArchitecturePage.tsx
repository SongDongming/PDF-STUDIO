import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { Icon } from "../components/Icons";
import {
  ARCHITECTURE_WORLD,
  architectureEdges,
  architectureGroups,
  architectureNodes,
  edgeKinds,
  type ArchitectureEdge,
  type ArchitectureFlow,
  type ArchitectureNode,
} from "./architectureData";

interface ViewState {
  x: number;
  y: number;
  scale: number;
}

interface EdgeGeometry {
  path: string;
  labelX: number;
  labelY: number;
}

const flowOptions: Array<{
  id: ArchitectureFlow;
  label: string;
  hint: string;
}> = [
  { id: "all", label: "后端全景", hint: "两条主链与数据底座" },
  { id: "query", label: "Agentic 检索", hint: "自主决策与证据回复" },
  { id: "compile", label: "文档编译", hint: "PDF 到可检索知识" },
  { id: "data", label: "数据底座", hint: "状态、对象与图谱" },
];

const nodeById = new Map(architectureNodes.map((node) => [node.id, node]));

function anchor(
  node: ArchitectureNode,
  target: ArchitectureNode,
): { x: number; y: number; axis: "horizontal" | "vertical"; direction: number } {
  const centerX = node.x + node.width / 2;
  const centerY = node.y + node.height / 2;
  const targetX = target.x + target.width / 2;
  const targetY = target.y + target.height / 2;
  const dx = targetX - centerX;
  const dy = targetY - centerY;
  if (Math.abs(dx) >= Math.abs(dy) * 0.78) {
    return {
      x: dx >= 0 ? node.x + node.width : node.x,
      y: centerY,
      axis: "horizontal",
      direction: dx >= 0 ? 1 : -1,
    };
  }
  return {
    x: centerX,
    y: dy >= 0 ? node.y + node.height : node.y,
    axis: "vertical",
    direction: dy >= 0 ? 1 : -1,
  };
}

function edgeGeometry(edge: ArchitectureEdge): EdgeGeometry | null {
  const source = nodeById.get(edge.source);
  const target = nodeById.get(edge.target);
  if (!source || !target) return null;
  const start = anchor(source, target);
  const end = anchor(target, source);
  let path: string;
  if (start.axis === "horizontal" && end.axis === "horizontal") {
    const curve = Math.max(46, Math.abs(end.x - start.x) * 0.44);
    path = `M ${start.x} ${start.y} C ${start.x + start.direction * curve} ${start.y}, ${end.x + end.direction * curve} ${end.y}, ${end.x} ${end.y}`;
  } else if (start.axis === "vertical" && end.axis === "vertical") {
    const curve = Math.max(58, Math.abs(end.y - start.y) * 0.38);
    path = `M ${start.x} ${start.y} C ${start.x} ${start.y + start.direction * curve}, ${end.x} ${end.y + end.direction * curve}, ${end.x} ${end.y}`;
  } else {
    const middleX = (start.x + end.x) / 2;
    const middleY = (start.y + end.y) / 2;
    path = `M ${start.x} ${start.y} C ${middleX} ${start.y}, ${end.x} ${middleY}, ${end.x} ${end.y}`;
  }
  return {
    path,
    labelX: (start.x + end.x) / 2,
    labelY: (start.y + end.y) / 2 - 7,
  };
}

function ArchitectureCodeModal({
  node,
  onClose,
}: {
  node: ArchitectureNode;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(node.code.snippet);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      // Clipboard permission denied or unavailable (non-secure context): no
      // crash, just leave the button in its idle state.
    }
  };

  return (
    <div className="architecture-code-overlay" onMouseDown={onClose}>
      <section
        className="architecture-code-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`${node.title} 核心代码`}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div className="architecture-code-identity">
            <span style={{ "--code-color": architectureGroups[node.group].color } as CSSProperties}>
              <Icon name={node.icon} size={20} />
            </span>
            <div>
              <small>{node.eyebrow}</small>
              <h2>{node.title}</h2>
            </div>
          </div>
          <button type="button" onClick={onClose} aria-label="关闭核心代码">
            <Icon name="close" size={20} />
          </button>
        </header>

        <div className="architecture-code-context">
          <p>{node.code.description}</p>
          <div>
            <span><Icon name="file" size={14} />{node.code.path}</span>
            <span><i />真实后端源码</span>
            <span>Python</span>
          </div>
        </div>

        <div className="architecture-code-toolbar">
          <span>CORE IMPLEMENTATION</span>
          <button type="button" onClick={() => void copyCode()}>
            <Icon name={copied ? "check" : "file"} size={14} />
            {copied ? "已复制" : "复制代码"}
          </button>
        </div>

        <pre className="architecture-code-block">
          {node.code.snippet.split("\n").map((line, index) => (
            <span key={`${node.id}-${index}`}>
              <i>{node.code.startLine + index}</i>
              <code>{line || " "}</code>
            </span>
          ))}
        </pre>

        <footer>
          <div>
            <strong>输入</strong>
            <span>{node.input}</span>
          </div>
          <div>
            <strong>输出</strong>
            <span>{node.output}</span>
          </div>
          <div>
            <strong>失败语义</strong>
            <span>{node.failure}</span>
          </div>
        </footer>
      </section>
    </div>
  );
}

export function ArchitecturePage() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    viewX: number;
    viewY: number;
  } | null>(null);
  const [flow, setFlow] = useState<ArchitectureFlow>("all");
  const [view, setView] = useState<ViewState>({ x: 30, y: 28, scale: 0.65 });
  const [dragging, setDragging] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<ArchitectureNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<ArchitectureNode | null>(null);

  const fitCanvas = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const rect = viewport.getBoundingClientRect();
    const padding = 42;
    const scale = Math.min(
      (rect.width - padding * 2) / ARCHITECTURE_WORLD.width,
      0.92,
    );
    const safeScale = Math.max(0.52, scale);
    setView({
      scale: safeScale,
      x: (rect.width - ARCHITECTURE_WORLD.width * safeScale) / 2,
      y: 16,
    });
  }, []);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const observer = new ResizeObserver(() => fitCanvas());
    observer.observe(viewport);
    requestAnimationFrame(fitCanvas);
    return () => observer.disconnect();
  }, [fitCanvas]);

  const geometries = useMemo(
    () =>
      architectureEdges.map((edge) => ({
        edge,
        geometry: edgeGeometry(edge),
      })),
    [],
  );

  const activeNode = (node: ArchitectureNode) =>
    flow === "all" || node.flows.includes(flow);
  const activeEdge = (edge: ArchitectureEdge) =>
    flow === "all" || edge.flows.includes(flow);

  const zoomAtCenter = (delta: number) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const rect = viewport.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    setView((current) => {
      const scale = Math.min(1.35, Math.max(0.4, current.scale + delta));
      const worldX = (centerX - current.x) / current.scale;
      const worldY = (centerY - current.y) / current.scale;
      return {
        scale,
        x: centerX - worldX * scale,
        y: centerY - worldY * scale,
      };
    });
  };

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;
    setView((current) => {
      const nextScale = Math.min(
        1.35,
        Math.max(0.4, current.scale * (event.deltaY > 0 ? 0.9 : 1.1)),
      );
      const worldX = (pointerX - current.x) / current.scale;
      const worldY = (pointerY - current.y) / current.scale;
      return {
        scale: nextScale,
        x: pointerX - worldX * nextScale,
        y: pointerY - worldY * nextScale,
      };
    });
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if ((event.target as Element).closest("button, .architecture-node-tooltip")) return;
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      viewX: view.x,
      viewY: view.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    setView((current) => ({
      ...current,
      x: drag.viewX + event.clientX - drag.startX,
      y: drag.viewY + event.clientY - drag.startY,
    }));
  };

  const finishDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
      setDragging(false);
    }
  };

  return (
    <main className="architecture-page">
      <header className="architecture-page-header">
        <div className="architecture-title">
          <span><Icon name="brain" size={22} /></span>
          <div>
            <small>BACKEND SYSTEM OBSERVATORY</small>
            <h1>项目功能架构</h1>
            <p>从 PDF 编译入库到 Agentic RAG 图文回答的真实后端数据流</p>
          </div>
        </div>

        <div className="architecture-metrics">
          <span><strong>2</strong><small>核心主链</small></span>
          <span><strong>4</strong><small>Agent 工具</small></span>
          <span><strong>4</strong><small>数据底座</small></span>
          <span className="is-live"><i /><strong>LIVE</strong><small>真实架构</small></span>
        </div>
      </header>

      <section className="architecture-toolbar" aria-label="架构视图筛选">
        <div className="architecture-flow-tabs">
          {flowOptions.map((option) => (
            <button
              key={option.id}
              type="button"
              data-flow={option.id}
              className={flow === option.id ? "is-active" : ""}
              onClick={() => setFlow(option.id)}
            >
              <strong>{option.label}</strong>
              <small>{option.hint}</small>
            </button>
          ))}
        </div>
        <div className="architecture-edge-legend">
          {Object.entries(edgeKinds).map(([id, item]) => (
            <span key={id}>
              <i style={{ "--edge-color": item.color } as CSSProperties} />
              {item.label}
            </span>
          ))}
        </div>
        <div className="architecture-zoom-controls">
          <button type="button" onClick={() => zoomAtCenter(-0.1)} aria-label="缩小画布">−</button>
          <span>{Math.round(view.scale * 100)}%</span>
          <button type="button" onClick={() => zoomAtCenter(0.1)} aria-label="放大画布">＋</button>
          <button type="button" onClick={fitCanvas} aria-label="适应画布">
            <Icon name="refresh" size={15} />
          </button>
        </div>
      </section>

      <section
        ref={viewportRef}
        className={`architecture-viewport${dragging ? " is-dragging" : ""}`}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={finishDrag}
        onPointerCancel={finishDrag}
      >
        <div className="architecture-canvas-stars" />
        <div
          className="architecture-world"
          style={{
            width: ARCHITECTURE_WORLD.width,
            height: ARCHITECTURE_WORLD.height,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
          }}
        >
          <div className="architecture-lane lane-query">
            <span>01 · AGENTIC RETRIEVAL FLOW</span>
            <strong>自主问答与证据闭环</strong>
          </div>
          <div className="architecture-lane lane-compile">
            <span>02 · MULTIMODAL COMPILATION FLOW</span>
            <strong>PDF 编译与知识派生</strong>
          </div>
          <div className="architecture-lane lane-data">
            <span>03 · DATA & RUNTIME FABRIC</span>
            <strong>持久化与运行底座</strong>
          </div>

          <svg
            className="architecture-edges"
            viewBox={`0 0 ${ARCHITECTURE_WORLD.width} ${ARCHITECTURE_WORLD.height}`}
            aria-hidden="true"
          >
            <defs>
              {Object.entries(edgeKinds).map(([id, item]) => (
                <marker
                  key={id}
                  id={`architecture-arrow-${id}`}
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={item.color} />
                </marker>
              ))}
              <filter id="architecture-line-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            {geometries.map(({ edge, geometry }, index) => {
              if (!geometry) return null;
              const isActive = activeEdge(edge);
              return (
                <g
                  key={edge.id}
                  className={`architecture-edge edge-${edge.kind}${isActive ? " is-active" : " is-muted"}`}
                  style={{ "--edge-index": index } as CSSProperties}
                >
                  <path className="architecture-edge-shadow" d={geometry.path} />
                  <path
                    className="architecture-edge-flow"
                    d={geometry.path}
                    markerEnd={`url(#architecture-arrow-${edge.kind})`}
                  />
                  {edge.label && isActive && (
                    <g transform={`translate(${geometry.labelX} ${geometry.labelY})`}>
                      <rect x="-52" y="-11" width="104" height="20" rx="10" />
                      <text textAnchor="middle" dominantBaseline="middle">{edge.label}</text>
                    </g>
                  )}
                </g>
              );
            })}
          </svg>

          {architectureNodes.map((node) => {
            const group = architectureGroups[node.group];
            return (
              <button
                key={node.id}
                type="button"
                data-node-id={node.id}
                className={`architecture-node node-${node.group}${activeNode(node) ? " is-active" : " is-muted"}`}
                style={{
                  left: node.x,
                  top: node.y,
                  width: node.width,
                  height: node.height,
                  "--node-color": group.color,
                  "--node-surface": group.surface,
                } as CSSProperties}
                onMouseEnter={() => setHoveredNode(node)}
                onMouseLeave={() => setHoveredNode((current) => current?.id === node.id ? null : current)}
                onFocus={() => setHoveredNode(node)}
                onBlur={() => setHoveredNode((current) => current?.id === node.id ? null : current)}
                onClick={() => setSelectedNode(node)}
              >
                <span className="architecture-node-icon"><Icon name={node.icon} size={20} /></span>
                <span className="architecture-node-copy">
                  <small>{node.eyebrow}</small>
                  <strong>{node.title}</strong>
                  <em>{node.subtitle}</em>
                </span>
                <i className="architecture-node-status" />
                <span className="architecture-node-open"><Icon name="arrowRight" size={14} /></span>
              </button>
            );
          })}

          {hoveredNode && (
            <aside
              className="architecture-node-tooltip"
              style={{
                left:
                  hoveredNode.x > 1540
                    ? hoveredNode.x - 326
                    : hoveredNode.x + hoveredNode.width + 16,
                top: Math.min(hoveredNode.y, ARCHITECTURE_WORLD.height - 252),
                "--node-color": architectureGroups[hoveredNode.group].color,
              } as CSSProperties}
            >
              <span>{architectureGroups[hoveredNode.group].label}</span>
              <h3>{hoveredNode.title}</h3>
              <p>{hoveredNode.summary}</p>
              <dl>
                <div><dt>输入</dt><dd>{hoveredNode.input}</dd></div>
                <div><dt>输出</dt><dd>{hoveredNode.output}</dd></div>
                <div><dt>异常</dt><dd>{hoveredNode.failure}</dd></div>
              </dl>
              <small><Icon name="file" size={13} />点击查看真实核心代码</small>
            </aside>
          )}
        </div>

        <div className="architecture-canvas-help">
          <span><Icon name="sliders" size={15} />滚轮缩放</span>
          <span><Icon name="network" size={15} />拖拽平移</span>
          <span><Icon name="file" size={15} />点击节点查看代码</span>
        </div>
      </section>

      {selectedNode && (
        <ArchitectureCodeModal
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
        />
      )}
    </main>
  );
}
