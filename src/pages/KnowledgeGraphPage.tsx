import { useEffect, useMemo, useRef, useState } from "react";
import {
  CanvasEvent,
  Graph,
  GraphEvent,
  NodeEvent,
  type EdgeData,
  type GraphData,
  type NodeData,
} from "@antv/g6";
import { CustomSelect } from "../components/CustomSelect";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useWorkspaceApi } from "../hooks/useWorkspaceApi";
import { api } from "../api/client";
import type {
  EvidenceLineage,
  GraphEdge,
  GraphNode,
  GraphView,
} from "../api/types";

type CommunityId = "core" | "memory" | "tools" | "retrieval" | "evidence";
type LayoutId = "atlas" | "radial" | "free";

interface CommunityDefinition {
  id: CommunityId;
  label: string;
  color: string;
  surface: string;
  stroke: string;
}

interface VisualNode {
  id: string;
  label: string;
  kind: GraphNode["kind"];
  detail: string;
  community: CommunityId;
  degree: number;
  importance: number;
  core: boolean;
}

interface SelectedNode extends VisualNode {}

interface GraphScene {
  data: GraphData;
  nodes: Map<string, VisualNode>;
  rootId: string;
  communityCounts: Record<CommunityId, number>;
}

const communities: CommunityDefinition[] = [
  {
    id: "core",
    label: "框架核心",
    color: "#5268e9",
    surface: "rgba(82,104,233,.065)",
    stroke: "rgba(82,104,233,.22)",
  },
  {
    id: "memory",
    label: "记忆系统",
    color: "#13a77d",
    surface: "rgba(19,167,125,.065)",
    stroke: "rgba(19,167,125,.22)",
  },
  {
    id: "tools",
    label: "工具调用",
    color: "#8060df",
    surface: "rgba(128,96,223,.065)",
    stroke: "rgba(128,96,223,.22)",
  },
  {
    id: "retrieval",
    label: "Agentic RAG",
    color: "#23a7c9",
    surface: "rgba(35,167,201,.065)",
    stroke: "rgba(35,167,201,.22)",
  },
  {
    id: "evidence",
    label: "评测与证据",
    color: "#ed8952",
    surface: "rgba(237,137,82,.065)",
    stroke: "rgba(237,137,82,.22)",
  },
];

const communityById = Object.fromEntries(
  communities.map((community) => [community.id, community]),
) as Record<CommunityId, CommunityDefinition>;

const communityCenters: Record<CommunityId, { x: number; y: number }> = {
  core: { x: 610, y: 350 },
  memory: { x: 960, y: 190 },
  tools: { x: 960, y: 545 },
  retrieval: { x: 270, y: 535 },
  evidence: { x: 245, y: 185 },
};

const communityKeywords: Array<[CommunityId, RegExp]> = [
  [
    "memory",
    /memory|checkpoint|checkpointer|saver|store|persist|thread|state|history|记忆|检查点|持久|会话|状态|存储/i,
  ],
  [
    "tools",
    /tool|function|api|hook|middleware|interrupt|command|call|工具|函数|调用|中间件|审批|命令/i,
  ],
  [
    "retrieval",
    /rag|retriev|embedding|vector|search|index|wiki|chunk|context|召回|检索|向量|索引|知识库|上下文/i,
  ],
  [
    "core",
    /deepagents|deep agents|langchain|langgraph|agent|framework|architecture|task|智能体|框架|架构|任务/i,
  ],
  [
    "evidence",
    /evidence|citation|claim|evaluation|metric|benchmark|test|review|document|pdf|page|asset|证据|引用|评测|测试|文档|页面|素材|主张/i,
  ],
];

const kindNames: Record<GraphNode["kind"], string> = {
  entity: "知识实体",
  claim: "事实主张",
  document: "文档",
  page: "PDF 页面",
  chunk: "检索片段",
  asset: "多模态素材",
  wiki: "Wiki 页面",
};


function nodeText(node: GraphNode) {
  return `${node.label} ${String(node.properties.entity_kind ?? "")} ${String(
    node.properties.description ??
      node.properties.summary ??
      node.properties.statement ??
      "",
  )}`;
}

function classifyCommunity(node: GraphNode): CommunityId {
  const text = nodeText(node);
  const semantic = communityKeywords.find(([, pattern]) => pattern.test(text));
  if (semantic) return semantic[0];
  if (node.kind === "wiki" || node.kind === "chunk") return "retrieval";
  if (
    node.kind === "claim" ||
    node.kind === "document" ||
    node.kind === "page" ||
    node.kind === "asset"
  ) {
    return "evidence";
  }
  return "core";
}

function compactGraphLabel(node: GraphNode) {
  const cleaned = node.label
    .replace(/\\(?:underline|text)\{([^}]*)\}/g, "$1")
    .replace(/\\[a-zA-Z]+/g, " ")
    .replace(/[#*`$[\]{}]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const limit =
    node.kind === "chunk" || node.kind === "claim" ? 30 : 34;
  return cleaned.length > limit
    ? `${cleaned.slice(0, Math.max(1, limit - 1)).trim()}…`
    : cleaned || kindNames[node.kind];
}

function detailForNode(node: GraphNode) {
  const detail =
    node.properties.description ??
    node.properties.summary ??
    node.properties.statement ??
    node.properties.entity_kind;
  return String(detail || `来自知识库的${kindNames[node.kind]}节点`);
}

function buildScene(
  graph: GraphView,
  communityFilter: CommunityId | "all",
): GraphScene {
  const degree = new Map<string, number>();
  for (const edge of graph.edges) {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
  }

  const graphRoot =
    graph.nodes
      .map((node) => {
        const labelBoost = /deepagents|deep agents/i.test(node.label) ? 120 : 0;
        const rootBoost = node.properties.root === true ? 100 : 0;
        return {
          node,
          score: (degree.get(node.id) ?? 0) * 3 + labelBoost + rootBoost,
        };
      })
      .sort((left, right) => right.score - left.score)[0]?.node ?? graph.nodes[0];

  const visualNodes = graph.nodes.map<VisualNode>((node) => {
    const nodeDegree = degree.get(node.id) ?? 0;
    const root = node.id === graphRoot?.id;
    const importance = Math.min(
      10,
      Math.log2(nodeDegree + 1) * 2.2 +
        (root ? 5 : 0) +
        (node.kind === "wiki" ? 1.5 : 0),
    );
    return {
      id: node.id,
      label: compactGraphLabel(node),
      kind: node.kind,
      detail: detailForNode(node),
      community: classifyCommunity(node),
      degree: nodeDegree,
      importance,
      core: root || importance >= 6.8,
    };
  });

  const filteredNodes =
    communityFilter === "all"
      ? visualNodes
      : visualNodes.filter((node) => node.community === communityFilter);
  const visibleIds = new Set(filteredNodes.map((node) => node.id));
  const filteredEdges = graph.edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  );
  const nodeMap = new Map(filteredNodes.map((node) => [node.id, node]));
  const positions = new Map<string, { x: number; y: number }>();
  for (const community of communities) {
    const members = visualNodes
      .filter((node) => node.community === community.id)
      .sort((left, right) => {
        if (left.id === graphRoot?.id) return -1;
        if (right.id === graphRoot?.id) return 1;
        return right.importance - left.importance;
      });
    const center = communityCenters[community.id];
    members.forEach((node, index) => {
      if (node.id === graphRoot?.id) {
        positions.set(node.id, center);
        return;
      }
      const spiralIndex =
        community.id === "core" && graphRoot ? Math.max(0, index - 1) : index;
      const angle =
        spiralIndex * 2.399963229728653 +
        (community.id === "memory" || community.id === "tools"
          ? Math.PI
          : 0);
      const radius = 44 + Math.sqrt(spiralIndex) * 24;
      positions.set(node.id, {
        x: center.x + Math.cos(angle) * radius,
        y: center.y + Math.sin(angle) * radius,
      });
    });
  }
  const communityCounts = Object.fromEntries(
    communities.map((community) => [
      community.id,
      visualNodes.filter((node) => node.community === community.id).length,
    ]),
  ) as Record<CommunityId, number>;

  const nodes: NodeData[] = filteredNodes.map((node) => ({
    id: node.id,
    combo: `community:${node.community}`,
    style: positions.get(node.id),
    data: {
      ...node,
      shape: node.core ? "rect" : "circle",
      labelVisible: node.core || node.importance >= 4.8,
    },
  }));
  const edges: EdgeData[] = filteredEdges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    data: {
      label: edge.relation,
      evidenceCount: edge.evidence_ids.length,
    },
  }));
  const activeCommunities = communities.filter((community) =>
    filteredNodes.some((node) => node.community === community.id),
  );

  return {
    data: {
      nodes,
      edges,
      combos: activeCommunities.map((community) => ({
        id: `community:${community.id}`,
        data: {
          community: community.id,
          label: community.label,
          count: communityCounts[community.id],
        },
      })),
    },
    nodes: nodeMap,
    rootId: graphRoot?.id ?? filteredNodes[0]?.id ?? "",
    communityCounts,
  };
}

function layoutOptions(layout: LayoutId) {
  if (layout === "atlas") {
    // "atlas" uses the preset node positions computed in buildScene; returning
    // undefined tells G6 to honour the per-node x/y instead of running a layout.
    return undefined;
  }
  if (layout === "free") {
    return {
      type: "force-atlas2",
      preventOverlap: true,
      kr: 28,
      kg: 5,
      ks: 0.12,
      iterations: 180,
    };
  }
  return {
    type: "combo-combined",
    comboPadding: 42,
    comboSpacing: 64,
    layout: (comboId: string | null) =>
      comboId
        ? {
            type: "circular",
            preventOverlap: true,
            nodeSpacing: 24,
            minNodeSpacing: 18,
            sortBy: "importance",
          }
        : {
            type: "circular",
            preventOverlap: true,
            radius: 390,
            nodeSpacing: 72,
            startAngle: -Math.PI / 2,
            endAngle: Math.PI * 1.5,
          },
  };
}

function labelForDatum(datum: NodeData) {
  return String(datum.data?.label ?? "");
}

function visualForDatum(datum: NodeData) {
  const community = String(datum.data?.community ?? "core") as CommunityId;
  return communityById[community] ?? communityById.core;
}

function applySemanticLabels(
  graph: Graph,
  selectedId: string | null,
  band: number,
) {
  if (graph.destroyed) return;
  const emphasizedNeighbors = new Set(
    selectedId
      ? graph
          .getNeighborNodesData(selectedId)
          .sort(
            (left, right) =>
              Number(right.data?.importance ?? 0) -
              Number(left.data?.importance ?? 0),
          )
          .slice(0, 3)
          .map((node) => String(node.id))
      : [],
  );
  const updates = graph.getNodeData().map((node) => {
    const importance = Number(node.data?.importance ?? 0);
    const visible =
      importance >= (band === 0 ? 6.8 : band === 1 ? 4.4 : 2.8) ||
      node.id === selectedId ||
      emphasizedNeighbors.has(String(node.id));
    return {
      id: node.id,
      style: {
        labelText: visible ? labelForDatum(node) : "",
      },
    };
  });
  try {
    graph.updateNodeData(updates);
    void graph.draw().catch(() => {});
  } catch {
    // The graph may be mid-rebuild; transient G6 draw errors are safe to skip.
  }
}

function applyFocusStates(
  graph: Graph,
  selectedId: string | null,
  query = "",
) {
  if (graph.destroyed) return;
  const stateMap: Record<string, string[]> = {};
  const relatedEdges = selectedId
    ? graph.getRelatedEdgesData(selectedId).map((edge) => String(edge.id))
    : [];
  const neighborIds = new Set(
    selectedId
      ? graph.getNeighborNodesData(selectedId).map((node) => String(node.id))
      : [],
  );
  const queryValue = query.trim().toLocaleLowerCase();

  for (const node of graph.getNodeData()) {
    const id = String(node.id);
    const label = labelForDatum(node).toLocaleLowerCase();
    if (queryValue && label.includes(queryValue)) {
      stateMap[id] = ["found"];
    } else if (!selectedId) {
      stateMap[id] = [];
    } else if (id === selectedId) {
      stateMap[id] = ["selected"];
    } else if (neighborIds.has(id)) {
      stateMap[id] = ["highlight"];
    } else {
      stateMap[id] = ["inactive"];
    }
  }
  for (const edge of graph.getEdgeData()) {
    const id = String(edge.id);
    stateMap[id] = selectedId
      ? relatedEdges.includes(id)
        ? ["highlight"]
        : ["inactive"]
      : [];
  }
  try {
    void graph.setElementState(stateMap, true).catch(() => {});
    if (selectedId) {
      void graph.frontElement([selectedId, ...neighborIds]).catch(() => {});
    }
  } catch {
    // The graph may be mid-rebuild; transient G6 state errors are safe to skip.
  }
}

export function KnowledgeGraphPage({
  onOpenPdf,
}: {
  onOpenPdf?: (
    documentId: string,
    page: number,
    elementId?: string | null,
  ) => void;
}) {
  const workspace = useWorkspaceApi();
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph | null>(null);
  const semanticBandRef = useRef(1);
  const selectedIdRef = useRef<string | null>(null);
  const [query, setQuery] = useState("");
  const [layout, setLayout] = useState<LayoutId>("atlas");
  const [depth, setDepth] = useState("2");
  const [communityFilter, setCommunityFilter] =
    useState<CommunityId | "all">("all");
  const [zoomPercent, setZoomPercent] = useState(72);
  const [evidence, setEvidence] = useState<EvidenceLineage[]>([]);
  const [displayGraph, setDisplayGraph] = useState<GraphView | null>(null);
  const [selected, setSelected] = useState<SelectedNode | null>(null);
  // Stabilize the graph reference on content, not object identity: the
  // workspace polls graph data every few seconds, which would otherwise tear
  // down and re-fit the G6 canvas on every poll (constant zooming).
  const graphSignature = useMemo(() => {
    const graph = displayGraph ?? workspace.graph;
    if (!graph) return "";
    return [
      graph.nodes.map((node) => node.id).join("|"),
      graph.edges
        .map((edge) => `${edge.source}>${edge.target}`)
        .join("|"),
    ].join("::");
  }, [displayGraph, workspace.graph]);
  const currentGraph = useMemo(() => {
    return displayGraph ?? workspace.graph;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphSignature]);
  const scene = useMemo(
    () => (currentGraph ? buildScene(currentGraph, communityFilter) : null),
    [communityFilter, currentGraph],
  );

  useEffect(() => {
    setDisplayGraph(workspace.graph);
  }, [workspace.graph]);

  useEffect(() => {
    if (!workspace.activeKnowledgeBase || !workspace.graph) {
      return;
    }
    if (!query.trim()) {
      setDisplayGraph(workspace.graph);
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      void api
        .getGraph(
          workspace.activeKnowledgeBase!.id,
          Number(depth),
          query,
          undefined,
          200,
        )
        .then((next) => {
          if (!cancelled) setDisplayGraph(next);
        })
        .catch(() => undefined);
    }, 260);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    depth,
    query,
    workspace.activeKnowledgeBase,
    workspace.graph,
    workspace.mode,
  ]);

  useEffect(() => {
    if (!scene?.nodes.size) {
      setSelected(null);
      return;
    }
    if (selected && scene.nodes.has(selected.id)) return;
    const root =
      scene.nodes.get(scene.rootId) ?? scene.nodes.values().next().value;
    if (root) setSelected(root);
  }, [scene, selected]);

  useEffect(() => {
    selectedIdRef.current = selected?.id ?? null;
  }, [selected?.id]);

  useEffect(() => {
    if (
      !workspace.activeKnowledgeBase ||
      !selected ||
      !currentGraph?.nodes.some((node) => node.id === selected.id)
    ) {
      setEvidence([]);
      return;
    }
    let cancelled = false;
    void api
      .getGraphEvidence(workspace.activeKnowledgeBase.id, selected.id)
      .then((items) => {
        if (!cancelled) setEvidence(items);
      })
      .catch(() => {
        if (!cancelled) setEvidence([]);
      });
    return () => {
      cancelled = true;
    };
  }, [
    currentGraph,
    selected,
    workspace.activeKnowledgeBase,
    workspace.mode,
  ]);

  useEffect(() => {
    if (!containerRef.current || !scene?.data.nodes?.length) return;
    let disposed = false;
    let transformFrame = 0;
    const graphHost = containerRef.current.parentElement;
    const graph = new Graph({
      container: containerRef.current,
      width: graphHost?.clientWidth ?? containerRef.current.clientWidth,
      height: graphHost?.clientHeight ?? containerRef.current.clientHeight,
      autoResize: true,
      padding: [86, 78, 92, 78],
      zoomRange: [0.18, 2.2],
      background: "transparent",
      data: scene.data,
      layout: layoutOptions(layout),
      animation: {
        duration: 420,
        easing: "ease-out",
      },
      node: {
        type: (datum) =>
          datum.data?.shape === "rect" ? "rect" : "circle",
        style: (datum) => {
          const visual = visualForDatum(datum);
          const core = Boolean(datum.data?.core);
          const importance = Number(datum.data?.importance ?? 0);
          const visible = Boolean(datum.data?.labelVisible);
          return {
            size: core
              ? [Math.min(132, 68 + labelForDatum(datum).length * 7), 42]
              : Math.max(14, Math.min(32, 15 + importance * 2.1)),
            radius: core ? 21 : undefined,
            fill: core ? "#ffffff" : visual.color,
            fillOpacity: core ? 0.96 : 0.9,
            stroke: core ? visual.color : "#ffffff",
            strokeOpacity: core ? 0.74 : 0.95,
            lineWidth: core ? 2.4 : 2,
            shadowColor: visual.color,
            shadowBlur: core ? 22 : 10,
            shadowOpacity: core ? 0.24 : 0.16,
            cursor: "pointer",
            label: true,
            labelText: visible ? labelForDatum(datum) : "",
            labelPlacement: core ? "center" : "bottom",
            labelOffsetY: core ? 0 : 7,
            labelFill: core ? "#33415f" : "#39465e",
            labelFontSize: core ? 12 : 10,
            labelFontWeight: core ? 720 : 600,
            labelBackground: !core,
            labelBackgroundFill: "rgba(255,255,255,.9)",
            labelBackgroundFillOpacity: 1,
            labelBackgroundRadius: 6,
            labelBackgroundPadding: [3, 5, 3, 5],
            labelMaxWidth: 148,
            zIndex: core ? 5 : 3,
          };
        },
        state: {
          selected: (datum) => ({
            halo: true,
            haloStroke: visualForDatum(datum).color,
            haloStrokeOpacity: 0.2,
            haloLineWidth: 22,
            lineWidth: 4,
            stroke: visualForDatum(datum).color,
            labelText: labelForDatum(datum),
            labelFontWeight: 760,
            labelFontSize: 12,
            zIndex: 20,
          }),
          highlight: (datum) => ({
            fillOpacity: 1,
            stroke: visualForDatum(datum).color,
            lineWidth: 3,
            zIndex: 12,
          }),
          inactive: {
            fillOpacity: 0.13,
            strokeOpacity: 0.15,
            labelFillOpacity: 0.08,
          },
          found: (datum) => ({
            halo: true,
            haloStroke: visualForDatum(datum).color,
            haloStrokeOpacity: 0.22,
            haloLineWidth: 18,
            lineWidth: 4,
            labelText: labelForDatum(datum),
            labelFontWeight: 760,
            zIndex: 22,
          }),
        },
      },
      edge: {
        type: "cubic",
        style: (datum) => ({
          stroke: "#9ba9ca",
          strokeOpacity: 0.34,
          lineWidth: Math.min(
            2,
            0.85 + Number(datum.data?.evidenceCount ?? 0) * 0.08,
          ),
          endArrow: true,
          endArrowSize: 4,
          label: true,
          labelText: "",
          labelFill: "#77839a",
          labelFontSize: 9,
          labelBackground: true,
          labelBackgroundFill: "rgba(255,255,255,.9)",
          labelBackgroundPadding: [2, 4, 2, 4],
        }),
        state: {
          highlight: (datum) => ({
            stroke: "#5d72df",
            strokeOpacity: 0.88,
            lineWidth: 2.4,
            halo: true,
            haloStroke: "#7d8ff0",
            haloStrokeOpacity: 0.14,
            haloLineWidth: 8,
            zIndex: 11,
          }),
          inactive: {
            strokeOpacity: 0.06,
            labelFillOpacity: 0,
          },
        },
      },
      combo: {
        type: "rect",
        style: (datum) => {
          const id = String(datum.data?.community ?? "core") as CommunityId;
          const community = communityById[id] ?? communityById.core;
          return {
            fill: community.surface,
            fillOpacity: 1,
            stroke: community.stroke,
            pointerEvents: "stroke",
            lineWidth: 1.2,
            lineDash: [5, 7],
            radius: 28,
            padding: [44, 30, 30, 30],
            label: true,
            labelText: `${community.label} · ${String(
              datum.data?.count ?? 0,
            )}`,
            labelPlacement: "top",
            labelOffsetY: -16,
            labelFill: community.color,
            labelFontSize: 12,
            labelFontWeight: 760,
            labelBackground: true,
            labelPointerEvents: "none",
            labelBackgroundPointerEvents: "none",
            labelBackgroundFill: "rgba(255,255,255,.86)",
            labelBackgroundPadding: [4, 7, 4, 7],
            zIndex: 0,
          };
        },
      },
      behaviors: [
        "drag-canvas",
        "zoom-canvas",
        "drag-element",
        {
          type: "hover-activate",
          degree: 1,
        },
        {
          type: "auto-adapt-label",
          padding: 4,
        },
        "optimize-viewport-transform",
      ],
      // Minimap removed: its debounced render fired after graph.destroy(),
      // producing unhandled getData errors during graph rebuilds.
      plugins: [],
    });

    graph.on(NodeEvent.CLICK, (event) => {
      const id = String(
        (event as { target?: { id?: string } }).target?.id ?? "",
      );
      const next = scene.nodes.get(id);
      if (!next) return;
      setSelected(next);
      applyFocusStates(graph, id, query);
    });
    graph.on(CanvasEvent.CLICK, () => {
      setSelected(null);
      applyFocusStates(graph, null, query);
    });
    graph.on(GraphEvent.AFTER_TRANSFORM, () => {
      window.cancelAnimationFrame(transformFrame);
      transformFrame = window.requestAnimationFrame(() => {
        if (disposed || graph.destroyed) return;
        const zoom = graph.getZoom();
        const band = zoom < 0.72 ? 0 : zoom < 1.35 ? 1 : 2;
        setZoomPercent(Math.round(zoom * 100));
        if (band !== semanticBandRef.current) {
          semanticBandRef.current = band;
          applySemanticLabels(graph, selectedIdRef.current, band);
        }
      });
    });
    graph.on(GraphEvent.AFTER_RENDER, () => {
      if (disposed) return;
      const initialId =
        selectedIdRef.current && scene.nodes.has(selectedIdRef.current)
          ? selectedIdRef.current
          : scene.nodes.has(scene.rootId)
            ? scene.rootId
            : null;
      applyFocusStates(graph, initialId, query);
      applySemanticLabels(graph, initialId, semanticBandRef.current);
      setZoomPercent(Math.round(graph.getZoom() * 100));
    });

    graphRef.current = graph;
    void graph
      .render()
      .then(() => {
        if (disposed || graph.destroyed) return;
        void graph
          .fitView(
            { when: "always", direction: "both" },
            { duration: 560, easing: "ease-out" },
          )
          .then(() => {
            if (disposed || graph.destroyed) return;
            const fittedZoom = graph.getZoom();
            const minimumReadableZoom =
              scene.nodes.size <= 12 ? 1.25 : scene.nodes.size <= 45 ? 0.92 : 0.74;
            const maximumReadableZoom =
              scene.nodes.size <= 12 ? 1.28 : scene.nodes.size <= 45 ? 1.08 : 1;
            const readableZoom = Math.min(
              maximumReadableZoom,
              Math.max(minimumReadableZoom, fittedZoom),
            );
            const zoomPromise =
              Math.abs(readableZoom - fittedZoom) > 0.01
                ? graph.zoomTo(
                    readableZoom,
                    {
                      duration: 360,
                      easing: "ease-out",
                    },
                    [
                      containerRef.current?.clientWidth
                        ? containerRef.current.clientWidth / 2
                        : 0,
                      containerRef.current?.clientHeight
                        ? containerRef.current.clientHeight / 2
                        : 0,
                    ],
                  )
                : Promise.resolve();
            void zoomPromise
              .then(() =>
                graph.fitCenter({
                  duration: 260,
                  easing: "ease-out",
                }),
              )
              .then(() => {
                if (!disposed && !graph.destroyed) {
                  setZoomPercent(Math.round(graph.getZoom() * 100));
                }
              })
              .catch(() => {});
          })
          .catch(() => {});
      })
      .catch(() => {});

    return () => {
      disposed = true;
      window.cancelAnimationFrame(transformFrame);
      graph.destroy();
      if (graphRef.current === graph) graphRef.current = null;
    };
  }, [
    depth,
    layout,
    query,
    scene,
    // NOTE: workspace.activeKnowledgeBase is intentionally NOT a dependency:
    // it is not used inside the effect, and the workspace poll changes its
    // reference every few seconds, which would tear down and re-fit the canvas.
    workspace.mode,
  ]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph || graph.destroyed) return;
    applyFocusStates(graph, selected?.id ?? null, query);
    applySemanticLabels(
      graph,
      selected?.id ?? null,
      semanticBandRef.current,
    );
  }, [query, selected?.id]);

  const liveRelations = useMemo(() => {
    if (!selected || !currentGraph) return [];
    return currentGraph.edges
      .filter(
        (edge) => edge.source === selected.id || edge.target === selected.id,
      )
      .slice(0, 6)
      .map((edge) => {
        const otherId =
          edge.source === selected.id ? edge.target : edge.source;
        const other = currentGraph.nodes.find((node) => node.id === otherId);
        return {
          id: otherId,
          label: other ? compactGraphLabel(other) : otherId,
          relation: edge.relation,
          kind: other ? kindNames[other.kind] : "知识节点",
        };
      });
  }, [currentGraph, selected]);

  const selectRelatedNode = (nodeId: string) => {
    const next = scene?.nodes.get(nodeId);
    if (next) setSelected(next);
  };

  const resetGraph = () => {
    setQuery("");
    setCommunityFilter("all");
    setDisplayGraph(workspace.graph);
    const graph = graphRef.current;
    if (graph && !graph.destroyed) {
      void graph
        .fitView(
          { when: "always", direction: "both" },
          { duration: 480, easing: "ease-out" },
        )
        .catch(() => {});
    }
  };

  return (
    <section className="workspace-page graph-page graph-page-g6">
      <PageHeader
        eyebrow="KNOWLEDGE GRAPH"
        title="知识图谱"
        description="探索文档中的实体、概念和证据关系，沿图谱发现隐藏关联。"
        actions={
          <button
            className="button primary"
            onClick={() =>
              graphRef.current
                ?.fitView(
                  { when: "always", direction: "both" },
                  { duration: 420, easing: "ease-out" },
                )
                .catch(() => {})
            }
          >
            <Icon name="network" size={17} />
            适配画布
          </button>
        }
      />

      <div className="graph-toolbar graph-atlas-toolbar workspace-card">
        <label className="search-field graph-search">
          <Icon name="search" size={17} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索实体、概念或证据"
          />
        </label>
        <CustomSelect
          compact
          value={layout}
          onChange={(value) => setLayout(value as LayoutId)}
          options={[
            { value: "atlas", label: "社区星图" },
            { value: "radial", label: "环形社区" },
            { value: "free", label: "自由探索" },
          ]}
        />
        <CustomSelect
          compact
          value={depth}
          onChange={setDepth}
          options={[
            { value: "1", label: "1 跳关系" },
            { value: "2", label: "2 跳关系" },
            { value: "3", label: "3 跳关系" },
          ]}
        />
        <div className="graph-semantic-zoom">
          <span>语义缩放</span>
          <input
            type="range"
            min="38"
            max="180"
            value={Math.min(180, Math.max(38, zoomPercent))}
            onChange={(event) => {
              const next = Number(event.target.value);
              setZoomPercent(next);
              void graphRef.current
                ?.zoomTo(next / 100, {
                  duration: 180,
                  easing: "ease-out",
                })
                .catch(() => {});
            }}
          />
          <strong>{zoomPercent}%</strong>
        </div>
      </div>

      <div className="graph-community-bar workspace-card">
        <button
          className={communityFilter === "all" ? "is-active" : ""}
          onClick={() => setCommunityFilter("all")}
        >
          全部社区
          <strong>{currentGraph?.nodes.length ?? 0}</strong>
        </button>
        {communities.map((community) => (
          <button
            key={community.id}
            className={communityFilter === community.id ? "is-active" : ""}
            onClick={() => setCommunityFilter(community.id)}
            style={{ "--community-color": community.color } as React.CSSProperties}
          >
            <i />
            {community.label}
            <strong>{scene?.communityCounts[community.id] ?? 0}</strong>
          </button>
        ))}
      </div>

      <div className="graph-shell graph-atlas-shell">
        <div className="workspace-card graph-canvas graph-atlas-canvas">
          <div className="graph-atlas-aurora" />
          <div className="graph-grid-bg" />
          <div className="g6-stage" ref={containerRef} />

          <div className="graph-breadcrumb">
            <span>全局图谱</span>
            <i>/</i>
            <span>
              {selected
                ? communityById[selected.community].label
                : "全部社区"}
            </span>
            {selected && (
              <>
                <i>/</i>
                <strong>{selected.label}</strong>
              </>
            )}
          </div>

          <div className="graph-stats graph-atlas-stats">
            <span>
              <strong>{currentGraph?.total_node_count ?? 0}</strong> 节点
            </span>
            <i />
            <span>
              <strong>{currentGraph?.total_edge_count ?? 0}</strong> 关系
            </span>
            <i />
            <span>
              <strong>{workspace.documents.length || 8}</strong> 文档
            </span>
            {currentGraph?.truncated && <em>当前显示语义子图</em>}
          </div>

          <div className="graph-zoom graph-atlas-controls">
            <button
              aria-label="放大"
              onClick={() =>
                graphRef.current
                  ?.zoomBy(1.16, {
                    duration: 160,
                    easing: "ease-out",
                  })
                  .catch(() => {})
              }
            >
              +
            </button>
            <button
              aria-label="缩小"
              onClick={() =>
                graphRef.current
                  ?.zoomBy(0.86, {
                    duration: 160,
                    easing: "ease-out",
                  })
                  .catch(() => {})
              }
            >
              −
            </button>
            <button aria-label="重置图谱" onClick={resetGraph}>
              <Icon name="refresh" size={15} />
            </button>
          </div>
        </div>

        <aside className="workspace-card entity-panel atlas-entity-panel">
          {selected ? (
            <>
              <div className="entity-panel-head">
                <span
                  className="entity-mark"
                  style={{
                    background: communityById[selected.community].color,
                    boxShadow: `0 0 0 7px ${
                      communityById[selected.community].surface
                    }`,
                  }}
                >
                  <Icon name="network" size={19} />
                </span>
                <div>
                  <small>{kindNames[selected.kind]}</small>
                  <h2>{selected.label}</h2>
                </div>
                <button aria-label="返回全局图谱" onClick={resetGraph}>
                  <Icon name="close" size={16} />
                </button>
              </div>

              <div className="entity-community-chip">
                <i
                  style={{
                    background: communityById[selected.community].color,
                  }}
                />
                {communityById[selected.community].label}
                <span>{selected.degree} 条一跳关系</span>
              </div>

              <p className="entity-summary">{selected.detail}。</p>

              <div className="entity-confidence">
                <span>可回溯证据</span>
                <strong>{evidence.length} 条</strong>
                <i>
                  <em
                    style={{ width: `${Math.min(100, evidence.length * 18)}%` }}
                  />
                </i>
              </div>

              <h3>关系路径</h3>
              <div className="relation-list atlas-relation-list">
                {liveRelations.length ? (
                  liveRelations.map((relation) => (
                    <button
                      key={`${relation.id}-${relation.relation}`}
                      onClick={() => selectRelatedNode(relation.id)}
                    >
                      <span>{relation.label}</span>
                      <small>{relation.relation}</small>
                      <em>{relation.kind}</em>
                      <Icon name="arrowRight" size={14} />
                    </button>
                  ))
                ) : (
                  <p className="graph-empty-copy">
                    当前节点在这个子图中暂无可展开关系。
                  </p>
                )}
              </div>

              <h3>证据来源</h3>
              {evidence.length ? (
                evidence.slice(0, 4).map((item, index) => {
                  const document = workspace.documents.find(
                    (candidate) => candidate.id === item.document_id,
                  );
                  return (
                    <button
                      className="entity-source"
                      key={`${item.document_id}-${item.page}-${item.chunk_id}-${item.element_id ?? ""}`}
                      onClick={() =>
                        onOpenPdf?.(
                          item.document_id,
                          item.page,
                          item.element_id,
                        )
                      }
                    >
                      <span className="mini-pdf">PDF</span>
                      <div>
                        <strong>
                          {document?.filename ?? `证据文档 ${index + 1}`}
                        </strong>
                        <small>
                          第 {item.page} 页 ·{" "}
                          {item.element_id ? "元素证据" : "页面证据"}
                        </small>
                      </div>
                      <Icon name="external" size={15} />
                    </button>
                  );
                })
              ) : (
                <p className="graph-empty-copy">
                  当前节点暂无直接 PDF 证据，可沿关系路径继续探索。
                </p>
              )}

              <button className="button ghost entity-chat">
                <Icon name="message" size={16} />
                围绕此实体提问
              </button>
            </>
          ) : (
            <div className="graph-selection-empty">
              <span>
                <Icon name="network" size={23} />
              </span>
              <strong>选择一个知识节点</strong>
              <p>点击图谱节点，查看它的关系路径与 PDF 证据。</p>
              <button className="button ghost" onClick={resetGraph}>
                返回全局图谱
              </button>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
