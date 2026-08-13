import type { IconName } from "../components/Icons";

export type ArchitectureFlow = "all" | "query" | "compile" | "data";
export type ArchitectureNodeGroup =
  | "entry"
  | "agent"
  | "tool"
  | "guard"
  | "pipeline"
  | "model"
  | "storage";
export type ArchitectureEdgeKind =
  | "control"
  | "text"
  | "visual"
  | "evidence"
  | "storage";

export interface ArchitectureCode {
  path: string;
  language: "python";
  startLine: number;
  description: string;
  snippet: string;
}

export interface ArchitectureNode {
  id: string;
  title: string;
  subtitle: string;
  eyebrow: string;
  group: ArchitectureNodeGroup;
  icon: IconName;
  x: number;
  y: number;
  width: number;
  height: number;
  flows: Exclude<ArchitectureFlow, "all">[];
  summary: string;
  input: string;
  output: string;
  failure: string;
  code: ArchitectureCode;
}

export interface ArchitectureEdge {
  id: string;
  source: string;
  target: string;
  kind: ArchitectureEdgeKind;
  flows: Exclude<ArchitectureFlow, "all">[];
  label?: string;
}

const code = (
  path: string,
  startLine: number,
  description: string,
  snippet: string,
): ArchitectureCode => ({
  path,
  language: "python",
  startLine,
  description,
  snippet: snippet.trim(),
});

export const ARCHITECTURE_WORLD = { width: 2040, height: 1120 };

export const architectureNodes: ArchitectureNode[] = [
  {
    id: "fastapi",
    title: "FastAPI 业务边界",
    subtitle: "REST · SSE · CORS",
    eyebrow: "BACKEND ENTRY",
    group: "entry",
    icon: "external",
    x: 52,
    y: 418,
    width: 190,
    height: 104,
    flows: ["query", "compile", "data"],
    summary: "统一承接文档、会话、检索、Wiki、图谱、任务和设置 API，并在应用生命周期中装配真实运行时。",
    input: "HTTP / SSE 请求",
    output: "领域路由与结构化响应",
    failure: "依赖不可用时返回明确状态，不使用演示数据冒充成功。",
    code: code(
      "backend/app/main.py",
      13,
      "应用生命周期负责连接 LangGraph 持久化并注册全部业务路由。",
      `
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    from app.services.agent_runtime_registry import agent_runtime_registry
    from app.services.langgraph_persistence import open_langgraph_persistence
    from app.store import store

    runtime = get_settings()
    async with open_langgraph_persistence(runtime.database_url) as persistence:
        agent_runtime_registry.configure(persistence, store)
        try:
            yield
        finally:
            agent_runtime_registry.clear()
`,
    ),
  },
  {
    id: "chat-api",
    title: "会话与 SSE",
    subtitle: "多轮消息入口",
    eyebrow: "CHAT ROUTE",
    group: "entry",
    icon: "message",
    x: 300,
    y: 154,
    width: 190,
    height: 96,
    flows: ["query"],
    summary: "保存用户消息、恢复线程历史，并把问题交给 Deep Agents；完成后以有序内容块通过 SSE 返回。",
    input: "thread_id + 用户问题",
    output: "text / image / table / formula",
    failure: "模型或证据校验失败时发布明确拒绝信息。",
    code: code(
      "backend/app/api/routes/chat.py",
      452,
      "同一入口根据运行时开关调用 Deep Agents，并保持用户与助手消息的持久合同。",
      `
async def answer_question(
    thread_id: str, content: str
) -> tuple[dict, dict]:
    thread = require_item("threads", thread_id, "会话")
    user = append_message(
        thread_id, "user", [TextBlock(markdown=content).model_dump()], []
    )
    rag = RagSettings(**store.settings["rag"])
    assistant = await _answer_with_deepagents(
        thread=thread,
        thread_id=thread_id,
        content=content,
        rag=rag,
    )
    return user, assistant
`,
    ),
  },
  {
    id: "deep-agents",
    title: "Deep Agents 运行时",
    subtitle: "LangChain · LangGraph",
    eyebrow: "AGENT RUNTIME",
    group: "agent",
    icon: "brain",
    x: 548,
    y: 154,
    width: 200,
    height: 96,
    flows: ["query"],
    summary: "DeepSeek 驱动的 Agent Loop，根据问题自主决定零检索、文本检索、图谱检索或视觉检查。",
    input: "消息历史 + 工具白名单",
    output: "结构化 AgenticModelAnswer",
    failure: "不可用时显式失败；不会静默切回旧 RAG。",
    code: code(
      "backend/app/services/agent_runtime_registry.py",
      17,
      "运行时注册表只暴露白名单工具，并注入 PostgreSQL Checkpointer 与 Store。",
      `
class AgentRuntimeRegistry:
    persistence: LangGraphPersistence | None = None
    build: DeepAgentsBuildResult | None = None

    def configure(self, persistence, store: object) -> None:
        self.persistence = persistence
        self.refresh(store)

    def refresh(self, store: object) -> DeepAgentsBuildResult:
        tools = build_agentic_tools(store)
        self.build = create_deepagents_runtime_from_env(
            tools=tools,
            allowed_tool_names=frozenset(tool.name for tool in tools),
            checkpointer=(
                self.persistence.checkpointer if self.persistence else None
            ),
            store=self.persistence.store if self.persistence else None,
        )
        return self.build
`,
    ),
  },
  {
    id: "tool-router",
    title: "自主工具决策",
    subtitle: "最多 4 次调用预算",
    eyebrow: "AGENT POLICY",
    group: "agent",
    icon: "sliders",
    x: 802,
    y: 154,
    width: 190,
    height: 96,
    flows: ["query"],
    summary: "通用问题直接回答；私有原文、跨文档关系和视觉细节分别路由到不同工具。",
    input: "问题意图 + 当前上下文",
    output: "工具调用或零检索直答",
    failure: "达到工具预算后停止继续搜索。",
    code: code(
      "backend/app/services/deepagents_runtime.py",
      380,
      "系统提示明确规定何时检索、何时跳过，以及视觉资产和引用的使用边界。",
      `
system_prompt = (
    "你必须根据问题决定是否检索，不是每轮都调用工具。"
    "通用解释、写作、翻译、计算、寒暄直接使用模型知识；"
    "依据 PDF 时调用 search_chunks；跨文档关系可调用 search_graph；"
    "只有需要核对视觉细节时才调用 inspect_visual。"
    "citation_id 和 asset_id 必须逐字来自工具结果。"
)
`,
    ),
  },
  {
    id: "search-chunks",
    title: "search_chunks",
    subtitle: "BM25 + Dense + RRF",
    eyebrow: "TEXT RETRIEVAL",
    group: "tool",
    icon: "search",
    x: 1050,
    y: 48,
    width: 194,
    height: 82,
    flows: ["query"],
    summary: "在当前会话绑定知识库中检索文字和 语义向量，租户范围由服务端注入。",
    input: "query · document_ids · top_k",
    output: "Chunk、分数、页码、asset_id",
    failure: "空召回返回 empty，不伪造文档依据。",
    code: code(
      "backend/app/services/agentic_tools.py",
      129,
      "工具从请求上下文读取知识库范围，并把每个命中写进本轮证据账本。",
      `
class SearchChunksTool:
    name = "search_chunks"

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.input_model.model_validate(arguments)
        scope, ledger = _context()
        _reserve_tool_call(scope, ledger, self.name)
        hits = await self.store.retrieve(
            query=payload.query,
            knowledge_base_id=scope.knowledge_base_id,
            rag=scope.rag,
            document_ids=payload.document_ids,
        )
        ledger.add_hits(hits, source=self.name)
        result = {
            "mode": "empty" if not hits else "hybrid",
            "hits": [{
                "citation_id": hit.as_evidence().citation.id,
                "chunk_id": hit.chunk.id,
                "page": hit.chunk.page,
                "text": hit.chunk.text,
                "asset_ids": hit.chunk.asset_ids,
                "score": hit.score,
            } for hit in hits],
        }
        ledger.cached_results[self.name] = result
        return result
`,
    ),
  },
  {
    id: "search-graph",
    title: "search_graph",
    subtitle: "实体匹配 · 1–2 跳",
    eyebrow: "GRAPH RETRIEVAL",
    group: "tool",
    icon: "network",
    x: 1050,
    y: 144,
    width: 194,
    height: 82,
    flows: ["query"],
    summary: "在 Neo4j 图谱中寻找实体和语义关系，再把路径回落到可引用的 PDF Chunk。",
    input: "关系问题 + hops",
    output: "语义路径 + PDF citation_id",
    failure: "没有可核验 PDF 证据的关系不会进入最终答案。",
    code: code(
      "backend/app/services/agentic_tools.py",
      221,
      "图谱工具限制跳数和节点类型，只保留语义边并收集路径关联的证据。",
      `
class SearchGraphTool:
    name = "search_graph"

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.input_model.model_validate(arguments)
        scope, ledger = _context()
        repository = self.store.graph_repository
        snapshot = repository.snapshot(scope.knowledge_base_id)
        ranked = sorted(
            (
                (self._node_score(payload.query, node.label, node.properties), node)
                for node in snapshot.nodes
                if node.kind in {"entity", "claim", "wiki"}
            ),
            key=lambda item: (-item[0], item[1].label.casefold(), item[1].id),
        )
        candidates = [node for score, node in ranked[:3] if score > 0]
        hops = min(payload.hops or scope.rag.graph_hops, 2)
        for candidate in candidates:
            local = repository.local_subgraph(
                scope.knowledge_base_id,
                candidate.id,
                hops=hops,
                limit=payload.limit,
            )
            # 语义边的 lineage 随后回落到 PDF chunk evidence。
`,
    ),
  },
  {
    id: "inspect-visual",
    title: "inspect_visual",
    subtitle: "真实像素 · 最多 3 项",
    eyebrow: "VISION TOOL",
    group: "tool",
    icon: "image",
    x: 1050,
    y: 240,
    width: 194,
    height: 82,
    flows: ["query"],
    summary: "仅能查看本轮已经检索到的图片、表格和公式，把真实像素交给 检索证据语义。",
    input: "已授权 asset_id",
    output: "多模态消息 + inspected assets",
    failure: "越权 asset_id 或总字节超限立即拒绝。",
    code: code(
      "backend/app/services/agentic_tools.py",
      390,
      "视觉工具首先检查资产是否属于当前证据账本，再从 MinIO 读取并编码真实像素。",
      `
requested = list(dict.fromkeys(payload.asset_ids))
blocked = sorted(set(requested) - ledger.allowed_asset_ids)
if blocked:
    raise PermissionError(
        "inspect_visual received assets outside retrieved evidence"
    )

for asset_id in requested:
    object_key = self.store.asset_keys.get(asset_id)
    raw = self.store.storage.get_bytes(object_key)
    content.append({
        "type": "image",
        "base64": base64.b64encode(raw).decode("ascii"),
        "mime_type": "image/png",
    })
`,
    ),
  },
  {
    id: "fetch-evidence",
    title: "fetch_evidence",
    subtitle: "页码 · BBox · 素材回取",
    eyebrow: "EVIDENCE TOOL",
    group: "tool",
    icon: "quote",
    x: 1050,
    y: 336,
    width: 194,
    height: 82,
    flows: ["query"],
    summary: "按 citation_id 回取本轮证据账本里的完整页码、BBox、原文和资产信息。",
    input: "已存在 citation_id",
    output: "受信 Evidence metadata",
    failure: "不属于本轮账本的引用 ID 立即拒绝。",
    code: code(
      "backend/app/services/agentic_tools.py",
      475,
      "证据回取工具不能访问全库，只允许读取当前请求账本已经登记的 citation。",
      `
class FetchEvidenceTool:
    name = "fetch_evidence"

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self.input_model.model_validate(arguments)
        scope, ledger = _context()
        _reserve_tool_call(scope, ledger, self.name)
        unknown = sorted(set(payload.citation_ids) - set(ledger.evidence))
        if unknown:
            raise PermissionError(
                "fetch_evidence received citations outside the current ledger"
            )
        return {"evidence": [
            {
                "citation": ledger.evidence[item].citation.model_dump(mode="json"),
                "text": ledger.evidence[item].text,
                "asset_ids": ledger.evidence[item].asset_ids,
            } for item in payload.citation_ids
        ]}
`,
    ),
  },
  {
    id: "evidence-ledger",
    title: "本轮证据账本",
    subtitle: "请求级隔离",
    eyebrow: "EVIDENCE LEDGER",
    group: "guard",
    icon: "shield",
    x: 1304,
    y: 154,
    width: 198,
    height: 96,
    flows: ["query"],
    summary: "汇总本次 Agent 调用得到的 citation、asset 和工具轨迹，是最终答案唯一可信来源。",
    input: "各工具真实命中",
    output: "允许引用的证据集合",
    failure: "会话结束即释放，不能跨轮借用旧证据。",
    code: code(
      "backend/app/services/agentic_tools.py",
      39,
      "EvidenceLedger 由 ContextVar 绑定到单次请求，集中记录证据、工具轨迹和已检查素材。",
      `
@dataclass(slots=True)
class EvidenceLedger:
    evidence: dict[str, GroundedEvidence] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    inspected_asset_ids: list[str] = field(default_factory=list)
    cached_results: dict[str, Any] = field(default_factory=dict)

    @property
    def items(self) -> list[GroundedEvidence]:
        return list(self.evidence.values())
`,
    ),
  },
  {
    id: "grounding-validator",
    title: "证据硬校验",
    subtitle: "Citation · Asset 白名单",
    eyebrow: "GROUNDING GATE",
    group: "guard",
    icon: "check",
    x: 1558,
    y: 154,
    width: 198,
    height: 96,
    flows: ["query"],
    summary: "逐个检查模型输出中的 citation_id、asset_id 与素材来源，阻断伪引用和伪图片。",
    input: "结构化回答 + EvidenceLedger",
    output: "ValidatedAnswer",
    failure: "任意引用越权都会拒绝发布整轮答案。",
    code: code(
      "backend/app/services/providers.py",
      117,
      "应用层从可信检索结果复制引用元数据，模型没有定义来源的权限。",
      `
citations_by_id = {item.citation.id: item.citation for item in evidence}
assets_by_id = {
    asset_id: item.citation.id
    for item in evidence
    for asset_id in item.asset_ids
}

for citation_id in parsed.citation_ids:
    if citation_id not in citations_by_id:
        raise GroundingValidationError(
            f"answer references unavailable citation: {citation_id}"
        )
`,
    ),
  },
  {
    id: "block-response",
    title: "有序 Block 回复",
    subtitle: "文字 · 图片 · 表格 · 公式",
    eyebrow: "SSE OUTPUT",
    group: "entry",
    icon: "send",
    x: 1812,
    y: 154,
    width: 178,
    height: 96,
    flows: ["query"],
    summary: "流式发布 Agent 阶段状态与 Markdown 增量，完成后以经过校验的多模态内容块和 PDF 引用收口。",
    input: "ValidatedAnswer",
    output: "SSE 生命周期 + ValidatedAnswer",
    failure: "绝不使用模板答案掩盖真实链路失败。",
    code: code(
      "backend/app/api/routes/chat.py",
      793,
      "LangGraph 原生流持续投递状态和回答增量，最终消息仍受证据校验约束。",
      `
async for part in runtime.astream(
    messages=messages,
    thread_id=thread_id,
):
    statuses, delta = projector.feed(part)
    for status_text in statuses:
        yield {"type": "agent.status", "status": status_text}
    if delta:
        yield {"type": "answer.delta", "delta": delta}

answer, validated = _validate_deepagent_output(...)
yield {
    "type": "answer.completed",
    "message": MessageView(**assistant).model_dump(mode="json"),
}
`,
    ),
  },
  {
    id: "deepseek-chat",
    title: "DeepSeek V4 Flash",
    subtitle: "问答 · 结构化输出",
    eyebrow: "LLM MODEL",
    group: "model",
    icon: "sparkles",
    x: 1304,
    y: 324,
    width: 198,
    height: 90,
    flows: ["query", "compile"],
    summary: "驱动 Agentic RAG 推理与结构化回答，基于检索证据生成带引用的最终答案。",
    input: "消息 / 检索证据",
    output: "结构化回答 / 引用",
    failure: "过载或格式不合法时保留明确可重试状态。",
    code: code(
      "backend/app/services/providers.py",
      203,
      "LLM 适配器使用严格结构化输出，并控制超时、重试与图像输入来源。",
      `
class DeepSeekProvider:
    provider_name = "moonshot"

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
        images: Sequence[VisionInput] = (),
        history: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        # 只接受受控 data:image 或 ms:// 图像引用。
`,
    ),
  },
  {
    id: "documents-api",
    title: "文档上传与编译 API",
    subtitle: "签名校验 · SHA-256",
    eyebrow: "DOCUMENT ROUTE",
    group: "entry",
    icon: "upload",
    x: 300,
    y: 548,
    width: 190,
    height: 96,
    flows: ["compile"],
    summary: "校验 PDF 类型和文件签名，保存不可变原件并创建后台编译任务。",
    input: "multipart PDF",
    output: "Document + object_key",
    failure: "非 PDF、空文件或对象存储失败立即拒绝。",
    code: code(
      "backend/app/api/routes/documents.py",
      155,
      "上传入口先完成格式与签名校验，再把 PDF 原件写入对象存储。",
      `
payload = await file.read()
if not payload or not payload.startswith(b"%PDF-"):
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": "invalid_pdf_signature", "message": "文件不是有效的 PDF"},
    )
digest = hashlib.sha256(payload).hexdigest()
object_key = f"sources/{knowledge_base_id}/{uuid4()}/{filename}"
stored = store.storage.put_bytes(
    object_key, payload, "application/pdf"
)
`,
    ),
  },
  {
    id: "job-runner",
    title: "任务调度与重试",
    subtitle: "进程内 LocalJobRunner",
    eyebrow: "JOB ORCHESTRATION",
    group: "pipeline",
    icon: "tasks",
    x: 548,
    y: 548,
    width: 200,
    height: 96,
    flows: ["compile", "data"],
    summary: "把耗时编译移出请求路径，串联 OCR、语义增强、索引、图谱和 Wiki 阶段。",
    input: "compile_document Job",
    output: "阶段进度与结果",
    failure: "错误码区分可重试与不可重试，支持安全重跑。",
    code: code(
      "backend/app/services/job_execution.py",
      20,
      "任务执行器明确维护可重试错误集合，并按阶段更新持久状态。",
      `
RETRYABLE_ERROR_CODES = frozenset({
    "connection_error",
    "ocr_timeout",
    "ocr_unavailable",
    "provider_unavailable",
    "rate_limited",
    "service_unavailable",
})

def error_is_retryable(code: str | None) -> bool:
    return bool(code and code in RETRYABLE_ERROR_CODES)
`,
    ),
  },
  {
    id: "page-renderer",
    title: "PDF 页面渲染",
    subtitle: "PyMuPDF · 144 DPI",
    eyebrow: "PAGE RENDER",
    group: "pipeline",
    icon: "file",
    x: 802,
    y: 548,
    width: 190,
    height: 96,
    flows: ["compile"],
    summary: "把不可变 PDF 渲染成稳定像素坐标系，为 OCR、BBox 和视觉裁图提供共同基准。",
    input: "PDF bytes",
    output: "PNG 页面 + 宽高",
    failure: "单页异常可降低 DPI 或使用真实文字层兜底。",
    code: code(
      "backend/app/services/ingestion.py",
      48,
      "PyMuPDFPageRenderer 在统一 DPI 下输出页图和稳定尺寸。",
      `
class PyMuPDFPageRenderer:
    def render(self, pdf: bytes) -> list[RenderedPage]:
        document = fitz.open(stream=pdf, filetype="pdf")
        pages: list[RenderedPage] = []
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
            pages.append(RenderedPage(
                page_number=index + 1,
                width=pixmap.width,
                height=pixmap.height,
                image=pixmap.tobytes("png"),
            ))
        return pages
`,
    ),
  },
  {
    id: "paddle-ocr",
    title: "PaddleOCR / Dspark",
    subtitle: "PP-DocLayoutV3 · OCR-VL",
    eyebrow: "LAYOUT AUTHORITY",
    group: "model",
    icon: "brain",
    x: 1050,
    y: 548,
    width: 194,
    height: 96,
    flows: ["compile"],
    summary: "识别页面元素、阅读顺序、文本和坐标，是页码、element_id 与 BBox 的权威来源。",
    input: "页面 PNG",
    output: "结构化 layout elements",
    failure: "失败页降 DPI 重试；仍失败时只使用真实 PDF 文字层。",
    code: code(
      "backend/app/services/ocr_client.py",
      24,
      "OCR 客户端合同返回页面级结构，调用方保留清晰的服务错误与重试边界。",
      `
class OCRClient(Protocol):
    def parse_page(
        self,
        page: RenderedPage,
        *,
        document_id: str,
        document_version: int,
    ) -> dict[str, Any]: ...

class OCRServiceError(RuntimeError):
    def __init__(self, message: str, *, code: str, retryable: bool):
        self.code = code
        self.retryable = retryable
        super().__init__(message)
`,
    ),
  },
  {
    id: "trusted-manifest",
    title: "可信版面 Manifest",
    subtitle: "Lineage · BBox · 阅读顺序",
    eyebrow: "IMMUTABLE CONTRACT",
    group: "guard",
    icon: "shield",
    x: 1304,
    y: 548,
    width: 198,
    height: 96,
    flows: ["compile"],
    summary: "冻结 Paddle 产生的文档版本、页面、元素、BBox 和顺序，后续模型只能追加语义。",
    input: "Paddle layout",
    output: "multimodal-pdf-v2 manifest",
    failure: "Lineage 不一致的增强结果不可发布。",
    code: code(
      "backend/app/services/ingestion.py",
      19,
      "V2 编译合同明确区分可信几何和后续语义覆盖层。",
      `
PIPELINE_VERSION = "multimodal-pdf-v2"
TRUSTED_LAYOUT_CONTRACT = "paddle-v3-layout-lineage-v1"
MULTIMODAL_KINDS = {"figure", "table", "formula"}

class CompilationError(RuntimeError):
    def __init__(self, message, *, stage, code, retryable=False):
        self.stage = stage
        self.code = code
        self.retryable = retryable
        super().__init__(message)
`,
    ),
  },
  {
    id: "llm-enrichment",
    title: "LLM 语义覆盖层",
    subtitle: "整页 + 元素双层增强",
    eyebrow: "SEMANTIC OVERLAY",
    group: "pipeline",
    icon: "image",
    x: 1558,
    y: 548,
    width: 198,
    height: 96,
    flows: ["compile"],
    summary: "描述图片、表格、公式和富媒体页面，为检索补充摘要、关系和关键词，不改写原始几何。",
    input: "裁图 + 可信 lineage",
    output: "multimodal-semantic-v2",
    failure: "增强失败保留 pending 状态和原始 Paddle 内容。",
    code: code(
      "backend/app/services/enrichment.py",
      24,
      "语义覆盖层拥有独立版本，发布前必须回显并核对原始 lineage。",
      `
ENRICHMENT_VERSION = "deepseek-element-v1"
PAGE_ENRICHMENT_VERSION = "deepseek-page-v1"
SEMANTIC_ARTIFACT_VERSION = "multimodal-semantic-v2"

class DeepSeekElementOutput(BaseModel):
    element_id: str
    document_id: str
    document_version: int
    page: int
    bbox: list[float]
    description: str
    search_text: str
`,
    ),
  },
  {
    id: "hybrid-indexer",
    title: "混合索引发布",
    subtitle: "Chunk · Embedding · Fingerprint",
    eyebrow: "INDEX PUBLISH",
    group: "pipeline",
    icon: "database",
    x: 1812,
    y: 548,
    width: 178,
    height: 96,
    flows: ["compile", "data"],
    summary: "把正文与 语义向量合并成 Chunk，生成稠密向量并以不可变 Manifest 发布。",
    input: "trusted chunks + semantic overlay",
    output: "Hybrid retrieval manifest",
    failure: "Embedding 不可用时明确降级为 lexical_fallback。",
    code: code(
      "backend/app/store.py",
      748,
      "编译主链在语义增强完成后发布索引，并把验证后的 Chunk 恢复到运行时检索器。",
      `
embedder = self._embedding_provider()
indexer = DocumentIndexer(storage=self.storage, embedder=embedder)
index_result = asyncio.run(indexer.index(
    result.manifest_key,
    document_title=str(document["title"]),
))
loaded = indexer.load_retriever(
    index_result.manifest_key,
    embedder=embedder if index_result.mode == "hybrid" else None,
)
self.retriever.upsert(loaded._chunks.values())
`,
    ),
  },
  {
    id: "knowledge-builder",
    title: "图谱与 LLM Wiki 构建",
    subtitle: "LLM 抽取 · Evidence links",
    eyebrow: "KNOWLEDGE DERIVATION",
    group: "pipeline",
    icon: "book",
    x: 1304,
    y: 744,
    width: 198,
    height: 94,
    flows: ["compile", "data"],
    summary: "跨全部 Chunk 抽取实体、主张和关系，构建可阅读 Wiki，并保留回溯 PDF 的证据边。",
    input: "已发布 Chunk 与元素",
    output: "GraphSnapshot + Wiki pages",
    failure: "知识派生失败不回滚已成功的文档索引。",
    code: code(
      "backend/app/store.py",
      903,
      "知识构建按批提取后确定性合并，再一次性编译图谱和 Wiki。",
      `
async def build_graph_and_wiki(
    self,
    knowledge_base_id: str,
    document_ids: list[str],
) -> dict[str, int]:
    batches = []
    for offset in range(0, len(chunks), 8):
        batches.append(await extractor.extract(
            chunks[offset:offset + 8],
            elements=elements,
        ))
    extraction = merge_graph_extractions(batches)
    return self.wiki_graph.compile(
        knowledge_base_id, extraction, source_fingerprint=fingerprint
    )
`,
    ),
  },
  {
    id: "embedding-model",
    title: "百炼 Embedding",
    subtitle: "text-embedding-v4 · 1024d",
    eyebrow: "DENSE MODEL",
    group: "model",
    icon: "brain",
    x: 1812,
    y: 744,
    width: 178,
    height: 94,
    flows: ["compile", "query"],
    summary: "为 Chunk 和查询生成稠密向量；OpenAI 槽位保留但当前因配额切换到百炼主路。",
    input: "文本批次",
    output: "1024 维向量",
    failure: "不可用时索引与检索清楚标记词法降级。",
    code: code(
      "backend/app/services/embeddings.py",
      64,
      "百炼适配器保留 query/document 语义区分，并固定索引维度合同。",
      `
class BailianEmbeddingProvider(_HttpEmbeddingProvider):
    provider_name = "aliyun-bailian"

    async def _embed(
        self,
        texts: Sequence[str],
        text_type: Literal["query", "document"],
    ) -> list[list[float]]:
        body = await self._post(
            "/api/v1/services/embeddings/text-embedding/text-embedding",
            {
                "model": self.model,
                "input": {"texts": list(texts)},
                "parameters": {
                    "text_type": text_type,
                    "dimension": self.dimensions,
                    "output_type": "dense",
                },
            },
        )
`,
    ),
  },
  {
    id: "postgres",
    title: "PostgreSQL",
    subtitle: "元数据 · 消息 · Checkpoint",
    eyebrow: "DURABLE STATE",
    group: "storage",
    icon: "database",
    x: 548,
    y: 940,
    width: 200,
    height: 88,
    flows: ["data", "query", "compile"],
    summary: "持久化知识库、文档、任务、会话与消息，同时承载 LangGraph checkpoint 和跨线程 Store。",
    input: "业务状态与 Agent state",
    output: "可恢复快照",
    failure: "非 PostgreSQL 环境才显式降级为进程内状态。",
    code: code(
      "backend/app/services/langgraph_persistence.py",
      36,
      "部署环境同时打开异步 Postgres Checkpointer 和 Store，并在启动时建表。",
      `
async with (
    AsyncPostgresSaver.from_conn_string(
        dsn, serde=serializer
    ) as checkpointer,
    AsyncPostgresStore.from_conn_string(dsn) as store,
):
    await checkpointer.setup()
    await store.setup()
    yield LangGraphPersistence(
        checkpointer=checkpointer,
        store=store,
        durable=True,
        backend="postgresql",
    )
`,
    ),
  },
  {
    id: "minio",
    title: "MinIO",
    subtitle: "PDF · 页图 · 裁图 · Manifest",
    eyebrow: "OBJECT STORAGE",
    group: "storage",
    icon: "folder",
    x: 802,
    y: 940,
    width: 190,
    height: 88,
    flows: ["data", "compile", "query"],
    summary: "保存不可变 PDF 原件、渲染页、元素裁图、语义覆盖层和索引 Manifest。",
    input: "二进制对象与 JSON",
    output: "稳定 object_key",
    failure: "对象缺失会阻断对应证据或视觉检查。",
    code: code(
      "backend/app/services/assets.py",
      20,
      "对象存储由统一协议隔离，业务层只持有 key 和内容类型。",
      `
class ObjectStorage(Protocol):
    def put_bytes(
        self, key: str, data: bytes, content_type: str
    ) -> StoredObject: ...

    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
`,
    ),
  },
  {
    id: "runtime-index",
    title: "运行时混合检索器",
    subtitle: "BM25 · Dense · RRF",
    eyebrow: "RETRIEVAL STATE",
    group: "storage",
    icon: "search",
    x: 1050,
    y: 940,
    width: 194,
    height: 88,
    flows: ["data", "query", "compile"],
    summary: "服务启动时从最新成功索引 Manifest 恢复，用 RRF 融合词法与向量结果。",
    input: "查询向量 + 词法 token",
    output: "排序后的 RetrievalHit",
    failure: "向量服务异常时仍可提供明确的词法检索。",
    code: code(
      "backend/app/services/retrieval.py",
      112,
      "检索器分别计算词法和稠密排名，再通过 Reciprocal Rank Fusion 合并。",
      `
lexical = self._lexical_search(query, candidates, rag.lexical_top_k)
dense = await self._dense_search(query, candidates, rag.dense_top_k)
scores: dict[str, float] = defaultdict(float)
for rank, hit in enumerate(lexical, start=1):
    scores[hit.chunk.id] += 1.0 / (60 + rank)
for rank, hit in enumerate(dense, start=1):
    scores[hit.chunk.id] += 1.0 / (60 + rank)
`,
    ),
  },
  {
    id: "neo4j",
    title: "Neo4j",
    subtitle: "实体 · 主张 · 证据边",
    eyebrow: "GRAPH STORE",
    group: "storage",
    icon: "network",
    x: 1304,
    y: 940,
    width: 198,
    height: 88,
    flows: ["data", "query", "compile"],
    summary: "保存语义图谱及 PDF 证据关系，为多跳查询和知识图谱页面提供子图。",
    input: "GraphSnapshot",
    output: "实体路径与 evidence_ids",
    failure: "图谱不可用不影响纯 Chunk 检索。",
    code: code(
      "backend/app/services/graph_repository.py",
      122,
      "图谱仓储对外只暴露版本化快照和有界局部子图。",
      `
class GraphRepository(Protocol):
    def replace(self, snapshot: GraphSnapshot) -> None: ...
    def snapshot(self, knowledge_base_id: str) -> GraphSnapshot: ...
    def local_subgraph(
        self,
        knowledge_base_id: str,
        node_id: str,
        *,
        hops: int,
        limit: int,
    ) -> GraphSnapshot: ...
`,
    ),
  },
  {
    id: "secrets",
    title: "受保护凭证存储",
    subtitle: "600 权限 · 不回显",
    eyebrow: "SECURITY BOUNDARY",
    group: "storage",
    icon: "key",
    x: 1558,
    y: 940,
    width: 198,
    height: 88,
    flows: ["data", "query", "compile"],
    summary: "API Key 只进入受保护服务端文件和进程环境，公开 API 仅返回是否已配置。",
    input: "SecretStr",
    output: "Provider 运行时配置",
    failure: "凭证不会进入日志、前端、OpenAPI 响应或业务快照。",
    code: code(
      "backend/app/services/provider_secret_store.py",
      24,
      "凭证存储通过原子替换和权限检查维护服务端边界。",
      `
class ProviderSecretStore:
    def update(self, values: dict[str, str]) -> None:
        if not values or set(values) - ALLOWED_ENV_KEYS:
            raise ProviderSecretStoreError(
                "credential update contains unsupported keys"
            )
        with self._lock:
            current = self._read()
            current.update({key: value.strip() for key, value in values.items()})
            descriptor, name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", dir=self.path.parent
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(current, handle, ensure_ascii=False, sort_keys=True)
            os.replace(Path(name), self.path)
            os.chmod(self.path, 0o600)
`,
    ),
  },
];

export const architectureEdges: ArchitectureEdge[] = [
  { id: "api-chat", source: "fastapi", target: "chat-api", kind: "control", flows: ["query"], label: "问题 / SSE" },
  { id: "chat-agent", source: "chat-api", target: "deep-agents", kind: "control", flows: ["query"] },
  { id: "agent-router", source: "deep-agents", target: "tool-router", kind: "control", flows: ["query"], label: "意图判断" },
  { id: "router-chunks", source: "tool-router", target: "search-chunks", kind: "text", flows: ["query"] },
  { id: "router-graph", source: "tool-router", target: "search-graph", kind: "text", flows: ["query"] },
  { id: "router-visual", source: "tool-router", target: "inspect-visual", kind: "visual", flows: ["query"] },
  { id: "ledger-fetch", source: "evidence-ledger", target: "fetch-evidence", kind: "evidence", flows: ["query"], label: "按 ID 回取" },
  { id: "fetch-validator", source: "fetch-evidence", target: "grounding-validator", kind: "evidence", flows: ["query"] },
  { id: "router-k3", source: "tool-router", target: "deepseek-chat", kind: "control", flows: ["query"], label: "零检索 / 最终推理" },
  { id: "chunks-ledger", source: "search-chunks", target: "evidence-ledger", kind: "evidence", flows: ["query"] },
  { id: "graph-ledger", source: "search-graph", target: "evidence-ledger", kind: "evidence", flows: ["query"] },
  { id: "visual-ledger", source: "inspect-visual", target: "evidence-ledger", kind: "visual", flows: ["query"] },
  { id: "ledger-validator", source: "evidence-ledger", target: "grounding-validator", kind: "evidence", flows: ["query"], label: "允许引用集合" },
  { id: "validator-response", source: "grounding-validator", target: "block-response", kind: "control", flows: ["query"], label: "校验通过" },
  { id: "agent-postgres", source: "deep-agents", target: "postgres", kind: "storage", flows: ["query", "data"], label: "checkpoint" },
  { id: "chat-postgres", source: "chat-api", target: "postgres", kind: "storage", flows: ["query", "data"] },
  { id: "visual-minio", source: "inspect-visual", target: "minio", kind: "storage", flows: ["query", "data"] },
  { id: "chunks-index", source: "search-chunks", target: "runtime-index", kind: "storage", flows: ["query", "data"] },
  { id: "graph-neo4j", source: "search-graph", target: "neo4j", kind: "storage", flows: ["query", "data"] },
  { id: "api-documents", source: "fastapi", target: "documents-api", kind: "control", flows: ["compile"] },
  { id: "documents-jobs", source: "documents-api", target: "job-runner", kind: "control", flows: ["compile"], label: "创建编译任务" },
  { id: "documents-minio", source: "documents-api", target: "minio", kind: "storage", flows: ["compile", "data"], label: "原始 PDF" },
  { id: "jobs-renderer", source: "job-runner", target: "page-renderer", kind: "control", flows: ["compile"] },
  { id: "renderer-paddle", source: "page-renderer", target: "paddle-ocr", kind: "visual", flows: ["compile"], label: "页面像素" },
  { id: "renderer-minio", source: "page-renderer", target: "minio", kind: "storage", flows: ["compile", "data"] },
  { id: "paddle-manifest", source: "paddle-ocr", target: "trusted-manifest", kind: "text", flows: ["compile"], label: "元素 + BBox" },
  { id: "manifest-enrichment", source: "trusted-manifest", target: "llm-enrichment", kind: "text", flows: ["compile"] },
  { id: "enrichment-k3", source: "llm-enrichment", target: "deepseek-chat", kind: "visual", flows: ["compile"], label: "裁图 / 整页" },
  { id: "enrichment-index", source: "llm-enrichment", target: "hybrid-indexer", kind: "text", flows: ["compile"] },
  { id: "index-embedding", source: "hybrid-indexer", target: "embedding-model", kind: "text", flows: ["compile"] },
  { id: "index-runtime", source: "hybrid-indexer", target: "runtime-index", kind: "storage", flows: ["compile", "data"], label: "发布 Manifest" },
  { id: "index-minio", source: "hybrid-indexer", target: "minio", kind: "storage", flows: ["compile", "data"] },
  { id: "index-builder", source: "hybrid-indexer", target: "knowledge-builder", kind: "text", flows: ["compile"] },
  { id: "builder-neo4j", source: "knowledge-builder", target: "neo4j", kind: "storage", flows: ["compile", "data"] },
  { id: "builder-postgres", source: "knowledge-builder", target: "postgres", kind: "storage", flows: ["compile", "data"] },
  { id: "models-secrets", source: "secrets", target: "deepseek-chat", kind: "storage", flows: ["query", "compile", "data"] },
  { id: "embedding-secrets", source: "secrets", target: "embedding-model", kind: "storage", flows: ["query", "compile", "data"] },
];

export const architectureGroups: Record<
  ArchitectureNodeGroup,
  { label: string; color: string; surface: string }
> = {
  entry: { label: "API 边界", color: "#72b7ff", surface: "rgba(57,135,255,.16)" },
  agent: { label: "Agent 运行时", color: "#9d83ff", surface: "rgba(126,88,255,.17)" },
  tool: { label: "检索工具", color: "#49d8d0", surface: "rgba(36,196,190,.15)" },
  guard: { label: "证据与合同", color: "#ffba70", surface: "rgba(245,149,62,.16)" },
  pipeline: { label: "编译流水线", color: "#70e2a5", surface: "rgba(49,193,122,.15)" },
  model: { label: "模型服务", color: "#d58bff", surface: "rgba(180,93,255,.16)" },
  storage: { label: "数据基础设施", color: "#8aa4c8", surface: "rgba(101,134,177,.17)" },
};

export const edgeKinds: Record<
  ArchitectureEdgeKind,
  { label: string; color: string }
> = {
  control: { label: "控制信号", color: "#6e8cff" },
  text: { label: "文本 / 向量", color: "#35d5c8" },
  visual: { label: "视觉像素", color: "#bd7bff" },
  evidence: { label: "证据引用", color: "#ffab64" },
  storage: { label: "持久化读写", color: "#7895bd" },
};
