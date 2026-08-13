import { useRef, useState, type ChangeEvent } from "react";
import { CustomSelect } from "../components/CustomSelect";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useWorkspaceApi } from "../hooks/useWorkspaceApi";

type CompileStatus = "ready" | "processing" | "queued" | "failed";

interface LibraryDocument {
  id: string;
  name: string;
  pages: number;
  size: string;
  elements: number;
  status: CompileStatus;
  progress: number;
  updatedAt: string;
}

const statusCopy: Record<CompileStatus, string> = {
  ready: "已入库",
  processing: "编译中",
  queued: "待编译",
  failed: "需重试",
};

export function KnowledgeBasePage() {
  const workspace = useWorkspaceApi();
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState("all");
  const [notice, setNotice] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const documents: LibraryDocument[] = workspace.documents.map((document) => ({
    id: document.id,
    name: document.filename,
    pages: document.page_count ?? 0,
    size: "服务端存储",
    elements: document.element_count,
    status:
      document.status === "ready"
        ? "ready"
        : document.status === "uploaded" || document.status === "queued"
          ? "queued"
          : document.status === "failed"
            ? "failed"
            : "processing",
    progress:
      workspace.jobs.find(
        (job) => job.document_id === document.id && job.is_current,
      )?.progress ?? (document.status === "ready" ? 100 : 0),
    updatedAt: new Date(document.updated_at).toLocaleString("zh-CN", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  const filtered = documents.filter((document) => {
    const matchesQuery = document.name.toLowerCase().includes(query.toLowerCase());
    const matchesScope = scope === "all" || document.status === scope;
    return matchesQuery && matchesScope;
  });

  const handleFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;
    setNotice(`正在上传 ${files.length} 份 PDF…`);
    try {
      const { documents: uploaded, job } = await workspace.uploadDocuments(files);
      setNotice(
        `已上传 ${uploaded.length} 份 PDF，知识库任务 ${job.id.slice(0, 8)} 正在自动编译、索引并更新 Wiki`,
      );
    } catch (error) {
      setNotice(`上传或启动建库失败：${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      event.target.value = "";
    }
  };

  const compile = async (id: string) => {
    try {
      const failed = workspace.jobs.find(
        (job) =>
          job.document_id === id &&
          job.is_current &&
          (job.status === "failed" || job.status === "partial") &&
          job.retryable,
      );
      const job = failed
        ? await workspace.retryJob(failed.id)
        : await workspace.compileDocument(id);
      setNotice(`编译任务 ${job.id.slice(0, 8)} 已进入单并发处理队列`);
    } catch (error) {
      setNotice(`创建编译任务失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const removeDocument = async (document: LibraryDocument) => {
    if (!window.confirm(`确定删除文档「${document.name}」吗？此操作不可撤销。`)) {
      return;
    }
    try {
      await workspace.deleteDocument(document.id);
      setNotice(`文档「${document.name}」已删除`);
    } catch (error) {
      setNotice(`删除失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  return (
    <section className="workspace-page knowledge-page">
      <PageHeader
        eyebrow="KNOWLEDGE BASE"
        title="知识库"
        description="上传 PDF、完成多模态编译，并管理可检索的知识资产。"
        actions={
          <>
            <button className="button secondary" onClick={async () => {
              try {
                const job = await workspace.compileKnowledgeBase();
                setNotice(`知识库更新任务 ${job.id.slice(0, 8)} 已启动，将补齐编译并更新 Wiki 与图谱`);
              } catch (error) {
                setNotice(`提交失败：${error instanceof Error ? error.message : "未知错误"}`);
              }
            }}>
              <Icon name="refresh" size={17} />
              更新知识库
            </button>
            <button className="button primary" onClick={() => fileRef.current?.click()}>
              <Icon name="upload" size={17} />
              上传文档
            </button>
            <input ref={fileRef} type="file" accept=".pdf" multiple hidden onChange={handleFiles} />
          </>
        }
      />

      {notice && (
        <div className="inline-notice">
          <Icon name="check" size={17} />
          <span>{notice}</span>
          <button onClick={() => setNotice("")} aria-label="关闭提示">×</button>
        </div>
      )}

      <div className="metric-grid">
        <article><span className="metric-icon blue"><Icon name="file" /></span><div><strong>{documents.length}</strong><small>知识文档</small></div><i>真实统计</i></article>
        <article><span className="metric-icon violet"><Icon name="database" /></span><div><strong>{documents.reduce((sum, item) => sum + item.pages, 0)}</strong><small>已解析页</small></div><i>真实统计</i></article>
        <article><span className="metric-icon green"><Icon name="image" /></span><div><strong>{workspace.documents.reduce((sum, item) => sum + item.asset_count, 0)}</strong><small>多模态元素</small></div><i>图表 · 公式</i></article>
        <article><span className="metric-icon orange"><Icon name="network" /></span><div><strong>{workspace.graph?.total_node_count ?? 0}</strong><small>知识节点</small></div><i>{workspace.graph?.total_edge_count ?? 0} 关系</i></article>
      </div>

      <div className="knowledge-layout">
        <article className="workspace-card library-card">
          <div className="card-toolbar">
            <div>
              <h2>文档资产</h2>
              <span>{filtered.length} 份结果</span>
            </div>
            <div className="toolbar-controls">
              <label className="search-field">
                <Icon name="search" size={17} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文件名" />
              </label>
              <CustomSelect
                compact
                value={scope}
                onChange={setScope}
                options={[
                  { value: "all", label: "全部状态" },
                  { value: "ready", label: "已入库" },
                  { value: "processing", label: "编译中" },
                  { value: "queued", label: "待编译" },
                  { value: "failed", label: "需处理" },
                ]}
              />
            </div>
          </div>

          <div className="document-table">
            <div className="document-table-head">
              <span>文档</span><span>解析结果</span><span>状态</span><span>更新时间</span><span />
            </div>
            {filtered.map((document) => (
              <div className="document-table-row" key={document.id}>
                <div className="document-cell-main">
                  <span className="mini-pdf">PDF</span>
                  <span><strong>{document.name}</strong><small>{document.pages} 页 · {document.size}</small></span>
                </div>
                <div className="element-count">
                  <strong>{document.elements || "—"}</strong>
                  <small>结构元素</small>
                </div>
                <div>
                  <span className={`status-pill ${document.status}`}>
                    {document.status === "processing" && <i />}
                    {statusCopy[document.status]}
                  </span>
                  {document.status === "processing" && (
                    <div className="mini-progress"><i style={{ width: `${document.progress}%` }} /></div>
                  )}
                </div>
                <small className="updated-at">{document.updatedAt}</small>
                <div className="row-actions">
                  {(document.status === "queued" || document.status === "failed") && (
                    <button title="开始编译" onClick={() => compile(document.id)}><Icon name="play" size={16} /></button>
                  )}
                  <button
                    title="删除文档"
                    onClick={() => void removeDocument(document)}
                  >
                    <Icon name="trash" size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </article>

        <aside className="workspace-card pipeline-card">
          <div className="pipeline-title">
            <span><Icon name="sparkles" size={18} /></span>
            <div><h2>编译流水线</h2><p>PaddleOCR + DeepSeek</p></div>
          </div>
          {[
            ["01", "页面渲染与布局检测", "识别段落、图表、公式与阅读顺序"],
            ["02", "语义增强", "LLM 生成标题、描述与结构化元数据"],
            ["03", "分块与向量化", "保留页码、坐标和素材引用"],
            ["04", "实体关系抽取", "同步更新 Wiki 与知识图谱"],
          ].map(([order, title, copy], index) => (
            <div className="pipeline-step" key={order}>
              <span>{order}</span>
              <div><strong>{title}</strong><small>{copy}</small></div>
              {index < 3 && <i />}
            </div>
          ))}
          <div className="pipeline-health">
            <span><i />后端编排服务在线</span>
            <span>{workspace.health?.status ?? "未知"}</span>
          </div>
        </aside>
      </div>
    </section>
  );
}
