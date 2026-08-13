import { useState } from "react";
import { CustomSelect } from "../components/CustomSelect";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useWorkspaceApi } from "../hooks/useWorkspaceApi";

const taskStatus = {
  running: "运行中",
  queued: "排队中",
  done: "已完成",
  partial: "部分完成",
  failed: "失败",
} as const;

const stageCopy: Record<string, string> = {
  waiting: "等待执行",
  planning: "分析待处理文档",
  loading_source: "读取源 PDF",
  rendering: "渲染 PDF 页面",
  ocr: "版面检测与 OCR",
  normalizing: "整理多模态元素",
  materializing_assets: "保存图片、表格与公式",
  writing_markdown: "生成结构化 Markdown",
  chunking: "切分可检索片段",
  writing_manifest: "发布编译清单",
  multimodal_enrichment: "LLM 语义增强",
  embedding_index: "生成混合检索索引",
  compiling_documents: "依次编译待入库文档",
  graph_wiki: "更新知识图谱与 LLM Wiki",
  completed: "处理完成",
  interrupted: "服务重启导致任务中断",
};

function stageLabel(raw: string): string {
  // "编译 2/3：ocr" -> "编译 2/3：版面检测与 OCR"
  const match = raw.match(/^(编译 \d+\/\d+：)(.+)$/);
  if (match) return `${match[1]}${stageCopy[match[2]] ?? match[2]}`;
  return stageCopy[raw] ?? raw;
}

export function TasksPage() {
  const workspace = useWorkspaceApi();
  const [scope, setScope] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState("");
  const items = workspace.jobs.map((job) => {
    const document = workspace.documents.find((item) => item.id === job.document_id);
    const status =
      job.status === "succeeded"
        ? "done"
        : job.status === "partial"
          ? "partial"
          : job.status === "running"
            ? "running"
            : job.status === "failed"
              ? "failed"
              : "queued";
    return {
      id: job.id,
      name: document?.filename ?? (
        job.kind === "build_wiki" ? "全库 Wiki 更新" :
        job.kind === "build_graph" ? "知识图谱构建" :
        job.kind === "rebuild_knowledge_base" ? "全知识库重建" :
        job.kind
      ),
      kind: job.kind.includes("wiki") ? "Wiki 生成" : job.kind.includes("graph") ? "图谱构建" : "文档编译",
      progress: job.progress,
      status,
      stage: job.error_message ?? stageLabel(job.stage),
      rawStage: job.stage,
      started: new Date(job.created_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
      updated: new Date(job.updated_at).toLocaleString("zh-CN"),
      time:
        !job.is_current
          ? "历史尝试"
          : job.status === "failed" || job.status === "partial"
            ? job.retryable
              ? "可以直接重试"
              : "需要检查配置或文档"
            : job.status === "succeeded"
              ? "处理完成"
              : "后端实时状态",
      retryable: job.retryable,
      isCurrent: job.is_current,
      attempt: job.attempt,
      errorCode: job.error_code,
      errorMessage: job.error_message,
      retryOf: job.retry_of,
    };
  });
  const currentItems = items.filter((task) => task.isCurrent);
  const filtered =
    scope === "history"
      ? items.filter((task) => !task.isCurrent)
      : currentItems.filter((task) => scope === "all" || task.status === scope);
  const selected = items.find((task) => task.id === selectedId) ?? null;

  const retry = async (id: string) => {
    try {
      const job = await workspace.retryJob(id);
      setActionNotice(`重试任务 ${job.id.slice(0, 8)} 已进入单并发队列`);
      setSelectedId(job.id);
    } catch (error) {
      setActionNotice(`无法重试：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const remove = async (id: string) => {
    if (!window.confirm("确定删除该任务吗？此操作不可撤销。")) return;
    try {
      await workspace.deleteJob(id);
      setActionNotice("任务已删除");
      setSelectedId((current) => (current === id ? null : current));
    } catch (error) {
      setActionNotice(`删除失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const counts = {
    running: currentItems.filter((item) => item.status === "running").length,
    queued: currentItems.filter((item) => item.status === "queued").length,
    done: currentItems.filter((item) => item.status === "done").length,
    failed: currentItems.filter((item) => item.status === "failed" || item.status === "partial").length,
    history: items.filter((item) => !item.isCurrent).length,
  };

  return (
    <section className="workspace-page tasks-page">
      <PageHeader
        eyebrow="TASK CENTER"
        title="任务中心"
        description="查看文档编译、Wiki 生成与图谱构建的实时进度和运行日志。"
        actions={<button className="button secondary" onClick={() => void workspace.refresh()}><Icon name="refresh" size={17} />刷新状态</button>}
      />
      <div className="task-summary">
        <article className="workspace-card"><span className="task-summary-icon running"><Icon name="play" /></span><div><strong>{counts.running}</strong><small>正在运行</small></div><i>实时</i></article>
        <article className="workspace-card"><span className="task-summary-icon queued"><Icon name="clock" /></span><div><strong>{counts.queued}</strong><small>等待执行</small></div><i>队列任务</i></article>
        <article className="workspace-card"><span className="task-summary-icon done"><Icon name="check" /></span><div><strong>{counts.done}</strong><small>已完成</small></div><i>处理成功</i></article>
        <article className="workspace-card"><span className="task-summary-icon failed"><Icon name="info" /></span><div><strong>{counts.failed}</strong><small>当前需处理</small></div><i>{counts.history} 条历史尝试</i></article>
      </div>

      {actionNotice && (
        <div className="inline-notice">
          <Icon name="info" size={17} />
          <span>{actionNotice}</span>
          <button onClick={() => setActionNotice("")} aria-label="关闭提示">×</button>
        </div>
      )}

      <article className="workspace-card task-list-card">
        <div className="card-toolbar">
          <div><h2>运行记录</h2><span>默认只显示每份文档的最新有效任务</span></div>
          <CustomSelect compact value={scope} onChange={setScope} options={[
            { value: "all", label: "当前任务" },
            { value: "running", label: "运行中" },
            { value: "queued", label: "排队中" },
            { value: "done", label: "已完成" },
            { value: "partial", label: "部分完成" },
            { value: "failed", label: "失败" },
            { value: "history", label: `历史尝试（${counts.history}）` },
          ]} />
        </div>
        <div className="task-list">
          {!filtered.length && (
            <div className="task-empty"><Icon name="check" size={18} />当前筛选下没有任务</div>
          )}
          {filtered.map((task) => (
            <div className={`task-row ${task.isCurrent ? "" : "is-history"} ${selectedId === task.id ? "is-selected" : ""}`} key={task.id}>
              <span className={`task-kind-icon ${task.status}`}><Icon name={task.kind === "文档编译" ? "file" : task.kind === "Wiki 生成" ? "book" : "network"} size={19} /></span>
              <div className="task-copy">
                <span><strong>{task.name}</strong><em>{task.kind}</em>{!task.isCurrent && <em className="history">历史</em>}</span>
                <small>{task.id.slice(0, 8)} · 第 {task.attempt} 次 · {task.stage}</small>
              </div>
              <div className="task-progress-cell">
                <span><strong>{task.progress}%</strong><small>{task.time}</small></span>
                <i><em style={{ width: `${task.progress}%` }} /></i>
              </div>
              <span className={`status-pill ${task.status}`}>{task.status === "running" && <i />}{taskStatus[task.status as keyof typeof taskStatus]}</span>
              <small className="task-start">{task.started}</small>
              <div className="row-actions">
                {(task.status === "failed" || task.status === "partial") && task.isCurrent && task.retryable && <button title="重试" onClick={() => retry(task.id)}><Icon name="refresh" size={16} /></button>}
                {(task.status === "failed" || task.status === "partial") && <button title="删除任务" onClick={() => remove(task.id)}><Icon name="trash" size={16} /></button>}
                <button title="查看详情" onClick={() => setSelectedId((current) => current === task.id ? null : task.id)}><Icon name="arrowRight" size={16} /></button>
              </div>
            </div>
          ))}
        </div>
      </article>

      {selected && (
        <article className="workspace-card task-detail-card">
          <div className="task-detail-heading">
            <div><span>任务详情</span><strong>{selected.name}</strong></div>
            <button onClick={() => setSelectedId(null)} aria-label="关闭详情">×</button>
          </div>
          <div className="task-detail-grid">
            <span><small>任务编号</small><strong>{selected.id}</strong></span>
            <span><small>执行阶段</small><strong>{selected.rawStage}</strong></span>
            <span><small>尝试次数</small><strong>第 {selected.attempt} 次</strong></span>
            <span><small>更新时间</small><strong>{selected.updated}</strong></span>
            <span><small>错误代码</small><strong>{selected.errorCode ?? "—"}</strong></span>
            <span><small>处理建议</small><strong>{selected.errorMessage ?? selected.time}</strong></span>
          </div>
        </article>
      )}

      <article className="workspace-card execution-log">
        <div><span className="live-dot" /><strong>任务状态流</strong><small>{selected ? `任务 ${selected.id.slice(0, 8)}` : "最近更新"}</small></div>
        <pre>
          {(selectedId ? workspace.jobs.filter((job) => job.id === selectedId) : workspace.jobs.filter((job) => job.is_current)).slice(0, 6).map((job) =>
            `${new Date(job.updated_at).toLocaleTimeString("zh-CN")}  ${job.kind} · ${stageLabel(job.stage)} · ${job.progress}%`,
          ).join("\n") || "当前没有运行任务"}
        </pre>
      </article>
    </section>
  );
}
