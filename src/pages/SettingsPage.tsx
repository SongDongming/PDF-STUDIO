import { useEffect, useState } from "react";
import type { ProviderStatus, RagSettings as RagSettingsValue } from "../api/types";
import { CustomSelect } from "../components/CustomSelect";
import { Icon, type IconName } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useWorkspaceApi } from "../hooks/useWorkspaceApi";

type SettingsTab = "models" | "ingestion" | "rag" | "security";

const tabs: Array<{ id: SettingsTab; label: string; copy: string; icon: IconName }> = [
  { id: "models", label: "模型服务", copy: "API 与模型分工", icon: "brain" },
  { id: "ingestion", label: "文档建库", copy: "OCR 与分块策略", icon: "database" },
  { id: "rag", label: "Agentic RAG", copy: "召回与推理边界", icon: "sliders" },
  { id: "security", label: "安全与连接", copy: "密钥和服务状态", icon: "shield" },
];

function SecretField({
  placeholder,
  connected = true,
  providerId,
  onSave,
}: {
  placeholder: string;
  connected?: boolean;
  providerId: string;
  onSave: (providerId: string, value: string) => Promise<void>;
}) {
  const [visible, setVisible] = useState(false);
  const [value, setValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveFailed, setSaveFailed] = useState(false);
  const save = async () => {
    if (!value.trim() || saving) return;
    setSaving(true);
    try {
      await onSave(providerId, value.trim());
      setValue("");
      setSaved(true);
      setSaveFailed(false);
    } catch {
      setSaved(false);
      setSaveFailed(true);
    } finally {
      setSaving(false);
    }
  };
  return (
    <div className="secret-field">
      <Icon name="key" size={16} />
      <input
        type={visible ? "text" : "password"}
        value={value}
        onChange={(event) => {
          setValue(event.target.value);
          setSaved(false);
        }}
        placeholder={connected ? "已配置 · 输入新密钥覆盖" : placeholder}
        autoComplete="new-password"
      />
      <button type="button" onClick={() => setVisible((value) => !value)}>{visible ? "隐藏" : "显示"}</button>
      <button type="button" disabled={!value.trim() || saving} onClick={() => void save()}>
        {saving ? "保存中" : "保存"}
      </button>
      <span className={connected || saved ? "is-connected" : saveFailed ? "is-failed" : ""}><i />{saveFailed ? "保存失败" : saved ? "已更新" : connected ? "已连接" : "未验证"}</span>
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  return <button type="button" role="switch" aria-checked={checked} className={`toggle ${checked ? "is-on" : ""}`} onClick={() => onChange(!checked)}><i /></button>;
}

function ModelsSettings({
  providers,
  onCredentialSave,
}: {
  providers: ProviderStatus[];
  onCredentialSave: (providerId: string, value: string) => Promise<void>;
}) {
  const [answerModel, setAnswerModel] = useState("deepseek-v4-flash");
  const [embedModel, setEmbedModel] = useState("glm-embedding");
  const configured = (id: string) =>
    providers.find((provider) => provider.id === id)?.configured ?? false;
  return (
    <>
      <div className="settings-section-heading"><div><h2>模型服务编排</h2><p>密钥仅保存在服务端受保护存储，浏览器不会获取凭证原文。</p></div><span className="safe-badge"><Icon name="shield" size={15} />安全存储</span></div>
      <div className="model-card-grid">
        <article className="model-config-card featured">
          <div className="model-card-title"><span className="model-logo deepseek">D</span><div><strong>问答推理模型</strong><small>检索证据并生成带引用的最终回答</small></div><em>核心</em></div>
          <label>模型</label>
          <CustomSelect value={answerModel} onChange={setAnswerModel} options={[
            { value: "deepseek-v4-flash", label: "DeepSeek V4 Flash", hint: "文本推理 · 混合检索问答" },
            { value: "deepseek-v4-pro", label: "DeepSeek V4 Pro", hint: "更强推理备用模型" },
          ]} />
          <label>DeepSeek API Key</label>
          <SecretField
            placeholder="输入 DeepSeek API Key"
            connected={configured("vision-chat")}
            providerId="vision-chat"
            onSave={onCredentialSave}
          />
          <div className="model-meta"><span>API 地址 <strong>api.deepseek.com</strong></span><span>推理 <strong>兼容 json_object</strong></span></div>
        </article>
        <article className="model-config-card">
          <div className="model-card-title"><span className="model-logo paddle">P</span><div><strong>OCR 与版面模型</strong><small>页面布局、文字、图表和公式坐标</small></div><em className="green">在线</em></div>
          <label>解析管线</label>
          <CustomSelect value="paddle-vl" onChange={() => undefined} options={[
            { value: "paddle-vl", label: "PaddleOCR-VL 1.6", hint: "布局检测 + 元素识别 + Markdown" },
            { value: "pp-ocr", label: "PP-OCRv6", hint: "纯文字高速识别" },
          ]} />
          <label>服务地址</label>
          <div className="plain-input"><span>OCR</span><input readOnly value="Dspark SSH 隧道 · 127.0.0.1:18111" /><i>受保护</i></div>
          <div className="model-meta"><span>处理方式 <strong>逐页布局识别</strong></span><span>运行设备 <strong>GB10</strong></span></div>
        </article>
        <article className="model-config-card">
          <div className="model-card-title"><span className="model-logo embedding">E</span><div><strong>Embedding 模型</strong><small>文本片段向量化与混合检索</small></div><em className="green">可用</em></div>
          <label>当前模型</label>
          <CustomSelect value={embedModel} onChange={setEmbedModel} options={[
            { value: "glm-embedding", label: "智谱 embedding-3", hint: "1024 维 · 当前主路" },
            { value: "openai-large", label: "OpenAI text-embedding-3-large", hint: "3072 维 · 备用" },
          ]} />
          <label>OpenAI 兼容 API Key</label>
          <SecretField
            placeholder="输入 OpenAI 兼容 API Key"
            connected={configured("embedding-openai")}
            providerId="embedding-openai"
            onSave={onCredentialSave}
          />
          <div className="model-meta"><span>向量维度 <strong>1024</strong></span><span>批处理 <strong>64 条/次</strong></span></div>
        </article>
      </div>
    </>
  );
}

function IngestionSettings() {
  const [table, setTable] = useState(true);
  const [formula, setFormula] = useState(true);
  const [imageCaption, setImageCaption] = useState(true);
  const [chunk, setChunk] = useState("semantic");
  return (
    <>
      <div className="settings-section-heading"><div><h2>文档编译策略</h2><p>定义 PDF 如何被解析、增强、切分并写入知识库。</p></div></div>
      <div className="settings-form-card">
        <h3>多模态元素</h3>
        {[
          ["表格结构恢复", "保留单元格、表头和跨页表格关系", table, setTable],
          ["公式识别", "同时保存 LaTeX、原始截图和页面坐标", formula, setFormula],
          ["图片语义描述", "由 LLM 为图表与插图生成可检索描述", imageCaption, setImageCaption],
        ].map(([title, copy, value, setter]) => (
          <div className="setting-toggle-row" key={String(title)}>
            <div><strong>{String(title)}</strong><small>{String(copy)}</small></div>
            <Toggle checked={Boolean(value)} onChange={setter as (value: boolean) => void} />
          </div>
        ))}
        <h3>分块与索引</h3>
        <div className="settings-field-grid">
          <label><span>分块策略</span><CustomSelect value={chunk} onChange={setChunk} options={[
            { value: "semantic", label: "结构感知语义分块", hint: "推荐：保持标题与多模态元素原子性" },
            { value: "fixed", label: "固定 Token 分块", hint: "适用于结构简单的纯文本文档" },
          ]} /></label>
          <label><span>目标片段长度</span><div className="number-input"><input type="number" defaultValue="900" /><em>tokens</em></div></label>
          <label><span>片段重叠</span><div className="number-input"><input type="number" defaultValue="120" /><em>tokens</em></div></label>
          <label><span>页面渲染精度</span><CustomSelect value="180" onChange={() => undefined} options={[
            { value: "144", label: "144 DPI · 快速" }, { value: "180", label: "180 DPI · 平衡" }, { value: "220", label: "220 DPI · 高质量" },
          ]} /></label>
        </div>
      </div>
    </>
  );
}

function RagSettings({
  value,
  onChange,
}: {
  value: RagSettingsValue | null;
  onChange: (value: RagSettingsValue) => void;
}) {
  const [hybrid, setHybrid] = useState(true);
  const [graph, setGraph] = useState(true);
  const [rerank, setRerank] = useState(true);
  const topK = value?.dense_top_k ?? 12;
  const hops = value?.max_tool_calls ?? 4;
  const [score, setScore] = useState(72);
  return (
    <>
      <div className="settings-section-heading"><div><h2>Agentic RAG 参数</h2><p>控制检索 Agent 可使用的工具、循环边界和证据门槛。</p></div><span className="agent-badge"><Icon name="brain" size={15} />Deep Agents</span></div>
      <div className="rag-settings-grid">
        <article className="settings-form-card">
          <h3>召回策略</h3>
          {[
            ["混合检索", "向量相似度 + BM25 关键词召回", hybrid, setHybrid],
            ["图谱扩展", "沿实体关系补充跨文档证据", graph, setGraph],
            ["候选重排", "对召回片段执行语义相关性重排", rerank, setRerank],
          ].map(([title, copy, value, setter]) => (
            <div className="setting-toggle-row" key={String(title)}><div><strong>{String(title)}</strong><small>{String(copy)}</small></div><Toggle checked={Boolean(value)} onChange={setter as (v: boolean) => void} /></div>
          ))}
        </article>
        <article className="settings-form-card">
          <h3>Agent 执行边界</h3>
          <label className="range-field"><span><strong>初始召回数量</strong><em>{topK} 条</em></span><input type="range" min="4" max="24" value={topK} onChange={(e) => value && onChange({ ...value, dense_top_k: Number(e.target.value), lexical_top_k: Number(e.target.value) })} /></label>
          <label className="range-field"><span><strong>最大工具轮次</strong><em>{hops} 轮</em></span><input type="range" min="1" max="8" value={hops} onChange={(e) => value && onChange({ ...value, max_tool_calls: Number(e.target.value) })} /></label>
          <label className="range-field"><span><strong>最低证据充分度</strong><em>{score}%</em></span><input type="range" min="40" max="95" value={score} onChange={(e) => setScore(Number(e.target.value))} /></label>
        </article>
      </div>
      <article className="settings-form-card tools-card">
        <h3>Agent 可用工具</h3>
        <div className="tool-chip-grid">
          {["hybrid_search", "graph_search", "fetch_chunk", "fetch_asset", "open_pdf_region", "search_wiki", "citation_check"].map((tool) => <span key={tool}><i /><code>{tool}</code><Icon name="check" size={13} /></span>)}
        </div>
      </article>
    </>
  );
}

function SecuritySettings({
  providers,
  onTest,
}: {
  providers: ProviderStatus[];
  onTest: (providerId: string) => Promise<void>;
}) {
  const items = providers;
  return (
    <>
      <div className="settings-section-heading"><div><h2>安全与服务连接</h2><p>所有连接均由后端代理，模型端口不会直接暴露给浏览器。</p></div></div>
      <div className="security-grid">
        {items.map((provider) => (
          <article className="service-card" key={provider.id}><span className={`service-state ${provider.health === "healthy" ? "connected" : "warning"}`}><i /></span><div><strong>{provider.provider}</strong><small>{provider.model ?? "未指定模型"}</small></div><em>{provider.health === "healthy" ? "可用" : provider.health}</em><button onClick={() => void onTest(provider.id)}><Icon name="refresh" size={15} />测试</button></article>
        ))}
      </div>
      <article className="settings-form-card security-note"><span><Icon name="shield" size={23} /></span><div><strong>凭证保护已启用</strong><p>API Key 使用服务端密钥引用，前端仅显示连接状态。日志会自动移除请求头和敏感字段。</p></div><button className="button ghost">查看安全策略</button></article>
    </>
  );
}

export function SettingsPage() {
  const workspace = useWorkspaceApi();
  const [tab, setTab] = useState<SettingsTab>("models");
  const [saved, setSaved] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [rag, setRag] = useState<RagSettingsValue | null>(null);

  useEffect(() => {
    if (workspace.settings) setRag(workspace.settings.rag);
  }, [workspace.settings]);

  const save = async () => {
    if (!rag) {
      setSaveMessage("后端尚未返回当前设置，请稍候再试");
      return;
    }
    try {
      await workspace.saveSettings({ rag });
      setSaved(true);
      setSaveMessage("设置已写入后端");
      window.setTimeout(() => setSaved(false), 1800);
    } catch (error) {
      setSaveMessage(`保存失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  const testProvider = async (providerId: string) => {
    try {
      const result = await workspace.testConnection(providerId);
      setSaveMessage(`${providerId}：${result.detail}`);
    } catch (error) {
      setSaveMessage(`连接测试失败：${error instanceof Error ? error.message : "未知错误"}`);
    }
  };
  const saveCredential = async (providerId: string, value: string) => {
    try {
      const result = await workspace.updateCredential(providerId, value);
      setSaveMessage(result.detail);
      await workspace.refresh();
    } catch (error) {
      setSaveMessage(`凭证保存失败：${error instanceof Error ? error.message : "未知错误"}`);
      throw error;
    }
  };
  return (
    <section className="workspace-page settings-page">
      <PageHeader
        eyebrow="SYSTEM SETTINGS"
        title="系统设置"
        description="配置模型服务、文档建库与 Agentic RAG 的运行策略。"
        actions={<button className="button primary" onClick={() => void save()}><Icon name="check" size={17} />{saved ? "已保存" : "保存设置"}</button>}
      />
      {saveMessage && <div className="inline-notice"><Icon name="info" size={17} /><span>{saveMessage}</span><button onClick={() => setSaveMessage("")}>×</button></div>}
      <div className="settings-shell">
        <aside className="workspace-card settings-nav">
          {tabs.map((item) => <button key={item.id} className={tab === item.id ? "is-active" : ""} onClick={() => setTab(item.id)}><span><Icon name={item.icon} size={18} /></span><div><strong>{item.label}</strong><small>{item.copy}</small></div><Icon name="arrowRight" size={15} /></button>)}
          <div className="settings-version"><span><i />{workspace.mode === "live" ? "真实服务已连接" : workspace.mode === "connecting" ? "正在连接后端" : "后端离线"}</span><small>{workspace.health ? `v${workspace.health.version}` : "等待连接"}</small></div>
        </aside>
        <main className="settings-content workspace-card">
          {tab === "models" && (
            <ModelsSettings
              providers={workspace.settings?.providers ?? []}
              onCredentialSave={saveCredential}
            />
          )}
          {tab === "ingestion" && <IngestionSettings />}
          {tab === "rag" && <RagSettings value={rag} onChange={setRag} />}
          {tab === "security" && <SecuritySettings providers={workspace.settings?.providers ?? []} onTest={testProvider} />}
        </main>
      </div>
    </section>
  );
}
