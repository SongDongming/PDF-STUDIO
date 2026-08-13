import { Icon, type IconName } from "./Icons";

export type ProductRoute =
  | "chat"
  | "architecture"
  | "knowledge"
  | "wiki"
  | "graph"
  | "tasks"
  | "settings";

const entries: Array<{
  id: ProductRoute;
  label: string;
  hint: string;
  icon: IconName;
}> = [
  { id: "chat", label: "智能问答", hint: "多模态检索", icon: "message" },
  { id: "architecture", label: "功能架构", hint: "后端数据流", icon: "brain" },
  { id: "knowledge", label: "知识库", hint: "文档与编译", icon: "database" },
  { id: "wiki", label: "LLM Wiki", hint: "结构化知识", icon: "book" },
  { id: "graph", label: "知识图谱", hint: "实体与关系", icon: "network" },
  { id: "tasks", label: "任务中心", hint: "流水线状态", icon: "tasks" },
  { id: "settings", label: "系统设置", hint: "模型与检索", icon: "settings" },
];

interface ProductNavProps {
  active: ProductRoute;
  onChange: (route: ProductRoute) => void;
  compact: boolean;
  onToggle: () => void;
  serviceMode?: "connecting" | "live" | "offline";
}

export function ProductNav({
  active,
  onChange,
  compact,
  onToggle,
  serviceMode = "offline",
}: ProductNavProps) {
  return (
    <aside className={`product-nav ${compact ? "is-compact" : ""}`}>
      <nav aria-label="产品导航">
        {entries.map((entry) => (
          <button
            key={entry.id}
            type="button"
            title={compact ? entry.label : undefined}
            className={active === entry.id ? "is-active" : ""}
            onClick={() => onChange(entry.id)}
          >
            <span className="product-nav-icon">
              <Icon name={entry.icon} size={19} />
            </span>
            {!compact && (
              <span className="product-nav-copy">
                <strong>{entry.label}</strong>
                <small>{entry.hint}</small>
              </span>
            )}
            {active === entry.id && <i />}
          </button>
        ))}
      </nav>

      <div className="product-nav-bottom">
        {!compact && (
          <div className="service-health">
            <span />
            <div>
              <strong>{serviceMode === "live" ? "真实服务已连接" : serviceMode === "connecting" ? "正在检查服务" : "后端离线"}</strong>
              <small>{serviceMode === "live" ? "后端健康检查通过" : "恢复后会自动重连"}</small>
            </div>
          </div>
        )}
        <button
          className="nav-collapse"
          type="button"
          onClick={onToggle}
          aria-label={compact ? "展开主导航" : "收起主导航"}
        >
          <Icon name={compact ? "arrowRight" : "arrowLeft"} size={18} />
          {!compact && <span>收起导航</span>}
        </button>
      </div>
    </aside>
  );
}
