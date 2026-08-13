import { useEffect, useState } from "react";
import type { WikiPage as ApiWikiPage } from "../api/types";
import { Icon } from "../components/Icons";
import { PageHeader } from "../components/PageHeader";
import { useWorkspaceApi } from "../hooks/useWorkspaceApi";

function cleanWikiText(value: string) {
  return value
    .replace(/\[\[([^|\]]+)\|[^\]]+\]\]/g, "$1")
    .replace(
      /\s*`?\[[0-9a-f-]{36}\s+p\.\d+\s+·\s+ch_[^\]]+\]`?/g,
      "",
    )
    .replace(/\s{2,}/g, " ")
    .trim();
}

function wikiInline(value: string) {
  return cleanWikiText(value)
    .split(/(`[^`]+`)/g)
    .filter(Boolean)
    .map((part, index) =>
      part.startsWith("`") && part.endsWith("`")
        ? <code key={index}>{part.slice(1, -1)}</code>
        : part,
    );
}

function WikiMarkdown({ markdown }: { markdown: string }) {
  return (
    <div className="wiki-live-markdown">
      {markdown.split("\n").map((rawLine, index) => {
        const line = rawLine.trim();
        if (!line || /^#\s+/.test(line)) return null;
        const heading = line.match(/^(#{2,4})\s+(.+)$/);
        if (heading) {
          const Heading = heading[1].length === 2 ? "h3" : "h4";
          return <Heading key={index}>{wikiInline(heading[2])}</Heading>;
        }
        if (/^[-*]\s+/.test(line)) {
          return (
            <div className="wiki-live-bullet" key={index}>
              <i />
              <p>{wikiInline(line.replace(/^[-*]\s+/, ""))}</p>
            </div>
          );
        }
        return <p key={index}>{wikiInline(line)}</p>;
      })}
    </div>
  );
}

export function WikiPage({
  onOpenPdf,
}: {
  onOpenPdf?: (documentId: string, page: number, elementId?: string | null) => void;
}) {
  const workspace = useWorkspaceApi();
  const sections = workspace.wikiPages.map((page) => ({
    id: page.slug,
    title: page.title,
    group: page.status === "published" ? "已发布" : "草稿",
    read: "实时页面",
    links: 0,
  }));
  const [activeId, setActiveId] = useState("");
  const [query, setQuery] = useState("");
  const [apiPage, setApiPage] = useState<ApiWikiPage | null>(null);
  const [wikiLoadFailed, setWikiLoadFailed] = useState(false);
  const active = sections.find((section) => section.id === activeId) ?? sections[0];
  const visible = sections.filter((item) => item.title.includes(query) || item.group.includes(query));

  useEffect(() => {
    if (!active || !workspace.wikiPages.length) {
      setApiPage(null);
      return;
    }
    let cancelled = false;
    setWikiLoadFailed(false);
    void workspace
      .loadWikiPage(active.id)
      .then((page) => {
        if (!cancelled) {
          setApiPage(page);
          setWikiLoadFailed(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setApiPage(null);
          setWikiLoadFailed(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [active?.id, workspace.wikiPages.length]);

  useEffect(() => {
    if (
      workspace.wikiPages.length &&
      !workspace.wikiPages.some((page) => page.slug === activeId)
    ) {
      const preferred =
        workspace.wikiPages.find((page) => page.title === "Agentic RAG") ??
        workspace.wikiPages[0];
      setActiveId(preferred.slug);
    }
  }, [activeId, workspace.wikiPages]);

  return (
    <section className="workspace-page wiki-page">
      <PageHeader
        eyebrow="LLM WIKI"
        title="知识 Wiki"
        description="由全库文档与知识图谱持续编译的可追溯技术百科（上传或删除文档后自动更新）。"
      />

      {!sections.length ? (
        <div className="workspace-card wiki-empty">
          <span><Icon name="book" size={24} /></span>
          <h3>暂无 Wiki 页面</h3>
          <p>当前知识库还没有生成 Wiki。上传 PDF 并完成编译后，LLM Wiki 会自动编译。</p>
        </div>
      ) : (
      <div className="wiki-shell">
        <aside className="workspace-card wiki-index">
          <label className="search-field wiki-search">
            <Icon name="search" size={17} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索 Wiki" />
          </label>
          <div className="wiki-index-summary">
            <span><strong>{workspace.wikiPages.length}</strong><small>知识页面</small></span>
            <span><strong>{workspace.graph?.total_node_count ?? 0}</strong><small>图谱节点</small></span>
          </div>
          <nav>
            {visible.map((section) => (
              <button
                key={section.id}
                className={activeId === section.id ? "is-active" : ""}
                onClick={() => setActiveId(section.id)}
              >
                <span className="wiki-item-icon"><Icon name="book" size={16} /></span>
                <span><small>{section.group}</small><strong>{section.title}</strong></span>
                <Icon name="arrowRight" size={15} />
              </button>
            ))}
          </nav>
        </aside>

        <article className="workspace-card wiki-article">
          <div className="wiki-breadcrumb">
            <span>知识库</span><i>/</i><span>{active.group}</span><i>/</i><strong>{active.title}</strong>
          </div>
          <div className="wiki-article-title">
            <div>
              <span className="wiki-generated">
                <Icon name="sparkles" size={14} />
                LLM 编译 · 证据可追溯
              </span>
              <h1>{active.title}</h1>
              <p>真实 Wiki · {apiPage?.citations.length ?? 0} 条引用</p>
            </div>
            <button title="在问答中打开"><Icon name="message" size={18} /></button>
          </div>

          <div className="wiki-toc">
            <strong>本页导览</strong>
            <a href="#definition">核心定义</a>
            <a href="#mechanism">作用机制</a>
            <a href="#factors">影响因素</a>
            <a href="#sources">参考来源</a>
          </div>

          <section id="definition">
            <h2>核心定义</h2>
            {wikiLoadFailed ? (
              <div className="wiki-load-error">
                <span><Icon name="info" size={18} /></span>
                <p>Wiki 页面加载失败。可切换其他页面或稍后重试。</p>
              </div>
            ) : apiPage ? (
              <WikiMarkdown markdown={apiPage.markdown} />
            ) : (
              <p>正在加载真实 Wiki 页面…</p>
            )}
          </section>

          <section id="sources" className="wiki-sources">
            <h2>参考来源</h2>
            {(apiPage?.citations.map((citation) => ({
              name: citation.document_title,
              pages: `第 ${citation.page} 页`,
              score: citation.score == null ? "—" : `${Math.round(citation.score * 100)}%`,
              citation,
            })) ?? []).map((item, index) => (
              <button
                key={`${item.name}-${item.pages}-${index}`}
                onClick={() =>
                  item.citation &&
                  onOpenPdf?.(
                    item.citation.document_id,
                    item.citation.page,
                    item.citation.element_id,
                  )
                }
              >
                <span>{index + 1}</span><div><strong>{item.name}</strong><small>{item.pages}</small></div><em>{item.score}</em><Icon name="external" size={15} />
              </button>
            ))}
          </section>
        </article>

        <aside className="workspace-card wiki-context">
          <h3>关联知识</h3>
          <div className="context-graph-mini">
            <span className="node central">{active.title.slice(0, 8)}</span>
            {(apiPage?.related_page_ids.slice(0, 4).map((id) =>
              workspace.wikiPages.find((page) => page.id === id)?.title ?? id
            ) ?? []).map((title, index) => (
              <span className={`node n${index + 1}`} key={title}>{title.slice(0, 7)}</span>
            ))}
            <i className="edge e1" /><i className="edge e2" /><i className="edge e3" /><i className="edge e4" />
          </div>
          <h3>实体摘要</h3>
          <dl>
            <div><dt>页面类型</dt><dd>知识图谱实体</dd></div>
            <div><dt>关联页面</dt><dd>{apiPage?.related_page_ids.length ?? 0} 条</dd></div>
            <div><dt>引用证据</dt><dd>{apiPage?.citations.length ?? 0} 条</dd></div>
            <div><dt>图谱版本</dt><dd>v{workspace.graph?.graph_version ?? 0}</dd></div>
          </dl>
          <button className="button ghost" onClick={() => { window.location.hash = "/graph"; }}>
            <Icon name="network" size={16} />在图谱中查看
          </button>
        </aside>
      </div>
      )}
    </section>
  );
}
