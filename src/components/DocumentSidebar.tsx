import { useRef, useState, type ChangeEvent } from "react";
import { Icon } from "./Icons";
import type { PdfDocument } from "../types";

interface DocumentSidebarProps {
  documents: PdfDocument[];
  activeId: string;
  onSelect: (id: string) => void;
  onOpen: (document: PdfDocument) => void;
  onUpload: (files: File[]) => Promise<void>;
  collapsed: boolean;
  onToggle: () => void;
}

function PdfMark() {
  return (
    <span className="pdf-mark" aria-hidden="true">
      <span className="pdf-fold" />
      <strong>PDF</strong>
    </span>
  );
}

export function DocumentSidebar({
  documents,
  activeId,
  onSelect,
  onOpen,
  onUpload,
  collapsed,
  onToggle,
}: DocumentSidebarProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadNotice, setUploadNotice] = useState("");
  const [uploading, setUploading] = useState(false);

  const handleFiles = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;
    setUploading(true);
    setUploadNotice(`正在上传 ${files.length} 份 PDF…`);
    try {
      await onUpload(files);
      setUploadNotice(`已上传 ${files.length} 份 PDF，正在自动编译入库`);
    } catch (error) {
      setUploadNotice(
        `上传失败：${error instanceof Error ? error.message : "未知错误"}`,
      );
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  if (collapsed) {
    return (
      <aside className="panel document-panel is-collapsed" aria-label="已收起的 PDF 文档列表">
        <button className="collapsed-expand" onClick={onToggle} aria-label="展开文档列表">
          <Icon name="arrowRight" size={18} />
        </button>
        <span className="collapsed-pdf-icon">
          <Icon name="file" size={22} />
        </span>
        <strong className="collapsed-count">{documents.length}</strong>
        <span className="collapsed-label">文档</span>
      </aside>
    );
  }

  return (
    <aside className="panel document-panel" aria-label="PDF 文档列表">
      <header className="document-header">
        <h1>
          <span>{documents.length}</span> 份文档
        </h1>
        <div className="header-actions">
          <button
            className="icon-button"
            aria-label="上传 PDF 并构建知识库"
            title="上传 PDF 并自动构建知识库"
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
          >
            <Icon name="upload" size={21} />
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,application/pdf"
            multiple
            hidden
            onChange={handleFiles}
          />
          <button className="icon-button" onClick={onToggle} aria-label="收起文档列表">
            <Icon name="arrowLeft" size={20} />
          </button>
          <button className="icon-button" aria-label="更多操作">
            <Icon name="more" size={20} />
          </button>
        </div>
      </header>

      <div className="document-list">
        {uploadNotice && (
          <div className={`document-upload-notice ${uploading ? "is-running" : ""}`}>
            <span><Icon name={uploading ? "refresh" : "check"} size={14} /></span>
            <small>{uploadNotice}</small>
            {!uploading && (
              <button onClick={() => setUploadNotice("")} aria-label="关闭上传提示">×</button>
            )}
          </div>
        )}
        {documents.map((document) => (
          <button
            className={`document-row ${
              activeId === document.id ? "is-active" : ""
            }`}
            key={document.id}
            onClick={() => {
              onSelect(document.id);
              onOpen(document);
            }}
          >
            <PdfMark />
            <span className="document-meta">
              <strong title={document.filename}>{document.filename}</strong>
              <small>{document.pages} 页</small>
            </span>
            <Icon name="more" size={18} />
          </button>
        ))}
      </div>
    </aside>
  );
}
