import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons";
import type { Session } from "../types";

interface SessionHeaderProps {
  sessions: Session[];
  activeIndex: number;
  onChange: (index: number) => void;
  onNewChat: () => void;
  onRename: () => void;
  onArchive: () => void;
}

export function SessionHeader({
  sessions,
  activeIndex,
  onChange,
  onNewChat,
  onRename,
  onArchive,
}: SessionHeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const switcherRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (
        switcherRef.current &&
        !switcherRef.current.contains(event.target as Node)
      ) {
        setMenuOpen(false);
      }
    };

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };

    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, []);

  return (
    <header className="session-header">
      <div className="session-switcher" ref={switcherRef}>
        <button
          className={`session-switcher-button ${menuOpen ? "is-open" : ""}`}
          onClick={() => setMenuOpen((value) => !value)}
          aria-expanded={menuOpen}
          aria-haspopup="listbox"
        >
          <span className="session-icon">
            <Icon name="message" size={20} />
          </span>
          <span className="session-current">
            <strong>{sessions[activeIndex]?.title ?? "新建研究会话"}</strong>
            <small>切换研究会话</small>
          </span>
          <Icon name="chevronDown" size={17} />
        </button>

        {menuOpen && (
          <div className="session-menu" role="listbox" aria-label="研究会话">
            <div className="session-menu-head">
              <span>最近会话</span>
              <small>{sessions.length} 个</small>
            </div>
            <div className="session-menu-list">
              {sessions.map((session, index) => (
                <button
                  role="option"
                  aria-selected={index === activeIndex}
                  className={`session-option ${
                    index === activeIndex ? "is-active" : ""
                  }`}
                  onClick={() => {
                    onChange(index);
                    setMenuOpen(false);
                  }}
                  key={session.id}
                >
                  <span className="session-option-icon">
                    <Icon name="message" size={16} />
                  </span>
                  <span className="session-option-copy">
                    <strong>{session.title}</strong>
                    <small>{session.question || "尚未开始提问"}</small>
                  </span>
                  {index === activeIndex && <span className="session-check">✓</span>}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="session-actions">
        <div className="session-manage-actions" aria-label="管理当前会话">
          <button onClick={onRename} aria-label="重命名当前会话" disabled={!sessions.length}>
            <Icon name="message" size={15} />
          </button>
          <button onClick={onArchive} aria-label="归档当前会话" disabled={!sessions.length}>
            <Icon name="trash" size={15} />
          </button>
        </div>
        <button className="new-chat-button" onClick={onNewChat}>
          <Icon name="plus" size={18} />
          新建会话
        </button>
      </div>
    </header>
  );
}
