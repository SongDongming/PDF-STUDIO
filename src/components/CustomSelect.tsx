import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons";

export interface SelectOption {
  value: string;
  label: string;
  hint?: string;
}

interface CustomSelectProps {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  label?: string;
  compact?: boolean;
}

export function CustomSelect({
  value,
  options,
  onChange,
  label,
  compact = false,
}: CustomSelectProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = options.find((option) => option.value === value) ?? options[0];

  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", escape);
    };
  }, []);

  return (
    <div className={`custom-select ${compact ? "is-compact" : ""}`} ref={rootRef}>
      {label && <span className="custom-select-label">{label}</span>}
      <button
        type="button"
        className={`custom-select-trigger ${open ? "is-open" : ""}`}
        onClick={() => setOpen((current) => !current)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span>
          <strong>{selected?.label}</strong>
          {!compact && selected?.hint && <small>{selected.hint}</small>}
        </span>
        <Icon name="chevronDown" size={16} />
      </button>
      {open && (
        <div className="custom-select-menu" role="listbox">
          {options.map((option) => (
            <button
              type="button"
              key={option.value}
              role="option"
              aria-selected={option.value === value}
              className={option.value === value ? "is-selected" : ""}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              <span>
                <strong>{option.label}</strong>
                {option.hint && <small>{option.hint}</small>}
              </span>
              {option.value === value && <Icon name="check" size={16} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
