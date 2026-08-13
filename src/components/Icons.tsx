import type { SVGProps } from "react";

export type IconName =
  | "upload"
  | "more"
  | "chevronDown"
  | "arrowLeft"
  | "arrowRight"
  | "plus"
  | "user"
  | "sparkles"
  | "paperclip"
  | "send"
  | "file"
  | "image"
  | "table"
  | "quote"
  | "message"
  | "info"
  | "database"
  | "book"
  | "network"
  | "tasks"
  | "settings"
  | "search"
  | "check"
  | "clock"
  | "play"
  | "trash"
  | "refresh"
  | "shield"
  | "key"
  | "brain"
  | "sliders"
  | "external"
  | "folder"
  | "close";

interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 18, ...props }: IconProps) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.9,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
    ...props,
  };

  switch (name) {
    case "upload":
      return (
        <svg {...common}>
          <path d="M12 16V3" />
          <path d="m7 8 5-5 5 5" />
          <path d="M5 13v7h14v-7" />
        </svg>
      );
    case "more":
      return (
        <svg {...common}>
          <circle cx="12" cy="5" r="1" fill="currentColor" stroke="none" />
          <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
          <circle cx="12" cy="19" r="1" fill="currentColor" stroke="none" />
        </svg>
      );
    case "chevronDown":
      return (
        <svg {...common}>
          <path d="m7 10 5 5 5-5" />
        </svg>
      );
    case "arrowLeft":
      return (
        <svg {...common}>
          <path d="m15 18-6-6 6-6" />
        </svg>
      );
    case "arrowRight":
      return (
        <svg {...common}>
          <path d="m9 18 6-6-6-6" />
        </svg>
      );
    case "plus":
      return (
        <svg {...common}>
          <path d="M12 5v14M5 12h14" />
        </svg>
      );
    case "user":
      return (
        <svg {...common}>
          <circle cx="12" cy="8" r="3.2" />
          <path d="M5.5 20c.8-4 3-6 6.5-6s5.7 2 6.5 6" />
        </svg>
      );
    case "sparkles":
      return (
        <svg {...common}>
          <path d="m12 3 1.2 3.3L16.5 8l-3.3 1.7L12 13l-1.2-3.3L7.5 8l3.3-1.7L12 3Z" />
          <path d="m18.5 14 .7 1.8L21 16.5l-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7.7-1.8Z" />
          <path d="m6 14 .8 2.2L9 17l-2.2.8L6 20l-.8-2.2L3 17l2.2-.8L6 14Z" />
        </svg>
      );
    case "paperclip":
      return (
        <svg {...common}>
          <path d="m20.5 11.5-8.7 8.7a5 5 0 0 1-7-7l9.4-9.4a3.4 3.4 0 0 1 4.8 4.8l-9.5 9.5a1.8 1.8 0 0 1-2.5-2.5l8.8-8.8" />
        </svg>
      );
    case "send":
      return (
        <svg {...common}>
          <path d="m22 2-7 20-4-9-9-4 20-7Z" />
          <path d="M22 2 11 13" />
        </svg>
      );
    case "file":
      return (
        <svg {...common}>
          <path d="M6 2h8l4 4v16H6z" />
          <path d="M14 2v5h5M9 12h6M9 16h6" />
        </svg>
      );
    case "image":
      return (
        <svg {...common}>
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <circle cx="8.5" cy="9" r="1.4" />
          <path d="m4 17 4.5-4.5 3.2 3.2 2.3-2.3 6 6" />
        </svg>
      );
    case "table":
      return (
        <svg {...common}>
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="M3 10h18M9 4v16M15 4v16" />
        </svg>
      );
    case "quote":
      return (
        <svg {...common}>
          <path d="M8 11H4a5 5 0 0 1 5-5v2a3 3 0 0 0-3 3v7h2zM18 11h-4a5 5 0 0 1 5-5v2a3 3 0 0 0-3 3v7h2z" />
        </svg>
      );
    case "message":
      return (
        <svg {...common}>
          <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" />
        </svg>
      );
    case "info":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5M12 8h.01" />
        </svg>
      );
    case "database":
      return (
        <svg {...common}>
          <ellipse cx="12" cy="5" rx="8" ry="3" />
          <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
          <path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
        </svg>
      );
    case "book":
      return (
        <svg {...common}>
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z" />
          <path d="M4 19.5V5.5M8 7h8M8 11h6" />
        </svg>
      );
    case "network":
      return (
        <svg {...common}>
          <circle cx="12" cy="5" r="2.5" />
          <circle cx="5" cy="18" r="2.5" />
          <circle cx="19" cy="18" r="2.5" />
          <path d="m10.8 7.2-4.6 8.6M13.2 7.2l4.6 8.6M7.5 18h9" />
        </svg>
      );
    case "tasks":
      return (
        <svg {...common}>
          <rect x="4" y="3" width="16" height="18" rx="2" />
          <path d="m8 8 1.3 1.3L12 6.5M14 8h3M8 14l1.3 1.3L12 12.5M14 14h3" />
        </svg>
      );
    case "settings":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.6v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z" />
        </svg>
      );
    case "search":
      return (
        <svg {...common}>
          <circle cx="10.5" cy="10.5" r="6.5" />
          <path d="m15.5 15.5 5 5" />
        </svg>
      );
    case "check":
      return (
        <svg {...common}>
          <path d="m5 12 4.2 4.2L19 6.5" />
        </svg>
      );
    case "clock":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v5l3.5 2" />
        </svg>
      );
    case "play":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="m10 8 6 4-6 4z" />
        </svg>
      );
    case "trash":
      return (
        <svg {...common}>
          <path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6" />
        </svg>
      );
    case "refresh":
      return (
        <svg {...common}>
          <path d="M20 7v5h-5M4 17v-5h5" />
          <path d="M18.5 9A7.5 7.5 0 0 0 6 6.5L4 9M5.5 15A7.5 7.5 0 0 0 18 17.5l2-2.5" />
        </svg>
      );
    case "shield":
      return (
        <svg {...common}>
          <path d="M12 3 4.5 6v5.5c0 4.8 3 8.2 7.5 9.5 4.5-1.3 7.5-4.7 7.5-9.5V6z" />
          <path d="m8.5 12 2.2 2.2 4.8-5" />
        </svg>
      );
    case "key":
      return (
        <svg {...common}>
          <circle cx="8" cy="15" r="4" />
          <path d="m11 12 8-8M16 7l2 2M14 9l2 2" />
        </svg>
      );
    case "brain":
      return (
        <svg {...common}>
          <path d="M9.5 4A3.5 3.5 0 0 0 6 7.5v.7A3.6 3.6 0 0 0 4 15a3.5 3.5 0 0 0 5.5 2.9V4ZM14.5 4A3.5 3.5 0 0 1 18 7.5v.7a3.6 3.6 0 0 1 2 6.8 3.5 3.5 0 0 1-5.5 2.9V4Z" />
          <path d="M6.5 9.5h3M14.5 8h3M14.5 14h3M7 15h2.5" />
        </svg>
      );
    case "sliders":
      return (
        <svg {...common}>
          <path d="M4 6h6M14 6h6M4 12h11M19 12h1M4 18h3M11 18h9" />
          <circle cx="12" cy="6" r="2" />
          <circle cx="17" cy="12" r="2" />
          <circle cx="9" cy="18" r="2" />
        </svg>
      );
    case "external":
      return (
        <svg {...common}>
          <path d="M14 4h6v6M20 4l-9 9" />
          <path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6" />
        </svg>
      );
    case "folder":
      return (
        <svg {...common}>
          <path d="M3 6h7l2 2h9v11H3z" />
        </svg>
      );
    case "close":
      return (
        <svg {...common}>
          <path d="m6 6 12 12M18 6 6 18" />
        </svg>
      );
  }
}
