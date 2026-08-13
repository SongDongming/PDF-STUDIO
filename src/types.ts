export type EvidenceKind = "figure" | "table" | "excerpt" | "diagram";

export interface PdfDocument {
  id: string;
  filename: string;
  pages: number;
}

export interface Session {
  id: string;
  title: string;
  question: string;
}

export type ChatContentBlock =
  | {
      type: "text";
      markdown: string;
    }
  | {
      type: "image" | "table" | "formula";
      assetId: string;
      caption: string | null;
      alt: string;
    };

export interface ChatTurn {
  id: string;
  question: string;
  answer: string;
  createdAt: string;
  rich: boolean;
  pending?: boolean;
  startedAt?: number;
  statusText?: string;
  blocks?: ChatContentBlock[];
  assets?: Array<{
    type: "image" | "table" | "formula";
    assetId: string;
    caption: string | null;
    alt: string;
  }>;
}

export interface SourceRef {
  id: string;
  evidenceId: string;
  page: number;
  filename: string;
  tone: "blue" | "green" | "violet";
}

export interface Evidence {
  id: string;
  order: number;
  kind: EvidenceKind;
  kindLabel: string;
  relevance: number;
  page: number;
  filename: string;
  summary: string;
  documentId?: string;
  elementId?: string | null;
  accent: "blue" | "green" | "violet" | "orange" | "rose";
}

export interface PdfViewerTarget {
  documentId: string;
  filename: string;
  page: number;
  totalPages: number;
  elementId?: string | null;
  relevance?: number;
}
