export interface Paper {
  paper_id: string;
  title: string;
}

export interface Citation {
  title: string;
  page: number;
}

export interface QAResult {
  answer: string;
  citations: Citation[];
}

export interface PaperSummary {
  paper_id: string;
  title: string;
  answer: string;
  citations: Citation[];
}

export interface SummaryResult {
  summaries: PaperSummary[];
}

export type ResearchIntent = 
| "qa" 
| "comparison" 
| "summary"
| "unanswerable";

export interface ResearchResponse {
  intent: ResearchIntent;
  result: QAResult | SummaryResult;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: ResearchIntent;
  citations?: Citation[];
  summaries?: PaperSummary[];
  timestamp: Date;
}

export interface UploadResponse {
  message: string;
  papers: Paper[];
}

export interface PapersResponse {
  papers: Paper[];
}

export function isSummaryResult(
  result: QAResult | SummaryResult
): result is SummaryResult {
  return "summaries" in result;
}
