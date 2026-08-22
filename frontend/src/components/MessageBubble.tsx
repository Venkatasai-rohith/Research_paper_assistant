import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import type { ChatMessage } from "../types";
import { CitationChips } from "./CitationChips";
import { preprocessMarkdown } from "../utils/markdownUtils";

interface MessageBubbleProps {
  message: ChatMessage;
}

const INTENT_LABELS: Record<string, string> = {
  qa: "Q&A",
  comparison: "Comparison",
  summary: "Summary",
};

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`message message--${message.role}`}>
      <div className="message__bubble">
        {isUser ? (
          message.content
        ) : (
          <>
            {message.summaries ? (
              <div className="summary-cards">
                {message.summaries.map((s) => (
                  <div key={s.paper_id} className="summary-card">
                    <div className="summary-card__title">{s.title}</div>
                    <div className="markdown-content">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeRaw, rehypeKatex]}
                      >
                        {preprocessMarkdown(s.answer)}
                      </ReactMarkdown>
                    </div>
                    <CitationChips citations={s.citations} />
                  </div>
                ))}
              </div>
            ) : (
              <>
                <div className="markdown-content">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeRaw, rehypeKatex]}
                  >
                    {preprocessMarkdown(message.content)}
                  </ReactMarkdown>
                </div>
                {message.citations && (
                  <CitationChips citations={message.citations} />
                )}
              </>
            )}
          </>
        )}
      </div>
      {!isUser && message.intent && (
        <div className="message__meta">
          <span className={`intent-badge intent-badge--${message.intent}`}>
            {INTENT_LABELS[message.intent] ?? message.intent}
          </span>
        </div>
      )}
    </div>
  );
}
