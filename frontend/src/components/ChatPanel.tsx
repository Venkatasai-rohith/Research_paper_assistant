import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";

interface ChatPanelProps {
  messages: ChatMessage[];
  loading: boolean;
  selectedCount: number;
  onSend: (question: string) => void;
}

const SUGGESTIONS = [
  "What is the main contribution of this paper?",
  "Summarize the key findings",
  "Compare the methodologies used",
  "What are the limitations mentioned?",
];

export function ChatPanel({
  messages,
  loading,
  selectedCount,
  onSend,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || loading || selectedCount === 0) return;
    onSend(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  }

  const canSend = input.trim().length > 0 && !loading && selectedCount > 0;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <main className="chat-area">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <div className="chat-welcome__icon">🔬</div>
            <h2 className="chat-welcome__title">Research Assistant</h2>
            <p className="chat-welcome__subtitle">
              Upload PDF research papers, select them from the library, and ask
              questions. The assistant automatically detects whether you want a
              Q&amp;A answer, a comparison, or a summary.
            </p>
            {selectedCount > 0 && (
              <div className="suggestion-chips">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    className="suggestion-chip"
                    onClick={() => onSend(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {loading && (
              <div className="message message--assistant">
                <div className="message__bubble">
                  <div className="loading-dots">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-input"
            placeholder={
              selectedCount === 0
                ? "Select at least one paper to ask a question…"
                : "Ask a question about your selected papers…"
            }
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={loading}
          />
          <button
            className="btn btn--primary"
            onClick={handleSend}
            disabled={!canSend}
          >
            Send
          </button>
        </div>
        <div className="chat-input__footer">
          <span className="chat-input__hint">
            {selectedCount === 0 ? (
              "Select papers from the sidebar"
            ) : (
              <>
                <strong>{selectedCount}</strong> paper
                {selectedCount !== 1 ? "s" : ""} in context · Enter to send,
                Shift+Enter for new line
              </>
            )}
          </span>
        </div>
      </div>
    </main>
  );
}
