import { useCallback, useEffect, useState } from "react";
import {
  checkHealth,
  deletePaper,
  fetchPapers,
  streamQuestion,
  uploadPapers,
} from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { PaperLibrary } from "./components/PaperLibrary";
import type { ChatMessage, Paper, ResearchIntent } from "./types";

function generateId() {
  return crypto.randomUUID();
}

export default function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  const loadPapers = useCallback(async () => {
    try {
      const data = await fetchPapers();
      setPapers(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load papers");
    }
  }, []);

  useEffect(() => {
    loadPapers();
    checkHealth().then(setApiOnline);
    const interval = setInterval(() => checkHealth().then(setApiOnline), 15000);
    return () => clearInterval(interval);
  }, [loadPapers]);

  function togglePaper(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAll() {
    setSelectedIds(new Set(papers.map((p) => p.paper_id)));
  }

  function clearSelection() {
    setSelectedIds(new Set());
  }

  async function handleUpload(files: File[]) {
    setUploading(true);
    setError(null);
    try {
      const uploaded = await uploadPapers(files);
      setPapers((prev) => [...prev, ...uploaded]);
      setSelectedIds((prev) => {
        const next = new Set(prev);
        uploaded.forEach((p) => next.add(p.paper_id));
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(id: string) {
    setError(null);
    try {
      await deletePaper(id);
      setPapers((prev) => prev.filter((p) => p.paper_id !== id));
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  }

  async function handleSend(question: string) {
    if (selectedIds.size === 0) return;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: question,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setError(null);

    try {
    const assistantId = generateId();

    // Create an empty assistant message immediately
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
      },
    ]);

    await streamQuestion(
      [...selectedIds],
      question,
      {
        onStart: (intent) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    intent: intent as ResearchIntent,
                  }
                : msg
            )
          );
        },

        onToken: (text) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content: msg.content + text,
                  }
                : msg
            )
          );
        },

        onCitations: (citations) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    citations,
                  }
                : msg
            )
          );
        },

        onError: (message) => {
          setError(message);
        },

        onEnd: () => {
          setLoading(false);
        },
      }
    );
  }
  catch (e) {
    setError(
      e instanceof Error ? e.message : "Request failed"
    );
  }
  finally {
    setLoading(false);
  }
  }
  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__brand">
          <div className="app-header__logo">R</div>
          <span className="app-header__title">Research Assistant</span>
        </div>
        <div className="app-header__status">
          <span
            className={`status-dot ${apiOnline ? "status-dot--online" : "status-dot--offline"}`}
          />
          {apiOnline === null
            ? "Checking API…"
            : apiOnline
              ? "API connected"
              : "API offline — start backend on :8000"}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button className="error-banner__close" onClick={() => setError(null)}>
            ✕
          </button>
        </div>
      )}

      <div className="app-body">
        <PaperLibrary
          papers={papers}
          selectedIds={selectedIds}
          onToggle={togglePaper}
          onSelectAll={selectAll}
          onClearSelection={clearSelection}
          onDelete={handleDelete}
          onUpload={handleUpload}
          uploading={uploading}
        />
        <ChatPanel
          messages={messages}
          loading={loading}
          selectedCount={selectedIds.size}
          onSend={handleSend}
        />
      </div>
    </div>
  );
}
