import type {
  Paper,
  PapersResponse,
  ResearchResponse,
  UploadResponse,
} from "../types";

export interface StreamCallbacks {
  onStart?: (intent: string) => void;
  onToken?: (text: string) => void;
  onCitations?: (citations: any[]) => void;
  onEnd?: () => void;
  onError?: (message: string) => void;
}

const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "/api" : "http://127.0.0.1:8000");

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchPapers(): Promise<Paper[]> {
  const data = await handleResponse<PapersResponse>(
    await fetch(`${API_BASE}/papers`)
  );
  return data.papers;
}

export async function uploadPapers(files: File[]): Promise<Paper[]> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const data = await handleResponse<UploadResponse>(
    await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    })
  );
  return data.papers;
}

export async function deletePaper(paperId: string): Promise<void> {
  await handleResponse(
    await fetch(`${API_BASE}/papers/${paperId}`, { method: "DELETE" })
  );
}

export async function askQuestion(
  paperIds: string[],
  question: string
): Promise<ResearchResponse> {
  return handleResponse<ResearchResponse>(
    await fetch(`${API_BASE}/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_ids: paperIds, question }),
    })
  );
}

export async function streamQuestion(
  paperIds: string[],
  question: string,
  callbacks: StreamCallbacks
) {
  const response = await fetch(`${API_BASE}/research`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      paper_ids: paperIds,
      question,
    }),
  });

  if (!response.ok) {
    throw new Error("Streaming request failed.");
  }

  if (!response.body) {
    throw new Error("No response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  while (true) {
  const { done, value } = await reader.read();

  if (done) break;

  buffer += decoder.decode(value, { stream: true });

  const events = buffer.split(/\r?\n\r?\n/);

  // Keep the last incomplete event in the buffer
  buffer = events.pop() ?? "";

  for (const eventText of events) {
    const lines = eventText.split("\n");

    let eventType = "";
    let data = "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      }

      if (line.startsWith("data:")) {
        data = line.slice(5).trim();
      }
    }

    if (!eventType || !data) continue;

    let payload;

    try {
      payload = JSON.parse(data);
    } catch (err) {
      console.error("Invalid SSE payload:", data);
      continue;
    }

    switch (eventType) {
      case "start":
        callbacks.onStart?.(payload.intent);
        break;

      case "token":
        callbacks.onToken?.(payload.text);
        break;

      case "citations":
        callbacks.onCitations?.(payload);
        break;

      case "end":
        callbacks.onEnd?.();
        break;

      case "error":
        callbacks.onError?.(payload.message);
        break;
    }
  }
}
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/`);
    return response.ok;
  } catch {
    return false;
  }
}
