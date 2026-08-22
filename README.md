# 📚 Research Assistant

An AI-powered research paper assistant that lets you upload PDF papers and have intelligent conversations about them. Ask questions, get summaries, and compare papers using a hybrid retrieval-augmented generation (RAG) pipeline.

---

## ✨ Features

- **📄 PDF Upload** — Upload one or more research papers (PDFs) and have them indexed automatically.
- **🔍 Intelligent Q&A** — Ask specific questions and get answers grounded in the papers, with citations.
- **📝 Summarization** — Generate brief, detailed, technical, student-friendly, or bullet-point summaries.
- **⚖️ Paper Comparison** — Ask the assistant to compare multiple uploaded papers side by side.
- **🔄 Streaming Responses** — Answers are streamed in real time via Server-Sent Events (SSE).
- **🧠 Hybrid Retrieval** — Combines dense vector search (FAISS) and sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF) for high-quality retrieval.
- **💾 Persistent Storage** — Uploaded papers and their vector indexes are saved and survive server restarts.

---

## 🏗️ Architecture

```
User → React Frontend → FastAPI Backend
                            ├── PDF Extraction (PyMuPDF)
                            ├── Text Chunking (LangChain)
                            ├── Embeddings (HuggingFace all-MiniLM-L6-v2)
                            ├── Vector Store (FAISS)
                            ├── BM25 Retriever (rank-bm25)
                            ├── Research Planner (LLM → intent detection)
                            └── Executors (QA / Summary / Comparison)
                                     └── Groq LLM (streaming)
```

### Request Flow

1. User sends a question with selected paper IDs.
2. The **Research Planner** (LLM-based) classifies the intent: `qa`, `summary`, `comparison`, or `unanswerable`.
3. The appropriate **Executor** is invoked.
4. The **Hybrid Retriever** fetches the most relevant chunks using RRF over FAISS + BM25.
5. The **LLM** generates a streaming response with inline citations.

---

## 🛠️ Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| LLM | Groq API (`langchain-groq`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Search | FAISS (`faiss-cpu`) |
| Keyword Search | BM25 (`rank-bm25`) |
| PDF Parsing | PyMuPDF (`fitz`) |
| Text Splitting | LangChain Text Splitters |
| Server | Uvicorn |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React 19 + TypeScript |
| Build Tool | Vite |
| Markdown | react-markdown + remark-gfm |
| Math Rendering | KaTeX + rehype-katex |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- A [Groq API key](https://console.groq.com/)

---

### Backend Setup

```bash
# 1. Navigate to backend
cd backend

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cd app
cp .env.example .env
# Edit .env and add your Groq API key
```

**`.env` file** (`backend/app/.env`):
```env
GROK_API_KEY=your_groq_api_key_here
```

```bash
# 5. Run the backend server
uvicorn main:app --reload
```

The backend will start at **http://localhost:8000**.

---

### Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

The frontend will start at **http://localhost:5173**.

> The Vite dev server automatically proxies `/api` requests to the backend, so no extra config is needed for local development.

---

## 📁 Project Structure

```
Research_assistant/
├── .gitignore
├── README.md
├── backend/
│   ├── requirements.txt          # Python dependencies
│   └── app/
│       ├── .env                  # Your secrets (never commit this)
│       ├── .env.example          # Template for environment variables
│       ├── main.py               # FastAPI app & API routes
│       ├── schemas.py            # Pydantic request/response models
│       ├── exceptions.py         # Custom exception classes
│       ├── prompt.py             # LLM prompt templates
│       ├── core/
│       │   ├── config.py         # App configuration & env loading
│       │   ├── streaming.py      # SSE streaming helpers
│       │   └── summary_config.py # Summary type instructions
│       ├── services/
│       │   ├── pdf_utils.py      # PDF text & abstract extraction
│       │   ├── chunking.py       # Document chunking
│       │   ├── embeddings.py     # HuggingFace embedding model
│       │   ├── vector_store.py   # FAISS vector store operations
│       │   ├── hybrid_retriever.py # BM25 + FAISS fusion (RRF)
│       │   ├── paper_manager.py  # Paper lifecycle management
│       │   ├── rag.py            # RAG answer generation
│       │   ├── summarizer.py     # Map-reduce summarization
│       │   ├── research_planner.py # LLM-based intent planning
│       │   ├── research_service.py # Orchestrates planner + executors
│       │   ├── citation_builder.py # Citation metadata builder
│       │   ├── evidence_plan.py  # EvidencePlan dataclass
│       │   ├── router.py         # Rule-based fallback router
│       │   └── llm.py            # Groq LLM client
│       ├── executors/
│       │   ├── qa_executor.py        # Q&A execution
│       │   ├── summary_executor.py   # Summarization execution
│       │   └── comparison_executor.py # Comparison execution
│       ├── utils/
│       │   └── text_utils.py     # Text normalization helpers
│       └── storage/
│           └── papers/           # Persisted paper data & indexes
└── frontend/
    ├── .env.example              # Template for frontend env vars
    ├── vite.config.ts            # Vite config (includes API proxy)
    ├── index.html
    └── src/
        ├── App.tsx               # Root component & layout
        ├── main.tsx              # App entry point
        ├── index.css             # Global styles
        ├── api/
        │   └── client.ts         # API client (fetch + SSE)
        ├── components/
        │   ├── ChatPanel.tsx     # Chat interface
        │   ├── MessageBubble.tsx # Message rendering
        │   ├── CitationChips.tsx # Citation display
        │   ├── PaperLibrary.tsx  # Paper list & selection
        │   └── UploadZone.tsx    # PDF drag-and-drop upload
        ├── types/
        │   └── index.ts          # TypeScript type definitions
        └── utils/
            └── markdownUtils.ts  # Markdown rendering helpers
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload one or more PDF files |
| `GET` | `/papers` | List all uploaded papers |
| `DELETE` | `/papers/{paper_id}` | Delete a paper |
| `POST` | `/research` | Ask a question (returns SSE stream) |

### `/research` Request Body
```json
{
  "paper_ids": ["uuid-1", "uuid-2"],
  "question": "What is the main contribution of this paper?"
}
```

### SSE Stream Events
```
event: start      → { "intent": "qa" | "summary" | "comparison" | "unanswerable" }
event: token      → { "text": "..." }
event: citations  → [{ "title": "...", "page": 3 }, ...]
event: end        → {}
```

---

## ⚙️ Configuration

### Backend (`backend/app/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROK_API_KEY` | ✅ Yes | — | Your Groq API key |
| `ALLOWED_ORIGINS` | No | `localhost:5173` | Comma-separated CORS origins |

### Frontend (`frontend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_URL` | No | Dev proxy | Production backend URL |

---

## 📄 License

MIT
