# Research Paper Assistant

An AI-powered research paper assistant that lets you upload PDF papers and have intelligent conversations about them. Ask questions, get summaries, and compare papers using a hybrid retrieval-augmented generation (RAG) pipeline.

---

## Features

- PDF Upload — Upload one or more research papers (PDFs) and have them indexed automatically.
- Intelligent Q&A  — Ask specific questions and get answers grounded in the papers, with citations.
- Summarization  — Generate brief, detailed, technical, student-friendly, or bullet-point summaries.
- Paper Comparison  — Ask the assistant to compare multiple uploaded papers side by side.
- Streaming Responses  — Answers are streamed in real time via Server-Sent Events (SSE).
- Hybrid Retrieval  — Combines dense vector search (FAISS) and sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF) for high-quality retrieval.
- Persistent Storage  — Uploaded papers and their vector indexes are saved and survive server restarts.
- RAG Evaluation Suite  — Benchmark planner routing, hybrid retrieval, Q&A, and summarization against a fixed test suite (`backend/eval`).

---

## Architecture

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

## Tech Stack

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
## Project Demo

[![Watch the Project Demo](https://img.youtube.com/vi/vBOHCKtW1bQ/0.jpg)](https://youtu.be/vBOHCKtW1bQ)

[Watch the full demo on YouTube](https://youtu.be/vBOHCKtW1bQ)

---

## Getting Started

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

## Project Structure

```
Research_assistant/
├── .gitignore
├── README.md
├── backend/
│   ├── requirements.txt          # Python dependencies
│   ├── eval/                     # RAG evaluation suite (see below)
│   │   ├── run_eval.py           # Master runner for all evaluators
│   │   ├── test_cases.py         # QA, comparison, summary, planner cases
│   │   ├── eval_retrieval.py     # Planner + retrieval metrics (P@K, MRR)
│   │   ├── eval_answer.py        # Answer faithfulness & relevance (LLM judge)
│   │   ├── eval_summary.py       # Summary coverage, coherence, conciseness
│   │   ├── rate_limit.py         # Throttling & retries for Groq 429 / quota
│   │   ├── requirements_eval.txt # Optional deps (RAGAS, pandas, etc.)
│   │   ├── EVALUATION_REPORT.md  # Latest benchmark write-up
│   │   └── results/              # Timestamped JSON/CSV reports (--save)
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

## API Endpoints

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

## Configuration

### Backend (`backend/app/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROK_API_KEY` |  Yes | — | Your Groq API key |
| `ALLOWED_ORIGINS` | No | `localhost:5173` | Comma-separated CORS origins |

### Frontend (`frontend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_URL` | No | Dev proxy | Production backend URL |

---

## RAG evaluation (`backend/eval`)

The **`eval`** folder is an offline benchmark for the full RAG stack. It loads the same papers, embeddings, and services as the running app and scores them on curated questions in `test_cases.py` (planner routing, hybrid retrieval, Q&A, comparisons, and summaries).

| Module | What it measures |
|--------|------------------|
| `eval_retrieval.py` | Research planner intent & paper routing; hybrid retriever Precision@K, Recall@K, MRR |
| `eval_answer.py` | End-to-end answers — faithfulness and relevance via LLM-as-judge; optional [RAGAS](https://github.com/explodinggradients/ragas) |
| `eval_summary.py` | Map-reduce summaries — coverage, coherence, conciseness (LLM-as-judge) |
| `run_eval.py` | Runs the modules above and optionally writes artifacts to `eval/results/` |

### Prerequisites

- Backend dependencies installed and `GROK_API_KEY` set in `backend/app/.env`.
- Benchmark PDFs already uploaded and indexed under `backend/app/storage/papers/` (the suite expects the papers referenced in `test_cases.py` to be present).

### Run the suite

From the **`backend/`** directory:

```bash
# Full suite (planner + retrieval + Q&A + summaries)
python eval/run_eval.py

# Subsets
python eval/run_eval.py --retrieval
python eval/run_eval.py --answer
python eval/run_eval.py --summary

# Persist JSON, CSV, and a Markdown report under eval/results/
python eval/run_eval.py --save

# Optional RAGAS metrics (extra packages)
pip install -r eval/requirements_eval.txt
python eval/run_eval.py --answer --ragas
```

### Groq rate limits

The suite makes many LLM calls (planner, long streamed answers, and two judge calls per QA case). On Groq free tiers you may see **429** or **daily token quota** errors.

**Lower load:**

```bash
# Retrieval only — no answer/summary generation (retrieval uses local FAISS/BM25)
python eval/run_eval.py --retrieval

# Fewer cases, longer pause between API calls
python eval/run_eval.py --max-cases 5 --delay 8

# Generate answers but skip LLM-as-judge (saves ~2 calls per case)
python eval/run_eval.py --answer --skip-judge --max-cases 3 --delay 10
```

**Environment variables** (optional): `EVAL_REQUEST_DELAY_SEC`, `EVAL_CASE_DELAY_SEC`, `EVAL_MAX_RETRIES`, `EVAL_BACKOFF_BASE_SEC`.

By default, eval runs **serial map-reduce summaries** (`SUMMARY_MAP_REDUCE_MAX_WORKERS=1`) because the app otherwise fires up to 8 parallel Groq calls per long paper. Use `--parallel-summaries` only if your Groq tier allows it.

If you hit **daily quota**, wait for the Groq limit to reset or upgrade the plan; the runner stops remaining LLM-heavy steps when quota is exhausted.

The HF Hub “unauthenticated requests” warning is harmless for local embeddings; set `HF_TOKEN` if model downloads are slow or rate-limited.

Individual modules can also be run as modules, e.g. `python -m eval.eval_retrieval`.

See `backend/eval/EVALUATION_REPORT.md` for example scores and methodology.

---

## License

MIT
