#  Financial Document RAG Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade Retrieval-Augmented Generation (RAG) system for intelligent financial document analysis — with hybrid search, voice I/O, real-time observability, and built-in evaluation metrics.**

[Features](#-features) · [Architecture](#-architecture) · [Tech Stack](#-tech-stack) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Evaluation](#-evaluation--metrics)

</div>

---

##  Overview

This project is a full-stack RAG chatbot purpose-built for financial documents (annual reports, 10-Ks, earnings releases, prospectuses). Users upload PDFs, and the system answers natural-language questions grounded exclusively in the uploaded content — eliminating hallucinations through a multi-stage retrieval and confidence-scoring pipeline.

Key engineering decisions:
- **Hybrid search** (FAISS dense + BM25 sparse) fused via Reciprocal Rank Fusion (RRF) — outperforms either retriever alone on financial terminology.
- **Cross-encoder reranking** (BGE-Reranker) with a smart skip mechanism — bypasses the expensive reranker when both retrievers strongly agree, cutting median latency ~45%.
- **Automatic web fallback** via Tavily when document confidence drops below a configurable threshold — so the chatbot answers "what is the current stock price?" gracefully rather than refusing.
- **Voice I/O** — ElevenLabs Scribe v2 for STT and multilingual TTS so analysts can dictate questions and hear answers read aloud.
- **Langfuse observability** — full distributed tracing of every pipeline stage (rewrite → NER → retrieval → rerank → generation), with live prompt versioning (no code redeploys to update system prompts).

---

##  Features

###  Retrieval Pipeline
| Stage | Implementation |
|---|---|
| **PDF Parsing** | PyMuPDF with layout-aware extraction |
| **Text Cleaning** | Custom cleaner for financial artifacts (headers, footnotes, tables) |
| **Chunking** | Token-aware chunking (700-token target, 100-token overlap, tiktoken) |
| **Embedding** | `BAAI/bge-large-en-v1.5` (1024-dim) via `sentence-transformers` |
| **Vector Search** | FAISS (flat cosine similarity) |
| **Keyword Search** | BM25 via `rank-bm25` |
| **Fusion** | Reciprocal Rank Fusion (RRF, k=60) with parallelized BM25 + FAISS |
| **Reranking** | `BAAI/bge-reranker-base` cross-encoder with conditional skip |
| **NER Boost** | spaCy-based entity extraction (companies, metrics, years, regions) for score boosting |

###  Generation & Safety
- **Query Rewriting** — LLM rewrites ambiguous or conversational queries into self-contained retrieval queries, aware of conversation history.
- **Context Compression** — reduces token cost and improves answer quality by pruning low-signal chunks before the LLM call.
- **Confidence Scoring** — blended signal from cross-encoder score (sigmoid-normalized) + entity coverage; triggers web fallback or "I don't know" at a configurable threshold.
- **Citation Extraction** — answer cites specific source pages and sections `[Source N]` style; UI renders these as tags.
- **No-answer Detection** — hardened prompt prevents fabrication; the system explicitly declines to answer when context is insufficient.
- **Web Search Fallback** — Tavily integration activates automatically for low-confidence or real-time queries (e.g., live stock prices).

###  Voice I/O
- **Speech-to-Text**: ElevenLabs Scribe v2 — browser `MediaRecorder` → WebM/Opus blob → ElevenLabs API → transcript appended to input field.
- **Text-to-Speech**: ElevenLabs multilingual v2 — per-message "read aloud" button streams MP3 audio in the browser with stop/play state management.

###  Observability
- **Langfuse tracing** — every request creates a trace with child spans for: `query-rewrite`, `ner-extraction`, `hybrid-retrieval`, `rerank`, and generation.
- **Live prompt versioning** — system prompts are managed in the Langfuse UI; the backend hot-reloads them every 5 minutes without a server restart.
- **In-process prompt cache** — startup warm-up pre-fetches all prompts; first request pays zero network latency for prompt retrieval.

###  Built-in Evaluation
Per-response metrics (powered by DeepEval + Groq judge):
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Contextual Precision** — are the retrieved chunks relevant?
- **Contextual Recall** — does retrieval cover what the correct answer needs?
- **Confidence Score** — internal blended metric visible in the chat UI.

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)              │
│   Upload Tab  │  Chat Tab  │  Voice I/O  │  Eval Metrics    │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST API (FastAPI)
┌───────────────────────▼─────────────────────────────────────┐
│                       Backend (FastAPI)                      │
│                                                             │
│  ┌──────────┐  ┌─────────────────────────────────────────┐  │
│  │ Ingestion│  │           Query Pipeline                │  │
│  │ Pipeline │  │                                         │  │
│  │          │  │  Query → Rewrite → NER → Hybrid         │  │
│  │ PDF Parse│  │  Retrieval (FAISS ∥ BM25) → RRF →      │  │
│  │ → Clean  │  │  Rerank → Entity Boost → Compress →     │  │
│  │ → Chunk  │  │  Confidence → Generate / Web Fallback   │  │
│  │ → Embed  │  │                                         │  │
│  │ → Index  │  └─────────────────────────────────────────┘  │
│  └──────────┘                                               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  FAISS Index │  │  BM25 Index  │  │  SQLite Metadata │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Langfuse (Tracing + Prompt Management)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                              │
    ┌────▼────┐                   ┌─────▼──────┐
    │  Groq   │                   │   Tavily   │
    │ LLM API │                   │ Web Search │
    └─────────┘                   └────────────┘
         │
    ┌────▼──────────┐
    │  ElevenLabs   │
    │  STT + TTS    │
    └───────────────┘
```

---

##  Tech Stack

### Backend
| Category | Technology |
|---|---|
| **Framework** | FastAPI 0.115, Uvicorn |
| **LLM** | Groq (Llama-3.3-70B) or OpenAI GPT (configurable) |
| **Embeddings** | `BAAI/bge-large-en-v1.5` via sentence-transformers |
| **Vector DB** | FAISS (CPU, persisted to disk) |
| **Keyword Search** | rank-bm25 |
| **Reranker** | `BAAI/bge-reranker-base` (cross-encoder) |
| **PDF Parsing** | PyMuPDF |
| **NER** | spaCy `en_core_web_sm` |
| **Metadata Store** | SQLite via SQLModel |
| **Observability** | Langfuse |
| **Web Search** | Tavily |
| **Tokenization** | tiktoken |
| **Evaluation** | DeepEval |
| **Caching** | cachetools (LRU, in-process) |
| **HTTP Client** | httpx (async) |

### Frontend
| Category | Technology |
|---|---|
| **Framework** | React 19, Vite 8 |
| **Voice STT** | ElevenLabs Scribe v2 (MediaRecorder API) |
| **Voice TTS** | ElevenLabs multilingual v2 |
| **Linting** | oxlint |
| **Styling** | Vanilla CSS |

---

##  Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- A [Groq](https://console.groq.com) API key *(free tier available)*
- A [Tavily](https://tavily.com) API key *(optional — web fallback)*
- A [Langfuse](https://langfuse.com) account *(optional — observability)*
- An [ElevenLabs](https://elevenlabs.io) API key *(optional — voice I/O)*

---

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fin-rag-phase1.git
cd fin-rag-phase1
```

---

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv311
# Windows
venv311\Scripts\activate
# macOS / Linux
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy NER model
python -m spacy download en_core_web_sm
```

#### Configure environment variables

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1

# Optional — web search fallback
TAVILY_API_KEY=tvly-your_tavily_api_key_here

# Optional — observability & prompt versioning
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Embedding model (default works, no change needed)
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5
```

> **Switching to OpenAI:** Set `LLM_PROVIDER=openai`, `LLM_API_KEY=sk-...`, `LLM_MODEL=gpt-4o`, `LLM_BASE_URL=https://api.openai.com/v1`. No code changes needed — the Groq API is OpenAI-compatible.

#### Start the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install
```

#### Configure environment variables

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000

# Optional — voice features
VITE_ELEVENLABS_API_KEY=your_elevenlabs_api_key
VITE_ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
```

#### Start the frontend

```bash
npm run dev
```

The UI will open at `http://localhost:5173`.

---

### 4. Usage

1. **Open** `http://localhost:5173` in your browser.
2. **Go to the Upload tab** → select a PDF (annual report, 10-K, etc.) → click **Upload & Index**.
   - Ingestion takes 1–3 minutes for large documents (parsing → chunking → embedding → indexing).
3. **Switch to the Chat tab** → ask questions in natural language:
   - *"What was the total revenue in 2023?"*
   - *"Summarize the key risk factors."*
   - *"How did operating income change year-over-year?"*
4. The response includes **citations** (page + section), a **confidence score**, and optionally a **web fallback** badge if live data was retrieved.
5. Click the **🔊 speaker icon** on any response to hear it read aloud. Click the **🎙️ mic icon** to dictate your question.

---

##  Project Structure

```
fin-rag-phase1/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── dependencies.py        # FastAPI DI container (wires all services)
│   │   │   └── routes/
│   │   │       ├── chat.py            # POST /chat — main conversation endpoint
│   │   │       ├── upload.py          # POST /upload — PDF ingestion
│   │   │       ├── document.py        # GET /documents — list indexed docs
│   │   │       ├── history.py         # GET /history/{session_id}
│   │   │       ├── health.py          # GET /health
│   │   │       ├── debug_query.py     # GET /debug/query — inspect retrieval
│   │   │       └── debug_answer.py    # GET /debug/answer — inspect full pipeline
│   │   ├── core/
│   │   │   ├── config.py              # Pydantic Settings (all tunables in one place)
│   │   │   ├── llm_client.py          # OpenAI-compatible client (Groq / OpenAI)
│   │   │   ├── langfuse_client.py     # Observability singleton
│   │   │   ├── prompt_manager.py      # Langfuse prompt versioning + in-process cache
│   │   │   └── query_cache.py         # TTL-based query result cache
│   │   ├── models/
│   │   │   └── schemas.py             # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── parser/
│   │   │   │   ├── pdf_parser.py      # PyMuPDF text extraction
│   │   │   │   └── cleaner.py         # Financial text normalization
│   │   │   ├── rag/
│   │   │   │   ├── chunker.py         # Token-aware semantic chunker
│   │   │   │   ├── ingestion_service.py   # Ingestion orchestrator
│   │   │   │   ├── query_pipeline_service.py  # Query orchestrator
│   │   │   │   ├── embeddings/        # Embedder (BGE-large)
│   │   │   │   ├── retriever/         # FAISS index, BM25 index, HybridRetriever (RRF)
│   │   │   │   ├── reranker/          # BGE-reranker cross-encoder
│   │   │   │   ├── ner/               # spaCy NER entity extraction
│   │   │   │   ├── query_rewriter/    # LLM-based query rewriting
│   │   │   │   ├── generation/        # AnswerService, ContextCompressor, ConfidenceScorer
│   │   │   │   ├── metadata/          # SQLite document/chunk metadata store
│   │   │   │   └── evaluation/        # DeepEval metric computation
│   │   │   └── web_search/            # Tavily client (async)
│   │   └── main.py                    # FastAPI app factory, startup/shutdown hooks
│   ├── requirements.txt
│   ├── rag_eval.py                    # Interactive CLI evaluation tool
│   ├── seed_prompts.py                # Seeds system prompts into Langfuse
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Main application component
│   │   ├── api.js                     # Backend API client
│   │   ├── hooks/
│   │   │   ├── useVoiceInput.js       # MediaRecorder → ElevenLabs STT
│   │   │   └── useVoiceOutput.js      # ElevenLabs TTS playback manager
│   │   └── services/
│   │       └── elevenlabs.js          # ElevenLabs STT + TTS API wrappers
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and ingest a PDF document |
| `POST` | `/chat` | Send a message; returns answer + citations + metrics |
| `GET` | `/documents` | List all indexed documents |
| `GET` | `/history/{session_id}` | Retrieve conversation history |
| `GET` | `/health` | Backend health check |
| `GET` | `/debug/query?q=...` | Inspect retrieval pipeline output |
| `GET` | `/debug/answer?q=...` | Full pipeline debug (rewrite → answer) |

Full interactive API documentation is available at `http://localhost:8000/docs` when the server is running.

### Example Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": null,
    "message": "What was the net income for fiscal year 2023?"
  }'
```

### Example Chat Response

```json
{
  "session_id": "uuid-...",
  "answer": "The net income for fiscal year 2023 was $4.2 billion [Source 1], representing a 12% increase compared to 2022 [Source 2].",
  "citations": [
    { "chunk_id": "...", "page_number": 42, "section": "Consolidated Statements of Income" },
    { "chunk_id": "...", "page_number": 43, "section": "Year-over-Year Comparison" }
  ],
  "confidence_score": 0.87,
  "used_web_fallback": false,
  "web_sources": [],
  "faithfulness": 0.95,
  "answer_relevancy": 0.92,
  "precision": 0.88,
  "recall": 0.84
}
```

---

##  Evaluation & Metrics

### Automated Evaluation (DeepEval + Groq)

Run the interactive CLI evaluator against a live backend:

```bash
cd backend
python rag_eval.py
```

You will be prompted for:
1. A question (e.g., *"What was revenue in 2023?"*)
2. An optional ground-truth answer (enables Contextual Precision + Recall)

| Metric | What it measures | Requires ground truth |
|---|---|---|
| **Faithfulness** | Answer is grounded in retrieved chunks (no hallucination) | ❌ |
| **Answer Relevancy** | Answer addresses the question asked | ❌ |
| **Contextual Precision** | Retrieved chunks are relevant to the expected answer | ✅ |
| **Contextual Recall** | Retrieved chunks cover what the expected answer needs | ✅ |

The judge model is pinned to `temperature=0` + a fixed seed for reproducible scores.

### Real-time Metrics in the UI

Every chat response displays live metrics directly below the answer bubble:
- **Confidence %** — blended reranker score + entity coverage
- **Faithfulness** — DeepEval score (0–1)
- **Relevancy** — DeepEval score (0–1)
- **Precision / Recall** — DeepEval contextual metrics

---

##  Configuration Reference

All tunables live in `backend/app/core/config.py` and are overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model identifier |
| `LLM_TEMPERATURE` | `0.0` | Deterministic outputs |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE_TOKENS` | `700` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `100` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking |
| `RERANK_TOP_K` | `5` | Final chunks passed to LLM |
| `RRF_K_CONSTANT` | `60` | RRF damping constant |
| `CONFIDENCE_THRESHOLD` | `0.45` | Below → web fallback or no-answer |
| `RERANK_SKIP_THRESHOLD` | `0.026` | Skip cross-encoder when RRF agrees strongly |
| `ENABLE_WEB_FALLBACK` | `true` | Enable Tavily web search |
| `TAVILY_MAX_RESULTS` | `5` | Max web search results |
| `QUERY_CACHE_TTL_SECONDS` | `300` | Cache lifetime for repeated queries |
| `LANGFUSE_PROMPT_CACHE_TTL_SECONDS` | `300` | Prompt hot-reload interval |

---

##  Continuous Integration (CI) & GitHub Actions

This repository includes an automated GitHub Actions CI pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) that verifies code quality, runs test suites, checks frontend builds, and validates Docker container images on every `push` and `pull_request` to `main`.

### Pipeline Architecture

```
Push / Pull Request
       │
       ├──► 1. Backend Pytest Suite (Python 3.11 + CPU PyTorch + spaCy + pytest-cov)
       ├──► 2. Frontend Lint & Build (Node 20 + oxlint + vite build)
       ├──► 3. Docker Image Build Check (Backend & Frontend images via Buildx)
       └──► 4. Docker Compose Integration Smoke Test (Container healthchecks)
```

### CI Triggers & Caching
- **Triggers**: Executes automatically on `push` to `main`/`master` and on `pull_request` targeting `main`/`master`.
- **Caching**:
  - `pip` wheel caching for Python dependencies.
  - `npm` module caching for Node dependencies.
  - `Docker GHA layer caching` (`cache-from: type=gha`) to make image builds fast.

---

### How to Understand and Debug CI Failures

When a CI run fails on GitHub, follow this 4-step workflow to diagnose and fix the failure:

#### Step 1: Locate the Failing Job and Step in GitHub
1. Navigate to the **Actions** tab in your GitHub repository.
2. Click on the failed workflow run (marked with a red `X`).
3. Click on the job that failed (e.g., `Backend Pytest Suite` or `Frontend Lint & Build`).
4. Click on the specific step that has a red error indicator to expand its logs.

#### Step 2: Identify Failure Categories

| Error Pattern | Root Cause | How to Fix |
|---|---|---|
| `pytest: command not found` or `ModuleNotFoundError` | Missing Python dependency or incorrect virtualenv context. | Check `requirements.txt` / `requirements-dev.txt` and ensure new packages are declared. |
| `AssertionError` in pytest | A unit test or API test failed due to logic changes. | Run `pytest` locally (see Step 3 below) and inspect tracebacks. |
| `oxlint` or `ESLint` error | Frontend linting violation (unused variable, syntax error). | Run `npm run lint` in `frontend/` locally to fix reported lines. |
| `vite build` error | TypeScript or JSX compilation failure. | Run `npm run build` locally in `frontend/` to view exact compile error. |
| `docker build` failed | Dockerfile instruction failure (missing package or file path). | Run `docker build -t test ./backend` locally to reproduce build step error. |

#### Step 3: Reproduce and Debug Failures Locally

Before pushing code to GitHub, run the CI checks locally on your machine:

**1. Debug Backend Tests Locally:**
```powershell
cd backend
python -m pytest --cov=app -vv
```

**2. Debug Frontend Lint & Build Locally:**
```powershell
cd frontend
npm run lint
npm run build
```

**3. Debug Docker Compose Integration Locally:**
```powershell
docker compose up --build -d
docker inspect --format="{{.State.Health.Status}}" fin-rag-backend
docker inspect --format="{{.State.Health.Status}}" fin-rag-frontend
curl -f http://localhost:8000/health
```

#### Step 4: Environment Variables & Mocks in CI
- CI runs in isolated GitHub runner VMs where live API keys (`GROQ_API_KEY`, `TAVILY_API_KEY`) are intentionally not provided.
- Backend tests use pytest mocks and fallback mechanisms. If adding a new feature that calls an external service, ensure a fallback or mock fixture is implemented in `tests/conftest.py`.

---

##  RAG Evaluation & Regression Testing

This project includes a scheduled RAG evaluation pipeline ([`.github/workflows/rag_eval.yml`](.github/workflows/rag_eval.yml)) that goes beyond unit tests to measure the actual *quality* of the retrieval-augmented generation pipeline.

### Conceptual Design: Why This Is Real

Most "scheduled" CI workflows automate meaningless tasks. This one is different:

| Property | Implementation |
|---|---|
| **Fixed ground truth** | `tests/eval/golden_qa.json` — 10 curated financial QA pairs with expected answers |
| **Real pipeline execution** | Starts a live `uvicorn` server, ingests a PDF, POSTs actual questions to `/debug/answer` |
| **LLM-as-judge scoring** | DeepEval metrics via `GroqJudge` (same model used in production, pinned `temperature=0, seed=42`) |
| **Regression detection** | Compares current run against a stored `baseline_metrics.json`; fails CI if any metric drops >10% |
| **Artifact-based baseline** | Each successful run updates the baseline artifact; the next run downloads it automatically |

### Workflow Schedule

| Trigger | What Runs |
|---|---|
| `push` to `main` | Fast CI: tests, lint, docker build (`ci.yml`) |
| Every Monday 02:00 UTC | Full RAG evaluation vs golden dataset (`rag_eval.yml`) |
| 1st of every month 03:00 UTC | Weekly eval + monthly trend report across last 4 runs |
| `workflow_dispatch` | Manual trigger — useful after intentional improvements |

### How to Read Evaluation Results

1. Go to **Actions → Scheduled RAG Evaluation** on GitHub.
2. Click on a run, then **RAG Pipeline Evaluation** job.
3. Each step shows the question-by-question scores.
4. The **"Write Summary to GitHub Actions UI"** step renders a markdown table with:
   - Current metric values vs baseline
   - 🔺 Improved / 🔻 Regressed / ➡️ Stable indicators
   - Any detected regressions highlighted in red.

### How to Add New Golden Test Cases

Edit [`backend/tests/eval/golden_qa.json`](backend/tests/eval/golden_qa.json) and add a new entry:

```json
{
  "id": "fin-011",
  "question": "Your question here",
  "expected_answer": "The expected answer grounded in the fixture document",
  "min_confidence": 0.40,
  "tags": ["your_tag"]
}
```

The `expected_answer` must be answerable from the synthetic `Horizon Capital Annual Report 2023` fixture embedded in [`run_rag_eval.py`](backend/tests/eval/run_rag_eval.py). If you add a question about new content, also add that content to `FIXTURE_REPORT_TEXT` in the harness.

### How to Update the Baseline (After Intentional Improvements)

When you make a meaningful improvement to the pipeline (e.g., new embedding model, better chunking), the new scores should become the reference baseline:

1. Trigger a manual run: **Actions → Scheduled RAG Evaluation → Run workflow**.
2. After a clean pass, download the `rag-eval-baseline` artifact.
3. Replace `backend/tests/eval/baseline_metrics.json` with the downloaded file.
4. Commit: `git commit -m "eval: update baseline after embedding model upgrade"`.

### Required GitHub Secret

Add `GROQ_API_KEY` to your repository secrets for the LLM judge to score answers:
**GitHub → Repository → Settings → Secrets and variables → Actions → New repository secret**

Without it, DeepEval scoring is skipped and only `confidence_score` thresholds are validated.

---

##  Contributing

Contributions are welcome! Please open an issue to discuss your proposed change before submitting a PR.

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feat/your-feature`
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

##  Author

**Ayesha Noor**

Built as a demonstration of production-level RAG engineering for financial document intelligence.

---


