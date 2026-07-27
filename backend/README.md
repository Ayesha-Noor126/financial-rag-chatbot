# Financial Document RAG Chatbot — Phase 1

## What's implemented
Document ingestion pipeline: PDF parsing (PyMuPDF) -> header/footer/page-number
stripping -> whitespace cleaning + section detection -> semantic chunking
(600-800 tokens, 100 overlap) -> embedding (BAAI/bge-large-en-v1.5, cached) ->
FAISS HNSW index + BM25 index -> SQLite metadata store.

## Run it
```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv311\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, TAVILY_API_KEY

# Important: exclude storage/ from reload watcher — otherwise FAISS/DB writes
# during upload restart the server mid-request (Swagger shows "Failed to fetch").
uvicorn app.main:app --reload --reload-exclude 'storage/*'
```

On Windows PowerShell you can also use:
```powershell
.\run-dev.ps1
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 — the UI talks to the API at http://localhost:8000.

Test ingestion:
```bash
curl -X POST http://localhost:8000/upload -F "file=@nestle_annual_report.pdf"
curl http://localhost:8000/health
```

**Upload takes 1–3 minutes** for a large PDF (parsing + embedding + FAISS). Swagger may show "Failed to fetch" if the server reloads during ingestion — use `--reload-exclude 'storage/*'` or run without `--reload`.

## Not yet implemented (coming in Phase 2+)
- Query rewriting, NER, hybrid retrieval + RRF, reranking (BAAI/bge-reranker-base)
- Context compression, grounded prompting, confidence scoring
- Tavily web-search fallback
- /chat, /history, /document endpoints, streaming responses
- React frontend
