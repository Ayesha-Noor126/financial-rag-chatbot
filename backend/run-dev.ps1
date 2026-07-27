# Dev server — excludes storage/ from reload watcher so FAISS/DB writes
# during ingestion don't restart uvicorn mid-request (causes Swagger "Failed to fetch").
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
.\venv311\Scripts\uvicorn.exe app.main:app --reload --reload-exclude "storage/*"
