"""
POST /upload

Design choice: the route itself contains zero business logic -- it validates
the incoming file, streams it to disk, and delegates to IngestionService.
This keeps the pipeline testable independent of HTTP (you can call
IngestionService.ingest_document() directly in a unit test with a local
file path, no TestClient / multipart form needed).

Ingestion is CPU-heavy (embedding + FAISS) and can take 1-3 minutes for a
large PDF. It runs in a worker thread via asyncio.to_thread so the event
loop stays responsive. Swagger/browsers may still show "Failed to fetch" if
uvicorn --reload restarts mid-request when storage/ files change -- run with
--reload-exclude 'storage/*' during development.
"""

import asyncio
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.dependencies import get_ingestion_service
from app.core.config import Settings, get_settings
from app.models.schemas import UploadResponse
from app.services.rag.ingestion_service import IngestionService

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 50


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_path = settings.upload_dir / f"{uuid.uuid4()}{suffix}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        size_mb = temp_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit ({size_mb:.1f}MB)",
            )

        result = await asyncio.to_thread(
            ingestion_service.ingest_document,
            str(temp_path),
            file.filename or "document.pdf",
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        # Keep the original file for potential re-processing/audit; if disk
        # usage becomes a concern, move to object storage (S3) instead of
        # deleting outright, since financial-document retention often has
        # compliance requirements.
        pass
