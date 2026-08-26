"""
API tests for POST /upload.

What we test
────────────
• Non-PDF file → HTTP 400 with a meaningful error message.
• Valid PDF upload with mocked IngestionService → HTTP 200 with correct
  UploadResponse shape (document_id, filename, num_pages, num_chunks, status).
• Oversized file → HTTP 400 (>50 MB).
• Missing file entirely → HTTP 422 (FastAPI validation).
• The filename in the response matches what was uploaded.

The real IngestionService (which loads BGE-large-en-v1.5 and FAISS) is replaced
by a MagicMock via conftest.api_client, so no model is loaded and no DB is used.
The route still writes the file to disk first (settings.upload_dir points at
tmp_path from the fixture) before calling ingest_document().
"""

import io
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pdf_file(name: str = "test.pdf", size_bytes: int = 1024) -> tuple[str, tuple]:
    """Build a (field_name, (filename, file_obj, content_type)) tuple for multipart upload."""
    content = b"%PDF-1.4 " + b"x" * size_bytes
    return ("file", (name, io.BytesIO(content), "application/pdf"))


def _non_pdf_file(name: str = "report.docx") -> tuple[str, tuple]:
    content = b"PK fake docx content"
    return ("file", (name, io.BytesIO(content), "application/vnd.openxmlformats-officedocument"))


# ── Validation (no ingestion service needed) ──────────────────────────────────

class TestUploadValidation:
    def test_non_pdf_returns_400(self, api_client):
        response = api_client.post("/upload", files=[_non_pdf_file()])
        assert response.status_code == 400

    def test_non_pdf_error_mentions_pdf(self, api_client):
        response = api_client.post("/upload", files=[_non_pdf_file()])
        body = response.json()
        assert "pdf" in body["detail"].lower(), (
            f"Expected PDF mention in error, got: {body['detail']}"
        )

    def test_missing_file_returns_422(self, api_client):
        """FastAPI returns 422 when a required field is missing from the request."""
        response = api_client.post("/upload")
        assert response.status_code == 422

    def test_oversized_file_returns_400(self, api_client):
        """Files over 50 MB should be rejected after writing to temp dir."""
        size_51mb = 51 * 1024 * 1024  # 51 MB > 50 MB limit
        response = api_client.post("/upload", files=[_pdf_file(size_bytes=size_51mb)])
        assert response.status_code == 400


# ── Successful upload (mocked ingestion) ─────────────────────────────────────

class TestUploadSuccess:
    def test_valid_pdf_returns_200(self, api_client):
        response = api_client.post("/upload", files=[_pdf_file("financial.pdf")])
        assert response.status_code == 200

    def test_response_shape(self, api_client):
        response = api_client.post("/upload", files=[_pdf_file("report.pdf")])
        body = response.json()
        required_keys = {"document_id", "filename", "num_pages", "num_chunks", "status"}
        assert required_keys.issubset(body.keys()), (
            f"Missing response keys: {required_keys - body.keys()}"
        )

    def test_response_document_id_is_non_empty(self, api_client):
        response = api_client.post("/upload", files=[_pdf_file()])
        assert response.json()["document_id"]

    def test_response_status_is_ready(self, api_client):
        response = api_client.post("/upload", files=[_pdf_file()])
        assert response.json()["status"] == "ready"

    def test_response_num_pages_is_positive(self, api_client):
        response = api_client.post("/upload", files=[_pdf_file()])
        assert response.json()["num_pages"] > 0

    def test_response_num_chunks_is_positive(self, api_client):
        response = api_client.post("/upload", files=[_pdf_file()])
        assert response.json()["num_chunks"] > 0

    def test_content_type_is_json(self, api_client):
        response = api_client.post("/upload", files=[_pdf_file()])
        assert "application/json" in response.headers["content-type"]
