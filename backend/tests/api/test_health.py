"""
API tests for GET /health.

What we test
────────────
• Returns HTTP 200.
• Response body has the expected keys: status, faiss_vectors,
  bm25_documents, documents_indexed.
• status value is "ok".
• All numeric fields are integers >= 0.

All underlying services are mocked via conftest.api_client.
"""

import pytest


class TestHealthEndpoint:
    def test_returns_200(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_response_has_required_keys(self, api_client):
        response = api_client.get("/health")
        body = response.json()
        expected_keys = {"status", "faiss_vectors", "bm25_documents", "documents_indexed"}
        assert expected_keys.issubset(body.keys()), (
            f"Missing keys: {expected_keys - body.keys()}"
        )

    def test_status_is_ok(self, api_client):
        response = api_client.get("/health")
        assert response.json()["status"] == "ok"

    def test_numeric_fields_are_non_negative_integers(self, api_client):
        body = api_client.get("/health").json()
        for key in ("faiss_vectors", "bm25_documents", "documents_indexed"):
            value = body[key]
            assert isinstance(value, int), f"{key} should be int, got {type(value)}"
            assert value >= 0, f"{key} should be >= 0, got {value}"

    def test_content_type_is_json(self, api_client):
        response = api_client.get("/health")
        assert "application/json" in response.headers["content-type"]
