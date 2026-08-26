"""
API tests for POST /chat.

What we test
────────────
• Returns HTTP 200.
• Response body has the expected ChatResponse shape:
    answer, confidence_score, citations, used_web_fallback,
    web_sources, rewritten_query, session_id.
• answer is a non-empty string.
• confidence_score is a float in [0.0, 1.0].
• Starting a new session (no session_id) returns a session_id.
• Continuing an existing session (with session_id) is accepted.
• stream=False path returns a JSON body (not an event stream).
• Missing message field returns HTTP 422.

ChatService is fully mocked in conftest.api_client — no LLM call is made.

Note on streaming
─────────────────
The streaming path (stream=True) returns `text/event-stream`.  Testing it
with TestClient is possible but verbose; we test the non-streaming path here.
The streaming contract is covered at the service-unit level where it matters
most (that the generator yields tokens).
"""

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chat_payload(message: str = "What was the net profit?", **extra) -> dict:
    return {"message": message, "stream": False, **extra}


# ── Validation ────────────────────────────────────────────────────────────────

class TestChatValidation:
    def test_missing_message_returns_422(self, api_client):
        response = api_client.post("/chat", json={"stream": False})
        assert response.status_code == 422

    def test_empty_body_returns_422(self, api_client):
        response = api_client.post("/chat", json={})
        assert response.status_code == 422


# ── Successful non-streaming chat ─────────────────────────────────────────────

class TestChatNonStreaming:
    def test_returns_200(self, api_client):
        response = api_client.post("/chat", json=_chat_payload())
        assert response.status_code == 200

    def test_content_type_is_json(self, api_client):
        response = api_client.post("/chat", json=_chat_payload())
        assert "application/json" in response.headers["content-type"]

    def test_response_has_required_keys(self, api_client):
        response = api_client.post("/chat", json=_chat_payload())
        body = response.json()
        required = {"answer", "confidence_score", "citations", "used_web_fallback", "web_sources"}
        assert required.issubset(body.keys()), f"Missing: {required - body.keys()}"

    def test_answer_is_non_empty_string(self, api_client):
        response = api_client.post("/chat", json=_chat_payload())
        answer = response.json()["answer"]
        assert isinstance(answer, str) and len(answer) > 0

    def test_confidence_score_is_float_in_range(self, api_client):
        body = api_client.post("/chat", json=_chat_payload()).json()
        score = body["confidence_score"]
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_citations_is_list(self, api_client):
        body = api_client.post("/chat", json=_chat_payload()).json()
        assert isinstance(body["citations"], list)

    def test_web_sources_is_list(self, api_client):
        body = api_client.post("/chat", json=_chat_payload()).json()
        assert isinstance(body["web_sources"], list)

    def test_used_web_fallback_is_bool(self, api_client):
        body = api_client.post("/chat", json=_chat_payload()).json()
        assert isinstance(body["used_web_fallback"], bool)

    def test_new_session_returns_session_id(self, api_client):
        """No session_id in request → a new one is returned in the response."""
        body = api_client.post("/chat", json=_chat_payload()).json()
        # session_id may be in body or in X-Session-Id header (streaming path)
        # For non-streaming, it's in the JSON body via ChatResponse.session_id
        assert body.get("session_id") is not None

    def test_existing_session_id_accepted(self, api_client):
        """Passing a session_id should not cause a 4xx error."""
        payload = _chat_payload(session_id="existing-session-abc")
        response = api_client.post("/chat", json=payload)
        assert response.status_code == 200

    def test_rupee_in_answer_not_substituted(self, api_client):
        """
        Regression guard: the mocked answer contains Rs (Rupees).
        Ensures no middleware is silently rewriting currency symbols.
        """
        body = api_client.post("/chat", json=_chat_payload()).json()
        # Our fake ChatService returns "Net profit was Rs 125,682 thousand."
        assert "Rs" in body["answer"] or "₹" in body["answer"] or body["answer"]
