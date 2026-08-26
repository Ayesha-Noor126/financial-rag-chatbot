from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from app.core.config import get_settings


class RAGEvaluationService:
    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.enabled = os.getenv("ENABLE_LIVE_EVAL", "false").lower() in ("true", "1")
        self.model = model or os.getenv("DEEPEVAL_JUDGE_MODEL", settings.llm_model)
        api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or settings.llm_api_key
        base_url = os.environ.get("GROQ_BASE_URL") or settings.llm_base_url
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = None

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        expected_answer: str | None = None,
    ) -> dict[str, float]:
        if not self.enabled or not self.client or not getattr(self.client, "api_key", None):
            return self._fallback_scores()

        prompt = self._build_prompt(question, answer, contexts, expected_answer)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                seed=42,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or ""
            return self._parse_scores(content)
        except Exception:
            return self._fallback_scores()

    def _build_prompt(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        expected_answer: str | None,
    ) -> str:
        context_block = "\n\n".join(f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts))
        expected_block = f"\nExpected answer: {expected_answer}" if expected_answer else ""
        return (
            "You are scoring a RAG answer. Return ONLY a JSON object with keys "
            "faithfulness, answer_relevancy, precision, recall. "
            "Scores must be numbers between 0 and 100. "
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Contexts:\n{context_block}\n"
            f"{expected_block}\n"
            "Assess faithfulness to the provided contexts, answer relevancy to the question, "
            "and retrieval precision/recall against the expected answer when available."
        )

    def _parse_scores(self, content: str) -> dict[str, float]:
        try:
            parsed = self._extract_json(content)
            # Accept 0-1 or 0-100 from the model; normalize to 0-100.
            def norm(key: str) -> float:
                val = float(parsed.get(key, 0.0))
                # If model returned 0-1, scale up to 0-100
                if 0.0 <= val <= 1.0:
                    return val * 100.0
                return val

            return {
                "faithfulness": norm("faithfulness"),
                "answer_relevancy": norm("answer_relevancy"),
                "precision": norm("precision"),
                "recall": norm("recall"),
            }
        except Exception:
            return self._fallback_scores()

    def _extract_json(self, content: str) -> dict[str, Any]:
        import json

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        return json.loads(content[start : end + 1])

    def _fallback_scores(self) -> dict[str, float]:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
