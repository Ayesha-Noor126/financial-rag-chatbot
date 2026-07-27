"""
Ask a question, get it scored immediately -- no pre-built question list.

Usage:
    python rag_eval.py

It will prompt you for a question, then optionally for the correct
answer. Faithfulness and Answer Relevancy always run (they only need the
question + answer + retrieved context). Contextual Precision and
Contextual Recall only run if you gave a correct answer -- those two
metrics are DEFINED as "did retrieval find what the correct answer
needed", so they can't be computed without a reference to compare against.
That's inherent to what the metrics mean, not something this script can
work around.

Matches your real /debug/answer route: GET /debug/answer?q=... returning
a ChatResponse JSON body (answer, citations, confidence_score, etc).

Setup:
    pip install deepeval httpx
    export GROQ_API_KEY=your_key_here
    Start your app first (e.g. uvicorn app.main:app --reload)

Consistency: the judge is pinned to temperature=0 + a fixed seed, so
asking the exact same question (with the exact same correct answer, if
given) scores the same way every time.
"""

import asyncio
import os

import httpx
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from openai import OpenAI

from app.core.config import get_settings
from app.services.rag.metadata.store import MetadataStore

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")
DEBUG_ANSWER_PATH = "/debug/answer"

JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "llama-3.3-70b-versatile")
JUDGE_SEED = 42


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            seed=JUDGE_SEED,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return JUDGE_MODEL


async def ask_and_evaluate(question: str, expected: str | None) -> None:
    settings = get_settings()
    store = MetadataStore(settings)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{APP_BASE_URL}{DEBUG_ANSWER_PATH}",
            params={"q": question},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

    chunk_ids = [c["chunk_id"] for c in data.get("citations", [])]
    chunks = store.get_chunks_by_ids(chunk_ids) if chunk_ids else []
    contexts = [c.text for c in chunks]
    answer = data["answer"]

    print(f"\nAnswer: {answer}\n")

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts,
    )

    judge = GroqJudge()
    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
    ]
    if expected:
        metrics.append(ContextualPrecisionMetric(model=judge, threshold=0.7))
        metrics.append(ContextualRecallMetric(model=judge, threshold=0.7))
    else:
        print(
            "(No correct answer given -- skipping Contextual Precision/Recall, "
            "since those need one to compare retrieval against.)\n"
        )

    for metric in metrics:
        metric.measure(test_case)
        print(f"{metric.__class__.__name__:<28} {metric.score:.2f}   reason: {metric.reason}")


def main():
    question = input("Ask your question: ").strip()
    expected = input("Correct answer (optional, press Enter to skip): ").strip() or None
    asyncio.run(ask_and_evaluate(question, expected))


if __name__ == "__main__":
    main()