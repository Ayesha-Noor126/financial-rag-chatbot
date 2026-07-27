from app.services.rag.evaluation.eval_service import RAGEvaluationService


def test_evaluation_service_returns_all_metrics_without_expected_answer():
    service = RAGEvaluationService()
    metrics = service.evaluate(
        question="What was the revenue?",
        answer="Revenue was $10 million.",
        contexts=["The company's revenue was $10 million last year."],
    )

    assert metrics["faithfulness"] >= 0.0
    assert metrics["answer_relevancy"] >= 0.0
    assert metrics["precision"] >= 0.0
    assert metrics["recall"] >= 0.0
    assert metrics["faithfulness"] <= 1.0
    assert metrics["answer_relevancy"] <= 1.0
    assert metrics["precision"] <= 1.0
    assert metrics["recall"] <= 1.0
