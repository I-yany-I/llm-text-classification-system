"""Tests for reproducible evaluation configuration."""

from unittest.mock import MagicMock

from evaluate_campus_kb import evaluate_cases


def test_evaluate_cases_passes_top_k_and_preserves_response_diagnostics():
    rag = MagicMock()
    rag.ask.return_value = {
        "status": "answered",
        "refusal_reason": None,
        "search_query": "标准化问题",
        "answer": "答案",
        "citations": [{"doc_id": "doc-1"}],
    }
    questions = [
        {
            "id": "q-1",
            "question": "口语问题",
            "expected_doc_ids": ["doc-1"],
            "should_refuse": False,
            "category": "it",
        }
    ]

    results = evaluate_cases(rag, questions, top_k=8)

    rag.ask.assert_called_once_with("口语问题", top_k=8)
    assert results[0]["status"] == "answered"
    assert results[0]["search_query"] == "标准化问题"
    assert results[0]["refusal_reason"] is None
