"""Tests for reproducible evaluation configuration."""

import json
import sys
from unittest.mock import MagicMock

from evaluate_campus_kb import evaluate_cases, resolve_questions_source


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


def test_resolve_questions_source_supports_named_splits():
    assert resolve_questions_source(None, "full") == "data/campus_kb/eval_questions.jsonl"
    assert resolve_questions_source(None, "dev") == "data/campus_kb/eval_questions_dev.jsonl"
    assert resolve_questions_source(None, "test") == "data/campus_kb/eval_questions_test.jsonl"
    assert resolve_questions_source("custom.jsonl", "dev") == "custom.jsonl"


def test_threshold_search_passes_config_path(tmp_path, monkeypatch):
    import tune_refusal_threshold as tuner

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "id": "q-1",
                "question": "问题",
                "expected_doc_ids": ["doc-1"],
                "should_refuse": False,
                "category": "it",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "threshold.json"
    fake_rag = MagicMock()
    fake_rag.ask.return_value = {
        "answer": "答案",
        "citations": [{"doc_id": "doc-1"}],
        "retrieved": [
            {
                "doc_id": "doc-1",
                "cross_encoder_score": 1.0,
                "dense_score": 0.8,
            }
        ],
    }
    rag_factory = MagicMock(return_value=fake_rag)
    monkeypatch.setattr(tuner, "CampusKBRAG", rag_factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_refusal_threshold.py",
            "--config",
            "custom.yaml",
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
        ],
    )

    tuner.main()

    rag_factory.assert_called_once_with(config_path="custom.yaml")


def test_threshold_search_uses_retrieved_candidates_when_current_gate_refuses(
    tmp_path, monkeypatch
):
    import tune_refusal_threshold as tuner

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "id": "q-1",
                "question": "问题",
                "expected_doc_ids": ["doc-1"],
                "should_refuse": False,
                "category": "it",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "threshold.json"
    fake_rag = MagicMock()
    fake_rag.ask.return_value = {
        "answer": "当前知识库没有足够依据",
        "citations": [],
        "retrieved": [
            {
                "doc_id": "doc-1",
                "cross_encoder_score": -1.0,
                "dense_score": 0.8,
            }
        ],
    }
    monkeypatch.setattr(tuner, "CampusKBRAG", MagicMock(return_value=fake_rag))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_refusal_threshold.py",
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
        ],
    )

    tuner.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    threshold_minus_two = next(
        row for row in payload["grid"] if row["refusal_ce_threshold"] == -2.0
    )
    assert threshold_minus_two["all"]["citation_hit_rate"] == 1.0
    thresholds = {row["refusal_ce_threshold"] for row in payload["grid"]}
    assert -4.0 in thresholds
    assert 2.0 in thresholds


def test_threshold_search_records_split_metadata(tmp_path, monkeypatch):
    import tune_refusal_threshold as tuner

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "id": "q-1",
                "question": "问题",
                "expected_doc_ids": ["doc-1"],
                "should_refuse": False,
                "category": "it",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "threshold.json"
    fake_rag = MagicMock()
    fake_rag.ask.return_value = {
        "answer": "答案",
        "citations": [{"doc_id": "doc-1"}],
        "retrieved": [
            {
                "doc_id": "doc-1",
                "cross_encoder_score": 1.0,
                "dense_score": 0.8,
            }
        ],
    }
    monkeypatch.setattr(tuner, "CampusKBRAG", MagicMock(return_value=fake_rag))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_refusal_threshold.py",
            "--questions",
            str(questions_path),
            "--output",
            str(output_path),
        ],
    )

    tuner.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["split"] == "full"
    assert payload["val_buckets"] == [0, 1, 2]
    assert payload["best_on_val"]["refusal_ce_threshold"] == 1.0
    assert payload["grid"][0]["all"]["n_total"] == 1
    assert "dev" in payload["grid"][0]


def test_threshold_search_replays_real_refusal_gate(tmp_path, monkeypatch):
    import tune_refusal_threshold as tuner

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {
                    "id": "q-answerable",
                    "question": "问题",
                    "expected_doc_ids": ["doc-1"],
                    "should_refuse": False,
                    "category": "it",
                },
                {
                    "id": "q-sentinel",
                    "question": "问题",
                    "expected_doc_ids": [],
                    "should_refuse": True,
                    "category": "out_of_scope",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "threshold.json"
    fake_rag = MagicMock()
    fake_rag.config = {
        "prompt": {
            "refusal_doc_ids": ["nju-support-unknown"],
            "refusal_dense_fallback_threshold": 0.43,
        },
        "retrieval": {"cross_encoder": {"enabled": True}},
    }
    fake_rag.ask.side_effect = [
        {
            "answer": "答案",
            "citations": [{"doc_id": "doc-1"}],
            "retrieved": [
                {"doc_id": "doc-1", "cross_encoder_score": -1.0, "dense_score": 0.2},
                {"doc_id": "doc-2", "cross_encoder_score": -2.0, "dense_score": 0.5},
            ],
        },
        {
            "answer": "当前知识库没有足够依据",
            "citations": [],
            "retrieved": [
                {
                    "doc_id": "nju-support-unknown",
                    "cross_encoder_score": 3.0,
                    "dense_score": 0.99,
                }
            ],
        },
    ]
    monkeypatch.setattr(tuner, "CampusKBRAG", MagicMock(return_value=fake_rag))

    payload = tuner.run_threshold_search(
        questions_path=str(questions_path), split="full", val_buckets=[]
    )

    threshold_minus_two = next(
        row for row in payload["grid"] if row["refusal_ce_threshold"] == -2.0
    )
    assert threshold_minus_two["all"]["citation_hit_rate"] == 1.0
    assert threshold_minus_two["all"]["refusal_accuracy"] == 1.0
    assert threshold_minus_two["all"]["false_refusal_rate"] == 0.0
