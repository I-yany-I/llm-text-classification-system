"""Tests for the campus KB one-click reproduction flow."""

from __future__ import annotations

import json
from pathlib import Path

import reproduce_eval


def test_run_reproduction_writes_all_artifacts_in_order(tmp_path, monkeypatch):
    calls = []

    class FakeRAG:
        def __init__(self, config_path=None):
            calls.append(("rag_init", config_path))

        def build_index(self, force=False):
            calls.append(("build_index", force))

    def fake_load_eval_questions(path):
        calls.append(("load_questions", path))
        return [{"id": "q-1", "question": "问题", "expected_doc_ids": [], "should_refuse": False, "category": "it"}]

    def fake_evaluate_cases(rag, questions, top_k=None):
        calls.append(("evaluate_cases", len(questions), top_k))
        return [{"should_refuse": False, "expected_doc_ids": [], "citations": []}]

    def fake_run_ablation(config_path, questions_path, split):
        calls.append(("run_ablation", config_path, questions_path, split))
        return {"split": split, "n_questions": 1, "rows": [{"name": "full", "overall": {}}]}

    def fake_run_threshold_search(config_path, questions_path, split, val_buckets):
        calls.append(("run_threshold_search", config_path, questions_path, split, val_buckets))
        return {"split": split, "val_buckets": [0, 1, 2], "best_on_val": {}, "grid": []}

    def fake_build_report(dev_path, test_path, ablation_path, threshold_path):
        calls.append(("build_report", Path(dev_path).name, Path(test_path).name, Path(ablation_path).name, Path(threshold_path).name))
        return {"markdown": "# Campus KB Evaluation Report\n", "splits": {}}

    monkeypatch.setattr(reproduce_eval, "CampusKBRAG", FakeRAG)
    monkeypatch.setattr(reproduce_eval, "load_eval_questions", fake_load_eval_questions)
    monkeypatch.setattr(reproduce_eval, "evaluate_cases", fake_evaluate_cases)
    monkeypatch.setattr(reproduce_eval, "summarize", lambda results: {"n_total": len(results)})
    monkeypatch.setattr(reproduce_eval, "per_category_summary", lambda results: {"it": {"n_total": len(results)}})
    monkeypatch.setattr(reproduce_eval, "run_ablation", fake_run_ablation)
    monkeypatch.setattr(reproduce_eval, "run_threshold_search", fake_run_threshold_search)
    monkeypatch.setattr(reproduce_eval, "build_report", fake_build_report)

    payload = reproduce_eval.run_reproduction(
        output_dir=tmp_path,
        config_path="custom.yaml",
        force_index=True,
        top_k=8,
    )

    assert calls[0] == ("rag_init", "custom.yaml")
    assert calls[1] == ("build_index", True)
    assert calls[2][0] == "load_questions"
    assert calls[3][0] == "evaluate_cases"
    assert calls[4][0] == "load_questions"
    assert calls[5][0] == "evaluate_cases"
    assert calls[6] == ("run_ablation", "custom.yaml", None, "test")
    assert calls[7] == ("run_threshold_search", "custom.yaml", None, "full", [0, 1, 2])
    assert calls[8][0] == "build_report"

    assert payload["report_markdown_path"].exists()
    assert payload["report_json_path"].exists()
    assert payload["eval_dev_path"].exists()
    assert payload["eval_test_path"].exists()
    assert payload["ablation_path"].exists()
    assert payload["threshold_path"].exists()

    report_text = payload["report_markdown_path"].read_text(encoding="utf-8")
    assert report_text.startswith("# Campus KB Evaluation Report")

    report_json = json.loads(payload["report_json_path"].read_text(encoding="utf-8"))
    assert report_json["splits"] == {}
