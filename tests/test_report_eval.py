"""Tests for the consolidated campus KB evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

from report_eval import build_report, render_markdown_report


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_render_markdown_report_includes_eval_and_ablation_tables():
    report = render_markdown_report(
        {
            "dev": {
                "n_questions": 52,
                "summary": {
                    "citation_hit_rate": 1.0,
                    "citation_recall_at_k": 0.9524,
                    "citation_mrr": 0.9378,
                    "refusal_accuracy": 1.0,
                    "false_refusal_rate": 0.0,
                },
            },
            "test": {
                "n_questions": 128,
                "summary": {
                    "citation_hit_rate": 1.0,
                    "citation_recall_at_k": 0.9826,
                    "citation_mrr": 0.9591,
                    "refusal_accuracy": 1.0,
                    "false_refusal_rate": 0.0,
                },
            },
        },
        {
            "split": "test",
            "n_questions": 128,
            "rows": [
                {
                    "name": "dense_only",
                    "overall": {
                        "citation_recall_at_8": 0.9302,
                        "citation_mrr": 0.8321,
                        "refusal_accuracy": 0.8333,
                    },
                },
                {
                    "name": "full",
                    "overall": {
                        "citation_recall_at_8": 0.9826,
                        "citation_mrr": 0.9591,
                        "refusal_accuracy": 1.0,
                    },
                },
            ],
        },
        {
            "split": "dev",
            "val_buckets": [0, 1, 2],
            "selection_split": "dev",
            "best_on_dev": {
                "refusal_ce_threshold": -0.25,
                "dev": {
                    "citation_hit_rate": 1.0,
                    "refusal_accuracy": 1.0,
                    "false_refusal_rate": 0.0,
                },
                "test": {
                    "citation_hit_rate": 1.0,
                    "refusal_accuracy": 1.0,
                    "false_refusal_rate": 0.0,
                },
            },
        },
    )

    assert "dev 52" in report
    assert "test 128" in report
    assert "dense_only" in report
    assert "-0.25" in report
    assert "Selection split: `dev`" in report


def test_build_report_returns_rendered_text_and_payload(tmp_path):
    dev_path = _write_json(
        tmp_path / "dev.json",
        {
            "n_questions": 52,
            "summary": {
                "citation_hit_rate": 1.0,
                "citation_recall_at_k": 0.9524,
                "citation_mrr": 0.9378,
                "refusal_accuracy": 1.0,
                "false_refusal_rate": 0.0,
            },
        },
    )
    test_path = _write_json(
        tmp_path / "test.json",
        {
            "n_questions": 128,
            "summary": {
                "citation_hit_rate": 1.0,
                "citation_recall_at_k": 0.9826,
                "citation_mrr": 0.9591,
                "refusal_accuracy": 1.0,
                "false_refusal_rate": 0.0,
            },
        },
    )
    ablation_path = _write_json(
        tmp_path / "ablation.json",
        {
            "split": "test",
            "n_questions": 128,
            "rows": [
                {
                    "name": "full",
                    "overall": {
                        "citation_recall_at_8": 0.9826,
                        "citation_mrr": 0.9591,
                        "refusal_accuracy": 1.0,
                    },
                }
            ],
        },
    )
    threshold_path = _write_json(
        tmp_path / "threshold.json",
        {
            "split": "full",
            "val_buckets": [0, 1, 2],
            "selection_split": "dev",
            "best_on_dev": {
                "refusal_ce_threshold": -0.25,
                "dev": {
                    "citation_hit_rate": 1.0,
                    "refusal_accuracy": 1.0,
                    "false_refusal_rate": 0.0,
                },
                "test": {
                    "citation_hit_rate": 1.0,
                    "refusal_accuracy": 1.0,
                    "false_refusal_rate": 0.0,
                },
            },
            "grid": [],
        },
    )

    payload = build_report(dev_path, test_path, ablation_path, threshold_path)

    assert payload["splits"]["dev"]["n_questions"] == 52
    assert payload["splits"]["test"]["summary"]["citation_recall_at_k"] == 0.9826
    assert payload["ablation"]["split"] == "test"
    assert payload["threshold"]["best_on_dev"]["refusal_ce_threshold"] == -0.25
    assert payload["markdown"].startswith("# Campus KB Evaluation Report")
