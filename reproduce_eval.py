"""One-command reproduction flow for the campus KB RAG project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from evaluate_ablation import run_ablation
from evaluate_campus_kb import (
    evaluate_cases,
    load_eval_questions,
    per_category_summary,
    resolve_questions_source,
)
from evaluation.metrics import summarize
from report_eval import build_report
from src.campus_kb_rag import CampusKBRAG
from src.campus_kb_rag.config import resolve_path
from tune_refusal_threshold import run_threshold_search


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _evaluate_split(rag: CampusKBRAG, split: str, top_k: int | None = None) -> Dict[str, Any]:
    questions = load_eval_questions(resolve_questions_source(None, split))
    results = evaluate_cases(rag, questions, top_k=top_k)
    return {
        "summary": summarize(results),
        "by_category": per_category_summary(results),
        "n_questions": len(results),
        "results": results,
    }


def run_reproduction(
    output_dir: str | Path,
    config_path: str | None = None,
    force_index: bool = False,
    top_k: int | None = None,
) -> Dict[str, Any]:
    root = resolve_path(output_dir)
    predictions_dir = root / "predictions"
    reports_dir = root / "reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rag = CampusKBRAG(config_path=config_path)
    rag.build_index(force=force_index)

    dev_payload = _evaluate_split(rag, "dev", top_k=top_k)
    test_payload = _evaluate_split(rag, "test", top_k=top_k)
    dev_path = _write_json(predictions_dir / "campus_kb_eval_dev.json", dev_payload)
    test_path = _write_json(predictions_dir / "campus_kb_eval_test.json", test_payload)

    ablation_payload = run_ablation(config_path=config_path, questions_path=None, split="test")
    ablation_path = _write_json(predictions_dir / "campus_kb_ablation_test.json", ablation_payload)

    threshold_payload = run_threshold_search(
        config_path=config_path,
        questions_path=None,
        split="full",
        val_buckets=[0, 1, 2],
    )
    threshold_path = _write_json(predictions_dir / "threshold_search.json", threshold_payload)

    report_payload = build_report(dev_path, test_path, ablation_path, threshold_path)
    report_markdown_path = reports_dir / "campus_kb_eval_report.md"
    report_json_path = reports_dir / "campus_kb_eval_report.json"
    report_markdown_path.write_text(report_payload["markdown"], encoding="utf-8")
    report_json_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": root,
        "eval_dev_path": dev_path,
        "eval_test_path": test_path,
        "ablation_path": ablation_path,
        "threshold_path": threshold_path,
        "report_markdown_path": report_markdown_path,
        "report_json_path": report_json_path,
        "report_payload": report_payload,
        "dev_payload": dev_payload,
        "test_payload": test_payload,
        "ablation_payload": ablation_payload,
        "threshold_payload": threshold_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full campus KB reproduction flow.")
    parser.add_argument("--config", default=None, help="主配置文件（可选）")
    parser.add_argument("--output-dir", default="artifacts/reproduction", help="输出目录")
    parser.add_argument("--force-index", action="store_true", help="强制重建索引")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="评估时显式指定引用数；省略则使用配置默认值",
    )
    args = parser.parse_args()

    payload = run_reproduction(
        output_dir=args.output_dir,
        config_path=args.config,
        force_index=args.force_index,
        top_k=args.top_k,
    )
    print(payload["report_payload"]["markdown"], end="")
    print(f"\nsaved {payload['report_markdown_path']}")
    print(f"saved {payload['report_json_path']}")


if __name__ == "__main__":
    main()
