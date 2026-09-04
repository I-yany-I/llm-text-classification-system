"""Evaluate retrieval and refusal behavior for the campus KB RAG system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from evaluation.metrics import summarize
from src.campus_kb_rag import CampusKBRAG
from src.campus_kb_rag.config import resolve_path


DEFAULT_EVAL_QUESTIONS = "data/campus_kb/eval_questions.jsonl"
DEFAULT_EVAL_SPLITS = {
    "full": "data/campus_kb/eval_questions.jsonl",
    "dev": "data/campus_kb/eval_questions_dev.jsonl",
    "test": "data/campus_kb/eval_questions_test.jsonl",
}


def resolve_questions_source(questions: str | None, split: str) -> str:
    if questions:
        return questions
    return DEFAULT_EVAL_SPLITS[split]


def load_eval_questions(path: str | Path) -> List[Dict]:
    resolved = resolve_path(path)
    questions = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions.append(
                {
                    "question": item["question"],
                    "expected_doc_ids": item.get("expected_doc_ids", []),
                    "should_refuse": item.get("should_refuse", False),
                    "category": item.get("category", ""),
                    "id": item.get("id", ""),
                }
            )
    return questions


def per_category_summary(results: List[Dict]) -> Dict:
    cats: Dict[str, List[Dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r)
    return {cat: summarize(items) for cat, items in cats.items()}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("top-k must be a positive integer")
    return parsed


def evaluate_cases(
    rag: CampusKBRAG, questions: List[Dict], top_k: int | None = None
) -> List[Dict]:
    results = []
    for case in questions:
        response = (
            rag.ask(case["question"], top_k=top_k)
            if top_k is not None
            else rag.ask(case["question"])
        )
        results.append(
            {
                "id": case.get("id", ""),
                "category": case.get("category", ""),
                "question": case["question"],
                "expected_doc_ids": case["expected_doc_ids"],
                "should_refuse": case["should_refuse"],
                "status": response.get("status"),
                "search_query": response.get("search_query"),
                "refusal_reason": response.get("refusal_reason"),
                "answer": response["answer"],
                "citations": response["citations"],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NJU campus KB RAG.")
    parser.add_argument("--config", default=None, help="主配置文件（可选；省略则用项目默认）")
    parser.add_argument(
        "--questions",
        default=None,
        help="评估问题集（可选；省略则根据 --split 选择内置集）",
    )
    parser.add_argument(
        "--split",
        choices=sorted(DEFAULT_EVAL_SPLITS),
        default="full",
        help="内置评估切分：full、dev 或 test",
    )
    parser.add_argument(
        "--output",
        default="artifacts/predictions/campus_kb_eval.json",
        help="评测汇总 JSON 的输出位置",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=None,
        help="显式指定返回引用数量；省略则使用配置中的 retrieval.final_top_k",
    )
    args = parser.parse_args()

    rag = CampusKBRAG(config_path=args.config)
    rag.build_index(force=False)

    questions = load_eval_questions(resolve_questions_source(args.questions, args.split))
    print(f"Loaded {len(questions)} evaluation questions.")

    results = evaluate_cases(rag, questions, top_k=args.top_k)

    overall = summarize(results)
    by_category = per_category_summary(results)

    payload = {
        "summary": overall,
        "by_category": by_category,
        "n_questions": len(results),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Overall ===")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== By Category ===")
    print(json.dumps(by_category, ensure_ascii=False, indent=2))
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
