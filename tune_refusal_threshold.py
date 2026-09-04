"""Search the Cross-Encoder refusal threshold without touching the test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.metrics import summarize
from evaluate_campus_kb import load_eval_questions, resolve_questions_source
from src.campus_kb_rag import CampusKBRAG


def _bucket(case_id: str, buckets: int = 10) -> int:
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def _parse_val_buckets(val_buckets: str | Sequence[int] | Iterable[int]) -> list[int]:
    if isinstance(val_buckets, str):
        return sorted({int(x) for x in val_buckets.split(",") if x.strip() != ""})
    return sorted({int(x) for x in val_buckets})


def _eval_split(source_split: str, case_id: str, dev_buckets: set[int]) -> str:
    if source_split == "full":
        return "dev" if _bucket(case_id) in dev_buckets else "test"
    return source_split


def _replay_refusal_gate(item: dict, threshold: float, rag: CampusKBRAG) -> list[dict]:
    """Apply the same evidence gate as CampusKBRAG._filter_low_confidence."""
    retrieved = item.get("retrieved") or []
    if not retrieved or item.get("refusal_reason") == "out_of_scope":
        return []

    prompt_cfg = rag.config.get("prompt", {})
    refusal_doc_ids = set(prompt_cfg.get("refusal_doc_ids", []))
    if retrieved[0].get("doc_id") in refusal_doc_ids:
        return []
    filtered = [item for item in retrieved if item.get("doc_id") not in refusal_doc_ids]
    if not filtered:
        return []

    top = filtered[0]
    ce_cfg = rag.config.get("retrieval", {}).get("cross_encoder", {})
    if ce_cfg.get("enabled") and top.get("cross_encoder_score") is not None:
        if float(top["cross_encoder_score"]) >= threshold:
            return filtered
        fallback_threshold = float(
            prompt_cfg.get("refusal_dense_fallback_threshold", 0.43)
        )
        fallback_pool = filtered[:3]
        best_dense = max(
            float(candidate.get("dense_score") or candidate.get("score") or 0.0)
            for candidate in fallback_pool
        )
        return filtered if best_dense >= fallback_threshold else []

    dense_threshold = float(prompt_cfg.get("refusal_threshold", 0.18))
    top_score = float(top.get("dense_score", top.get("score", 0.0)))
    return filtered if top_score >= dense_threshold else []


def run_threshold_search(
    config_path: str | None = None,
    questions_path: str | None = None,
    split: str = "full",
    val_buckets: str | Sequence[int] | Iterable[int] = "0,1,2",
) -> dict:
    rag = CampusKBRAG(config_path=config_path)
    rag.build_index(force=False)
    questions = load_eval_questions(resolve_questions_source(questions_path, split))
    sorted_val_buckets = _parse_val_buckets(val_buckets)
    val_bucket_set = set(sorted_val_buckets)

    scored = []
    for case in questions:
        response = rag.ask(case["question"])
        retrieved = response.get("retrieved") or []
        top = retrieved[0] if retrieved else {}
        scored.append(
            {
                **case,
                "retrieved": retrieved,
                "answer": response.get("answer", ""),
                "ce": None if top.get("cross_encoder_score") is None else float(top["cross_encoder_score"]),
                "dense": None if top.get("dense_score") is None else float(top["dense_score"]),
                "top_doc": top.get("doc_id"),
                "refusal_reason": response.get("refusal_reason"),
                "eval_split": _eval_split(split, str(case.get("id", "")), val_bucket_set),
            }
        )

    thresholds = [round(x * 0.25 - 4.0, 2) for x in range(0, 25)]
    rows = []
    for threshold in thresholds:

        def apply(split_name: str) -> dict:
            subset = []
            for item in scored:
                if item["eval_split"] != split_name:
                    continue
                citations = _replay_refusal_gate(item, threshold, rag)
                answer = item["answer"] if citations else "当前知识库没有足够依据"
                subset.append({**item, "citations": citations, "answer": answer})
            return summarize(subset)

        rows.append(
            {
                "refusal_ce_threshold": threshold,
                "dev": apply("dev"),
                "test": apply("test"),
                "all": None,
            }
        )
        all_items = []
        for item in scored:
            citations = _replay_refusal_gate(item, threshold, rag)
            answer = item["answer"] if citations else "当前知识库没有足够依据"
            all_items.append({**item, "citations": citations, "answer": answer})
        rows[-1]["all"] = summarize(all_items)

    def objective(summary: dict) -> float:
        return (
            float(summary.get("citation_hit_rate", 0))
            + float(summary.get("refusal_accuracy", 0))
            - float(summary.get("false_refusal_rate", 0))
        )

    selection_split = "dev" if any(row["dev"]["n_total"] for row in rows) else "all"
    # Prefer the stricter threshold when several candidates have the same
    # validation objective, reducing unnecessary evidence acceptance.
    best = max(
        rows,
        key=lambda row: (objective(row[selection_split]), row["refusal_ce_threshold"]),
    )
    return {
        "split": split,
        "val_buckets": sorted_val_buckets,
        "selection_split": selection_split,
        "n_questions": len(questions),
        "best_on_dev": best,
        # Keep the old key readable for already-written report tooling.
        "best_on_val": best,
        "grid": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="主配置文件（可选）")
    parser.add_argument("--questions", default=None)
    parser.add_argument(
        "--split",
        choices=("full", "dev", "test"),
        default="full",
        help="内置评估切分：推荐 full，以 dev 桶选阈值并在 test 桶报告；也可直接指定 dev/test",
    )
    parser.add_argument("--output", default="artifacts/predictions/threshold_search.json")
    parser.add_argument("--val-buckets", default="0,1,2", help="full 模式下分配给 dev 的 hash buckets")
    args = parser.parse_args()

    payload = run_threshold_search(args.config, args.questions, args.split, args.val_buckets)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_on_val"], ensure_ascii=False, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
