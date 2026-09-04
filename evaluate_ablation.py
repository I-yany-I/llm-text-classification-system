"""Run retrieval ablations on the campus KB evaluation split."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from evaluate_campus_kb import (
    evaluate_cases,
    load_eval_questions,
    per_category_summary,
    resolve_questions_source,
)
from evaluation.metrics import summarize
from src.campus_kb_rag import CampusKBRAG
from src.campus_kb_rag.config import DEFAULT_CONFIG_PATH, resolve_path


def make_ablation_configs(base_cfg: Dict) -> List[Tuple[str, Dict]]:
    def clone() -> Dict:
        return copy.deepcopy(base_cfg)

    variants = []

    dense_only = clone()
    dense_only["retrieval"]["hybrid_enabled"] = False
    dense_only["retrieval"]["query_rewrite"] = False
    dense_only["retrieval"]["cross_encoder"]["enabled"] = False
    variants.append(("dense_only", dense_only))

    hybrid = clone()
    hybrid["retrieval"]["hybrid_enabled"] = True
    hybrid["retrieval"]["query_rewrite"] = False
    hybrid["retrieval"]["cross_encoder"]["enabled"] = False
    variants.append(("hybrid", hybrid))

    hybrid_rewrite = clone()
    hybrid_rewrite["retrieval"]["hybrid_enabled"] = True
    hybrid_rewrite["retrieval"]["query_rewrite"] = True
    hybrid_rewrite["retrieval"]["cross_encoder"]["enabled"] = False
    variants.append(("hybrid_rewrite", hybrid_rewrite))

    full = clone()
    full["retrieval"]["hybrid_enabled"] = True
    full["retrieval"]["query_rewrite"] = True
    full["retrieval"]["cross_encoder"]["enabled"] = True
    variants.append(("full", full))

    return variants


def _load_base_config(path: str | Path | None) -> Dict:
    config_path = resolve_path(path or DEFAULT_CONFIG_PATH)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _evaluate_variant(config: Dict, questions: List[Dict]) -> Dict:
    with tempfile.TemporaryDirectory(prefix="campus-kb-ablation-") as tmpdir:
        temp_path = Path(tmpdir) / "variant.yaml"
        temp_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        rag = CampusKBRAG(config_path=str(temp_path))
        rag.build_index(force=False)
        results = evaluate_cases(rag, questions)
        return {
            "overall": summarize(results),
            "by_category": per_category_summary(results),
        }


def run_ablation(
    config_path: str | Path | None,
    questions_path: str | Path | None,
    split: str,
) -> Dict:
    base_cfg = _load_base_config(config_path)
    questions = load_eval_questions(resolve_questions_source(questions_path, split))
    rows = []
    for name, cfg in make_ablation_configs(base_cfg):
        metrics = _evaluate_variant(cfg, questions)
        rows.append({"name": name, **metrics})
    return {
        "split": split,
        "n_questions": len(questions),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval ablation on campus KB.")
    parser.add_argument("--config", default=None, help="主配置文件（可选）")
    parser.add_argument(
        "--questions",
        default=None,
        help="问题集路径（可选；省略则根据 --split 选择内置集）",
    )
    parser.add_argument(
        "--split",
        choices=("full", "dev", "test"),
        default="test",
        help="内置评估切分：full、dev 或 test",
    )
    parser.add_argument(
        "--output",
        default="artifacts/predictions/campus_kb_ablation.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    payload = run_ablation(args.config, args.questions, args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
