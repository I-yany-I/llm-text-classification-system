"""Build a compact report from campus KB evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "-"


def _render_eval_section(dev_test_payload: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "## Evaluation",
        "",
        f"- dev {dev_test_payload.get('dev', {}).get('n_questions', '-') } questions",
        f"- test {dev_test_payload.get('test', {}).get('n_questions', '-') } questions",
        "",
        "| Split | Questions | Hit Rate | Recall@K | MRR | Refusal Acc. | False Refusal |",
        "|------|----------:|---------:|---------:|----:|-------------:|--------------:|",
    ]
    for split in ("dev", "test"):
        payload = dev_test_payload.get(split, {})
        summary = payload.get("summary", {})
        lines.append(
            "| {split} | {n} | {hit} | {recall} | {mrr} | {refusal} | {false_refusal} |".format(
                split=split,
                n=payload.get("n_questions", "-"),
                hit=_pct(summary.get("citation_hit_rate")),
                recall=_pct(summary.get("citation_recall_at_k")),
                mrr=_pct(summary.get("citation_mrr")),
                refusal=_pct(summary.get("refusal_accuracy")),
                false_refusal=_pct(summary.get("false_refusal_rate")),
            )
        )
    return "\n".join(lines)


def _render_ablation_section(ablation_payload: Dict[str, Any] | None) -> str:
    if not ablation_payload:
        return ""

    lines = [
        "## Ablation",
        "",
        f"- Split: `{ablation_payload.get('split', '-')}`",
        f"- Questions: `{ablation_payload.get('n_questions', '-')}`",
        "",
        "| Variant | Recall@8 | MRR | Refusal Acc. |",
        "|--------|---------:|----:|-------------:|",
    ]
    for row in ablation_payload.get("rows", []):
        overall = row.get("overall", {})
        lines.append(
            "| {name} | {recall} | {mrr} | {refusal} |".format(
                name=row.get("name", "-"),
                recall=_pct(overall.get("citation_recall_at_8")),
                mrr=_pct(overall.get("citation_mrr")),
                refusal=_pct(overall.get("refusal_accuracy")),
            )
        )
    return "\n".join(lines)


def _render_threshold_section(threshold_payload: Dict[str, Any] | None) -> str:
    if not threshold_payload:
        return ""

    best = threshold_payload.get("best_on_dev") or threshold_payload.get("best_on_val", {})
    lines = [
        "## Threshold Search",
        "",
        f"- Split: `{threshold_payload.get('split', '-')}`",
        f"- Selection split: `{threshold_payload.get('selection_split', 'dev')}`",
        f"- Dev buckets: `{threshold_payload.get('val_buckets', '-')}`",
        f"- Best CE threshold: `{best.get('refusal_ce_threshold', '-')}`",
        "",
        "| Set | Hit Rate | Recall@K | Refusal Acc. | False Refusal |",
        "|----|---------:|---------:|-------------:|--------------:|",
    ]
    for key in ("dev", "test", "all"):
        # Read the legacy `val` key so older generated artifacts remain renderable.
        summary = best.get(key, {})
        if key == "dev" and not summary:
            summary = best.get("val", {})
        lines.append(
            "| {key} | {hit} | {recall} | {refusal} | {false_refusal} |".format(
                key=key,
                hit=_pct(summary.get("citation_hit_rate")),
                recall=_pct(summary.get("citation_recall_at_k")),
                refusal=_pct(summary.get("refusal_accuracy")),
                false_refusal=_pct(summary.get("false_refusal_rate")),
            )
        )
    return "\n".join(lines)


def render_markdown_report(
    dev_test_payload: Dict[str, Dict[str, Any]],
    ablation_payload: Dict[str, Any] | None,
    threshold_payload: Dict[str, Any] | None,
) -> str:
    sections = ["# Campus KB Evaluation Report", "", _render_eval_section(dev_test_payload)]
    ablation_section = _render_ablation_section(ablation_payload)
    if ablation_section:
        sections.extend(["", ablation_section])
    threshold_section = _render_threshold_section(threshold_payload)
    if threshold_section:
        sections.extend(["", threshold_section])
    return "\n".join(sections).strip() + "\n"


def build_report(
    dev_path: str | Path,
    test_path: str | Path,
    ablation_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
) -> Dict[str, Any]:
    splits = {
        "dev": _load_json(dev_path),
        "test": _load_json(test_path),
    }
    ablation_payload = _load_json(ablation_path) if ablation_path else None
    threshold_payload = _load_json(threshold_path) if threshold_path else None
    markdown = render_markdown_report(splits, ablation_payload, threshold_payload)
    return {
        "splits": splits,
        "ablation": ablation_payload,
        "threshold": threshold_payload,
        "markdown": markdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render campus KB evaluation report.")
    parser.add_argument("--dev", required=True, help="dev split evaluation JSON")
    parser.add_argument("--test", required=True, help="test split evaluation JSON")
    parser.add_argument("--ablation", default=None, help="ablation JSON")
    parser.add_argument("--threshold", default=None, help="threshold search JSON")
    parser.add_argument(
        "--output",
        default="artifacts/reports/campus_kb_eval_report.md",
        help="markdown report output",
    )
    parser.add_argument(
        "--json-output",
        default="artifacts/reports/campus_kb_eval_report.json",
        help="structured JSON output",
    )
    args = parser.parse_args()

    payload = build_report(args.dev, args.test, args.ablation, args.threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload["markdown"], encoding="utf-8")
    json_output_path = Path(args.json_output)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(payload["markdown"], end="")
    print(f"\nsaved {output_path}")
    print(f"saved {json_output_path}")


if __name__ == "__main__":
    main()
