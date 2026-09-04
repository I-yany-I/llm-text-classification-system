# Campus RAG Evaluation Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small reporting utility that consolidates dev/test evaluation results, threshold search output, and ablation results into one reproducible Markdown/JSON report.

**Architecture:** Keep the existing evaluation and ablation scripts as the source of truth. Add a pure formatting layer that reads their JSON outputs and renders a stable summary for README, resume notes, or manual inspection. Also tighten the threshold-search output schema so its best-on-dev / test comparison is explicit and machine-readable.

**Tech Stack:** Python 3.9+, pytest, JSON, pathlib

---

### Task 1: Lock the report format with tests

**Files:**
- Create: `tests/test_report_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

from report_eval import build_report, render_markdown_report


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_render_markdown_report_includes_eval_and_ablation_tables():
    report = render_markdown_report(
        {
            "dev": {"n_questions": 52, "summary": {"citation_hit_rate": 1.0, "citation_recall_at_k": 0.9524, "citation_mrr": 0.9378, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0}},
            "test": {"n_questions": 128, "summary": {"citation_hit_rate": 1.0, "citation_recall_at_k": 0.9826, "citation_mrr": 0.9591, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0}},
        },
        {
            "split": "test",
            "n_questions": 128,
            "rows": [
                {"name": "dense_only", "overall": {"citation_recall_at_8": 0.9302, "citation_mrr": 0.8321, "refusal_accuracy": 0.8333}},
                {"name": "full", "overall": {"citation_recall_at_8": 0.9826, "citation_mrr": 0.9591, "refusal_accuracy": 1.0}},
            ],
        },
        {
            "split": "dev",
            "val_buckets": [0, 1, 2],
            "best_on_val": {
                "refusal_ce_threshold": -0.25,
                "val": {"citation_hit_rate": 1.0, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0},
                "test": {"citation_hit_rate": 1.0, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0},
            },
        },
    )

    assert "dev 52" in report
    assert "test 128" in report
    assert "dense_only" in report
    assert "-0.25" in report


def test_build_report_returns_rendered_text_and_payload(tmp_path):
    dev_path = _write_json(tmp_path / "dev.json", {"n_questions": 52, "summary": {"citation_hit_rate": 1.0, "citation_recall_at_k": 0.9524, "citation_mrr": 0.9378, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0}})
    test_path = _write_json(tmp_path / "test.json", {"n_questions": 128, "summary": {"citation_hit_rate": 1.0, "citation_recall_at_k": 0.9826, "citation_mrr": 0.9591, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0}})
    ablation_path = _write_json(tmp_path / "ablation.json", {"split": "test", "n_questions": 128, "rows": [{"name": "full", "overall": {"citation_recall_at_8": 0.9826, "citation_mrr": 0.9591, "refusal_accuracy": 1.0}}]})
    threshold_path = _write_json(tmp_path / "threshold.json", {"split": "dev", "val_buckets": [0, 1, 2], "best_on_val": {"refusal_ce_threshold": -0.25, "val": {"citation_hit_rate": 1.0, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0}, "test": {"citation_hit_rate": 1.0, "refusal_accuracy": 1.0, "false_refusal_rate": 0.0}}, "grid": []})

    payload = build_report(dev_path, test_path, ablation_path, threshold_path)

    assert payload["splits"]["dev"]["n_questions"] == 52
    assert payload["splits"]["test"]["summary"]["citation_recall_at_k"] == 0.9826
    assert payload["ablation"]["split"] == "test"
    assert payload["threshold"]["best_on_val"]["refusal_ce_threshold"] == -0.25
    assert payload["markdown"].startswith("# Campus KB Evaluation Report")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_report_eval.py -q
```

Expected: fail because `report_eval.py` does not exist yet.

- [ ] **Step 3: Keep the report API small**

Add a new `report_eval.py` module with two pure helpers:

```python
def render_markdown_report(dev_test_payload: dict, ablation_payload: dict | None, threshold_payload: dict | None) -> str:
    sections = ["# Campus KB Evaluation Report", "", _render_eval_section(dev_test_payload)]
    if ablation_payload:
        sections.extend(["", _render_ablation_section(ablation_payload)])
    if threshold_payload:
        sections.extend(["", _render_threshold_section(threshold_payload)])
    return "\n".join(sections).strip() + "\n"


def build_report(dev_path: str | Path, test_path: str | Path, ablation_path: str | Path | None = None, threshold_path: str | Path | None = None) -> dict:
    splits = {"dev": _load_json(dev_path), "test": _load_json(test_path)}
    ablation = _load_json(ablation_path) if ablation_path else None
    threshold = _load_json(threshold_path) if threshold_path else None
    return {
        "splits": splits,
        "ablation": ablation,
        "threshold": threshold,
        "markdown": render_markdown_report(splits, ablation, threshold),
    }
```

The CLI should accept explicit paths, read JSON, render Markdown, and optionally write a JSON summary next to it.

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
python -m pytest tests/test_report_eval.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add report_eval.py tests/test_report_eval.py
git commit -m "feat: add campus rag eval report"
```

### Task 2: Make threshold-search output easier to consume

**Files:**
- Modify: `tune_refusal_threshold.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Add a regression test for the output schema**

```python
def test_threshold_search_records_best_row_and_splits(tmp_path, monkeypatch):
    import json
    import tune_refusal_threshold as tuner

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(json.dumps({"id": "q-1", "question": "问题", "expected_doc_ids": ["doc-1"], "should_refuse": False, "category": "it"}) + "\n", encoding="utf-8")
    output_path = tmp_path / "threshold.json"
    fake_rag = MagicMock()
    fake_rag.ask.return_value = {"answer": "答案", "citations": [{"doc_id": "doc-1"}], "retrieved": [{"doc_id": "doc-1", "cross_encoder_score": 1.0, "dense_score": 0.8}]}
    monkeypatch.setattr(tuner, "CampusKBRAG", MagicMock(return_value=fake_rag))
    monkeypatch.setattr(sys, "argv", ["tune_refusal_threshold.py", "--questions", str(questions_path), "--output", str(output_path)])

    tuner.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["split"] == "dev"
    assert payload["val_buckets"] == [0, 1, 2]
    assert payload["best_on_val"]["refusal_ce_threshold"] == -2.0
    assert payload["grid"][0]["all"]["n_total"] == 1
    assert "test" in payload["grid"][0]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
python -m pytest tests/test_evaluate.py -k "threshold_search_records_best_row_and_splits" -q
```

- [ ] **Step 3: Simplify and clarify the output payload**

Remove the redundant `all: apply("val") | {}` placeholder path, keep `val`, `test`, and `all` as explicit summaries, and add a top-level `split` / `val_buckets` record so the JSON is self-describing.

- [ ] **Step 4: Run the full evaluation tests**

Run:

```powershell
python -m pytest tests/test_evaluate.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add tune_refusal_threshold.py tests/test_evaluate.py
git commit -m "feat: clarify threshold search output"
```

### Task 3: Verify the report on real artifacts

**Files:**
- Modify: none, unless the CLI needs a tiny help-text tweak

- [ ] **Step 1: Run the report generator on current artifacts**

Run:

```powershell
python report_eval.py --dev artifacts/predictions/campus_kb_eval_dev_phase7_20260830.json --test artifacts/predictions/campus_kb_eval_test_phase7_20260830.json --ablation artifacts/predictions/campus_kb_ablation_test_phase7_20260830.json --threshold artifacts/predictions/threshold_search.json --output artifacts/reports/campus_kb_eval_report.md
```

- [ ] **Step 2: Inspect the generated Markdown**

Confirm the report includes the split metrics, ablation table, and threshold summary without malformed numbers or duplicate sections.

- [ ] **Step 3: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

- [ ] **Step 4: Commit the verified report utility**

```powershell
git add report_eval.py tests/test_report_eval.py tune_refusal_threshold.py tests/test_evaluate.py
git commit -m "chore: add campus rag evaluation report"
```

### Plan Self-Review

- The report utility is a small standalone addition and does not change the RAG runtime.
- The threshold-search schema change is isolated and test-backed.
- The plan keeps the existing evaluation JSON files as source data.
- No placeholder requirements remain.
