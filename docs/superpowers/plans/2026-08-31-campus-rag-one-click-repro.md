# Campus RAG One-Click Reproduction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one command that rebuilds the campus KB index, runs dev/test evaluation, runs ablation and threshold search, and emits the consolidated report.

**Architecture:** Keep all existing evaluation logic in the current scripts and helper functions. Add a thin orchestration script that calls those helpers in order, writes each artifact to a stable path, and stops on the first failure. The script should not introduce new model logic or new metrics; it should only coordinate the existing pipeline and report outputs.

**Tech Stack:** Python 3.9+, pytest, pathlib, JSON

---

### Task 1: Lock the orchestration contract with tests

**Files:**
- Create: `tests/test_reproduce_eval.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from unittest.mock import MagicMock

import reproduce_eval


def test_run_reproduction_calls_steps_in_order(tmp_path, monkeypatch):
    calls = []

    class FakeRAG:
        def __init__(self, config_path=None):
            calls.append(("rag_init", config_path))

        def build_index(self, force=False):
            calls.append(("build_index", force))

    monkeypatch.setattr(reproduce_eval, "CampusKBRAG", FakeRAG)
    monkeypatch.setattr(reproduce_eval, "load_eval_questions", lambda path: [{"question": "q"}])
    monkeypatch.setattr(reproduce_eval, "evaluate_cases", lambda rag, questions, top_k=None: [{"should_refuse": False}])
    monkeypatch.setattr(reproduce_eval, "summarize", lambda results: {"n_total": len(results)})
    monkeypatch.setattr(reproduce_eval, "per_category_summary", lambda results: {"it": {}})
    monkeypatch.setattr(reproduce_eval, "run_ablation", lambda *args, **kwargs: {"split": "test", "n_questions": 1, "rows": []})
    monkeypatch.setattr(reproduce_eval, "run_threshold_search", lambda *args, **kwargs: {"split": "dev", "val_buckets": [0, 1, 2], "best_on_val": {}, "grid": []})
    monkeypatch.setattr(reproduce_eval, "build_report", lambda *args, **kwargs: {"markdown": "# report", "splits": {}})

    payload = reproduce_eval.run_reproduction(
        output_dir=tmp_path,
        config_path="custom.yaml",
        force_index=True,
        top_k=8,
    )

    assert calls[0] == ("rag_init", "custom.yaml")
    assert calls[1] == ("build_index", True)
    assert payload["report_markdown_path"].exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests/test_reproduce_eval.py -q
```

Expected: fail because `reproduce_eval.py` does not exist yet.

- [ ] **Step 3: Add a minimal orchestration API**

Create `reproduce_eval.py` with:

```python
def run_reproduction(
    output_dir: str | Path,
    config_path: str | None = None,
    force_index: bool = False,
    top_k: int | None = None,
) -> dict:
    ...
```

The function should:

1. build the index;
2. evaluate `dev` and `test`;
3. run ablation on `test`;
4. run threshold search on `dev`;
5. render the consolidated report.

- [ ] **Step 4: Run the focused test**

Run:

```powershell
python -m pytest tests/test_reproduce_eval.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add reproduce_eval.py tests/test_reproduce_eval.py
git commit -m "feat: add campus rag reproduction script"
```

### Task 2: Wire the existing helpers into the orchestrator

**Files:**
- Modify: `evaluate_campus_kb.py`
- Modify: `evaluate_ablation.py`
- Modify: `tune_refusal_threshold.py`
- Modify: `report_eval.py`

- [ ] **Step 1: Expose reusable helper functions**

Add small pure helpers so the orchestrator can call the existing logic without shelling out. Keep CLI behavior unchanged.

- [ ] **Step 2: Implement the orchestration output paths**

Write dev/test JSON to:

```text
artifacts/predictions/campus_kb_eval_dev_phase7_20260831.json
artifacts/predictions/campus_kb_eval_test_phase7_20260831.json
artifacts/predictions/campus_kb_ablation_test_phase7_20260831.json
artifacts/predictions/threshold_search.json
artifacts/reports/campus_kb_eval_report.md
artifacts/reports/campus_kb_eval_report.json
```

- [ ] **Step 3: Run the focused tests**

Run:

```powershell
python -m pytest tests/test_evaluate.py tests/test_report_eval.py tests/test_reproduce_eval.py -q
```

- [ ] **Step 4: Commit**

```powershell
git add evaluate_campus_kb.py evaluate_ablation.py tune_refusal_threshold.py report_eval.py reproduce_eval.py tests/test_evaluate.py tests/test_report_eval.py tests/test_reproduce_eval.py
git commit -m "feat: wire campus rag reproduction flow"
```

### Task 3: Document the single-command workflow

**Files:**
- Modify: `README.md`
- Modify: `INTERVIEW_PREP.md` only if the new command should be mentioned

- [ ] **Step 1: Add the command to the README**

Document the new usage:

```powershell
python reproduce_eval.py
```

- [ ] **Step 2: Validate the command output**

Run the script on current artifacts and confirm all generated paths exist and the report text matches the current split metrics.

- [ ] **Step 3: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

- [ ] **Step 4: Commit**

```powershell
git add README.md INTERVIEW_PREP.md reproduce_eval.py tests/test_reproduce_eval.py evaluate_campus_kb.py evaluate_ablation.py tune_refusal_threshold.py report_eval.py tests/test_evaluate.py tests/test_report_eval.py
git commit -m "docs: add campus rag reproduction workflow"
```

### Plan Self-Review

- The plan keeps all heavy logic in existing helpers.
- The orchestrator is thin and deterministic.
- Output paths are explicit and stable.
- No new metrics or model behavior are introduced.
