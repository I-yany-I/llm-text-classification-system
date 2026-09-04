# Campus KB Eval Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the campus KB evaluation set so it covers more categories, harder multi-hop cases, and more refusal boundaries without inflating the score from repetitive paraphrases.

**Architecture:** Keep the evaluation corpus as a single JSONL source of truth, add a regression test that enforces balance and uniqueness, then append curated questions that reuse the existing knowledge-base document IDs. The new samples should bias toward finance, international, refusal, and multi-document prompts so the reported metrics better reflect generalization rather than a few easy clusters.

**Tech Stack:** Python, pytest, JSONL data files

---

### Task 1: Lock the target distribution with a regression test

**Files:**
- Create: `tests/test_eval_questions.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from collections import Counter
from pathlib import Path


def test_eval_questions_are_balanced_and_unique():
    rows = [
        json.loads(line)
        for line in Path("data/campus_kb/eval_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    counts = Counter(row["category"] for row in rows)

    assert len(rows) >= 180
    assert len({row["id"] for row in rows}) == len(rows)
    assert counts["finance"] >= 12
    assert counts["international"] >= 12
    assert counts["refusal"] >= 25
    assert sum(1 for row in rows if len(row.get("expected_doc_ids", [])) >= 2) >= 40
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest -q tests/test_eval_questions.py -v`
Expected: FAIL because the current corpus is still smaller and less balanced.

- [ ] **Step 3: Keep the test as the guardrail**

Do not relax the thresholds after the fact unless the corpus expansion plan changes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_eval_questions.py
git commit -m "test: lock campus eval set balance"
```

### Task 2: Expand the evaluation corpus

**Files:**
- Modify: `data/campus_kb/eval_questions.jsonl`

- [ ] **Step 1: Add curated finance, international, refusal, and multi-hop items**

```jsonl
{"id":"q-fin-127","question":"学费和住宿费是一起交还是分开交？能用微信提现吗？","expected_doc_ids":["nju-fin-tuition"],"should_refuse":false,"category":"finance"}
{"id":"q-fin-128","question":"学校财务缴费页面支持哪些支付方式？","expected_doc_ids":["nju-fin-tuition"],"should_refuse":false,"category":"finance"}
{"id":"q-intl-127","question":"交换生出国前要办哪些学校材料？","expected_doc_ids":["nju-intl-visa","nju-ac-transcript-en","nju-ac-enrollment"],"should_refuse":false,"category":"international"}
{"id":"q-refuse-127","question":"学校附近哪家奶茶店最好喝？","expected_doc_ids":[],"should_refuse":true,"category":"refusal"}
```

- [ ] **Step 2: Add more answerable edge cases**

```jsonl
{"id":"q-it-127","question":"邮箱登录不了怎么办，先查密码还是先查账号状态？","expected_doc_ids":["nju-it-auth-password","nju-it-auth-login"],"should_refuse":false,"category":"it"}
{"id":"q-academic-127","question":"补考通过之后还要不要重新算 GPA？","expected_doc_ids":["nju-ac-exam-makeup","nju-ac-gpa"],"should_refuse":false,"category":"academic"}
{"id":"q-student-127","question":"医保和校医院就诊流程分别怎么走？","expected_doc_ids":["nju-stu-insurance","nju-stu-clinic"],"should_refuse":false,"category":"student"}
```

- [ ] **Step 3: Re-run the corpus test and ensure the new totals clear the thresholds**

Run: `python -m pytest -q tests/test_eval_questions.py -v`
Expected: PASS after the corpus reaches the target balance.

- [ ] **Step 4: Commit**

```bash
git add data/campus_kb/eval_questions.jsonl
git commit -m "data: expand campus eval set"
```

### Task 3: Re-evaluate and record the new baseline

**Files:**
- Modify: `README.md` if summary metrics change materially

- [ ] **Step 1: Run full evaluation**

Run: `python evaluate_campus_kb.py --output artifacts/predictions/campus_kb_eval_phase4_20260830.json`

- [ ] **Step 2: Check the new metrics**

Expected: improved coverage on finance, international, refusal, and multi-document cases without regressions in refusal accuracy.

- [ ] **Step 3: Update the README summary if the numbers are now the project baseline**

Only change the published metrics block if the new run is intended to replace the previous baseline.

- [ ] **Step 4: Commit**

```bash
git add README.md artifacts/predictions/campus_kb_eval_phase4_20260830.json
git commit -m "docs: refresh campus eval baseline"
```
