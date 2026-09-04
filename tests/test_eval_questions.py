"""Regression checks for the campus KB evaluation corpus."""

from __future__ import annotations

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


def test_eval_questions_have_dev_test_split():
    base = Path("data/campus_kb")
    dev_rows = [
        json.loads(line)
        for line in (base / "eval_questions_dev.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    test_rows = [
        json.loads(line)
        for line in (base / "eval_questions_test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    full_rows = [
        json.loads(line)
        for line in (base / "eval_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(dev_rows) == 52
    assert len(test_rows) == 128
    assert len(dev_rows) + len(test_rows) == len(full_rows)
    assert {row["id"] for row in dev_rows}.isdisjoint(
        {row["id"] for row in test_rows}
    )
