"""Tests for citation ranking diagnostics."""

import pytest

from evaluation.metrics import (
    citation_mrr,
    citation_recall_at_k,
    summarize,
)


def test_citation_mrr_uses_first_expected_document_rank():
    results = [
        {
            "should_refuse": False,
            "expected_doc_ids": ["doc-a"],
            "citations": [{"doc_id": "doc-b"}, {"doc_id": "doc-a"}],
        },
        {
            "should_refuse": False,
            "expected_doc_ids": ["doc-c"],
            "citations": [{"doc_id": "doc-c"}],
        },
    ]

    assert citation_mrr(results) == pytest.approx(0.75)


def test_citation_recall_at_k_limits_citations_to_requested_cutoff():
    result = {
        "should_refuse": False,
        "expected_doc_ids": ["doc-a", "doc-b"],
        "citations": [
            {"doc_id": "doc-a"},
            {"doc_id": "doc-x"},
            {"doc_id": "doc-b"},
        ],
    }

    assert citation_recall_at_k([result], 1) == pytest.approx(0.5)
    assert citation_recall_at_k([result], 3) == pytest.approx(1.0)


def test_summarize_exposes_common_recall_cutoffs_and_mrr():
    result = {
        "should_refuse": False,
        "expected_doc_ids": ["doc-a"],
        "citations": [{"doc_id": "doc-a"}],
    }

    summary = summarize([result])

    assert summary["citation_mrr"] == 1.0
    assert summary["citation_recall_at_1"] == 1.0
    assert summary["citation_recall_at_3"] == 1.0
    assert summary["citation_recall_at_5"] == 1.0
    assert summary["citation_recall_at_8"] == 1.0
