"""Evaluation metrics for the citation-oriented campus KB RAG system."""

from __future__ import annotations

from typing import Dict, Iterable, List


def citation_hit_rate(results: Iterable[Dict]) -> float:
    """至少命中一个期望文档的问题占比（排除 should_refuse 的问题）。"""
    rows = [r for r in results if not r.get("should_refuse")]
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        expected = set(row.get("expected_doc_ids", []))
        if not expected:
            continue
        retrieved = {c.get("doc_id") for c in row.get("citations", [])}
        if expected.intersection(retrieved):
            hits += 1
    evaluable = [r for r in rows if r.get("expected_doc_ids")]
    return hits / len(evaluable) if evaluable else 0.0


def citation_recall_at_k(results: Iterable[Dict], k: int | None = None) -> float:
    """所有期望文档 ID 中被检索到的比例（排除 should_refuse）。"""
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k <= 0):
        raise ValueError("k must be a positive integer")

    rows = [r for r in results if not r.get("should_refuse")]
    total_expected = 0
    total_hit = 0
    for row in rows:
        expected = set(row.get("expected_doc_ids", []))
        if not expected:
            continue
        citations = row.get("citations", [])
        if k is not None:
            citations = citations[:k]
        retrieved = {c.get("doc_id") for c in citations}
        total_expected += len(expected)
        total_hit += len(expected.intersection(retrieved))
    return total_hit / total_expected if total_expected > 0 else 0.0


def citation_mrr(results: Iterable[Dict]) -> float:
    """平均首个期望文档倒数排名（排除 should_refuse）。"""
    rows = [
        r
        for r in results
        if not r.get("should_refuse") and r.get("expected_doc_ids")
    ]
    if not rows:
        return 0.0

    reciprocal_ranks = []
    for row in rows:
        expected = set(row.get("expected_doc_ids", []))
        reciprocal_rank = 0.0
        for rank, citation in enumerate(row.get("citations", []), start=1):
            if citation.get("doc_id") in expected:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def refusal_accuracy(results: Iterable[Dict]) -> float:
    """知识库外问题被正确拒答的比例（should_refuse=True 的问题）。"""
    rows = [r for r in results if r.get("should_refuse")]
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        answer = row.get("answer", "")
        citations = row.get("citations", [])
        if not citations and ("没有足够依据" in answer or "超出" in answer or "无法回答" in answer):
            correct += 1
    return correct / len(rows)


def false_refusal_rate(results: Iterable[Dict]) -> float:
    """有期望文档但系统拒答（引用为空）的比例（越低越好）。"""
    rows = [r for r in results if not r.get("should_refuse") and r.get("expected_doc_ids")]
    if not rows:
        return 0.0
    false_refusals = 0
    for row in rows:
        if not row.get("citations"):
            false_refusals += 1
    return false_refusals / len(rows)


def summarize(results: List[Dict]) -> Dict[str, float]:
    rows = list(results)
    return {
        "citation_hit_rate": round(citation_hit_rate(rows), 4),
        "citation_recall_at_k": round(citation_recall_at_k(rows), 4),
        "citation_mrr": round(citation_mrr(rows), 4),
        "citation_recall_at_1": round(citation_recall_at_k(rows, 1), 4),
        "citation_recall_at_3": round(citation_recall_at_k(rows, 3), 4),
        "citation_recall_at_5": round(citation_recall_at_k(rows, 5), 4),
        "citation_recall_at_8": round(citation_recall_at_k(rows, 8), 4),
        "refusal_accuracy": round(refusal_accuracy(rows), 4),
        "false_refusal_rate": round(false_refusal_rate(rows), 4),
        "n_total": len(rows),
        "n_answerable": sum(1 for r in rows if not r.get("should_refuse")),
        "n_refusal": sum(1 for r in rows if r.get("should_refuse")),
    }
