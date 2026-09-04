"""Tests for RAG pipeline: refusal logic, citation formatting, end-to-end ask flow.

Run from project root:
    python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campus_kb_rag.pipeline import CampusKBRAG


# ---------------------------------------------------------------------------
# Refusal logic (_filter_low_confidence)
# ---------------------------------------------------------------------------

class TestRefusalLogic:
    @pytest.fixture
    def rag_with_config(self):
        """Create a CampusKBRAG with a specific config, mocking heavy deps."""
        config = {
            "knowledge_base": {"path": "data/kb.jsonl"},
            "index": {"dir": "vector_store/", "faiss_path": "faiss.index", "metadata_path": "chunks.json"},
            "retrieval": {
                "embedding_model": "test-model",
                "hybrid_enabled": True,
                "dense_top_k": 12,
                "bm25_top_k": 12,
                "final_top_k": 5,
                "rrf_k": 60,
                "cross_encoder": {"enabled": False},
            },
            "prompt": {
                "refusal_threshold": 0.18,
                "refusal_ce_threshold": 0.0,
                "refusal_dense_fallback_threshold": 0.43,
                "refusal_doc_ids": ["nju-support-unknown"],
            },
            "generation": {"backend": "extractive"},
        }
        # Avoid loading models by patching the retriever init
        with patch("src.campus_kb_rag.pipeline.CampusKBRetriever", autospec=True):
            with patch("src.campus_kb_rag.pipeline.CampusAnswerGenerator", autospec=True):
                rag = CampusKBRAG.__new__(CampusKBRAG)
                rag.config = config
                return rag

    def test_empty_retrieved_returns_empty(self, rag_with_config):
        evidence, reason = rag_with_config._filter_low_confidence([])
        assert evidence == []
        assert reason == "low_confidence"

    def test_refusal_doc_id_triggers_rejection(self, rag_with_config):
        retrieved = [{"doc_id": "nju-support-unknown", "score": 0.85, "text": "..."}]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert evidence == []
        assert reason == "sentinel_document"

    def test_score_below_threshold_rejected(self, rag_with_config):
        retrieved = [{"doc_id": "nju-it-vpn", "score": 0.10, "text": "..."}]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert evidence == []
        assert reason == "low_confidence"

    def test_score_above_threshold_accepted(self, rag_with_config):
        retrieved = [{"doc_id": "nju-it-vpn", "score": 0.50, "text": "..."}]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert len(evidence) == 1
        assert evidence[0]["doc_id"] == "nju-it-vpn"
        assert reason is None

    def test_threshold_exactly_at_boundary(self, rag_with_config):
        # score == threshold (0.18): code uses '<' not '<=', so equal is accepted
        retrieved = [{"doc_id": "nju-it-vpn", "score": 0.18, "text": "..."}]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert len(evidence) == 1  # 0.18 < 0.18 is False -> accepted
        assert reason is None

    def test_refusal_doc_id_checked_before_score(self, rag_with_config):
        """Even with high score, refusal doc_id should trigger rejection."""
        retrieved = [{"doc_id": "nju-support-unknown", "score": 0.99, "text": "..."}]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert evidence == []
        assert reason == "sentinel_document"

    def test_lower_sentinel_document_is_removed_from_evidence(self, rag_with_config):
        retrieved = [
            {"doc_id": "nju-it-vpn", "score": 0.90, "text": "VPN"},
            {"doc_id": "nju-support-unknown", "score": 0.80, "text": "unknown"},
        ]

        evidence, reason = rag_with_config._filter_low_confidence(retrieved)

        assert [item["doc_id"] for item in evidence] == ["nju-it-vpn"]
        assert reason is None

    def test_different_refusal_doc_ids(self, rag_with_config):
        rag_with_config.config["prompt"]["refusal_doc_ids"] = ["custom-refusal", "fallback-doc"]
        retrieved = [{"doc_id": "custom-refusal", "score": 0.90, "text": "..."}]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert evidence == []
        assert reason == "sentinel_document"

    def test_dense_score_used_when_cross_encoder_off(self, rag_with_config):
        retrieved = [
            {
                "doc_id": "nju-it-vpn",
                "score": -4.2,
                "dense_score": 0.51,
                "cross_encoder_score": -4.2,
                "text": "...",
            }
        ]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert len(evidence) == 1
        assert reason is None

    def test_cross_encoder_negative_logit_refused(self, rag_with_config):
        rag_with_config.config["retrieval"]["cross_encoder"]["enabled"] = True
        retrieved = [
            {
                "doc_id": "nju-it-vpn",
                "dense_score": 0.31,
                "cross_encoder_score": -1.2,
                "score": 0.31,
                "text": "...",
            }
        ]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert evidence == []
        assert reason == "low_confidence"

    def test_topic_preferred_negative_ce_accepted_with_dense(self, rag_with_config):
        rag_with_config.config["retrieval"]["cross_encoder"]["enabled"] = True
        retrieved = [
            {
                "doc_id": "nju-intl-passport",
                "dense_score": 0.78,
                "cross_encoder_score": -0.41,
                "score": 0.78,
                "topic_preferred": True,
                "text": "...",
            }
        ]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert len(evidence) == 1
        assert reason is None

    def test_strong_dense_match_can_rescue_negative_ce(self, rag_with_config):
        rag_with_config.config["retrieval"]["cross_encoder"]["enabled"] = True
        retrieved = [
            {
                "doc_id": "nju-fin-tuition",
                "dense_score": 0.72,
                "cross_encoder_score": -2.1,
                "score": 0.72,
                "text": "...",
            }
        ]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert len(evidence) == 1
        assert reason is None

    def test_best_among_top_candidates_can_rescue_negative_ce(
        self, rag_with_config
    ):
        rag_with_config.config["retrieval"]["cross_encoder"]["enabled"] = True
        retrieved = [
            {
                "doc_id": "nju-it-auth-activate",
                "dense_score": 0.31,
                "cross_encoder_score": -2.1,
                "score": 0.31,
                "text": "...",
            },
            {
                "doc_id": "nju-it-auth-password",
                "dense_score": 0.44,
                "cross_encoder_score": -3.2,
                "score": 0.44,
                "text": "...",
            },
        ]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert len(evidence) == 2
        assert reason is None

    def test_out_of_scope_skips_retrieval(self, rag_with_config):
        rag_with_config.retriever = MagicMock()
        rag_with_config.generator = MagicMock()
        rag_with_config.generator.generate.return_value = "no evidence"
        result = rag_with_config.ask("学校附近哪里有打印店？价格怎么样？")
        rag_with_config.retriever.search.assert_not_called()
        assert result["status"] == "refused"
        assert result["refusal_reason"] == "out_of_scope"
        assert result["search_query"] is None
        assert result["citations"] == []
        assert result["retrieved"] == []

    def test_cross_encoder_positive_logit_accepted(self, rag_with_config):
        rag_with_config.config["retrieval"]["cross_encoder"]["enabled"] = True
        retrieved = [
            {
                "doc_id": "nju-it-vpn",
                "dense_score": 0.51,
                "cross_encoder_score": 1.2,
                "score": 0.51,
                "text": "...",
            }
        ]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        assert len(evidence) == 1
        assert reason is None

    def test_no_prompt_config_defaults(self, rag_with_config):
        """Without prompt config keys, should still work with defaults."""
        rag_with_config.config.pop("prompt")
        retrieved = [{"doc_id": "some-doc", "score": 0.50, "text": "..."}]
        evidence, reason = rag_with_config._filter_low_confidence(retrieved)
        # Default threshold is 0.18, score 0.50 > 0.18 → accepted
        assert len(evidence) == 1
        assert reason is None

    def test_empty_query_returns_input_required(self, rag_with_config):
        rag_with_config.retriever = MagicMock()
        rag_with_config.generator = MagicMock()
        result = rag_with_config.ask("  \n")
        assert result["status"] == "input_required"
        assert result["query"] == ""
        assert result["search_query"] is None
        assert result["refusal_reason"] is None
        rag_with_config.retriever.search.assert_not_called()

    @pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
    def test_invalid_top_k_fails_before_retrieval(self, rag_with_config, top_k):
        rag_with_config.retriever = MagicMock()
        rag_with_config.generator = MagicMock()
        with pytest.raises(ValueError, match="top_k"):
            rag_with_config.ask("校园卡怎么办", top_k=top_k)
        rag_with_config.retriever.search.assert_not_called()

    def test_answered_result_contains_rewritten_query_and_scores(self, rag_with_config):
        rag_with_config.retriever = MagicMock()
        rag_with_config.generator = MagicMock()
        rag_with_config.generator.generate.return_value = "请先挂失，再补办。"
        rag_with_config.retriever.search.return_value = [
            {
                "doc_id": "nju-it-card",
                "title": "校园卡补办",
                "department": "信息化中心",
                "source": "https://example.com",
                "updated_at": "2026-08-01",
                "text": "先挂失，再携带证件补办。",
                "score": 0.8,
                "dense_score": 0.8,
                "rrf_score": 0.03,
                "cross_encoder_score": 1.2,
            }
        ]
        result = rag_with_config.ask("校卡怎么补办")
        assert result["status"] == "answered"
        assert result["search_query"]
        assert result["refusal_reason"] is None
        assert result["citations"][0]["dense_score"] == pytest.approx(0.8)
        assert result["citations"][0]["rrf_score"] == pytest.approx(0.03)
        assert result["citations"][0]["cross_encoder_score"] == pytest.approx(1.2)

    def test_ask_passes_rewritten_query_only_to_sparse_retrieval(self, rag_with_config):
        rag_with_config.retriever = MagicMock()
        rag_with_config.generator = MagicMock()
        rag_with_config.generator.generate.return_value = "请使用 VPN。"
        rag_with_config.retriever.search.return_value = [
            {
                "doc_id": "nju-it-vpn",
                "title": "VPN 使用说明",
                "department": "信息化中心",
                "source": "test",
                "updated_at": "2026-01-01",
                "text": "校外访问请使用 VPN。",
                "score": 0.8,
                "dense_score": 0.8,
            }
        ]

        query = "在家怎么访问校园网？"
        result = rag_with_config.ask(query)

        rag_with_config.retriever.search.assert_called_once_with(
            query,
            top_k=None,
            sparse_query=result["search_query"],
        )
        assert "VPN" in result["search_query"]


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

class TestCitations:
    def test_formats_single_evidence(self):
        evidence = [
            {
                "doc_id": "nju-it-vpn",
                "title": "校园VPN使用说明",
                "department": "信息化中心",
                "source": "https://example.com",
                "updated_at": "2026-05-01",
                "score": 0.85,
            }
        ]
        citations = CampusKBRAG._citations(evidence)
        assert len(citations) == 1
        assert citations[0]["index"] == 1
        assert citations[0]["doc_id"] == "nju-it-vpn"
        assert citations[0]["title"] == "校园VPN使用说明"
        assert citations[0]["score"] == 0.85

    def test_formats_multiple_evidence(self):
        evidence = [
            {"doc_id": "doc-a", "title": "A", "department": "", "source": "", "updated_at": "", "score": 0.9},
            {"doc_id": "doc-b", "title": "B", "department": "", "source": "", "updated_at": "", "score": 0.7},
            {"doc_id": "doc-c", "title": "C", "department": "", "source": "", "updated_at": "", "score": 0.5},
        ]
        citations = CampusKBRAG._citations(evidence)
        assert len(citations) == 3
        assert [c["index"] for c in citations] == [1, 2, 3]

    def test_empty_evidence(self):
        assert CampusKBRAG._citations([]) == []

    def test_missing_keys_default_to_none(self):
        evidence = [{"doc_id": "test"}]
        citations = CampusKBRAG._citations(evidence)
        assert citations[0]["title"] is None
        assert citations[0]["department"] is None
