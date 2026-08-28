"""Tests for hybrid retriever: tokenizer, RRF fusion, dense/BM25 search.

Run from project root:
    python -m pytest tests/test_retriever.py -v
"""

import sys
import math
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campus_kb_rag.retriever import (
    _tokenize,
    _rrf,
    prefer_topic_docs,
    CampusKBRetriever,
)
from src.campus_kb_rag.documents import KBChunk


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_english_tokens(self):
        tokens = _tokenize("hello world vpn_config")
        assert "hello" in tokens
        assert "world" in tokens
        assert "vpn_config" in tokens

    def test_chinese_characters(self):
        tokens = _tokenize("统一身份认证")
        # Should contain individual CJK chars
        assert "统" in tokens
        assert "一" in tokens
        assert "身" in tokens

    def test_chinese_bigrams(self):
        tokens = _tokenize("统一身份认证")
        # Should contain bigrams
        assert "统一" in tokens
        assert "一身" in tokens
        assert "身份" in tokens

    def test_mixed_text(self):
        tokens = _tokenize("VPN使用说明 校园网")
        assert "vpn" in tokens
        # Should have both CJK chars and bigrams
        cjk_chars = [t for t in tokens if "一" <= t <= "鿿"]
        assert len(cjk_chars) > 0

    def test_empty_string(self):
        assert _tokenize("") == []
        assert _tokenize(None) == []

    def test_numbers_preserved(self):
        tokens = _tokenize("test123 abc")
        assert "test123" in tokens
        assert "abc" in tokens

    def test_jieba_words_added(self):
        tokens = _tokenize("统一身份认证")
        assert "统一身份认证" in tokens or "统一" in tokens


# ---------------------------------------------------------------------------
# _rrf
# ---------------------------------------------------------------------------

class TestRRF:
    def test_basic_fusion(self):
        list_a = [0, 1, 2]
        list_b = [2, 0, 1]
        scores = _rrf([list_a, list_b], rrf_k=60)
        assert 0 in scores
        assert 1 in scores
        assert 2 in scores
        # Item 0 appears at rank 0 in A and rank 1 in B → higher total
        assert scores[0] > 0

    def test_k_parameter_effect(self):
        """Larger k makes scores closer together."""
        scores_k60 = _rrf([[0, 1, 2], [2, 0, 1]], rrf_k=60)
        scores_k10 = _rrf([[0, 1, 2], [2, 0, 1]], rrf_k=10)
        # With k=10, rank differences matter more → larger variance
        range_k60 = max(scores_k60.values()) - min(scores_k60.values())
        range_k10 = max(scores_k10.values()) - min(scores_k10.values())
        assert range_k10 > range_k60

    def test_single_list(self):
        scores = _rrf([[0, 1, 2]], rrf_k=60)
        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]

    def test_empty_list(self):
        scores = _rrf([], rrf_k=60)
        assert scores == {}

    def test_negative_indices_skipped(self):
        """Negative indices (like -1 from FAISS) should be ignored."""
        scores = _rrf([[0, -1, 1]], rrf_k=60)
        assert -1 not in scores
        assert 0 in scores
        assert 1 in scores

    def test_rank_order_preserved(self):
        """Higher-ranked items should get higher RRF scores."""
        list_a = [0, 1, 2, 3, 4]
        list_b = [0, 1, 2, 3, 4]
        scores = _rrf([list_a, list_b], rrf_k=60)
        for i in range(4):
            assert scores[i] > scores[i + 1], f"Rank {i} should score higher than rank {i+1}"


# ---------------------------------------------------------------------------
# CampusKBRetriever — search logic (requires mocking)
# ---------------------------------------------------------------------------

class TestRetrieverSearch:
    @pytest.fixture
    def mock_config(self):
        return {
            "knowledge_base": {
                "_resolved_path": "/fake/kb.jsonl",
            },
            "index": {
                "_resolved_dir": "/fake/index/",
                "_resolved_faiss_path": "/fake/index/faiss.index",
                "_resolved_metadata_path": "/fake/index/chunks.json",
            },
            "retrieval": {
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "hybrid_enabled": True,
                "dense_top_k": 12,
                "bm25_top_k": 12,
                "final_top_k": 5,
                "rrf_k": 60,
                "cross_encoder": {"enabled": False},
            },
        }

    def test_search_calls_dense_and_bm25(self, mock_config):
        with patch("src.campus_kb_rag.retriever.SentenceTransformer", autospec=True):
            retriever = CampusKBRetriever(mock_config)

        # Mock the internal methods
        retriever._encode_query = MagicMock(return_value=np.zeros((1, 4), dtype="float32"))
        retriever._dense_search = MagicMock(return_value=([0, 1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6, 0.5]))
        retriever._bm25_search = MagicMock(return_value=[2, 0, 4, 1, 3])
        retriever.load = MagicMock()  # prevent actual load

        # Need chunks for _result
        retriever.chunks = [
            KBChunk(
                chunk_id=f"doc-{i}-0",
                doc_id=f"doc-{i}",
                title=f"title-{i}",
                text=f"text content {i}",
                department="IT",
                source="test",
                tags=["test"],
                updated_at="2026-01-01",
            )
            for i in range(5)
        ]
        retriever.index = MagicMock()
        retriever.index.ntotal = 5

        results = retriever.search("测试问题")

        assert len(results) <= 5
        assert retriever._dense_search.called
        assert retriever._bm25_search.called

    def test_search_dense_only_when_hybrid_disabled(self, mock_config):
        mock_config["retrieval"]["hybrid_enabled"] = False
        with patch("src.campus_kb_rag.retriever.SentenceTransformer", autospec=True):
            retriever = CampusKBRetriever(mock_config)

        retriever._encode_query = MagicMock(return_value=np.zeros((1, 4), dtype="float32"))
        retriever._dense_search = MagicMock(return_value=([0, 1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6, 0.5]))
        retriever._bm25_search = MagicMock()
        retriever.load = MagicMock()  # prevent actual load

        retriever.chunks = [
            KBChunk(
                chunk_id=f"doc-{i}-0",
                doc_id=f"doc-{i}",
                title=f"title-{i}",
                text=f"text content {i}",
                department="IT",
                source="test",
                tags=["test"],
                updated_at="2026-01-01",
            )
            for i in range(5)
        ]
        retriever.index = MagicMock()
        retriever.index.ntotal = 5

        retriever.search("测试")
        assert not retriever._bm25_search.called

    def test_cross_encoder_rerank_does_not_overwrite_dense_score(self, mock_config):
        mock_config["retrieval"]["cross_encoder"] = {
            "enabled": True,
            "model_name": "unused",
            "rerank_pool": 3,
        }
        with patch("src.campus_kb_rag.retriever.SentenceTransformer", autospec=True):
            retriever = CampusKBRetriever(mock_config)

        retriever.load = MagicMock()
        retriever._encode_query = MagicMock(return_value=np.zeros((1, 4), dtype="float32"))
        retriever._dense_search = MagicMock(return_value=([0, 1, 2], [0.61, 0.40, 0.22]))
        retriever._bm25_search = MagicMock(return_value=[1, 0, 2])
        retriever.chunks = [
            KBChunk(
                chunk_id=f"doc-{i}-0",
                doc_id=f"doc-{i}",
                title=f"title-{i}",
                text=f"text content {i}",
                department="IT",
                source="test",
                tags=["test"],
                updated_at="2026-01-01",
            )
            for i in range(3)
        ]
        retriever.index = MagicMock()
        retriever.index.ntotal = 3

        fake_ce = MagicMock()
        fake_ce.predict.return_value = np.asarray([-2.0, 4.0, 0.5], dtype="float32")
        with patch("sentence_transformers.CrossEncoder", return_value=fake_ce):
            results = retriever.search("测试问题")

        assert [item["doc_id"] for item in results] == ["doc-1", "doc-2", "doc-0"]
        assert results[0]["dense_score"] == pytest.approx(0.40)
        assert results[0]["score"] == pytest.approx(0.40)
        assert results[0]["cross_encoder_score"] == pytest.approx(4.0)

    def test_retriever_does_not_load_embedder_on_construction(self, mock_config):
        with patch("src.campus_kb_rag.retriever.SentenceTransformer") as model:
            retriever = CampusKBRetriever(mock_config)
        model.assert_not_called()
        assert retriever._embedder is None

    def test_embedder_is_created_once_on_first_use(self, mock_config):
        fake_embedder = MagicMock()
        with patch(
            "src.campus_kb_rag.retriever.SentenceTransformer",
            return_value=fake_embedder,
        ) as model:
            retriever = CampusKBRetriever(mock_config)
            assert retriever._get_embedder() is fake_embedder
            assert retriever._get_embedder() is fake_embedder
        model.assert_called_once_with(retriever.embedding_model_name)


# ---------------------------------------------------------------------------
# KBChunk
# ---------------------------------------------------------------------------

class TestKBChunk:
    def test_to_dict(self):
        chunk = KBChunk(
            chunk_id="nju-it-vpn-0",
            doc_id="nju-it-vpn",
            title="VPN使用说明",
            text="关于校园VPN的使用方法...",
            department="信息化中心",
            source="https://example.com",
            tags=["VPN", "校外访问"],
            updated_at="2026-05-01",
        )
        d = chunk.to_dict()
        assert d["doc_id"] == "nju-it-vpn"
        assert d["title"] == "VPN使用说明"
        assert "VPN" in d["tags"]

    def test_from_dict(self):
        d = {
            "chunk_id": "nju-it-vpn-0",
            "doc_id": "nju-it-vpn",
            "title": "VPN使用说明",
            "text": "content",
            "department": "IT",
            "source": "url",
            "tags": ["vpn"],
            "updated_at": "2026-01-01",
        }
        chunk = KBChunk(**d)
        assert chunk.doc_id == "nju-it-vpn"
        assert chunk.title == "VPN使用说明"


def test_prefer_passport_over_visa():
    retrieved = [
        {"doc_id": "nju-intl-visa", "text": "签证"},
        {"doc_id": "nju-intl-passport", "text": "护照"},
    ]
    out = prefer_topic_docs("办护照学校能开什么证明？", retrieved)
    assert out[0]["doc_id"] == "nju-intl-passport"
    assert out[0]["topic_preferred"] is True


def test_prefer_major_transfer():
    retrieved = [
        {"doc_id": "nju-ac-degree", "text": "学位"},
        {"doc_id": "nju-ac-major-transfer", "text": "转专业"},
    ]
    out = prefer_topic_docs("想转专业到另一个院系", retrieved)
    assert out[0]["doc_id"] == "nju-ac-major-transfer"
