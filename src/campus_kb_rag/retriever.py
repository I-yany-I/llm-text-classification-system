"""Hybrid retrieval for campus knowledge-base chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from src.campus_kb_rag.config import resolve_model_source
from src.campus_kb_rag.documents import KBChunk, chunk_documents, load_documents


# Optional retrieval backends are loaded only when an index is built or queried.
faiss = None
SentenceTransformer = None


def _get_faiss():
    global faiss
    if faiss is None:
        import faiss as faiss_module

        faiss = faiss_module
    return faiss


def _tokenize(text: str) -> List[str]:
    raw = text or ""
    lowered = raw.lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    cjk_bigrams = [lowered[i : i + 2] for i in range(len(lowered) - 1)]
    cjk_bigrams = [t for t in cjk_bigrams if re.search(r"[\u4e00-\u9fff]", t)]
    words: List[str] = []
    try:
        import jieba

        words = [token.strip().lower() for token in jieba.cut(raw) if token.strip()]
    except Exception:
        words = []
    return ascii_tokens + cjk_chars + cjk_bigrams + words


def _rrf(ranked_lists: List[List[int]], rrf_k: int) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            if idx < 0:
                continue
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1.0)
    return scores


# Promote the official page when the query names a topic the Cross-Encoder
# routinely confuses (护照 vs 签证, 转专业 vs 学位证明).
_TOPIC_PREFER = (
    (("护照",), ("签证",), "nju-intl-passport"),
    (("转专业",), (), "nju-ac-major-transfer"),
)


def prefer_topic_docs(query: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not retrieved:
        return retrieved
    text = query or ""
    for required, excluded, doc_id in _TOPIC_PREFER:
        if not all(token in text for token in required):
            continue
        if any(token in text for token in excluded):
            continue
        match = [item for item in retrieved if item.get("doc_id") == doc_id]
        if not match:
            continue
        rest = [item for item in retrieved if item.get("doc_id") != doc_id]
        promoted = dict(match[0])
        promoted["topic_preferred"] = True
        extras = [dict(item) for item in match[1:]]
        return [promoted, *extras, *rest]
    return retrieved


class CampusKBRetriever:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        kb_cfg = config["knowledge_base"]
        idx_cfg = config["index"]
        ret_cfg = config["retrieval"]

        self.kb_path = Path(kb_cfg["_resolved_path"])
        self.index_dir = Path(idx_cfg["_resolved_dir"])
        self.faiss_path = Path(idx_cfg["_resolved_faiss_path"])
        self.metadata_path = Path(idx_cfg["_resolved_metadata_path"])
        self.embedding_model_name = resolve_model_source(ret_cfg["embedding_model"])
        self._embedder = None
        self.chunks: List[KBChunk] = []
        self.index = None
        self.bm25 = None
        self._cross_encoder = None

    def _get_embedder(self):
        if self._embedder is None:
            model_cls = SentenceTransformer
            if model_cls is None:
                from sentence_transformers import SentenceTransformer as model_cls

            self._embedder = model_cls(self.embedding_model_name)
        return self._embedder

    def build(self, force: bool = False) -> None:
        if not force and self.faiss_path.exists() and self.metadata_path.exists():
            self.load()
            return

        docs = load_documents(self.kb_path)
        kb_cfg = self.config["knowledge_base"]
        self.chunks = chunk_documents(
            docs,
            chunk_size=int(kb_cfg.get("chunk_size", 420)),
            chunk_overlap=int(kb_cfg.get("chunk_overlap", 80)),
        )
        if not self.chunks:
            raise ValueError(f"No chunks loaded from {self.kb_path}")

        embeddings = self._get_embedder().encode(
            [self._embed_text(c) for c in self.chunks],
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        faiss_module = _get_faiss()
        faiss_module.normalize_L2(embeddings)

        self.index = faiss_module.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25Okapi([_tokenize(self._embed_text(c)) for c in self.chunks])

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss_module.write_index(self.index, str(self.faiss_path))
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.faiss_path.exists() or not self.metadata_path.exists():
            self.build(force=True)
            return
        self.index = _get_faiss().read_index(str(self.faiss_path))
        with self.metadata_path.open("r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [KBChunk(**item) for item in raw_chunks]
        self.bm25 = BM25Okapi([_tokenize(self._embed_text(c)) for c in self.chunks])

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            self.load()

        ret_cfg = self.config["retrieval"]
        dense_k = int(ret_cfg.get("dense_top_k", 12))
        bm25_k = int(ret_cfg.get("bm25_top_k", 12))
        final_k = int(top_k or ret_cfg.get("final_top_k", 5))

        query_vec = self._encode_query(query)
        dense_ids, dense_scores = self._dense_search(query_vec, dense_k)
        dense_map = {idx: score for idx, score in zip(dense_ids, dense_scores)}
        if ret_cfg.get("hybrid_enabled", True):
            bm25_ids = self._bm25_search(query, bm25_k)
            fused = _rrf([dense_ids, bm25_ids], int(ret_cfg.get("rrf_k", 60)))
            candidate_ids = sorted(fused, key=lambda idx: -fused[idx])[: max(final_k, bm25_k)]
            rrf_map = fused
        else:
            candidate_ids = dense_ids
            rrf_map = {}

        candidates = []
        for idx in candidate_ids:
            dense_score = dense_map.get(idx)
            if dense_score is None:
                dense_score = self._dense_score_at(query_vec, idx)
            item = self._result(idx, dense_score)
            item["dense_score"] = float(dense_score)
            if idx in rrf_map:
                item["rrf_score"] = float(rrf_map[idx])
            candidates.append(item)
        ce_cfg = ret_cfg.get("cross_encoder", {})
        if ce_cfg.get("enabled", False) and candidates:
            candidates = self._rerank_cross_encoder(query, candidates, ce_cfg)
        candidates = prefer_topic_docs(query, candidates)
        return candidates[:final_k]

    def _encode_query(self, query: str) -> np.ndarray:
        query_vec = self._get_embedder().encode([query], convert_to_numpy=True).astype("float32")
        _get_faiss().normalize_L2(query_vec)
        return query_vec

    def _dense_score_at(self, query_vec: np.ndarray, idx: int) -> float:
        assert self.index is not None
        vector = np.asarray(self.index.reconstruct(int(idx)), dtype="float32").reshape(-1)
        return float(np.dot(query_vec.reshape(-1), vector))

    def _dense_search(self, query_vec: np.ndarray, k: int) -> tuple[List[int], List[float]]:
        assert self.index is not None
        k = max(1, min(k, self.index.ntotal))
        distances, indices = self.index.search(query_vec, k)
        ids = [int(i) for i in indices[0] if i >= 0]
        scores = [float(s) for s in distances[0][: len(ids)]]
        return ids, scores

    def _bm25_search(self, query: str, k: int) -> List[int]:
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float64)
        if scores.size == 0:
            return []
        k = max(1, min(k, scores.size))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [int(i) for i in idx]

    def _rerank_cross_encoder(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        ce_cfg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        from sentence_transformers import CrossEncoder

        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(resolve_model_source(str(ce_cfg["model_name"])))
        pool = candidates[: int(ce_cfg.get("rerank_pool", len(candidates)))]
        pairs = [(query, item["text"][:1600]) for item in pool]
        ce_scores = self._cross_encoder.predict(pairs)
        for item, score in zip(pool, ce_scores):
            item["cross_encoder_score"] = float(score)
        pool.sort(key=lambda x: x.get("cross_encoder_score", x.get("score", 0.0)), reverse=True)
        return pool + candidates[len(pool) :]

    def _result(self, idx: int, score: float) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        item = chunk.to_dict()
        item["score"] = float(score)
        return item

    @staticmethod
    def _embed_text(chunk: KBChunk) -> str:
        tags = " ".join(chunk.tags or [])
        return f"{chunk.title} {chunk.department} {tags}\n{chunk.text}"
