"""End-to-end campus KB RAG pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from src.campus_kb_rag.config import load_config, resolve_path
from src.campus_kb_rag.generator import CampusAnswerGenerator
from src.campus_kb_rag.retriever import CampusKBRetriever
from src.campus_kb_rag.rewrite import rewrite_query
from src.campus_kb_rag.scope import is_out_of_scope


class CampusKBRAG:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self._resolve_config_paths()
        self.retriever = CampusKBRetriever(self.config)
        self.generator = CampusAnswerGenerator(self.config)

    def build_index(self, force: bool = False) -> None:
        self.retriever.build(force=force)

    def ask(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        self._validate_top_k(top_k)
        normalized = " ".join((query or "").split())
        if not normalized:
            return {
                "status": "input_required",
                "query": "",
                "search_query": None,
                "answer": "请输入具体的校园办事问题。",
                "citations": [],
                "retrieved": [],
                "refusal_reason": None,
            }

        if is_out_of_scope(normalized):
            answer = self.generator.generate(normalized, [])
            return {
                "status": "refused",
                "query": normalized,
                "search_query": None,
                "answer": answer,
                "citations": [],
                "retrieved": [],
                "refusal_reason": "out_of_scope",
            }

        search_query = rewrite_query(normalized) if self.config.get("retrieval", {}).get("query_rewrite", True) else normalized
        retrieved = self.retriever.search(search_query, top_k=top_k)
        evidence, refusal_reason = self._filter_low_confidence(retrieved)
        answer = self.generator.generate(normalized, evidence)
        return {
            "status": "answered" if evidence else "refused",
            "query": normalized,
            "search_query": search_query,
            "answer": answer,
            "citations": self._citations(evidence),
            "retrieved": retrieved,
            "refusal_reason": refusal_reason,
        }

    @staticmethod
    def _validate_top_k(top_k: int | None) -> None:
        if top_k is None:
            return
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    def _filter_low_confidence(
        self, retrieved: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], str | None]:
        if not retrieved:
            return [], "low_confidence"
        prompt_cfg = self.config.get("prompt", {})
        refusal_doc_ids = set(prompt_cfg.get("refusal_doc_ids", []))
        if retrieved[0].get("doc_id") in refusal_doc_ids:
            return [], "sentinel_document"
        top = retrieved[0]
        ce_cfg = self.config.get("retrieval", {}).get("cross_encoder", {})
        if ce_cfg.get("enabled") and top.get("cross_encoder_score") is not None:
            ce_threshold = float(prompt_cfg.get("refusal_ce_threshold", 0.0))
            if float(top["cross_encoder_score"]) >= ce_threshold:
                return retrieved, None
            # Topic routing already put the official page first; dense cosine is
            # a second gate so a negative CE logit does not false-refuse it.
            if top.get("topic_preferred") and float(top.get("dense_score") or 0.0) >= 0.50:
                return retrieved, None
            return [], "low_confidence"
        threshold = float(prompt_cfg.get("refusal_threshold", 0.18))
        top_score = float(top.get("dense_score", top.get("score", 0.0)))
        if top_score < threshold:
            return [], "low_confidence"
        return retrieved, None

    @staticmethod
    def _citations(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for i, item in enumerate(evidence, start=1):
            citation = {
                "index": i,
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "department": item.get("department"),
                "source": item.get("source"),
                "updated_at": item.get("updated_at"),
                "score": item.get("score"),
            }
            for score_name in ("dense_score", "rrf_score", "cross_encoder_score"):
                if score_name in item:
                    citation[score_name] = item[score_name]
            citations.append(citation)
        return citations

    def _resolve_config_paths(self) -> None:
        kb_cfg = self.config.setdefault("knowledge_base", {})
        idx_cfg = self.config.setdefault("index", {})
        kb_cfg["_resolved_path"] = str(resolve_path(kb_cfg["path"]))
        idx_cfg["_resolved_dir"] = str(resolve_path(idx_cfg["dir"]))
        idx_cfg["_resolved_faiss_path"] = str(resolve_path(idx_cfg["faiss_path"]))
        idx_cfg["_resolved_metadata_path"] = str(resolve_path(idx_cfg["metadata_path"]))
        idx_cfg["_resolved_manifest_path"] = str(
            resolve_path(idx_cfg.get("manifest_path", f"{idx_cfg['dir']}/manifest.json"))
        )
