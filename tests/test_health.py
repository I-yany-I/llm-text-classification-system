"""Tests for lightweight campus RAG health diagnostics."""

from unittest.mock import MagicMock

import health


def test_collect_health_reports_required_checks_without_loading_models(
    tmp_path, monkeypatch
):
    kb_path = tmp_path / "kb.jsonl"
    faiss_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "chunks.json"
    manifest_path = tmp_path / "manifest.json"
    for path in (kb_path, faiss_path, metadata_path, manifest_path):
        path.write_text("ready", encoding="utf-8")

    fake_rag = MagicMock()
    fake_rag.config = {"_config_path": str(tmp_path / "config.yaml")}
    fake_rag.retriever.kb_path = kb_path
    fake_rag.retriever.faiss_path = faiss_path
    fake_rag.retriever.metadata_path = metadata_path
    fake_rag.retriever.manifest_path = manifest_path
    fake_rag.retriever.embedding_model_name = str(tmp_path / "embedding-model")
    fake_rag.retriever._embedder = None
    fake_rag.retriever._cross_encoder = None
    fake_rag.retriever.load = MagicMock()
    fake_rag.config["retrieval"] = {"cross_encoder": {"enabled": False}}
    monkeypatch.setattr(health, "CampusKBRAG", MagicMock(return_value=fake_rag))

    report = health.collect_health()

    assert report["status"] == "ok"
    assert {check["name"] for check in report["checks"]} >= {
        "config",
        "knowledge_base",
        "index_artifacts",
        "index_manifest",
    }
    fake_rag.retriever.load.assert_called_once()
    assert fake_rag.retriever._embedder is None
    assert fake_rag.retriever._cross_encoder is None
