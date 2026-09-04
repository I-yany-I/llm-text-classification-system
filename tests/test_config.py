"""Tests for config path / local model resolution."""

from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.campus_kb_rag import config as cfg


def test_hub_id_passthrough_when_missing():
    assert cfg.resolve_model_source("sentence-transformers/does-not-exist") == (
        "sentence-transformers/does-not-exist"
    )


def test_existing_directory_is_used(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert cfg.resolve_model_source(str(tmp_path)) == str(tmp_path.resolve())


def test_project_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "LOCAL_MODELS_DIR", tmp_path)
    model_dir = tmp_path / "sentence-transformers" / "paraphrase-multilingual-MiniLM-L12-v2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    resolved = cfg.resolve_model_source("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    assert resolved == str(model_dir.resolve())


def test_flattened_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "LOCAL_MODELS_DIR", tmp_path)
    model_dir = tmp_path / "cross-encoder--mmarco-mMiniLMv2-L12-H384-v1"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_bytes(b"x")
    resolved = cfg.resolve_model_source("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    assert resolved == str(model_dir.resolve())


def test_file_sha256_changes_when_content_changes(tmp_path):
    source = tmp_path / "kb.jsonl"
    source.write_text("first", encoding="utf-8")
    first = cfg.file_sha256(source)
    source.write_text("second", encoding="utf-8")
    assert cfg.file_sha256(source) != first


def test_model_identifier_is_stable_for_hub_id():
    assert cfg.model_identifier("sentence-transformers/demo") == (
        "sentence-transformers/demo"
    )


def test_default_final_top_k_is_eight():
    loaded = cfg.load_config()
    assert loaded["retrieval"]["final_top_k"] == 8


def _write_config(tmp_path, overrides):
    config = {
        "knowledge_base": {
            "path": "data/kb.jsonl",
            "chunk_size": 420,
            "chunk_overlap": 80,
        },
        "index": {
            "dir": "vector_store",
            "faiss_path": "vector_store/faiss.index",
            "metadata_path": "vector_store/chunks.json",
        },
        "retrieval": {
            "embedding_model": "test-model",
            "dense_top_k": 12,
            "bm25_top_k": 12,
            "final_top_k": 8,
            "rrf_k": 60,
            "cross_encoder": {
                "enabled": False,
                "model_name": "test-cross-encoder",
            },
        },
        "generation": {"backend": "extractive"},
        "prompt": {
            "refusal_threshold": 0.18,
            "refusal_ce_threshold": -0.25,
        },
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("section", "values", "field"),
    [
        ("knowledge_base", {"chunk_size": 0}, "knowledge_base.chunk_size"),
        ("knowledge_base", {"chunk_overlap": 420}, "knowledge_base.chunk_overlap"),
        ("retrieval", {"dense_top_k": 0}, "retrieval.dense_top_k"),
        ("retrieval", {"rrf_k": -1}, "retrieval.rrf_k"),
        ("generation", {"backend": "unknown"}, "generation.backend"),
    ],
)
def test_load_config_rejects_invalid_values(tmp_path, section, values, field):
    path = _write_config(tmp_path, {section: values})
    with pytest.raises(ValueError, match=field):
        cfg.load_config(path)


def test_load_config_rejects_empty_required_path(tmp_path):
    path = _write_config(tmp_path, {"knowledge_base": {"path": ""}})
    with pytest.raises(ValueError, match="knowledge_base.path"):
        cfg.load_config(path)


def test_load_config_rejects_enabled_cross_encoder_without_model(tmp_path):
    path = _write_config(
        tmp_path,
        {"retrieval": {"cross_encoder": {"enabled": True, "model_name": ""}}},
    )
    with pytest.raises(ValueError, match="retrieval.cross_encoder.model_name"):
        cfg.load_config(path)
