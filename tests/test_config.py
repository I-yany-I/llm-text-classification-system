"""Tests for config path / local model resolution."""

from pathlib import Path
import sys

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
