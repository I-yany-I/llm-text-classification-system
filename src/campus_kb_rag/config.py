"""Configuration loading for the campus KB RAG pipeline."""

from __future__ import annotations

import hashlib
import numbers
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "campus_kb.yaml"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"

_MODEL_MARKERS = (
    "config.json",
    "modules.json",
    "pytorch_model.bin",
    "model.safetensors",
)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identifier(source: str | Path) -> str:
    source_path = Path(str(source))
    if source_path.exists():
        return str(source_path.resolve())
    return str(source)


def _looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in _MODEL_MARKERS)


def resolve_model_source(name: str | Path) -> str:
    """Prefer a local snapshot so Hub downloads are optional.

    Lookup order:
    1. The given path, if it already exists.
    2. ``models/<hf-id>`` under the project root.
    3. ``models/<org>--<name>`` for flattened Hub cache names.
    4. The original Hub id / path (let sentence-transformers fetch it).
    """
    raw = Path(str(name))
    if _looks_like_model_dir(raw):
        return str(raw.resolve())

    relative = resolve_path(raw)
    if _looks_like_model_dir(relative):
        return str(relative)

    hub_id = str(name).replace("\\", "/").strip()
    candidates = [
        LOCAL_MODELS_DIR / hub_id,
        LOCAL_MODELS_DIR / hub_id.replace("/", "--"),
    ]
    for candidate in candidates:
        if _looks_like_model_dir(candidate):
            return str(candidate.resolve())
    return hub_id


def _require_mapping(cfg: Dict[str, Any], field: str) -> Dict[str, Any]:
    value = cfg.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _positive_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _non_empty_path(value: Any, field: str) -> None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty path")


def _numeric(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field} must be numeric")


def _validate_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")

    kb_cfg = _require_mapping(cfg, "knowledge_base")
    index_cfg = _require_mapping(cfg, "index")
    retrieval_cfg = _require_mapping(cfg, "retrieval")

    for field in ("path",):
        _non_empty_path(kb_cfg.get(field), f"knowledge_base.{field}")
    _positive_int(kb_cfg.get("chunk_size", 420), "knowledge_base.chunk_size")
    overlap = kb_cfg.get("chunk_overlap", 80)
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError("knowledge_base.chunk_overlap must be an integer")
    if overlap < 0 or overlap >= int(kb_cfg.get("chunk_size", 420)):
        raise ValueError(
            "knowledge_base.chunk_overlap must satisfy "
            "0 <= knowledge_base.chunk_overlap < knowledge_base.chunk_size"
        )

    for field in ("dir", "faiss_path", "metadata_path"):
        _non_empty_path(index_cfg.get(field), f"index.{field}")

    _non_empty_path(
        retrieval_cfg.get("embedding_model"),
        "retrieval.embedding_model",
    )
    for field in ("dense_top_k", "bm25_top_k", "final_top_k", "rrf_k"):
        _positive_int(retrieval_cfg.get(field, 12 if field != "rrf_k" else 60), f"retrieval.{field}")
    if "max_chunks_per_doc" in retrieval_cfg:
        _positive_int(
            retrieval_cfg["max_chunks_per_doc"],
            "retrieval.max_chunks_per_doc",
        )

    cross_encoder_cfg = retrieval_cfg.get("cross_encoder", {})
    if not isinstance(cross_encoder_cfg, dict):
        raise ValueError("retrieval.cross_encoder must be a mapping")
    enabled = cross_encoder_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("retrieval.cross_encoder.enabled must be a boolean")
    if "rerank_pool" in cross_encoder_cfg:
        _positive_int(
            cross_encoder_cfg["rerank_pool"],
            "retrieval.cross_encoder.rerank_pool",
        )
    if enabled:
        _non_empty_path(
            cross_encoder_cfg.get("model_name"),
            "retrieval.cross_encoder.model_name",
        )

    generation_cfg = cfg.get("generation", {})
    if not isinstance(generation_cfg, dict):
        raise ValueError("generation must be a mapping")
    backend = generation_cfg.get("backend", "extractive")
    if backend not in {"extractive", "llm"}:
        raise ValueError("generation.backend must be extractive or llm")

    prompt_cfg = cfg.get("prompt", {})
    if not isinstance(prompt_cfg, dict):
        raise ValueError("prompt must be a mapping")
    for field in (
        "refusal_threshold",
        "refusal_ce_threshold",
        "refusal_dense_fallback_threshold",
    ):
        if field in prompt_cfg:
            _numeric(prompt_cfg[field], f"prompt.{field}")


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = resolve_path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _validate_config(cfg)
    cfg["_config_path"] = str(path)
    return cfg
