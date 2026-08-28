"""Configuration loading for the campus KB RAG pipeline."""

from __future__ import annotations

import hashlib
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


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = resolve_path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return cfg
