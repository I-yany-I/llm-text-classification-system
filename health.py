"""Lightweight health checks for the campus KB RAG application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.campus_kb_rag import CampusKBRAG
from src.campus_kb_rag.config import resolve_model_source


def _add_check(
    report: Dict[str, Any],
    name: str,
    status: str,
    detail: str,
) -> None:
    report["checks"].append({"name": name, "status": status, "detail": detail})


def collect_health(config_path: str | None = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "status": "error",
        "config_path": config_path,
        "checks": [],
    }
    try:
        rag = CampusKBRAG(config_path=config_path)
    except Exception as exc:
        _add_check(report, "config", "error", f"{type(exc).__name__}: {exc}")
        return report

    report["config_path"] = rag.config.get("_config_path")
    _add_check(report, "config", "ok", "configuration loaded and validated")

    retriever = rag.retriever
    kb_path = Path(retriever.kb_path)
    if kb_path.is_file():
        _add_check(report, "knowledge_base", "ok", str(kb_path))
    else:
        _add_check(report, "knowledge_base", "error", f"missing file: {kb_path}")

    artifact_paths = {
        "faiss": Path(retriever.faiss_path),
        "metadata": Path(retriever.metadata_path),
        "manifest": Path(retriever.manifest_path),
    }
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        _add_check(
            report,
            "index_artifacts",
            "error",
            "missing: " + ", ".join(missing),
        )
    else:
        _add_check(report, "index_artifacts", "ok", "all index files exist")
        try:
            retriever.load()
        except Exception as exc:
            _add_check(
                report,
                "index_manifest",
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        else:
            _add_check(report, "index_manifest", "ok", "index contract validated")

    embedding_source = resolve_model_source(retriever.embedding_model_name)
    if Path(embedding_source).is_dir():
        _add_check(report, "embedding_model", "ok", embedding_source)
    else:
        _add_check(
            report,
            "embedding_model",
            "warning",
            f"hub source or uncached path: {embedding_source}",
        )

    ce_cfg = rag.config.get("retrieval", {}).get("cross_encoder", {})
    if ce_cfg.get("enabled", False):
        ce_source = resolve_model_source(str(ce_cfg["model_name"]))
        if Path(ce_source).is_dir():
            _add_check(report, "cross_encoder", "ok", ce_source)
        else:
            _add_check(
                report,
                "cross_encoder",
                "warning",
                f"hub source or uncached path: {ce_source}",
            )

    report["status"] = (
        "error"
        if any(check["status"] == "error" for check in report["checks"])
        else "ok"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check campus KB RAG readiness.")
    parser.add_argument("--config", default=None, help="主配置文件（可选）")
    args = parser.parse_args()
    report = collect_health(config_path=args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
