"""Tests for the lightweight Gradio callback."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock


def test_answer_includes_structured_diagnostics(monkeypatch):
    import app

    fake_rag = MagicMock()
    fake_rag.ask.return_value = {
        "status": "refused",
        "refusal_reason": "low_confidence",
        "answer": "当前知识库没有足够依据。",
        "citations": [],
        "retrieved": [],
    }
    monkeypatch.setattr(app, "rag", fake_rag)

    answer, citations, retrieved = app.answer("未知问题", 5)

    assert "当前知识库没有足够依据" in answer
    assert citations == []
    assert "status=refused" in retrieved
    assert "refusal_reason=low_confidence" in retrieved


def test_importing_app_does_not_load_embedding_model():
    script = r'''
import builtins

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "faiss" or name == "sentence_transformers":
        raise AssertionError(f"eager backend import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import app  # noqa: F401
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
