"""Startup tests that must not require retrieval backends to be imported."""

from __future__ import annotations

import subprocess
import sys


def test_importing_pipeline_does_not_import_heavy_retrieval_backends():
    script = r'''
import builtins

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "faiss" or name == "sentence_transformers":
        raise AssertionError(f"eager backend import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from src.campus_kb_rag.pipeline import CampusKBRAG  # noqa: F401
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
