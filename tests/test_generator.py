"""Regression tests for lazy answer generation."""

from src.campus_kb_rag.generator import CampusAnswerGenerator


def test_generator_construction_does_not_load_llm():
    generator = CampusAnswerGenerator({"generation": {"backend": "llm"}})
    assert generator._pipe is None
