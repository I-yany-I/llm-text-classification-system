"""Tests for campus knowledge-base JSONL loading and validation."""

import json

import pytest

from src.campus_kb_rag.documents import load_documents


def _write_lines(path, rows):
    path.write_text(
        "\n".join(
            row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _valid_row(doc_id="doc-1"):
    return {
        "id": doc_id,
        "title": "校园卡补办",
        "text": "先挂失，再补办。",
        "tags": ["校园卡"],
    }


def test_loader_reports_malformed_json_with_line(tmp_path):
    source = tmp_path / "kb.jsonl"
    _write_lines(source, [_valid_row(), "{"])

    with pytest.raises(ValueError, match=r"kb\.jsonl:2"):
        load_documents(source)


@pytest.mark.parametrize(
    "row",
    [
        {**_valid_row(), "id": ""},
        {**_valid_row(), "title": " "},
        {**_valid_row(), "text": ""},
        {**_valid_row(), "tags": "校园卡"},
        {**_valid_row(), "tags": [1]},
    ],
)
def test_loader_rejects_invalid_record_with_line(tmp_path, row):
    source = tmp_path / "kb.jsonl"
    _write_lines(source, [row])

    with pytest.raises(ValueError, match=r"kb\.jsonl:1"):
        load_documents(source)


def test_loader_rejects_duplicate_document_id_with_line(tmp_path):
    source = tmp_path / "kb.jsonl"
    _write_lines(source, [_valid_row(), _valid_row()])

    with pytest.raises(ValueError, match=r"kb\.jsonl:2.*doc-1"):
        load_documents(source)
