"""Knowledge-base document loading and chunking."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


@dataclass
class KBDocument:
    id: str
    title: str
    department: str
    source: str
    updated_at: str
    tags: List[str]
    text: str


@dataclass
class KBChunk:
    chunk_id: str
    doc_id: str
    title: str
    department: str
    source: str
    updated_at: str
    tags: List[str]
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_documents(jsonl_path: Path) -> List[KBDocument]:
    docs: List[KBDocument] = []
    seen_ids = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no} invalid JSON: {exc.msg}") from exc

            if not isinstance(raw, dict):
                raise ValueError(f"{jsonl_path}:{line_no} record must be a JSON object")

            for field in ("id", "title", "text"):
                value = raw.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{jsonl_path}:{line_no} field {field} must be a non-empty string"
                    )

            doc_id = raw["id"]
            if doc_id in seen_ids:
                raise ValueError(
                    f"{jsonl_path}:{line_no} duplicate document id: {doc_id}"
                )

            tags = raw.get("tags", [])
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise ValueError(
                    f"{jsonl_path}:{line_no} field tags must be a list of strings"
                )

            seen_ids.add(doc_id)
            docs.append(
                KBDocument(
                    id=doc_id,
                    title=raw["title"],
                    department=str(raw.get("department", "")),
                    source=str(raw.get("source", "")),
                    updated_at=str(raw.get("updated_at", "")),
                    tags=list(tags),
                    text=raw["text"],
                )
            )
    return docs


def _sliding_window(text: str, size: int, overlap: int) -> Iterable[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return
    size = max(80, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    start = 0
    while start < len(normalized):
        yield normalized[start : start + size]
        if start + size >= len(normalized):
            break
        start += size - overlap


def chunk_documents(
    docs: List[KBDocument],
    chunk_size: int = 420,
    chunk_overlap: int = 80,
) -> List[KBChunk]:
    chunks: List[KBChunk] = []
    for doc in docs:
        for idx, text in enumerate(_sliding_window(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                KBChunk(
                    chunk_id=f"{doc.id}#{idx}",
                    doc_id=doc.id,
                    title=doc.title,
                    department=doc.department,
                    source=doc.source,
                    updated_at=doc.updated_at,
                    tags=doc.tags,
                    text=text,
                )
            )
    return chunks
