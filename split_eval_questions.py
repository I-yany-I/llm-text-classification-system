"""Split the campus KB evaluation corpus into dev/test JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.campus_kb_rag.config import resolve_path


def _bucket(case_id: str, buckets: int = 10) -> int:
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def _load_rows(path: str | Path) -> list[dict]:
    resolved = resolve_path(path)
    rows = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_rows(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split eval questions into dev/test sets.")
    parser.add_argument(
        "--source",
        default="data/campus_kb/eval_questions.jsonl",
        help="Full evaluation corpus to split",
    )
    parser.add_argument(
        "--dev-out",
        default="data/campus_kb/eval_questions_dev.jsonl",
        help="Output JSONL path for the development split",
    )
    parser.add_argument(
        "--test-out",
        default="data/campus_kb/eval_questions_test.jsonl",
        help="Output JSONL path for the test split",
    )
    parser.add_argument(
        "--dev-buckets",
        default="0,1,2",
        help="Hash buckets assigned to dev (out of 10)",
    )
    args = parser.parse_args()

    dev_buckets = {int(x) for x in args.dev_buckets.split(",") if x.strip()}
    rows = _load_rows(args.source)
    dev_rows = [row for row in rows if _bucket(str(row.get("id", ""))) in dev_buckets]
    test_rows = [row for row in rows if _bucket(str(row.get("id", ""))) not in dev_buckets]

    _write_rows(args.dev_out, dev_rows)
    _write_rows(args.test_out, test_rows)
    print(
        json.dumps(
            {
                "source": str(resolve_path(args.source)),
                "dev_rows": len(dev_rows),
                "test_rows": len(test_rows),
                "dev_out": str(Path(args.dev_out)),
                "test_out": str(Path(args.test_out)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
