# Campus RAG Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the campus RAG configuration, knowledge-base loading, retrieval parameters, and index rebuild workflow fail clearly and recover safely without changing the RAG architecture or evaluation data.

**Architecture:** Keep `CampusKBRAG`, `CampusKBRetriever`, JSONL documents, FAISS, BM25, RRF, Cross-Encoder, and the existing manifest contract. Add validation at the configuration/document boundaries and publish rebuilt FAISS, metadata, and manifest artifacts through a temporary build directory with rollback on publication failure.

**Tech Stack:** Python 3.10+, PyYAML, pytest, NumPy, FAISS, rank-bm25, Sentence-Transformers.

---

## File Map

- Modify `src/campus_kb_rag/config.py`: validate the loaded YAML contract and keep path/model helpers unchanged.
- Modify `src/campus_kb_rag/documents.py`: validate JSONL syntax, required fields, tag types, and duplicate IDs with line-aware errors.
- Modify `src/campus_kb_rag/retriever.py`: validate retrieval settings and isolate index artifact writing/publication.
- Modify `tests/test_config.py`: cover invalid configuration values and field-specific messages.
- Modify `tests/test_retriever.py`: cover invalid retrieval settings and atomic build behavior.
- Create `tests/test_documents.py`: cover malformed and invalid knowledge-base records.
- Run existing `tests/test_pipeline.py`, `tests/test_startup.py`, `tests/test_app.py`, and the full suite as regression coverage.

## Task 1: Validate Loaded Configuration

**Files:**
- Modify: `src/campus_kb_rag/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for invalid configuration**

Add tests that write temporary YAML files and assert the dotted field name is
included in the `ValueError`:

```python
import yaml
import pytest


def write_config(tmp_path, overrides):
    config = {
        "knowledge_base": {
            "path": "data/kb.jsonl",
            "chunk_size": 420,
            "chunk_overlap": 80,
        },
        "index": {
            "dir": "vector_store",
            "faiss_path": "vector_store/faiss.index",
            "metadata_path": "vector_store/chunks.json",
        },
        "retrieval": {
            "embedding_model": "test-model",
            "dense_top_k": 12,
            "bm25_top_k": 12,
            "final_top_k": 8,
            "rrf_k": 60,
            "cross_encoder": {
                "enabled": False,
                "model_name": "test-cross-encoder",
            },
        },
        "generation": {"backend": "extractive"},
        "prompt": {
            "refusal_threshold": 0.18,
            "refusal_ce_threshold": -0.25,
        },
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("section", "values", "field"),
    [
        ("knowledge_base", {"chunk_size": 0}, "knowledge_base.chunk_size"),
        ("knowledge_base", {"chunk_overlap": 420}, "knowledge_base.chunk_overlap"),
        ("retrieval", {"dense_top_k": 0}, "retrieval.dense_top_k"),
        ("retrieval", {"rrf_k": -1}, "retrieval.rrf_k"),
        ("generation", {"backend": "unknown"}, "generation.backend"),
    ],
)
def test_load_config_rejects_invalid_values(tmp_path, section, values, field):
    path = write_config(tmp_path, {section: values})
    with pytest.raises(ValueError, match=field):
        cfg.load_config(path)


def test_load_config_rejects_empty_required_path(tmp_path):
    path = write_config(tmp_path, {"knowledge_base": {"path": ""}})
    with pytest.raises(ValueError, match="knowledge_base.path"):
        cfg.load_config(path)


def test_load_config_rejects_enabled_cross_encoder_without_model(tmp_path):
    path = write_config(
        tmp_path,
        {"retrieval": {"cross_encoder": {"enabled": True, "model_name": ""}}},
    )
    with pytest.raises(ValueError, match="retrieval.cross_encoder.model_name"):
        cfg.load_config(path)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_config.py -k "rejects_invalid or empty_required or enabled_cross_encoder" -q
```

Expected: the tests fail because `load_config()` currently returns YAML
without validating the contract.

- [ ] **Step 3: Implement minimal configuration validation**

Add a private `_validate_config(cfg)` called by `load_config()` after
`safe_load`. Use explicit helpers so YAML booleans are not accepted as integer
values:

```python
def _require_mapping(cfg: dict, field: str) -> dict:
    value = cfg.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _positive_int(value, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _non_empty_path(value, field: str) -> None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty path")
```

Validate the fields listed in the design, defaulting optional sections to
empty mappings only where the current runtime already supplies defaults.
Return the original configuration with `_config_path` unchanged.

- [ ] **Step 4: Run configuration tests**

Run:

```powershell
python -m pytest tests/test_config.py -q
```

Expected: all configuration tests pass, including the existing model-source
resolution tests.

- [ ] **Step 5: Commit the focused change**

```powershell
git add src/campus_kb_rag/config.py tests/test_config.py
git commit -m "feat: validate campus rag configuration"
```

## Task 2: Validate JSONL Knowledge-Base Documents

**Files:**
- Modify: `src/campus_kb_rag/documents.py`
- Create: `tests/test_documents.py`

- [ ] **Step 1: Write failing document-loader tests**

Create tests for malformed JSON, missing/empty required values, invalid tags,
and duplicate IDs:

```python
import json
import pytest

from src.campus_kb_rag.documents import load_documents


def write_lines(path, rows):
    path.write_text(
        "\n".join(
            row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def valid_row(doc_id="doc-1"):
    return {
        "id": doc_id,
        "title": "校园卡补办",
        "text": "先挂失，再补办。",
        "tags": ["校园卡"],
    }


def test_loader_reports_malformed_json_with_line(tmp_path):
    source = tmp_path / "kb.jsonl"
    write_lines(source, [valid_row(), "{"])
    with pytest.raises(ValueError, match=r"kb\.jsonl:2"):
        load_documents(source)


@pytest.mark.parametrize(
    "row",
    [
        {**valid_row(), "id": ""},
        {**valid_row(), "title": " "},
        {**valid_row(), "text": ""},
        {**valid_row(), "tags": "校园卡"},
        {**valid_row(), "tags": [1]},
    ],
)
def test_loader_rejects_invalid_record_with_line(tmp_path, row):
    source = tmp_path / "kb.jsonl"
    write_lines(source, [row])
    with pytest.raises(ValueError, match=r"kb\.jsonl:1"):
        load_documents(source)


def test_loader_rejects_duplicate_document_id_with_line(tmp_path):
    source = tmp_path / "kb.jsonl"
    write_lines(source, [valid_row(), valid_row()])
    with pytest.raises(ValueError, match=r"kb\.jsonl:2.*doc-1"):
        load_documents(source)
```

- [ ] **Step 2: Run the tests and verify the expected failures**

Run:

```powershell
python -m pytest tests/test_documents.py -q
```

Expected: failures show that malformed JSON and invalid records currently
escape as raw `JSONDecodeError` or are silently coerced.

- [ ] **Step 3: Implement line-aware document validation**

In `load_documents`, parse each line in a `try/except json.JSONDecodeError` and
raise `ValueError(f"{jsonl_path}:{line_no} invalid JSON: ...")`. Before
constructing `KBDocument`, require non-empty string `id`, `title`, and `text`;
require `tags` to be a list whose members are strings; and track IDs in a set.
Do not add date parsing or network freshness checks.

- [ ] **Step 4: Run document and existing chunking tests**

Run:

```powershell
python -m pytest tests/test_documents.py tests/test_retriever.py::TestKBChunk -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit the focused change**

```powershell
git add src/campus_kb_rag/documents.py tests/test_documents.py
git commit -m "feat: validate campus knowledge base records"
```

## Task 3: Reject Invalid Retrieval Settings Before Heavy Work

**Files:**
- Modify: `src/campus_kb_rag/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
@pytest.mark.parametrize(
    "field",
    ["dense_top_k", "bm25_top_k", "final_top_k", "rrf_k"],
)
def test_retriever_rejects_invalid_retrieval_setting(mock_config, field):
    mock_config["retrieval"][field] = 0
    with pytest.raises(ValueError, match=field):
        CampusKBRetriever(mock_config)


def test_invalid_retrieval_setting_fails_before_embedder_access(mock_config):
    mock_config["retrieval"]["dense_top_k"] = -1
    with patch("src.campus_kb_rag.retriever.SentenceTransformer") as model:
        with pytest.raises(ValueError, match="dense_top_k"):
            CampusKBRetriever(mock_config)
    model.assert_not_called()
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```powershell
python -m pytest tests/test_retriever.py -k "invalid_retrieval_setting" -q
```

Expected: current construction accepts invalid retrieval values.

- [ ] **Step 3: Implement the minimal retriever validation**

Add `_validate_retrieval_config(ret_cfg)` and call it before assigning any
retrieval runtime state. Validate positive integer values for
`dense_top_k`, `bm25_top_k`, `final_top_k`, and `rrf_k`; validate
`cross_encoder.rerank_pool` when present; and reject an enabled Cross-Encoder
without a non-empty model name. Keep runtime `top_k` validation unchanged.

- [ ] **Step 4: Run the complete retriever tests**

Run:

```powershell
python -m pytest tests/test_retriever.py -q
```

Expected: all existing lazy-loading, manifest, ordering, and search-boundary
tests remain green.

- [ ] **Step 5: Commit the focused change**

```powershell
git add src/campus_kb_rag/retriever.py tests/test_retriever.py
git commit -m "feat: validate campus rag retrieval settings"
```

## Task 4: Publish Rebuilt Index Artifacts Safely

**Files:**
- Modify: `src/campus_kb_rag/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write a failing test for failed publication rollback**

Add a test that creates an existing three-file artifact set, patches the fake
FAISS writer to create new output, forces the second publication replacement
to fail, and asserts the old file bytes remain unchanged:

```python
def test_failed_build_preserves_existing_artifacts(tmp_path, monkeypatch):
    retriever, fake_faiss, old_files = make_build_retriever(tmp_path)
    fake_faiss.write_index.side_effect = lambda index, path: Path(path).write_bytes(
        b"new-faiss"
    )

    real_replace = os.replace
    calls = {"count": 0}

    def fail_second_replace(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated publish failure"):
        retriever.build(force=True)

    for path, content in old_files.items():
        assert path.read_bytes() == content
    assert not list(tmp_path.glob(".campus-kb-build-*"))
```

The helper should use a fake embedder and fake FAISS implementation, matching
the existing `test_build_writes_manifest_without_loading_real_model` pattern.

- [ ] **Step 2: Run the rollback test and verify failure**

Run:

```powershell
python -m pytest tests/test_retriever.py::test_failed_build_preserves_existing_artifacts -q
```

Expected: the test fails because the current build writes directly to final
paths and has no rollback boundary.

- [ ] **Step 3: Implement temporary artifact publication**

Refactor `build()` into these bounded operations:

1. Load documents, chunk them, encode vectors, and construct the in-memory
   FAISS/BM25 state exactly as today.
2. Create a unique temporary directory beside `index_dir`, such as
   `tempfile.mkdtemp(prefix=".campus-kb-build-", dir=str(index_dir.parent))`.
3. Write FAISS, metadata, and manifest to temporary paths.
4. Parse the temporary metadata and manifest and read the temporary FAISS
   index before publication.
5. Copy existing final artifacts to a backup directory.
6. Replace the three final files with `os.replace` in the order FAISS,
   metadata, manifest.
7. If any replacement fails, restore every existing artifact from the backup
   and remove newly created targets that had no previous file.
8. Always remove only the temporary build and backup directories created by
   this call.

Keep the in-memory `self.index`, `self.chunks`, and `self.bm25` assignment
after successful publication. Existing manifest validation remains the loader
contract for readers that observe publication between file replacements.

- [ ] **Step 4: Run focused index tests**

Run:

```powershell
python -m pytest tests/test_retriever.py -q
```

Expected: all existing manifest tests plus the rollback test pass.

- [ ] **Step 5: Commit the focused change**

```powershell
git add src/campus_kb_rag/retriever.py tests/test_retriever.py
git commit -m "feat: publish campus rag index artifacts atomically"
```

## Task 5: Run Regression Checks and Update Phase Documentation

**Files:**
- Modify: `README.md`
- Modify: `INTERVIEW_PREP.md` only if a documented command is inaccurate

- [ ] **Step 1: Write a documentation check**

Run:

```powershell
rg -n "configuration|JSONL|duplicate|rebuild|manifest|--force" README.md
```

Use the result to identify missing user-facing instructions; do not change
code in this step.

- [ ] **Step 2: Update the documentation**

Document that:

- malformed KB records fail with file/line diagnostics;
- changing config, corpus, model, or chunking requires `--force`;
- a failed rebuild preserves the last complete artifact set;
- invalid retrieval parameters fail during startup/configuration rather than
  after a model has loaded.

- [ ] **Step 3: Run static and focused checks**

Use a project interpreter that has pytest installed. The following PowerShell
snippet searches the current project and nearby project environments, then
fails explicitly if none is available:

```powershell
$python = Get-ChildItem -Path .,.. -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
  Where-Object {
    & $_.FullName -c "import pytest" 2>$null
    $LASTEXITCODE -eq 0
  } |
  Select-Object -First 1
if (-not $python) {
  throw "No Python interpreter with pytest was found."
}
```

Run the selected interpreter with:

```powershell
& $python.FullName -m pytest tests/test_config.py tests/test_documents.py tests/test_retriever.py tests/test_pipeline.py tests/test_startup.py tests/test_app.py -q
& $python.FullName -m compileall -q src app.py build_campus_kb_index.py evaluate_campus_kb.py
git diff --check
```

- [ ] **Step 4: Run the full available test suite**

```powershell
& $python.FullName -m pytest -q
```

Expected: all tests pass without loading real model weights. If no installed
interpreter has pytest, report that exact environment limitation rather than
claiming the suite passed.

- [ ] **Step 5: Inspect the final diff**

```powershell
git status --short
git diff --stat
git diff --check
```

Confirm that only phase-one files changed in the implementation diff and that
the user's pre-existing staged/unstaged corpus and evaluation changes were not
reverted or restaged.

- [ ] **Step 6: Commit documentation and verified implementation**

```powershell
git add README.md INTERVIEW_PREP.md src tests
git commit -m "chore: harden campus rag local workflow"
```

If the sandbox still cannot update the branch reference, leave the working
tree intact and report the exact Git permission failure.

## Plan Self-Review

- Configuration requirements map to Task 1.
- JSONL record and duplicate-ID requirements map to Task 2.
- Retrieval parameter preconditions map to Task 3.
- Temporary writes, rollback, cleanup, and manifest compatibility map to Task 4.
- Regression and documentation requirements map to Task 5.
- No task changes the corpus, evaluation questions, public response schema, or
  retrieval architecture.
- Every production behavior change has a preceding failing-test step.
