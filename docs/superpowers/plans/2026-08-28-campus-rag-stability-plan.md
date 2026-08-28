# Campus RAG Stability Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the campus RAG pipeline safe and diagnosable on a 16 GB local machine without changing its retrieval architecture or the existing 126-question evaluation contract.

**Architecture:** Keep BM25 + FAISS dense retrieval, RRF fusion, optional Cross-Encoder reranking, and extractive/LLM generation. Move model construction behind cached accessors, add a manifest-backed index contract, and make `ask()` return explicit status and refusal metadata. The Gradio layer consumes those fields but does not own pipeline logic.

**Tech Stack:** Python 3.10+, PyYAML, FAISS, rank-bm25, Sentence-Transformers, NumPy, pytest, Gradio.

---

## File Map

- Modify `src/campus_kb_rag/config.py`: add stable file hashing and normalized model identifiers used by the index contract.
- Modify `src/campus_kb_rag/retriever.py`: defer embedding-model construction, create/read/validate the manifest, and expose actionable index errors.
- Modify `src/campus_kb_rag/pipeline.py`: validate inputs, return status/search/refusal fields, and preserve infrastructure exceptions.
- Modify `src/campus_kb_rag/generator.py`: keep generation lazy and make refusal text independent from infrastructure failures.
- Modify `app.py`: display the structured status and diagnostic fields without changing the question-answer interaction.
- Modify `tests/test_config.py`: test deterministic source fingerprints and model identifiers.
- Modify `tests/test_retriever.py`: test lazy loading, manifest creation/validation, count mismatches, and no implicit rebuild.
- Modify `tests/test_pipeline.py`: test input validation, answer/refusal states, reasons, rewritten query, and scored citations.
- Create `tests/test_generator.py`: add a focused regression test for lazy generation construction.
- Create `tests/test_app.py`: verify the Gradio callback exposes structured diagnostics without loading models.
- Modify `README.md`: document the build-before-query workflow, manifest validation, and lightweight default startup behavior.

## Task 1: Add Configuration Primitives

**Files:**
- Modify `src/campus_kb_rag/config.py`
- Test `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add tests for a deterministic SHA-256 fingerprint and a stable model source identifier. The identifier must use the resolved local path when a local model is selected, while a Hub id remains unchanged.

```python
def test_file_sha256_changes_when_content_changes(tmp_path):
    source = tmp_path / "kb.jsonl"
    source.write_text("first", encoding="utf-8")
    first = cfg.file_sha256(source)
    source.write_text("second", encoding="utf-8")
    assert cfg.file_sha256(source) != first


def test_model_identifier_is_stable_for_hub_id():
    assert cfg.model_identifier("sentence-transformers/demo") == (
        "sentence-transformers/demo"
    )
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```powershell
python -m pytest tests/test_config.py::test_file_sha256_changes_when_content_changes tests/test_config.py::test_model_identifier_is_stable_for_hub_id -q
```

Expected: FAIL because `file_sha256` and `model_identifier` do not exist.

- [ ] **Step 3: Implement the minimal helpers**

In `config.py`, import `hashlib` and add:

```python
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identifier(source: str | Path) -> str:
    return str(Path(source).resolve()) if Path(str(source)).exists() else str(source)
```

- [ ] **Step 4: Run the focused tests**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/campus_kb_rag/config.py tests/test_config.py
git commit -m "feat: add campus rag index fingerprints"
```

## Task 2: Make Retriever Construction Lightweight

**Files:**
- Modify `src/campus_kb_rag/retriever.py`
- Test `tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

Patch `SentenceTransformer` with a constructor spy and assert construction does not call it. Add a second test that calls `_get_embedder()` twice and asserts one construction.

```python
def test_retriever_does_not_load_embedder_on_construction(mock_config):
    with patch("src.campus_kb_rag.retriever.SentenceTransformer") as model:
        retriever = CampusKBRetriever(mock_config)
    model.assert_not_called()
    assert retriever._embedder is None


def test_embedder_is_created_once_on_first_use(mock_config):
    fake_embedder = MagicMock()
    with patch(
        "src.campus_kb_rag.retriever.SentenceTransformer",
        return_value=fake_embedder,
    ) as model:
        retriever = CampusKBRetriever(mock_config)
        assert retriever._get_embedder() is fake_embedder
        assert retriever._get_embedder() is fake_embedder
    model.assert_called_once_with(retriever.embedding_model_name)
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
python -m pytest tests/test_retriever.py::test_retriever_does_not_load_embedder_on_construction tests/test_retriever.py::test_embedder_is_created_once_on_first_use -q
```

Expected: FAIL because construction currently instantiates `SentenceTransformer` and `_get_embedder` is absent.

- [ ] **Step 3: Implement lazy embedding access**

Replace `self.embedder = SentenceTransformer(...)` with `self._embedder = None`, add:

```python
def _get_embedder(self):
    if self._embedder is None:
        self._embedder = SentenceTransformer(self.embedding_model_name)
    return self._embedder
```

Use `_get_embedder()` in `build()` and `_encode_query()`. Keep the Cross-Encoder cache unchanged.

- [ ] **Step 4: Run focused and existing retriever tests**

```powershell
python -m pytest tests/test_retriever.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/campus_kb_rag/retriever.py tests/test_retriever.py
git commit -m "perf: lazy load campus embedding model"
```

## Task 3: Add and Enforce the Index Manifest

**Files:**
- Modify `src/campus_kb_rag/retriever.py`
- Test `tests/test_retriever.py`

- [ ] **Step 1: Write failing manifest tests**

Add temporary-file tests that construct a three-chunk fake index and manifest. Test that a valid manifest loads, a changed source fingerprint raises `IndexIncompatibleError` containing `rebuild`, and a missing manifest raises instead of calling `build(force=True)`.

```python
def test_load_accepts_valid_manifest(retriever_with_temp_index):
    retriever_with_temp_index.load()
    assert len(retriever_with_temp_index.chunks) == 3


def test_load_rejects_changed_source(retriever_with_temp_index, tmp_path):
    retriever_with_temp_index.kb_path.write_text("changed", encoding="utf-8")
    with pytest.raises(IndexIncompatibleError, match="rebuild"):
        retriever_with_temp_index.load()


def test_load_does_not_rebuild_when_manifest_is_missing(retriever_with_temp_index):
    retriever_with_temp_index.manifest_path.unlink()
    retriever_with_temp_index.build = MagicMock()
    with pytest.raises(IndexIncompatibleError, match="manifest"):
        retriever_with_temp_index.load()
    retriever_with_temp_index.build.assert_not_called()
```

- [ ] **Step 2: Run the manifest tests and verify failure**

```powershell
python -m pytest tests/test_retriever.py -k manifest -q
```

Expected: FAIL because there is no manifest path, manifest writer, validator, or dedicated error.

- [ ] **Step 3: Implement the manifest contract**

Add `IndexIncompatibleError(ValueError)` and a `manifest_path` derived from the configured index directory. Add `_manifest_payload(vector_dim)` using `file_sha256`, `model_identifier`, chunk parameters, and counts. Write it after FAISS and metadata are written.

Add `_validate_manifest()` that:

1. checks the manifest exists and parses as JSON;
2. compares schema, model id, source hash, chunk size, overlap, chunk count, vector count, and FAISS dimension;
3. raises `IndexIncompatibleError("... rebuild ...")` with the first mismatched field;
4. validates metadata count against `index.ntotal` before assigning loaded state.

Change `load()` to fail on missing artifacts or manifest. Only `build(force=True)` may create a new index. Change `build(force=False)` to validate an existing artifact set before returning.

- [ ] **Step 4: Run focused tests and existing retriever tests**

```powershell
python -m pytest tests/test_retriever.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/campus_kb_rag/retriever.py tests/test_retriever.py
git commit -m "feat: validate campus rag index manifest"
```

## Task 4: Define Structured Pipeline Results

**Files:**
- Modify `src/campus_kb_rag/pipeline.py`
- Test `tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Add tests asserting:

```python
def test_empty_query_returns_input_required(pipeline):
    result = pipeline.ask("  \n")
    assert result["status"] == "input_required"
    assert result["refusal_reason"] is None
    pipeline.retriever.search.assert_not_called()


def test_invalid_top_k_fails_before_retrieval(pipeline):
    with pytest.raises(ValueError, match="top_k"):
        pipeline.ask("校园卡怎么办", top_k=0)
    pipeline.retriever.search.assert_not_called()


def test_out_of_scope_result_has_reason(pipeline):
    result = pipeline.ask("讲一个笑话")
    assert result["status"] == "refused"
    assert result["refusal_reason"] == "out_of_scope"
    assert result["search_query"] is None


def test_answered_result_contains_rewritten_query_and_scores(pipeline):
    result = pipeline.ask("校卡怎么补办")
    assert result["status"] == "answered"
    assert result["search_query"]
    assert result["refusal_reason"] is None
    assert result["citations"][0]["dense_score"] == pytest.approx(0.8)
```

Use the existing fake retriever/generator fixtures and set `filter_low_confidence` to return both evidence and a reason, e.g. `([], "low_confidence")` for refusal.

- [ ] **Step 2: Run focused tests and verify failure**

```powershell
python -m pytest tests/test_pipeline.py -q
```

Expected: FAIL because current responses lack `status`, `search_query`, `refusal_reason`, and the filter only returns a list.

- [ ] **Step 3: Implement the minimal response contract**

Add a private `_validate_top_k` that accepts `None` or positive integers and raises `ValueError` otherwise. Make `_filter_low_confidence` return `(evidence, reason)` while preserving existing threshold behavior. Set:

- empty input: `input_required`;
- scope rejection: `refused/out_of_scope`;
- sentinel document: `refused/sentinel_document`;
- threshold rejection: `refused/low_confidence`;
- non-empty evidence: `answered`.

Set `search_query` to the rewritten query only when retrieval runs. Extend `_citations` with `dense_score`, `rrf_score`, and `cross_encoder_score` when present, while retaining the existing `score` field.

- [ ] **Step 4: Run pipeline and full unit tests**

```powershell
python -m pytest tests/test_pipeline.py tests/test_rewrite.py tests/test_scope.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/campus_kb_rag/pipeline.py tests/test_pipeline.py
git commit -m "feat: expose structured campus rag outcomes"
```

## Task 5: Keep Generation and Application Lightweight

**Files:**
- Modify `src/campus_kb_rag/generator.py`
- Modify `app.py`
- Create `tests/test_generator.py`
- Create `tests/test_app.py`

- [ ] **Step 1: Write failing startup/display tests**

Test that constructing `CampusAnswerGenerator` does not import or instantiate the text-generation pipeline, and that `app.answer` includes status and refusal reason in its diagnostic output when returned by the pipeline.

```python
def test_generator_construction_does_not_load_llm():
    generator = CampusAnswerGenerator({"generation": {"backend": "llm"}})
    assert generator._pipe is None
```

- [ ] **Step 2: Run focused tests and verify failure**

```powershell
python -m pytest tests/test_generator.py tests/test_app.py -q
```

Expected: the app diagnostic assertion fails before the display fields are added; the generator regression test passes because generation is already lazy.

- [ ] **Step 3: Implement diagnostic display only**

Keep `_get_pipeline()` lazy. Update `app.answer` to include a compact diagnostic header such as `status=...` and `refusal_reason=...` in the existing retrieved/evidence output. Do not instantiate a second pipeline or alter user-facing answer generation.

- [ ] **Step 4: Run app/generator tests**

```powershell
python -m pytest tests/test_generator.py tests/test_app.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/campus_kb_rag/generator.py app.py tests/test_generator.py tests/test_app.py
git commit -m "feat: expose campus rag diagnostics without eager loading"
```

## Task 6: Update Documentation and Verify End to End

**Files:**
- Modify `README.md`
- Modify `INTERVIEW_PREP.md` only if current commands or response examples become inaccurate.

- [ ] **Step 1: Write documentation checks**

Use repository text checks to require the documented build command, manifest behavior, and structured response keys.

```powershell
rg -n "manifest|build_index|status|refusal_reason|search_query" README.md
```

Expected before the edit: at least one required concept is absent.

- [ ] **Step 2: Update documentation**

Document:

- `python build_campus_kb_index.py --force` as the explicit build path;
- search requires a valid manifest and will not silently rebuild;
- default extractive mode does not load the optional generation model;
- `ask()` response statuses and refusal reasons;
- how to use a local model directory to avoid Hub downloads.

- [ ] **Step 3: Run the complete test and static checks**

```powershell
python -m pytest -q
ruff check src tests app.py build_campus_kb_index.py evaluate_campus_kb.py
git diff --check
```

Expected: all tests and lint checks pass with no whitespace errors.

- [ ] **Step 4: Run the lightweight import smoke test**

```powershell
python -m pytest tests/test_app.py::test_importing_app_does_not_load_embedding_model -q
```

Expected: PASS, with no embedding model constructor call and no real model download.

- [ ] **Step 5: Inspect process memory before any heavyweight evaluation**

Confirm the user-owned Python process is no longer running or has released enough memory. Do not terminate it. Only then run the existing fake/offline evaluation and, if safe, the full 126-question evaluation.

- [ ] **Step 6: Commit documentation and verified implementation**

```powershell
git add README.md INTERVIEW_PREP.md src tests app.py
git commit -m "chore: verify campus rag stability workflow"
```

## Self-Review Checklist

- [x] The plan covers lazy embedding and generation construction.
- [x] The plan covers manifest creation, all specified mismatch classes, and no implicit rebuild.
- [x] The plan covers input validation and all refusal reasons in the design.
- [x] The plan preserves retrieval architecture and evaluation data.
- [x] Every code change has a preceding failing test step, except documentation and configuration-only verification.
- [x] Every task names concrete files and commands.
- [x] No task requires a second heavyweight model process while the user's process is active.
