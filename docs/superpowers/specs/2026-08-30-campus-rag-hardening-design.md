# Campus RAG Hardening Design

Date: 2026-08-30

## Objective

Improve the Nanjing University campus KB RAG project in priority order,
starting with correctness and local operability. The first phase must make
configuration errors, malformed knowledge-base records, and interrupted index
builds fail clearly and recoverably without changing the current retrieval
architecture or evaluation corpus.

## Scope

This phase covers:

- validating configuration values at load time;
- validating JSONL knowledge-base records and duplicate document IDs;
- validating retrieval parameters before model or index access;
- writing index artifacts through a temporary build set and publishing them
  atomically;
- preserving the existing manifest contract and explicit `--force` rebuild
  behavior;
- adding focused regression tests for each new failure mode.

This phase does not:

- replace embedding, BM25, FAISS, RRF, or Cross-Encoder components;
- change the knowledge-base documents or evaluation questions;
- retune refusal thresholds;
- change the public answer schema;
- add a service, database, or external runtime dependency.

## Current Data Flow

The pipeline currently resolves configuration, constructs a retriever and
generator, loads or builds a FAISS index, performs hybrid retrieval, optionally
reranks candidates, applies refusal gates, and generates an extractive or LLM
answer. The hardening changes remain within configuration, document loading,
index publication, and retriever input validation.

## Design

### Configuration validation

`load_config()` continues to read YAML and resolve the configuration path, then
validates the fields required by the current pipeline:

- `knowledge_base.path`, `index.dir`, `index.faiss_path`, and
  `index.metadata_path` are non-empty path values;
- `chunk_size` is a positive integer and `chunk_overlap` is an integer in the
  range `0 <= chunk_overlap < chunk_size`;
- `dense_top_k`, `bm25_top_k`, `final_top_k`, and `rrf_k` are positive integers;
- `cross_encoder.enabled` is boolean-like only when represented as a YAML
  boolean, and an enabled Cross-Encoder has a non-empty model name;
- `generation.backend` is `extractive` or `llm`;
- `prompt.refusal_threshold` and `prompt.refusal_ce_threshold` are numeric.

Validation errors are `ValueError` instances that identify the dotted
configuration field. Existing defaults remain unchanged.

### Knowledge-base validation

`load_documents()` validates each non-empty JSONL record before constructing a
`KBDocument`:

- required `id`, `title`, and `text` values must be non-empty strings;
- `tags`, when present, must be a list of strings;
- document IDs must be unique within the file;
- malformed JSON and missing or invalid fields include the source path and line
  number in the error.

The loader keeps optional metadata permissive enough for the current corpus:
`department`, `source`, and `updated_at` remain string-like values. No network
request or date freshness check is introduced.

### Retrieval parameter validation

The retriever validates configured search sizes and `rrf_k` when it is
constructed or before the first search. Runtime `top_k` keeps the existing
positive-integer contract. Invalid values fail before embedding, FAISS, BM25,
or Cross-Encoder work begins.

### Atomic index publication

Index building writes FAISS, chunk metadata, and manifest into a unique
temporary directory located beside the configured index directory. Each
artifact is fully written and readable before publication. The final files are
then replaced in a deterministic order using same-directory replacement.

The implementation must preserve the previous complete index until the new
artifact set is ready. Temporary build directories are cleaned after success
or failure. A failed build raises the original actionable error and does not
silently rebuild or delete the existing valid index.

Because the three files cannot be replaced as one filesystem transaction, the
existing manifest validation remains the final consistency gate. A reader that
observes a mixed set must reject it as incompatible rather than serve
unverified chunks.

### Compatibility

The existing manifest fields remain required:

- schema version;
- embedding model identifier;
- source SHA-256;
- chunk size and overlap;
- chunk count;
- vector count;
- vector dimension.

No current manifest is considered compatible merely because files exist.
Changes to these fields continue to require an explicit forced rebuild.

## Error Handling

- Missing or unreadable YAML: preserve the file error and identify the config
  path.
- Invalid configuration: raise `ValueError` with the dotted field name and
  expected constraint.
- Malformed JSONL or invalid record: raise `ValueError` with path and line.
- Duplicate document ID: raise `ValueError` naming the duplicate ID and line.
- Missing or invalid index artifact: preserve the existing actionable index
  exception behavior.
- Atomic build failure: leave the previous complete artifact set untouched,
  remove only the temporary build directory created by the current run, and
  propagate the failure.

Infrastructure failures remain exceptions and are never converted into an
ordinary low-confidence refusal.

## Testing Strategy

Add or extend tests for:

1. invalid chunk sizes, overlap, retrieval sizes, backend names, and missing
   required paths;
2. malformed JSON, missing required fields, empty required values, invalid tag
   types, and duplicate document IDs;
3. invalid configured retrieval parameters failing before embedding access;
4. successful index builds publishing a complete manifest and metadata set;
5. failed index publication preserving an existing artifact set and cleaning
   temporary output;
6. existing lazy-loading, manifest mismatch, search input, pipeline status, and
   evaluation tests remaining green.

Tests use temporary files and fake embedders/FAISS objects. No test should
download or load real model weights.

## Acceptance Criteria

- Invalid configuration and malformed corpus data fail with actionable,
  location-specific messages.
- Invalid retrieval parameters are rejected before any heavy backend call.
- A failed forced rebuild cannot destroy a previously valid index.
- A successful build leaves a mutually validated FAISS/metadata/manifest set.
- Existing public pipeline response fields and evaluation data remain
  unchanged.
- Focused tests and the full available test suite pass in the project runtime.
- No unrelated user changes are staged or committed.
