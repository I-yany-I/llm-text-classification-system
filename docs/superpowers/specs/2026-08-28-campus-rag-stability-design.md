# Campus RAG Stability Design

Date: 2026-08-28

## Objective

Improve the campus knowledge-base RAG project for reliable local demonstration on a 16 GB Windows machine. The application must start without loading large models, detect stale or incompatible indexes, expose clear answer and refusal states, and preserve the existing 126-question evaluation contract.

## Scope

This change keeps the existing retrieval architecture:

- deterministic query rewriting;
- BM25 sparse retrieval;
- Sentence-Transformer and FAISS dense retrieval;
- reciprocal-rank fusion;
- optional Cross-Encoder reranking;
- evidence-based refusal;
- extractive generation by default and optional LLM generation.

The change does not replace models, introduce services, alter the knowledge-base corpus, retune thresholds, or claim new evaluation results.

## Design

### Lazy model loading

`CampusKBRetriever` stores the resolved embedding model source during construction but does not instantiate `SentenceTransformer`. A private accessor creates and caches the embedder only when index building or dense query encoding first needs it. Cross-Encoder loading remains lazy and cached.

Constructing `CampusKBRAG`, including at the module level in `app.py`, therefore loads configuration and lightweight Python objects only. It must not allocate embedding or generation model weights.

### Index manifest and validation

Index building writes a JSON manifest beside the FAISS index and chunk metadata. The manifest records:

- a schema version;
- the configured embedding model identifier;
- the embedding vector dimension;
- a SHA-256 fingerprint of the knowledge-base source file;
- chunk size and overlap;
- chunk count;
- index vector count.

Index loading validates the manifest before serving a query. Missing legacy manifests, changed knowledge-base content, changed chunk parameters, changed embedding model identifiers, vector-dimension mismatches, and chunk/index count mismatches produce a dedicated actionable error telling the operator to rebuild the index. The loader must not silently rebuild during a user query because that can unexpectedly download models and consume memory.

`build(force=False)` may reuse a valid index. If index artifacts exist but validation fails, it reports the incompatibility; `build(force=True)` explicitly rebuilds all artifacts.

### Input and response contract

`CampusKBRAG.ask` normalizes whitespace and validates `top_k`. `top_k` must be a positive integer when supplied. Empty questions return an input-required result without invoking retrieval.

Every normal response includes:

- `status`: `answered`, `refused`, or `input_required`;
- `query`: the normalized user question;
- `search_query`: the rewritten query used for retrieval, or `null` when retrieval was skipped;
- `answer`;
- `citations`;
- `retrieved`;
- `refusal_reason`: `out_of_scope`, `low_confidence`, `sentinel_document`, or `null`.

Low-confidence filtering returns both accepted evidence and a reason instead of losing the reason as an empty list. Citations are generated only from accepted evidence and include the available dense, fusion, and reranker scores for inspection.

Configuration, model, and index failures remain exceptions with actionable messages rather than being represented as ordinary refusals. This prevents infrastructure failures from being misreported as “the knowledge base has no answer.”

### Application behavior

The Gradio application keeps its current interaction and default extractive mode. It displays answer status and refusal reason in the diagnostic output while preserving user-facing Chinese answers and citations. Importing the application module must not load embedding, reranker, or generation model weights.

## Error Handling

- Missing knowledge-base source: raise `FileNotFoundError` naming the expected file.
- Missing index artifacts during search: raise an index-not-ready error with the build command.
- Invalid or stale manifest: raise an index-incompatible error listing the mismatched field and rebuild action.
- Corrupt JSON, FAISS, or manifest content: wrap the underlying failure in an index-load error that identifies the artifact.
- Invalid `top_k`: raise `ValueError` before model or index access.
- Missing model files or failed Hub access: preserve the underlying model-load error and prepend the resolved model source.

## Testing Strategy

Tests use small temporary files and fake embedders/indexes so they do not allocate real model weights.

Required tests:

1. Constructing `CampusKBRetriever` and `CampusKBRAG` does not instantiate `SentenceTransformer`.
2. First embedding operation instantiates the model once and later operations reuse it.
3. A valid manifest permits index loading.
4. Knowledge-base, chunking, model, vector-dimension, and count mismatches are rejected with rebuild guidance.
5. Search never silently rebuilds a missing or invalid index.
6. Empty questions and invalid `top_k` avoid retrieval and return or raise the defined result.
7. Answered, out-of-scope, sentinel-document, and low-confidence paths expose the correct structured status and reason.
8. Citations contain only accepted evidence and retain available retrieval scores.
9. Existing rewrite, scope, retrieval ordering, and refusal-threshold tests continue to pass.

## Acceptance Criteria

- Importing and constructing the default pipeline does not load model weights.
- Existing indexes cannot be used after their model, source corpus, or chunking contract changes.
- Search does not trigger a surprise index rebuild.
- Every successful `ask` result has an explicit status and deterministic refusal reason.
- The full unit-test suite and configured static checks pass.
- A lightweight smoke test proves startup behavior without running a second heavyweight model process.
- The 126-question dataset and metric definitions remain unchanged. Full model evaluation is run only when sufficient memory is available.
