# QA branch — production flow

```text
User query
  -> Agent Core
      -> VN/EN query variants
      -> answer type / entities / relations / OCR flags
  -> KIS shared retrieval
  -> question-aware retrieval over all query variants
  -> candidate fusion
  -> temporal evidence window
  -> grounding / spatial / counting / OCR evidence
  -> adaptive frame selection
  -> Qwen-VL answer
  -> independent visual verification when needed
  -> answer validation + evidence ranking
  -> submission rows
```

## Canonical debug command

Run from the project root:

```bash
python -m qa.query_runner "What is the man holding?" --event "A man is standing on a stage" --top-k 5
```

For exact competition-facing rows:

```bash
python -m qa.query_runner "What is the man holding?" --event "A man is standing on a stage" --top-k 100 --submission
```

For JSON diagnostics:

```bash
python -m qa.query_runner "How many people are visible?" --event "A group of people on a stage" --top-k 5 --json
```

## Important runtime behavior

`agent_core` owns query translation/normalization. The old `deep-translator` QA layer was removed.

`QA_VLM_VERIFY_ENABLED` defaults to `true` for constrained questions. Deterministic counting only overrides the VLM when multi-frame agreement is strong and no unsupported semantic filter is present.

Spatial `in_front_of`/`behind` relations are never hard-counted from 2-D boxes; they are treated as visual hints and sent to VLM verification.

Temporal questions keep start/middle/end evidence anchors instead of truncating the evidence window before frame selection.

## Tests

```bash
pytest -q
```

The suite covers Agent Core integration, query parsing regressions, counting, spatial safety, frame selection, cache/output helpers, and the existing Agent Core tests.

The self-contained orchestration test can also be run with:

```bash
python -m qa.test_sample_data_e2e
```

This test uses a deterministic fake VLM and does not measure real visual accuracy.

## QA notebook

Canonical interactive validation notebook:

```text
qa/AIC_QA_QUERY_UPDATED.ipynb
```

It performs repository/import smoke tests, Milvus preflight, structured-input inspection, the 50-case benchmark, category metrics, stage diagnostics, and regression tests. Set `RUN_LIVE_BENCHMARK = True` only after the Milvus preflight reports ready.

## Root-level regression command

Run from `qa_merged/`:

```bash
pytest -q
```

The root `pytest.ini` exposes both `qa/` and `agent_core/src/` so Agent Core tests no longer fail during collection.
