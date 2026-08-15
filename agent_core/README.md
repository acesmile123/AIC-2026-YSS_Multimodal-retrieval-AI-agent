# AIC 2026 Agent Core

Query-understanding layer for the three preliminary-round tasks:

- **KIS**: semantic parsing and single-event retrieval planning.
- **QA**: event retrieval plus a separate VQA question/answer type.
- **TRAKE**: whole-video retrieval followed by ordered semantic-keyframe alignment.

The implementation uses the official `google-genai` SDK, Gemini structured output,
and one common Pydantic schema consumed by the CLIP/object/OCR/ASR/VQA/TRAKE branches.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`, then:

```python
from aic_agent_core import route_query

task_type, structured_query = route_query(
    "Tìm cảnh một diễn giả mặc áo đỏ phát biểu ngoài trời, phía sau có cây xanh.",
    query_id="q_001",
)

print(task_type)                         # TaskType.KIS
print(structured_query.model_dump())     # does not contain task_type
```

For FastAPI or another async orchestrator:

```python
from aic_agent_core import aroute_query

task_type, structured_query = await aroute_query(raw_text)
```

## Downstream dispatch

Use `task_type` for the hard branch, `query_variants` for text/CLIP retrieval,
`entities` for object filtering, and `needs_ocr`/`needs_asr` to enable those indexes.
For TRAKE, `events` is guaranteed to contain at
least two chronologically indexed events; downstream alignment should enforce
`frame_1 < ... < frame_N` within each retrieved candidate video.

## Interactive test

After adding `GOOGLE_API_KEY` to `.env`, run:

```powershell
.\.venv\Scripts\python.exe -m aic_agent_core
```

Paste a Vietnamese query, optionally enter a query ID, and the complete JSON is
printed to the terminal. If query ID is omitted, the router creates one automatically.

## Tests

```powershell
pytest
```

Tests inject a fake Gemini client, so they do not consume API quota.
