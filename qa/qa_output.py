from typing import Iterable, Mapping, List

REQUIRED_KEYS = ("video_id", "frame_id", "answer")


def validate_submission_rows(rows: Iterable[Mapping]) -> List[dict]:
    rows = list(rows)
    if len(rows) > 100:
        raise ValueError("AIC screening accepts at most 100 answers per query")
    validated = []
    for i, row in enumerate(rows, 1):
        missing = [k for k in REQUIRED_KEYS if k not in row]
        if missing:
            raise ValueError(f"Row {i} missing required fields: {missing}")
        try:
            frame_id = int(row["frame_id"])
        except (TypeError, ValueError):
            raise ValueError(f"Row {i} frame_id must be an integer")
        answer = str(row["answer"]).strip()
        if not answer:
            raise ValueError(f"Row {i} answer must be non-empty")
        if len(answer) > 100:
            raise ValueError(f"Row {i} answer exceeds the 100-character limit")
        validated.append({"video_id": str(row["video_id"]), "frame_id": frame_id, "answer": answer})
    return validated


def format_submission(rows: Iterable[Mapping]) -> str:
    rows = validate_submission_rows(rows)
    return "\n".join(f"{r['video_id']},{r['frame_id']},{r['answer']}" for r in rows)


def format_debug(rows: Iterable[Mapping]) -> str:
    return "\n".join(repr(dict(r)) for r in rows)
