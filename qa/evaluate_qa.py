from __future__ import annotations

"""Offline QA benchmark evaluator.

Ground-truth JSONL rows support:
  {question, video_id, frame_id, answer, answer_type?}
Prediction rows support the public QA output plus optional answer_type and
confidence/diagnostics fields.

The evaluator reports exact answer accuracy, exact frame accuracy, temporal
frame tolerance accuracy, video recall, accuracy by answer type, and optional
stage diagnostics. This is intended to answer *where* the QA system fails, not
just whether the final answer matched.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _key(row: dict[str, Any]) -> str:
    # Question should be unique in the benchmark; index duplicates are handled
    # by stable occurrence order in _pair_rows.
    return _norm(row.get("question"))


def _pair_rows(gt_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> list[tuple[dict, dict | None]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for pred in pred_rows:
        buckets[_key(pred)].append(pred)
    used: Counter[str] = Counter()
    pairs = []
    for gt in gt_rows:
        k = _key(gt)
        idx = used[k]
        choices = buckets.get(k, [])
        pred = choices[idx] if idx < len(choices) else None
        used[k] += 1
        pairs.append((gt, pred))
    return pairs


def _answer_match(gt: dict[str, Any], pred: dict[str, Any]) -> bool:
    return _norm(gt.get("answer")) == _norm(pred.get("answer"))


def _frame_match(gt: dict[str, Any], pred: dict[str, Any], tolerance: int = 0) -> bool:
    gvid, pvid = str(gt.get("video_id")), str(pred.get("video_id"))
    gf, pf = _safe_int(gt.get("frame_id")), _safe_int(pred.get("frame_id"))
    return gvid == pvid and gf is not None and pf is not None and abs(gf - pf) <= tolerance


def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _answer_type(row: dict[str, Any]) -> str:
    return str(row.get("answer_type") or "UNKNOWN").upper()


def evaluate(
    gt_rows: list[dict],
    pred_rows: list[dict],
    *,
    frame_tolerance: int = 3,
    diagnostics_rows: list[dict] | None = None,
) -> dict:
    pairs = _pair_rows(gt_rows, pred_rows)
    total = len(gt_rows)
    answered = 0
    exact_answer = 0
    exact_frame = 0
    tolerant_frame = 0
    video_recall = 0
    full_exact = 0
    by_type = defaultdict(lambda: {"total": 0, "answer_correct": 0, "frame_correct": 0, "video_found": 0})

    for gt, pred in pairs:
        if pred is None:
            continue
        answered += 1
        atype = _answer_type(gt)
        bucket = by_type[atype]
        bucket["total"] += 1
        aok = _answer_match(gt, pred)
        fok = _frame_match(gt, pred, 0)
        ftok = _frame_match(gt, pred, frame_tolerance)
        vok = str(gt.get("video_id")) == str(pred.get("video_id"))
        exact_answer += int(aok)
        exact_frame += int(fok)
        tolerant_frame += int(ftok)
        video_recall += int(vok)
        full_exact += int(aok and fok)
        bucket["answer_correct"] += int(aok)
        bucket["frame_correct"] += int(ftok)
        bucket["video_found"] += int(vok)

    out = {
        "questions": total,
        "predictions_matched": answered,
        "coverage": _ratio(answered, total),
        "answer_accuracy": _ratio(exact_answer, total),
        "frame_accuracy_exact": _ratio(exact_frame, total),
        "frame_accuracy_tolerant": _ratio(tolerant_frame, total),
        "video_recall": _ratio(video_recall, total),
        "full_exact_accuracy": _ratio(full_exact, total),
        "frame_tolerance": frame_tolerance,
        "by_answer_type": {},
    }

    for atype, stats in sorted(by_type.items()):
        out["by_answer_type"][atype] = {
            "questions": stats["total"],
            "answer_accuracy": _ratio(stats["answer_correct"], stats["total"]),
            "frame_accuracy_tolerant": _ratio(stats["frame_correct"], stats["total"]),
            "video_recall": _ratio(stats["video_found"], stats["total"]),
        }

    if diagnostics_rows:
        failures = Counter()
        stage_counts = Counter()
        for row in diagnostics_rows:
            for item in row.get("diagnostics", []) or []:
                stage = str(item.get("stage", "unknown"))
                stage_counts[stage] += 1
                if item.get("ok") is False or item.get("error"):
                    failures[stage] += 1
        out["diagnostics"] = {
            "stage_events": dict(stage_counts),
            "stage_failures": dict(failures),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ground_truth", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("--frame-tolerance", type=int, default=3)
    p.add_argument("--diagnostics", type=Path, default=None)
    args = p.parse_args()
    diagnostics = _load(args.diagnostics) if args.diagnostics else None
    result = evaluate(
        _load(args.ground_truth),
        _load(args.predictions),
        frame_tolerance=max(0, args.frame_tolerance),
        diagnostics_rows=diagnostics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
