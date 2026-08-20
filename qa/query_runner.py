from __future__ import annotations

import argparse
import json
import os

from .pipeline import QASystem
from .qa_types import build_structured_query
from .qa_output import validate_submission_rows, format_submission


def _format_mmss(seconds):
    if seconds is None:
        return "--:--"
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "--:--"
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def main():
    p=argparse.ArgumentParser(description="AIC QA: KIS retrieval -> frame evidence -> VLM -> ranked answers")
    p.add_argument("question", help="Question to answer")
    p.add_argument("--event", default=None, help="KIS/video retrieval description; defaults to the question")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true", help="Print exact required QA rows as JSON")
    p.add_argument("--submission", action="store_true", help="Print only competition-facing CSV rows")
    p.add_argument("--tuples", action="store_true", help="Print exact Python-style list of (video_id, frame_id, answer, score)")
    args=p.parse_args()

    event=args.event or args.question
    qa=QASystem.from_shared_kis()
    rows=qa.solve_qa(build_structured_query(event), args.question, top_k_qa=args.top_k)
    if not rows:
        raise SystemExit("No answer rows returned. Check KIS/Milvus/data paths.")
    if args.submission:
        print(format_submission(rows))
    elif args.tuples:
        tuples=[(r["video_id"], int(r["frame_id"]), r["answer"], float(r.get("score", 0.0))) for r in rows]
        print(repr(tuples))
    elif args.json:
        print(json.dumps(validate_submission_rows(rows), ensure_ascii=False, indent=2))
    else:
        print(f"QUERY: {args.question}")
        for i, row in enumerate(rows, 1):
            try:
                ts = qa.frame_loader.timestamp_sec(row["video_id"], int(row["frame_id"]))
            except Exception:
                ts = None
            print(f"#{i} {row['video_id']} | frame={row['frame_id']} | time={_format_mmss(ts)} | answer={str(row['answer']).strip()}")

if __name__ == "__main__":
    main()
