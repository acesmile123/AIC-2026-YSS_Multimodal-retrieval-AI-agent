from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "sample_data"
os.environ.setdefault("QA_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("QA_DATA_DIR", str(DATA_ROOT))
os.environ.setdefault("QA_KEYFRAMES_DIR", str(DATA_ROOT))
os.environ.setdefault("QA_VLM_VERIFY_ENABLED", "false")
os.environ.setdefault("QA_QUESTION_SEMANTIC_ENABLED", "false")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.pipeline import QASystem
from qa.qa_types import build_structured_query
from qa.qa_output import validate_submission_rows
from qa import qa_config
from qa.frame_loader import TemporalFrameLoader
qa_config.DATA_DIR = str(DATA_ROOT)
qa_config.KEYFRAMES_DIR = str(DATA_ROOT)

class CaptionKIS:
    """Small deterministic KIS stand-in with no external sample-data dependency."""
    def __init__(self, rows):
        self.rows = rows

    @staticmethod
    def _score(query: str, caption: str) -> float:
        q = set(str(query).lower().split())
        c = set(str(caption).lower().split())
        if not q or not c:
            return 0.0
        return len(q & c) / max(1, len(q))

    def retrieve(self, structured_query):
        query = structured_query.get("raw_query", "")
        out = []
        for r in self.rows:
            s = self._score(query, r.get("caption", ""))
            if s > 0:
                out.append((r["video_id"], int(r["frame_id"]), float(s)))
        return sorted(out, key=lambda x: x[2], reverse=True)[:20]

    def retrieve_for_question(self, question, top_k=100):
        return self.retrieve({"raw_query": question})[:top_k]


class DeterministicVLM:
    """Dependency-free VLM double; it proves orchestration, not visual accuracy."""
    def generate_answer(self, frames, question, evidence_context=""):
        text=(question or "").lower()
        if any(k in text for k in ("what is visible", "what can be seen", "what is in")):
            return "a visual scene", 0.9
        return "unknown", 0.2

    def verify_answer(self, frames, question, answer, evidence_context=""):
        return 1.0 if answer and answer != "unknown" else 0.2

def main():
    import tempfile
    from PIL import Image, ImageDraw

    with tempfile.TemporaryDirectory(prefix="qa_e2e_") as tmp:
        data_dir = Path(tmp)
        video_id = "E2E_VIDEO"
        video_dir = data_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            {"video_id": video_id, "frame_id": 1, "caption": "a city skyline with a large sign in the foreground"},
            {"video_id": video_id, "frame_id": 2, "caption": "a city skyline with a large sign in the foreground"},
        ]
        (data_dir / "captions.json").write_text(json.dumps(rows), encoding="utf-8")
        (data_dir / f"{video_id}.csv").write_text("frame_idx,n\\n1,1\\n2,2\\n", encoding="utf-8")
        for idx in (1, 2):
            image = Image.new("RGB", (320, 240), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 40, 300, 200), outline="black", width=3)
            draw.text((60, 100), "CITY SIGN", fill="black")
            image.save(video_dir / f"{idx:04d}.jpg")

        os.environ["QA_DATA_DIR"] = str(data_dir)
        os.environ["QA_KEYFRAMES_DIR"] = str(data_dir)
        qa_config.DATA_DIR = str(data_dir)
        qa_config.KEYFRAMES_DIR = str(data_dir)
        qa_config.OBJECT_GROUNDING_ENABLED = False

        kis=CaptionKIS(rows)
        qa=QASystem(kis=kis, frame_loader=TemporalFrameLoader(base_dir=str(data_dir), data_dir=str(data_dir)))
        qa.vlm=DeterministicVLM()
        structured=build_structured_query("a city skyline with a large sign in the foreground")
        rows_out=qa.solve_qa(structured, "What is visible in the video?", top_k_qa=5)
        assert rows_out, "No QA rows returned"
        public=validate_submission_rows(rows_out)
        assert public and {"video_id","frame_id","answer"}.issubset(public[0])
        assert any(x.get("stage") == "kis" for x in qa.last_run_diagnostics)

        # Interactive query path: question itself is used as the KIS query, so
        # question-aware refinement must be skipped instead of duplicating KIS.
        _ = qa.query("a city skyline with a large sign in the foreground", top_k=1)
        assert any(
            x.get("stage") == "question_refinement" and x.get("skipped") == "duplicate_query"
            for x in qa.last_run_diagnostics
        )
        print("PASS self-contained QA E2E")
        print("TOP-K")
        for row in rows_out:
            print(row)
        print("REQUIRED_OUTPUT")
        print(public[0])
        print("DIAGNOSTICS")
        for item in qa.last_run_diagnostics:
            print(item)

if __name__ == "__main__":
    main()
