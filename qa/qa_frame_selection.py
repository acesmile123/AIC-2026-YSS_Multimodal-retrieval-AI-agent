from __future__ import annotations

from typing import List, Sequence

from .qa_question_analysis import QuestionAnalysis, analyze_question


class AdaptiveFrameSelector:
    """Select evidence frames without throwing away temporal anchors.

    Temporal/action questions use start/middle/end anchors first. Spatial/count
    questions prefer the frame with the richest target evidence, while retaining
    temporal context around it. The final VLM engine can still reduce the list
    to its GPU-safe input budget.
    """

    _WIDE_EVIDENCE_TYPES = {"COUNTING", "SPATIAL", "RELATIONSHIP", "COMPARISON"}
    _TEMPORAL_TYPES = {"ACTION", "TEMPORAL", "RELATIONSHIP"}

    def select(
        self,
        question: str,
        records: Sequence[dict],
        analysis: QuestionAnalysis | None = None,
        limit: int = 7,
    ) -> List[dict]:
        analysis = analysis or analyze_question(question)
        recs = list(records)
        if not recs:
            return []
        limit = max(1, int(limit))
        if len(recs) <= limit:
            return recs

        active_types = set(analysis.types) | {analysis.primary_type}
        if active_types & self._TEMPORAL_TYPES or analysis.needs_temporal:
            # Uniform anchors are much safer than simply taking the first N
            # frames, especially for after/before/then questions.
            idxs = []
            for x in (0.0, 0.5, 1.0):
                idx = round(x * (len(recs) - 1))
                if idx not in idxs:
                    idxs.append(idx)
            # Add evenly spaced context if the VLM budget allows it.
            if limit > len(idxs):
                for i in range(1, limit - 2):
                    idx = round(i * (len(recs) - 1) / max(1, limit - 1))
                    if idx not in idxs:
                        idxs.append(idx)
            return [recs[i] for i in sorted(idxs)[:limit]]

        if active_types & self._WIDE_EVIDENCE_TYPES:
            center = len(recs) // 2
            scored = sorted(
                enumerate(recs),
                key=lambda item: (
                    len(item[1].get("detections", [])),
                    -abs(item[1].get("frame_id", 0) - recs[center].get("frame_id", 0)),
                ),
                reverse=True,
            )
            chosen = sorted(idx for idx, _ in scored[:limit])
            return [recs[i] for i in chosen]

        center = len(recs) // 2
        order = sorted(range(len(recs)), key=lambda i: abs(i - center))
        return [recs[i] for i in order[:limit]]
