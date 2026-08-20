from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence

from .qa_grounding import Detection, canonical_label, is_animal_label


def _box_iou(a, b):
    ay1, ax1, ay2, ax2 = a; by1, bx1, by2, bx2 = b
    inter = max(0, min(ay2, by2) - max(ay1, by1)) * max(0, min(ax2, bx2) - max(ax1, bx1))
    ua = max(0, ay2-ay1) * max(0, ax2-ax1); ub = max(0, by2-by1) * max(0, bx2-bx1)
    return inter/(ua+ub-inter) if ua+ub-inter > 0 else 0.0


@dataclass(frozen=True)
class CountResult:
    target: str
    count: int
    confidence: float
    source: str = "grounding"
    qualified: bool = False
    evidence_frames: int = 1
    notes: str = ""
    agreement: float = 1.0


class CountingEngine:
    """Counting with explicit semantics.

    For ordinary visual QA questions, the intended quantity is almost always
    "how many are visible in the relevant scene", not "how many distinct
    tracks survived across an arbitrary temporal window". Therefore the primary
    multi-frame estimator is frame-count consensus. Tracking is retained only
    as a secondary signal and for callers that explicitly want unique objects.
    """
    def __init__(self, min_score: float = 0.35):
        self.min_score = min_score

    def _target(self, detections: Sequence[Detection], target: str) -> List[Detection]:
        target_c = canonical_label(target)
        if target_c in {"object", "objects", "thing", "things", "animal", "animals"}:
            selected = [d for d in detections if d.score >= self.min_score]
            if target_c in {"animal", "animals"}:
                selected = [d for d in selected if is_animal_label(d.canonical)]
            return selected
        return [d for d in detections if d.score >= self.min_score and (
            d.canonical == target_c or (target_c == "vehicle" and d.canonical == "car"))]

    def _frame_count(self, detections: Sequence[Detection], target: str) -> CountResult:
        selected = self._target(detections, target)
        count = len(selected)
        if not selected:
            return CountResult(canonical_label(target), 0, 0.0, source="grounding", notes="no_target_detected", agreement=0.0)
        mean_score = sum(d.score for d in selected) / len(selected)
        conf = min(0.99, 0.55 + 0.45 * mean_score)
        return CountResult(canonical_label(target), count, conf, source="single_frame", agreement=1.0)

    def count(self, detections: Sequence[Detection], target: str = "object") -> CountResult:
        return self._frame_count(detections, target)

    def count_frame_consensus(self, frame_detections: Sequence[Sequence[Detection]], target: str, min_agreement: float = 0.34) -> CountResult:
        frames = [self._frame_count(ds, target) for ds in frame_detections]
        if not frames:
            return CountResult(canonical_label(target), 0, 0.0, source="frame_consensus", evidence_frames=0, agreement=0.0)
        valid = [f for f in frames if f.confidence > 0]
        if not valid:
            # Preserve a genuine zero only when several frames agree that the
            # target is absent; a lone empty detector result is not proof.
            zeros = sum(1 for f in frames if f.count == 0)
            conf = min(0.85, 0.45 + 0.45 * zeros / len(frames)) if zeros else 0.0
            return CountResult(canonical_label(target), 0, conf, source="frame_consensus", evidence_frames=len(frames), notes="all_frames_empty", agreement=zeros / max(1, len(frames)))

        counts = [f.count for f in valid]
        freq = Counter(counts)
        best_count, best_votes = max(freq.items(), key=lambda kv: (kv[1], -kv[0]))
        agreement = best_votes / max(1, len(valid))
        # Median provides a robust tie-break for unstable detectors.
        med = int(round(float(median(counts))))
        final_count = best_count if agreement >= min_agreement else med
        quality = sum(f.confidence for f in valid) / len(valid)
        confidence = min(0.98, 0.45 + 0.35 * agreement + 0.20 * quality)
        note = f"counts={counts};mode={best_count};agreement={agreement:.2f}"
        return CountResult(canonical_label(target), final_count, confidence, source="frame_consensus", evidence_frames=len(frame_detections), notes=note, agreement=agreement)

    def count_conditional(self, frame_detections: Sequence[Sequence[Detection]], target: str, relations: Sequence[dict], spatial_reasoner, min_relation_confidence: float = 0.55) -> CountResult:
        """Apply all resolvable spatial relations per frame, then use count consensus.

        If a reference cannot be grounded, deterministic counting is refused rather
        than silently reverting to a global count. This is essential for questions
        such as "how many people are in front of the podium?".
        """
        if not relations:
            return self.count_frame_consensus(frame_detections, target)
        per_frame: List[List[Detection]] = []
        unresolved = False
        target_c = canonical_label(target)
        for detections in frame_detections:
            current = self._target(detections, target_c)
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                relation = rel.get("type") or rel.get("relation")
                reference_text = rel.get("reference") or rel.get("object") or rel.get("reference_entity") or ""
                refs = spatial_reasoner.reference_candidates(detections, reference_text)
                if not refs:
                    unresolved = True
                    current = []
                    break
                current = spatial_reasoner.filter_relation(current, refs, relation, min_confidence=min_relation_confidence)
            per_frame.append(current)
        if unresolved:
            return CountResult(target_c, 0, 0.0, source="grounding_unresolved", qualified=True, evidence_frames=len(frame_detections), notes="reference_not_grounded", agreement=0.0)
        result = self.count_frame_consensus(per_frame, target_c, min_agreement=0.50)
        return CountResult(target_c, result.count, min(result.confidence, 0.92), source="qualified_frame_consensus", qualified=True, evidence_frames=len(frame_detections), notes=result.notes, agreement=result.agreement)

    def count_across_frames(self, frame_detections: Sequence[Sequence[Detection]], target: str, already_filtered: bool = False) -> CountResult:
        """Legacy unique-object tracker. Use only when callers explicitly need tracks."""
        target_c = canonical_label(target)
        tracks: List[Dict] = []
        next_id = 0
        for detections in frame_detections:
            current = list(detections) if already_filtered else self._target(detections, target_c)
            assigned = set(); candidates = []
            for di, det in enumerate(current):
                for ti, tr in enumerate(tracks):
                    if ti in assigned or tr["label"] != det.canonical: continue
                    iou = _box_iou(tr["box"], det.box)
                    cx, cy = det.center; tx, ty = tr["center"]
                    dist = ((cx-tx)**2 + (cy-ty)**2) ** 0.5
                    score = 0.78*iou + 0.22*max(0.0, 1.0-dist)
                    candidates.append((score, iou, dist, di, ti))
            used_dets = set()
            for _, iou, dist, di, ti in sorted(candidates, reverse=True):
                if di in used_dets or ti in assigned: continue
                if iou >= 0.18 or dist <= 0.18:
                    det=current[di]; tracks[ti].update(box=det.box, center=det.center)
                    tracks[ti]["seen"] += 1; tracks[ti]["score_sum"] += det.score
                    assigned.add(ti); used_dets.add(di)
            for di, det in enumerate(current):
                if di in used_dets: continue
                tracks.append({"id": next_id, "label": det.canonical, "box": det.box, "center": det.center, "seen": 1, "score_sum": det.score})
                next_id += 1
            for ti, tr in enumerate(tracks):
                if ti not in assigned: tr["last_frame_gap"] = tr.get("last_frame_gap", 0) + 1
        if not tracks:
            return CountResult(target_c, 0, 0.0, source="tracking", evidence_frames=len(frame_detections))
        min_seen = 2 if len(frame_detections) >= 2 else 1
        stable = [t for t in tracks if t["seen"] >= min_seen]
        if len(frame_detections) >= 3:
            stable = [t for t in stable if t["seen"] / len(frame_detections) >= 0.34]
        if not stable:
            return CountResult(target_c, 0, 0.0, source="tracking", evidence_frames=len(frame_detections))
        stability = sum(min(1.0, t["seen"] / max(1, len(frame_detections))) for t in stable) / len(stable)
        quality = sum(min(1.0, t["score_sum"] / max(1, t["seen"])) for t in stable) / len(stable)
        conf = min(0.97, 0.40 + 0.30*stability + 0.30*quality)
        return CountResult(target_c, len(stable), conf, source="tracking", evidence_frames=len(frame_detections))

    @staticmethod
    def render_summary(detections: Sequence[Detection]) -> Dict[str, int]:
        return dict(Counter(d.canonical for d in detections))
