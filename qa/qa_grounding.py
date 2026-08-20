from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


ALIASES = {
    "cattle": "cow", "bull": "cow", "ox": "cow", "buffalo": "buffalo",
    "buffaloes": "buffalo", "person": "person", "man": "person", "woman": "person",
    "boy": "person", "girl": "person", "human": "person",
    "car": "car", "automobile": "car", "vehicle": "vehicle", "land vehicle": "vehicle",
    "dog": "dog", "cat": "cat", "horse": "horse", "sheep": "sheep", "goat": "goat",
    "lamb": "sheep", "chicken": "chicken", "hen": "chicken", "rooster": "chicken",
    "bird": "bird", "duck": "duck", "goose": "goose", "deer": "deer",
    "zebra": "zebra", "elephant": "elephant", "giraffe": "giraffe", "lion": "lion",
    "tiger": "tiger", "bear": "bear", "monkey": "monkey", "animal": "animal",
}

ANIMAL_LABELS = {
    "cow", "buffalo", "horse", "dog", "cat", "sheep", "goat", "chicken",
    "bird", "duck", "goose", "deer", "zebra", "elephant", "giraffe", "lion",
    "tiger", "bear", "monkey", "animal",
}


def is_animal_label(label: str) -> bool:
    return canonical_label(label) in ANIMAL_LABELS


def canonical_label(label: str) -> str:
    x = str(label or "").strip().lower()
    return ALIASES.get(x, x)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]
    raw_label: str = ""

    @property
    def canonical(self) -> str:
        return canonical_label(self.label)

    @property
    def area(self) -> float:
        y1, x1, y2, x2 = self.box
        return max(0.0, y2-y1) * max(0.0, x2-x1)

    @property
    def center(self) -> tuple[float, float]:
        y1, x1, y2, x2 = self.box
        return ((x1+x2)/2.0, (y1+y2)/2.0)


def _iou(a, b) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    y1, x1 = max(ay1, by1), max(ax1, bx1)
    y2, x2 = min(ay2, by2), min(ax2, bx2)
    inter = max(0.0, y2-y1) * max(0.0, x2-x1)
    union = max(0.0, ay2-ay1)*max(0.0, ax2-ax1) + max(0.0, by2-by1)*max(0.0, bx2-bx1) - inter
    return inter/union if union > 0 else 0.0


class GroundingEngine:
    """Structured grounding from supplied detection metadata.

    If an external detector/grounder is provided it may implement `detect(image)`
    and return either Detection objects or dicts with label/score/box. Otherwise
    the AIC supplied object metadata can be converted through `from_metadata`.
    """
    def __init__(self, detector=None, score_threshold: float = 0.20, iou_threshold: float = 0.50):
        self.detector = detector
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def from_metadata(self, metadata: Dict[str, Any]) -> List[Detection]:
        # Preferred QA-vector representation: compact list of normalized detections.
        normalized = metadata.get("detections") or []
        if normalized:
            out = []
            for item in normalized:
                if not isinstance(item, dict):
                    continue
                try:
                    label = str(item.get("label", ""))
                    score_f = float(item.get("score", 0.0))
                    box_t = tuple(float(x) for x in item.get("box", ()))
                except Exception:
                    continue
                if score_f < self.score_threshold or len(box_t) != 4:
                    continue
                out.append(Detection(label, score_f, box_t, label))
            return self._nms(out)

        # Backward-compatible official BTC JSON representation.
        labels = metadata.get("detection_class_entities", [])
        scores = metadata.get("detection_scores", [])
        boxes = metadata.get("detection_boxes", [])
        out = []
        for label, score, box in zip(labels, scores, boxes):
            try: score_f = float(score)
            except Exception: continue
            if score_f < self.score_threshold or not box or len(box) != 4: continue
            try: box_t = tuple(float(x) for x in box)
            except Exception: continue
            out.append(Detection(str(label), score_f, box_t, str(label)))
        return self._nms(out)

    def detect(self, image, metadata: Optional[Dict[str, Any]] = None) -> List[Detection]:
        if self.detector is not None:
            raw = self.detector.detect(image)
            detections = []
            for item in raw:
                if isinstance(item, Detection): detections.append(item)
                else:
                    detections.append(Detection(str(item.get("label","")), float(item.get("score",0.0)), tuple(float(x) for x in item.get("box",(0,0,0,0)))))
            return self._nms([d for d in detections if d.score >= self.score_threshold])
        return self.from_metadata(metadata or {})

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        kept: List[Detection] = []
        for det in sorted(detections, key=lambda d: d.score, reverse=True):
            if any(det.canonical == prev.canonical and _iou(det.box, prev.box) >= self.iou_threshold for prev in kept):
                continue
            kept.append(det)
        return kept

    @staticmethod
    def filter_label(detections: Iterable[Detection], target: str) -> List[Detection]:
        t = canonical_label(target)
        return [d for d in detections if d.canonical == t or (t == "animal" and is_animal_label(d.canonical))]
