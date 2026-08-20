from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .qa_grounding import ALIASES, ANIMAL_LABELS, Detection, canonical_label

# Scene-vocabulary references that fall outside the detector's own label set
# (podiums, furniture, generic "patient" -> person, etc.). Layered on top of
# `qa_grounding.ALIASES` rather than duplicating it, so any label the detector
# actually knows about -- including every animal in this dataset's domain --
# is automatically resolvable as a spatial reference too.
_EXTRA_REFERENCE_ALIASES = {
    "podium": "podium", "microphone": "microphone", "micro": "microphone",
    "table": "table", "desk": "desk", "chair": "chair", "patient": "person",
}

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)


def _known_reference_labels() -> set:
    return set(ALIASES.keys()) | set(ALIASES.values()) | ANIMAL_LABELS | set(_EXTRA_REFERENCE_ALIASES.values())


def _singularize(token: str) -> str:
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


@dataclass(frozen=True)
class Relation:
    subject: str
    relation: str
    object: str
    confidence: float = 1.0


def _gap(a: Detection, b: Detection):
    ay1, ax1, ay2, ax2 = a.box
    by1, bx1, by2, bx2 = b.box
    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0.0, max(by1 - ay2, ay1 - by2))
    return dx, dy


class SpatialReasoner:
    """Conservative spatial reasoning over normalized boxes.

    The old implementation used fixed center-distance thresholds. This version
    prefers box geometry, overlap-aware relations, and normalized confidence.
    It is still only a visual prior: unresolved references must be verified by
    the VLM instead of being forced into a deterministic answer.
    """

    def relations(self, detections: Sequence[Detection]) -> List[Relation]:
        out: List[Relation] = []
        for i, a in enumerate(detections):
            ax, ay = a.center
            for j, b in enumerate(detections):
                if i == j:
                    continue
                bx, by = b.center
                dx, dy = ax - bx, ay - by
                dist = (dx * dx + dy * dy) ** 0.5
                horiz = abs(dx)
                vert = abs(dy)
                near_score = max(0.0, 1.0 - min(1.0, ((_gap(a, b)[0] ** 2 + _gap(a, b)[1] ** 2) ** 0.5) / 0.35))
                # Center ordering is only emitted when the separation is clear.
                if horiz > 0.12 and horiz >= vert * 0.9:
                    out.append(Relation(a.canonical, "left_of" if dx < 0 else "right_of", b.canonical, min(1.0, horiz * 1.5)))
                if vert > 0.12 and vert >= horiz * 0.9:
                    out.append(Relation(a.canonical, "above" if dy < 0 else "below", b.canonical, min(1.0, vert * 1.5)))
                if near_score >= 0.45 or dist < 0.22:
                    out.append(Relation(a.canonical, "near", b.canonical, max(near_score, 0.45)))
                if 0.0 < max(_gap(a, b)) <= 0.16 and near_score >= 0.30:
                    out.append(Relation(a.canonical, "next_to", b.canonical, max(near_score, 0.30)))
                # Do NOT convert image-y ordering into a deterministic 3-D
                # in-front/behind relation. Perspective makes that heuristic
                # systematically unsafe. Such relations are emitted only as a
                # low-confidence cue for the VLM, never as hard counting evidence.
                if dy > 0.18 and abs(dx) < 0.24:
                    out.append(Relation(a.canonical, "in_front_of", b.canonical, 0.40))
                if dy < -0.18 and abs(dx) < 0.24:
                    out.append(Relation(a.canonical, "behind", b.canonical, 0.40))
        return out

    @staticmethod
    def _reference_label(reference: str) -> str:
        """Resolve a natural-language reference phrase to a canonical label.

        The reference text coming out of question analysis is a full noun
        phrase -- "the buffalo", "a red car", "the podium on stage" -- not a
        bare label. Matching the whole phrase against a fixed alias table
        (the previous behavior) only ever succeeds when the phrase happens to
        be exactly one known word with nothing else attached, which silently
        breaks reference-conditioned counting/spatial questions ("how many
        cows are near the buffalo?") for almost every real phrasing. This
        instead tokenizes the phrase and checks each token -- including a
        simple plural fallback -- against the detector's full known-label
        vocabulary plus a small set of non-detector scene nouns.
        """
        text = (reference or "").strip().lower()
        if not text:
            return ""
        known = _known_reference_labels()
        for needle, label in _EXTRA_REFERENCE_ALIASES.items():
            if needle in text:
                return label
        # Prefer the last matching token: in an English noun phrase the head
        # noun is typically rightmost ("the red fire truck" -> "truck"),
        # so scanning right-to-left favors the referent over its modifiers.
        tokens = _WORD_RE.findall(text)
        for token in reversed(tokens):
            candidate = canonical_label(token)
            if candidate in known:
                return candidate
            singular = _singularize(candidate)
            if singular in known:
                return singular
        return canonical_label(text) if text in known else ""

    def filter_relation(self, targets: Sequence[Detection], references: Sequence[Detection], relation: str, min_confidence: float = 0.50) -> List[Detection]:
        relation = str(relation or "").strip().lower().replace(" ", "_")
        out: List[Detection] = []
        for target in targets:
            best = 0.0
            for ref in references:
                if target is ref:
                    continue
                ax, ay = target.center
                bx, by = ref.center
                dx, dy = ax - bx, ay - by
                dist = (dx * dx + dy * dy) ** 0.5
                gx, gy = _gap(target, ref)
                if relation == "left_of": conf = min(1.0, max(0.0, -dx) * 1.8) if dx < 0 else 0.0
                elif relation == "right_of": conf = min(1.0, max(0.0, dx) * 1.8) if dx > 0 else 0.0
                elif relation == "above": conf = min(1.0, max(0.0, -dy) * 1.8) if dy < 0 else 0.0
                elif relation == "below": conf = min(1.0, max(0.0, dy) * 1.8) if dy > 0 else 0.0
                elif relation in {"near", "next_to"}: conf = max(0.0, 1.0 - min(1.0, (gx * gx + gy * gy) ** 0.5 / 0.35))
                elif relation in {"in_front_of", "behind"}:
                    # 2-D boxes cannot establish true depth. Refuse deterministic
                    # filtering; the independent VLM verifier must decide.
                    conf = 0.0
                else: conf = 0.0
                best = max(best, conf)
            if best >= min_confidence:
                out.append(target)
        return out

    def reference_candidates(self, detections: Sequence[Detection], reference: str) -> List[Detection]:
        label = self._reference_label(reference)
        if not label:
            return []
        # Some VLM/object detectors emit a broad category; preserve it rather
        # than inventing a finer class.
        if label == "vehicle":
            return [d for d in detections if d.canonical in {"vehicle", "car"}]
        if label == "person":
            return [d for d in detections if d.canonical == "person"]
        return [d for d in detections if d.canonical == canonical_label(label)]

    @staticmethod
    def closest_to_camera(detections: Sequence[Detection]):
        return max(detections, key=lambda d: d.area, default=None)
