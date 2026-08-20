from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .qa_grounding import Detection
from .qa_attributes import AttributeExtractor


class SceneUnderstanding:
    def __init__(self):
        self.attributes = AttributeExtractor()

    def summarize(self, detections: Sequence[Detection], caption: str = "") -> dict:
        counts = Counter(d.canonical for d in detections)
        dominant = []
        for label, _ in counts.most_common(5):
            dominant.append(label)
        has_person = counts.get("person", 0) > 0
        animals = sum(counts.get(k,0) for k in ("cow","buffalo","horse","dog","cat","animal"))
        scene={
            "objects": dict(counts),
            "animals_visible": animals,
            "people_visible": counts.get("person",0),
            "dominant_objects": dominant,
            "caption": caption,
            "environment": self._environment(caption),
            "has_people": has_person,
        }
        return self.attributes.enrich_scene(scene, caption)

    @staticmethod
    def _environment(caption: str) -> str:
        c = (caption or "").lower()
        if any(x in c for x in ("field","grass","rice field","meadow")): return "outdoor_field"
        if any(x in c for x in ("road","street","highway")): return "road"
        if any(x in c for x in ("room","office","kitchen")): return "indoor"
        if any(x in c for x in ("water","river","lake","boat")): return "water"
        return "unknown"
