from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .qa_grounding import Detection


@dataclass(frozen=True)
class TemporalState:
    frame_id: int
    objects: tuple[str, ...]


class TemporalReasoner:
    def build_states(self, records: Sequence[dict]) -> List[TemporalState]:
        states=[]
        for r in records:
            labels=tuple(sorted({d.canonical for d in r.get("detections", [])}))
            states.append(TemporalState(int(r["frame_id"]), labels))
        return states

    def events(self, states: Sequence[TemporalState]) -> List[dict]:
        out=[]
        for prev,curr in zip(states, states[1:]):
            appeared=sorted(set(curr.objects)-set(prev.objects))
            disappeared=sorted(set(prev.objects)-set(curr.objects))
            if appeared:
                out.append({"type":"appeared","frame_id":curr.frame_id,"objects":appeared})
            if disappeared:
                out.append({"type":"disappeared","frame_id":curr.frame_id,"objects":disappeared})
        return out

    @staticmethod
    def direction(question: str) -> str:
        q=(question or '').lower()
        if any(x in q for x in ("next","tiếp theo","sau đó","sau khi")): return "future"
        if any(x in q for x in ("before","trước khi")): return "past"
        return "center"
