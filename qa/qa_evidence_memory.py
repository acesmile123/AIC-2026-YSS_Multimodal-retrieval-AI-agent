from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EvidenceRecord:
    video_id: str
    frame_id: int
    caption: str = ""
    scene: dict = field(default_factory=dict)
    detections: list = field(default_factory=list)
    ocr_text: str = ""
    temporal_index: int = 0
    ocr_confidence: float = 0.0


class EvidenceMemory:
    def __init__(self, max_records: int = 512):
        self.max_records = max_records
        self._records: Dict[Tuple[str,int], EvidenceRecord] = {}

    def put(self, record: EvidenceRecord) -> None:
        key = (record.video_id, int(record.frame_id))
        self._records[key] = record
        if len(self._records) > self.max_records:
            self._records.pop(next(iter(self._records)))

    def get(self, video_id: str, frame_id: int) -> Optional[EvidenceRecord]:
        return self._records.get((video_id, int(frame_id)))

    def related(self, video_id: str, frame_ids: List[int]) -> List[EvidenceRecord]:
        return [r for fid in frame_ids if (r := self.get(video_id, fid)) is not None]

    def clear(self) -> None:
        self._records.clear()

    def stats(self) -> dict:
        return {"records": len(self._records), "max_records": self.max_records}
