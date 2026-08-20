from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class VectorObjectMetadataLookup:
    """QA-only lookup for object metadata stored in the frame vector DB.

    KIS retrieval is intentionally untouched. This class is queried only after
    KIS has returned candidate (video_id, frame_id) pairs. It reads the enriched
    metadata from the same Milvus collection used by CLIP retrieval and falls
    back to the official BTC object JSON when the collection is older/un-enriched.
    """

    def __init__(
        self,
        collection_name: str = "clip_keyframes",
        milvus_uri: str = "http://localhost:19530",
        data_dir: str | Path = "data",
        enable_milvus: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.milvus_uri = milvus_uri
        self.data_dir = Path(data_dir)
        self._cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._client = None
        self._milvus_ready = False
        if enable_milvus:
            try:
                from pymilvus import MilvusClient
                self._client = MilvusClient(uri=milvus_uri)
                self._milvus_ready = bool(self._client.has_collection(collection_name))
            except Exception:
                self._client = None
                self._milvus_ready = False

    def get(self, key_or_video: Any, frame_id: int | None = None) -> dict[str, Any]:
        if frame_id is None:
            if isinstance(key_or_video, (tuple, list)) and len(key_or_video) == 2:
                video_id, frame_id = str(key_or_video[0]), int(key_or_video[1])
            else:
                raise TypeError("get() requires (video_id, frame_id) or a 2-item tuple/list")
        else:
            video_id = str(key_or_video)
            frame_id = int(frame_id)

        key = (video_id, frame_id)
        if key in self._cache:
            return self._cache[key]

        meta = self._get_from_vector_db(video_id, frame_id)
        if not meta:
            meta = self._get_from_official_btc_json(video_id, frame_id)
        self._cache[key] = meta or {}
        return self._cache[key]

    def _get_from_vector_db(self, video_id: str, frame_id: int) -> dict[str, Any]:
        if not self._milvus_ready or self._client is None:
            return {}
        try:
            rows = self._client.query(
                collection_name=self.collection_name,
                filter=f'video_id == "{video_id}" and frame_id == {int(frame_id)}',
                output_fields=["video_id", "frame_id", "object_metadata", "object_count", "object_labels"],
                limit=1,
            )
            if not rows:
                return {}
            row = rows[0]
            meta = row.get("object_metadata")
            if isinstance(meta, dict) and meta:
                return meta
            # Old/partially rebuilt collections may have only scalar fields.
            labels = row.get("object_labels") or ""
            count = row.get("object_count")
            if labels or count is not None:
                return {
                    "source": "milvus_enriched",
                    "status": "partial",
                    "detection_count": int(count or 0),
                    "labels": [x for x in str(labels).split("|") if x],
                    "detections": [],
                }
        except Exception:
            return {}
        return {}

    def _get_from_official_btc_json(self, video_id: str, frame_id: int) -> dict[str, Any]:
        csv_path = self.data_dir / f"{video_id}.csv"
        if not csv_path.exists():
            return {}
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            rows = df[df["frame_idx"].astype(int) == int(frame_id)]
            if rows.empty:
                return {}
            keyframe_n = int(rows.iloc[0]["n"])
        except Exception:
            return {}

        object_id = video_id.replace("L21_V", "")
        json_path = self.data_dir / f"object_{object_id}" / f"{keyframe_n:03d}.json"
        if not json_path.exists():
            return {
                "source": "btc_official_object_detection",
                "status": "missing",
                "keyframe_n": keyframe_n,
                "detection_count": 0,
                "labels": [],
                "detections": [],
            }
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return self._normalize_official(raw, keyframe_n, json_path.name)

    @staticmethod
    def _normalize_official(raw: dict[str, Any], keyframe_n: int, filename: str) -> dict[str, Any]:
        labels = list(raw.get("detection_class_entities") or [])
        scores = list(raw.get("detection_scores") or [])
        boxes = list(raw.get("detection_boxes") or [])
        detections = []
        for label, score, box in zip(labels, scores, boxes):
            try:
                score_f = float(score)
                box_f = [float(x) for x in box]
            except Exception:
                continue
            detections.append({"label": str(label), "score": score_f, "box": box_f})
        return {
            "source": "btc_official_object_detection",
            "status": "ok",
            "keyframe_n": int(keyframe_n),
            "source_file": filename,
            "detection_count": len(detections),
            "labels": [d["label"] for d in detections],
            "detections": detections,
        }
