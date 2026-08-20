"""Build a QA-only Milvus collection enriched with official BTC object detections.

The existing KIS collection and retrieval code are never modified. This command
copies frame_id + CLIP embedding from the KIS collection into a separate QA
collection and attaches the official BTC object detections for each keyframe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metadata_for_frame(data_dir: Path, video_id: str, frame_id: int) -> dict[str, Any]:
    import pandas as pd

    csv_path = data_dir / f"{video_id}.csv"
    if not csv_path.exists():
        return {"source": "btc_official_object_detection", "status": "missing_csv", "detections": [], "labels": [], "detection_count": 0}
    df = pd.read_csv(csv_path)
    rows = df[df["frame_idx"].astype(int) == int(frame_id)]
    if rows.empty:
        return {"source": "btc_official_object_detection", "status": "missing_frame_mapping", "detections": [], "labels": [], "detection_count": 0}
    keyframe_n = int(rows.iloc[0]["n"])
    object_id = video_id.replace("L21_V", "")
    path = data_dir / f"object_{object_id}" / f"{keyframe_n:03d}.json"
    if not path.exists():
        return {"source": "btc_official_object_detection", "status": "missing_detection", "keyframe_n": keyframe_n, "detections": [], "labels": [], "detection_count": 0}
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = list(raw.get("detection_class_entities") or [])
    scores = list(raw.get("detection_scores") or [])
    boxes = list(raw.get("detection_boxes") or [])
    detections = []
    for label, score, box in zip(labels, scores, boxes):
        try:
            detections.append({"label": str(label), "score": float(score), "box": [float(x) for x in box]})
        except Exception:
            continue
    return {
        "source": "btc_official_object_detection",
        "status": "ok",
        "keyframe_n": keyframe_n,
        "source_file": path.name,
        "detection_count": len(detections),
        "labels": [d["label"] for d in detections],
        "detections": detections,
    }


def build(source_collection: str, target_collection: str, uri: str, data_dir: str) -> None:
    from pymilvus import MilvusClient, DataType

    client = MilvusClient(uri=uri)
    if not client.has_collection(source_collection):
        raise RuntimeError(f"Source collection not found: {source_collection}")
    if client.has_collection(target_collection):
        client.drop_collection(target_collection)

    # Source schema metadata lets us discover the embedding dimension without
    # changing any KIS collection fields.
    src_info = client.describe_collection(collection_name=source_collection)
    fields = src_info.get("schema", {}).get("fields", [])
    dim = next((int(f.get("params", {}).get("dim")) for f in fields if f.get("name") == "embedding"), 512)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("video_id", DataType.VARCHAR, max_length=256)
    schema.add_field("frame_id", DataType.INT64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("object_metadata", DataType.JSON)
    schema.add_field("object_count", DataType.INT64)
    schema.add_field("object_labels", DataType.VARCHAR, max_length=8192)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
    client.create_collection(collection_name=target_collection, schema=schema, index_params=index_params)

    iterator = client.query_iterator(collection_name=source_collection, filter="", output_fields=["video_id", "frame_id", "embedding"], batch_size=1000)
    total = 0
    batch_rows = []
    while True:
        rows = iterator.next()
        if not rows:
            break
        for row in rows:
            video_id = str(row["video_id"])
            frame_id = int(row["frame_id"])
            meta = _metadata_for_frame(Path(data_dir), video_id, frame_id)
            batch_rows.append({
                "video_id": video_id,
                "frame_id": frame_id,
                "embedding": row["embedding"],
                "object_metadata": meta,
                "object_count": int(meta.get("detection_count", 0)),
                "object_labels": "|".join(meta.get("labels", [])),
            })
        if len(batch_rows) >= 500:
            client.insert(collection_name=target_collection, data=batch_rows)
            total += len(batch_rows)
            batch_rows = []
    iterator.close()
    if batch_rows:
        client.insert(collection_name=target_collection, data=batch_rows)
        total += len(batch_rows)
    client.flush(target_collection)
    print(f"Built QA object-enriched collection: {target_collection}; rows={total}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="clip_keyframes")
    p.add_argument("--target", default="clip_keyframes_qa_enriched")
    p.add_argument("--uri", default="http://localhost:19530")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    build(args.source, args.target, args.uri, args.data_dir)


if __name__ == "__main__":
    main()
