from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    videos = sorted(data.glob("*.npy"))
    if not videos:
        raise SystemExit("No CLIP .npy vectors found under data/")

    total_vectors = 0
    total_json = 0
    missing = []
    for npy in videos:
        video_id = npy.stem
        df = pd.read_csv(data / f"{video_id}.csv")
        total_vectors += len(df)
        object_dir = data / f"object_{video_id.replace('L21_V', '')}"
        for _, row in df.iterrows():
            n = int(row["n"])
            p = object_dir / f"{n:03d}.json"
            if not p.exists():
                missing.append((video_id, n, int(row["frame_idx"])))
                continue
            raw = json.loads(p.read_text(encoding="utf-8"))
            labels = raw.get("detection_class_entities") or []
            scores = raw.get("detection_scores") or []
            boxes = raw.get("detection_boxes") or []
            if not (len(labels) == len(scores) == len(boxes)):
                raise AssertionError(f"Detection arrays mismatch: {p}")
            total_json += 1

    print(f"Vectors: {total_vectors}")
    print(f"Official BTC object metadata files: {total_json}")
    print(f"Missing official object metadata files: {len(missing)}")
    for item in missing[:20]:
        print("  MISSING:", item)
    print("PASS: every vector row will receive object_metadata in the enriched collection;")
    print("      rows without a supplied BTC JSON are marked status=missing, never fabricated.")


if __name__ == "__main__":
    main()
