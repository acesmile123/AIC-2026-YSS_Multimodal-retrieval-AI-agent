from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

from qa.vector_object_metadata import VectorObjectMetadataLookup


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    data = project / "data"
    # The official bundle contains data/object_001/*.json and L21_V001.csv.
    lookup = VectorObjectMetadataLookup(enable_milvus=False, data_dir=data)
    meta = lookup.get("L21_V001", 90)
    assert meta.get("source") == "btc_official_object_detection"
    assert meta.get("status") == "ok"
    assert meta.get("keyframe_n") == 2
    assert meta.get("detection_count", 0) > 0
    assert meta.get("detections")

    missing = lookup.get("L21_V001", 0)
    assert missing.get("source") == "btc_official_object_detection"
    assert missing.get("status") == "missing"
    assert missing.get("keyframe_n") == 1

    print("PASS: BTC object metadata lookup -> frame_id 90 maps to keyframe 2.")
    print("PASS: missing keyframe metadata is explicit, never fabricated.")


if __name__ == "__main__":
    main()
