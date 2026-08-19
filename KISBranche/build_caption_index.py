import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


CAPTION_JSON = "data/captions.json"
INDEX_PATH = "index/caption.index"
MAPPING_PATH = "index/caption_mapping.json"


def build_caption_index():

    os.makedirs("index", exist_ok=True)

    with open(CAPTION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[Caption] Found {len(data)} records")

    if len(data) == 0:
        raise ValueError("captions.json is empty")

    print("[Caption] Loading SentenceTransformer...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = []

    for item in data:

        text = item.get("retrieval_text", "")

        if not text:
            text = item.get("caption", "")

        texts.append(text)

    print("[Caption] Example retrieval text:")
    print(texts[0])
    print("[Caption] Encoding retrieval_text...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    mapping = []

    for item in data:

        mapping.append({
            "video_id": item["video_id"],
            "frame_id": int(item["frame_id"]),
            "keyframe_n": int(item["keyframe_n"]),
            "image_path": item.get("image_path", "")
        })

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(
        MAPPING_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mapping,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Records      : {len(data)}")
    print(f"Index dim    : {dim}")
    print(f"Vectors      : {index.ntotal}")
    print(f"Index path   : {INDEX_PATH}")
    print(f"Mapping path : {MAPPING_PATH}")
    print("======================================")


if __name__ == "__main__":
    build_caption_index()

