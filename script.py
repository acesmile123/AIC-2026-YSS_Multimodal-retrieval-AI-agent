import os
import json
import numpy as np
import pandas as pd
import faiss

CLIP_DIR = "data/clip_features"
CSV_DIR = "data/keyframes"
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "clip.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "mapping.json")


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    all_features = []
    mapping = []

    global_id = 0

    npy_files = sorted(
        f for f in os.listdir(CLIP_DIR)
        if f.endswith(".npy")
    )

    if len(npy_files) == 0:
        raise RuntimeError(
            f"Không tìm thấy file .npy trong {CLIP_DIR}"
        )
    print(f"Tìm thấy {len(npy_files)} file .npy")

    for file_name in npy_files:
        video_id = os.path.splitext(file_name)[0]

        npy_path = os.path.join(
            CLIP_DIR,
            file_name
        )

        csv_path = os.path.join(
            CSV_DIR,
            video_id + ".csv"
        )


        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Không tìm thấy CSV cho {video_id}: "
                f"{csv_path}"
            )

        features = np.load(npy_path)

        print(
            f"\n{video_id}"
        )

        print(
            f"CLIP shape: {features.shape}"
        )

        # float16 -> float32
        features = features.astype(np.float32)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError(
                f"{file_name} có shape không hợp lệ: "
                f"{features.shape}"
            )

        num_features = features.shape[0]
        df = pd.read_csv(csv_path)

        print(
            f"CSV rows: {len(df)}"
        )
        if num_features != len(df):

            raise ValueError(
                f"Số vector CLIP và số keyframe "
                f"không khớp cho {video_id}: "
                f"{num_features} != {len(df)}"
            )
        faiss.normalize_L2(features)

        all_features.append(features)

        for i in range(num_features):

            frame_idx = int(
                df.iloc[i]["frame_idx"]
            )

            mapping.append({
                "faiss_id": global_id,
                "video_id": video_id,
                "frame_id": frame_idx
            })

            global_id += 1

    all_features = np.vstack(
        all_features
    )

    print("\n======================")
    print("TOTAL DATA")
    print("======================")

    print(
        "Total vectors:",
        all_features.shape[0]
    )

    print(
        "Embedding dimension:",
        all_features.shape[1]
    )


    embedding_dim = all_features.shape[1]

    index = faiss.IndexFlatIP(
        embedding_dim
    )

    index.add(all_features)

    print(
        "FAISS vectors:",
        index.ntotal
    )
    faiss.write_index(
        index,
        INDEX_PATH
    )

    print(
        f"Saved index: {INDEX_PATH}"
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

    print(
        f"Saved mapping: {MAPPING_PATH}"
    )
    print("\nBUILD SUCCESS!")
