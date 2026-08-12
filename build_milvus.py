import os
import numpy as np
import pandas as pd
from pymilvus import MilvusClient


CLIP_DIR = "data"
CSV_DIR = "data"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "clip_keyframes"

EMBEDDING_DIM = 512


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def get_or_create_collection(client):
    if client.has_collection(COLLECTION_NAME):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype="INT64", is_primary=True)
    schema.add_field(field_name="video_id", datatype="VARCHAR", max_length=256)
    schema.add_field(field_name="frame_id", datatype="INT64")
    schema.add_field(field_name="embedding", datatype="FLOAT_VECTOR",dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()

    index_params.add_index( field_name="embedding", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})

    client.create_collection( collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


def build_index():
    client = MilvusClient( uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    get_or_create_collection(client)

    npy_files = [
        f for f in os.listdir(CLIP_DIR)
        if f.endswith(".npy")
    ]

    if not npy_files:
        raise RuntimeError(f"Không tìm thấy file .npy trong {CLIP_DIR}")

    total_inserted = 0

    for file_name in sorted(npy_files):
        video_id = os.path.splitext(file_name)[0]

        npy_path = os.path.join(CLIP_DIR, file_name)

        csv_path = os.path.join( CSV_DIR, video_id + ".csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")

        features = np.load(npy_path).astype(np.float32)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2:
            raise ValueError(f"{file_name} có shape không hợp lệ: {features.shape}")

        if features.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"{video_id}: dim {features.shape[1]} "f"!= {EMBEDDING_DIM}")

        df = pd.read_csv(csv_path)

        if "frame_idx" not in df.columns:
            raise ValueError(f"{csv_path} thiếu cột frame_idx")

        if len(features) != len(df):
            raise ValueError(f"{video_id}: {len(features)} vectors "f"!= {len(df)} frames")

        features = normalize(features)

        rows = [
            {
                "video_id": video_id,
                "frame_id": int(df.iloc[i]["frame_idx"]),
                "embedding": features[i].tolist()
            }
            for i in range(len(features))
        ]

        result = client.insert(collection_name=COLLECTION_NAME, data=rows)

        total_inserted += result["insert_count"]

        print(f"{video_id}: " f"{result['insert_count']} vectors")

    client.flush(COLLECTION_NAME)

    print(f"\nTotal inserted: {total_inserted}")
    print("BUILD SUCCESS!")

