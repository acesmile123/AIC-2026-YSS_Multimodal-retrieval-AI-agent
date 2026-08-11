import json
import numpy as np
import faiss


INDEX_PATH = "index/clip.index"
MAPPING_PATH = "index/mapping.json"


class ClipRetriever:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)

        # Load mapping
        with open(
            MAPPING_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            self.mapping = json.load(f)

        print(
            f"Loaded {self.index.ntotal} vectors"
        )


    def search(self, query_vector, top_k=10):
        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )
        #Check dimension
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape[1] != self.index.d:

            raise ValueError(
                f"Query vector có {query_vector.shape[1]} chiều, "
                f"nhưng FAISS cần {self.index.d} chiều."
            )
        faiss.normalize_L2(query_vector)

        scores, ids = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for score, faiss_id in zip(
            scores[0],
            ids[0]
        ):
            if faiss_id == -1:
                continue

            info = self.mapping[faiss_id]

            results.append({
                "video_id": info["video_id"],
                "frame_id": info["frame_id"],
                "score": float(score)
            })
        return results