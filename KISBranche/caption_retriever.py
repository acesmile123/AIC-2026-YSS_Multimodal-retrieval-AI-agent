import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class CaptionRetriever:

    def __init__(
        self,
        json_path="data/captions.json",
        index_path="index/caption.index",
        mapping_path="index/caption_mapping.json"
    ):
        print("[Caption] Loading SentenceTransformer...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        self.json_path = json_path
        self.index_path = index_path
        self.mapping_path = mapping_path

        print("[Caption] Loading captions JSON...")

        with open(json_path, "r", encoding="utf-8") as f:
            self.caption_data = json.load(f)

        print(
            f"[Caption] Loaded {len(self.caption_data)} caption records"
        )

        print("[Caption] Loading FAISS index...")

        self.index = faiss.read_index(index_path)

        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        print(
            f"[Caption] Loaded {self.index.ntotal} caption vectors"
        )

        if self.index.ntotal != len(self.caption_data):
            print(
                "[Warning] Number of vectors and JSON records "
                "do not match!"
            )

    def encode_query(self, query):
        """
        Encode user query using the same
        SentenceTransformer used when building the index.
        """

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        return vector

    def search(self, query, top_k=50):

        query_vector = self.encode_query(query)

        top_k = min(top_k, self.index.ntotal)

        scores, ids = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], ids[0]):

            if idx == -1:
                continue
            if isinstance(self.mapping, dict):
                item = self.mapping.get(str(idx))

                if item is None:
                    continue

            else:
                if idx >= len(self.mapping):
                    continue

                item = self.mapping[idx]

            results.append({
                "video_id": item["video_id"],
                "frame_id": int(item["frame_id"]),
                "score": float(score)
            })

        return results

    def search_with_text(self, query, top_k=20):

        results = self.search(query, top_k)

        output = []

        for r in results:

            video_id = r["video_id"]
            frame_id = r["frame_id"]

            text = None

            for item in self.caption_data:

                if (
                    item["video_id"] == video_id
                    and int(item["frame_id"]) == frame_id
                ):
                    text = item.get(
                        "retrieval_text",
                        item.get("caption", "")
                    )
                    break

            output.append({
                "video_id": video_id,
                "frame_id": frame_id,
                "score": r["score"],
                "retrieval_text": text
            })

        return output

