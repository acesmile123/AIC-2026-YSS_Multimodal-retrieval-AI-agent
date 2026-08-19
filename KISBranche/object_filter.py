import os
import json
import pandas as pd


class ObjectMetadataLookup:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.csv_cache = {}

    def _get_keyframe_n(self, video_id, frame_id):
        if video_id not in self.csv_cache:
            csv_path = os.path.join(self.data_dir, video_id + ".csv")

            if not os.path.exists(csv_path):
                return None

            self.csv_cache[video_id] = pd.read_csv(csv_path)

        df = self.csv_cache[video_id]
        rows = df[df["frame_idx"] == frame_id]

        if rows.empty:
            return None

        return int(rows.iloc[0]["n"])

    def get(self, video_id, frame_id=None):
        if frame_id is None:
            if isinstance(video_id, (tuple, list)) and len(video_id) == 2:
                video_id, frame_id = video_id
            else:
                raise TypeError(
                    "ObjectMetadataLookup.get() requires (video_id, frame_id) "
                    "or a 2-item tuple/list key."
                )

        n = self._get_keyframe_n(video_id, frame_id)

        if n is None:
            return None

        object_id = video_id.replace("L21_V", "")
        object_dir = f"object_{object_id}"

        json_path = os.path.join(
            self.data_dir,
            object_dir,
            f"{n:03d}.json"
        )

        if not os.path.exists(json_path):
            return None

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)