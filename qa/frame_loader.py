import os
import threading
import bisect
from typing import Dict, List, Optional, Tuple

from PIL import Image

from . import qa_config
from .qa_cache import LRUCache, frame_cache_key
from .qa_logging import get_logger

logger = get_logger("frame_loader")
_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")


class TemporalFrameLoader:
    """Resolve competition frame IDs to physical keyframe images.

    The AIC sample data stores the submitted frame index in CSV metadata and
    stores images by keyframe ordinal (`n`).  This loader therefore resolves
    `frame_idx -> n -> image path` before falling back to direct filenames.
    """

    def __init__(self, base_dir: str = qa_config.KEYFRAMES_DIR,
                 data_dir: str = qa_config.DATA_DIR,
                 cache_size: int = qa_config.FRAME_CACHE_SIZE,
                 raw_frame_cache_size: int = qa_config.RAW_FRAME_CACHE_SIZE):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self._dir_cache: Dict[str, List[str]] = {}
        self._mapping_cache: Dict[str, Dict[int, int]] = {}
        self._lock = threading.Lock()
        self._frame_cache = LRUCache(max_size=cache_size)
        self._image_cache = LRUCache(max_size=raw_frame_cache_size)
        if not os.path.exists(self.base_dir):
            logger.warning("Keyframe root does not exist: %s", self.base_dir)

    def _list_dir(self, video_dir: str) -> List[str]:
        with self._lock:
            cached = self._dir_cache.get(video_dir)
        if cached is not None:
            return cached
        try:
            listing = os.listdir(video_dir)
        except OSError:
            listing = []
        with self._lock:
            self._dir_cache.setdefault(video_dir, listing)
            return self._dir_cache[video_dir]

    def _load_mapping(self, video_id: str) -> Dict[int, int]:
        with self._lock:
            if video_id in self._mapping_cache:
                return self._mapping_cache[video_id]
        csv_path = os.path.join(self.data_dir, f"{video_id}.csv")
        mapping: Dict[int, int] = {}
        try:
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        frame_idx = int(float(row.get("frame_idx", "")))
                        n = int(float(row.get("n", row.get("keyframe_n", ""))))
                        mapping[frame_idx] = n
                    except (TypeError, ValueError):
                        continue
        except OSError:
            pass
        with self._lock:
            self._mapping_cache[video_id] = mapping
        return mapping

    def _resolve_keyframe_ordinal(self, video_id: str, frame_id: int) -> Optional[int]:
        mapping = self._load_mapping(video_id)
        return mapping.get(int(frame_id))

    def resolve_nearest_frame_id(self, video_id: str, frame_id: int) -> Optional[int]:
        """Return the mapped raw frame index nearest to `frame_id`.

        Temporal sampling often requests offsets that are not themselves
        stored keyframe indices. Falling back to the nearest mapped keyframe
        preserves the intended temporal evidence instead of silently dropping
        the frame.
        """
        mapping = self._load_mapping(video_id)
        if not mapping:
            return int(frame_id) if self.get_frame_path(video_id, int(frame_id)) else None
        keys = sorted(mapping.keys())
        pos = bisect.bisect_left(keys, int(frame_id))
        if pos == 0:
            return keys[0]
        if pos == len(keys):
            return keys[-1]
        before, after = keys[pos-1], keys[pos]
        return before if abs(int(frame_id)-before) <= abs(after-int(frame_id)) else after

    def resolve_frame_record(self, video_id: str, frame_id: int) -> Optional[Tuple[int, str]]:
        """Return (resolved_frame_id, physical_image_path)."""
        direct = self.get_frame_path(video_id, int(frame_id))
        if direct:
            return int(frame_id), direct
        nearest = self.resolve_nearest_frame_id(video_id, int(frame_id))
        if nearest is None:
            return None
        path = self.get_frame_path(video_id, nearest)
        return (nearest, path) if path else None

    def get_frame_path(self, video_id: str, frame_id: int) -> Optional[str]:
        try:
            frame_id = int(frame_id)
        except (TypeError, ValueError):
            return None

        video_dir = os.path.join(self.base_dir, video_id)
        if not os.path.isdir(video_dir):
            return None

        # Primary: official mapping frame_idx -> keyframe ordinal.
        ordinal = self._resolve_keyframe_ordinal(video_id, frame_id)
        candidates = []
        if ordinal is not None:
            candidates.extend(f"{ordinal:04d}.{ext}" for ext in _IMAGE_EXTENSIONS)
            candidates.extend(f"{ordinal:03d}.{ext}" for ext in _IMAGE_EXTENSIONS)
            candidates.extend(f"{ordinal}.{ext}" for ext in _IMAGE_EXTENSIONS)

        # Compatibility: datasets where image filename itself is the frame ID.
        candidates.extend(f"{frame_id:04d}.{ext}" for ext in _IMAGE_EXTENSIONS)
        candidates.extend(f"{frame_id:05d}.{ext}" for ext in _IMAGE_EXTENSIONS)
        candidates.extend(f"{frame_id:06d}.{ext}" for ext in _IMAGE_EXTENSIONS)
        candidates.extend(f"{frame_id}.{ext}" for ext in _IMAGE_EXTENSIONS)

        seen = set()
        for fname in candidates:
            if fname in seen:
                continue
            seen.add(fname)
            path = os.path.join(video_dir, fname)
            if os.path.isfile(path):
                return path

        return None

    def load_frame(self, video_id: str, frame_id: int) -> Optional[Image.Image]:
        path = self.get_frame_path(video_id, frame_id)
        if path is None:
            return None
        cached = self._image_cache.get(path)
        if cached is not None:
            return cached
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return None
        self._image_cache.set(path, image)
        return image

    def load_temporal_frames(self, video_id: str, center_frame_id: int,
                            window: int = qa_config.TEMPORAL_WINDOW_SIZE,
                            step: int = qa_config.FRAME_STEP) -> List[Image.Image]:
        key = frame_cache_key(video_id, center_frame_id, window, step)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        frames: List[Image.Image] = []
        seen_paths = set()
        for offset in [i * step for i in range(-window, window + 1)]:
            fid = max(0, int(center_frame_id) + offset)
            record = self.resolve_frame_record(video_id, fid)
            if record is None:
                continue
            _resolved_id, path = record
            if path in seen_paths:
                continue
            seen_paths.add(path)
            img = self.load_frame(video_id, _resolved_id)
            if img is not None:
                frames.append(img)
        if frames:
            self._frame_cache.set(key, frames)
        return frames

    def load_temporal_records(self, video_id: str, center_frame_id: int,
                              window: int = qa_config.TEMPORAL_WINDOW_SIZE,
                              step: int = qa_config.FRAME_STEP) -> List[dict]:
        records = []
        seen = set()
        for offset in [i * step for i in range(-window, window + 1)]:
            requested = max(0, int(center_frame_id) + offset)
            record = self.resolve_frame_record(video_id, requested)
            if not record:
                continue
            resolved_id, path = record
            if resolved_id in seen:
                continue
            seen.add(resolved_id)
            image = self.load_frame(video_id, resolved_id)
            if image is not None:
                records.append({"requested_frame_id": requested, "frame_id": resolved_id, "path": path, "image": image})
        return records

    def resolve_frame_path(self, video_id: str, frame_id: int) -> Optional[str]:
        return self.get_frame_path(video_id, frame_id)

    def timestamp_sec(self, video_id: str, frame_id: int) -> Optional[float]:
        """Return the official BTC timestamp (pts_time) for a frame_idx.

        The BTC CSV is the source of truth: frame_idx -> pts_time. This is
        debug/evaluation metadata only and is never used as retrieval input.
        """
        csv_path = os.path.join(self.data_dir, f"{video_id}.csv")
        try:
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                target = int(frame_id)
                for row in reader:
                    try:
                        if int(float(row.get("frame_idx", ""))) == target:
                            return float(row.get("pts_time", ""))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            return None
        return None

    def mapping_stats(self, video_id: str) -> dict:
        mapping = self._load_mapping(video_id)
        return {"video_id": video_id, "mapped_frame_ids": len(mapping)}

    def cache_stats(self) -> dict:
        return self._frame_cache.stats()
