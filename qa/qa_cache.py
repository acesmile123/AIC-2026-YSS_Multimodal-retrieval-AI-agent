# qa_cache.py
"""
Cache bounded (giới hạn kích thước) cho nhánh Q&A.

Mục tiêu: trong một phiên chạy có nhiều câu hỏi hỏi lại trên cùng
video/frame (rất phổ biến khi người dùng thử nhiều câu hỏi cho cùng một
kết quả KIS), việc load lại ảnh từ đĩa và chạy lại VLM là lãng phí. Cache
này KHÔNG dùng functools.lru_cache trực tiếp vì đối số (PIL.Image, dict
structured_query) không hashable / không nên dùng làm key; thay vào đó ta
tự xây key là tuple các giá trị nguyên thủy (str/int).

Cache là in-memory, single-process, không thread-safe theo nghĩa
lock-free hoàn toàn nhưng dùng threading.Lock để an toàn cơ bản nếu được
gọi từ nhiều thread (vd. một web server phục vụ nhiều request).
"""
import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple


class LRUCache:
    """Cache LRU đơn giản, thread-safe, giới hạn số lượng entry."""

    def __init__(self, max_size: int = 128):
        if max_size <= 0:
            raise ValueError("max_size phải > 0")
        self.max_size = max_size
        self._data: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)  # evict LRU nhất

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total else 0.0
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0


def frame_cache_key(video_id: str, center_frame_id: int, window: int, step: int) -> Tuple:
    return ("frames", video_id, int(center_frame_id), window, step)


def answer_cache_key(video_id: str, frame_id: int, question: str) -> Tuple:

    return ("answer", video_id, int(frame_id), question.strip().lower())
