# qa_text_similarity.py
"""
Tiện ích so khớp/gộp nhóm chuỗi câu trả lời, dùng chung bởi:

  - qa_answer_aggregator.py  (gộp nhiều câu trả lời -> MỘT câu trả lời)
  - qa_candidate_ranker.py   (xếp hạng candidate cho output cạnh tranh)

Trước đây logic normalize/similarity nằm trực tiếp trong
qa_answer_aggregator.py và bị lặp lại khi cần dùng cho ranker. Tách ra
module riêng, không phụ thuộc torch/transformers, để cả hai nơi dùng
chung một định nghĩa "hai câu trả lời có cùng ý hay không" và test được
độc lập, không cần GPU.
"""
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, Dict, List, TypeVar

T = TypeVar("T")

_PUNCT_RE = re.compile(r"[.,!?;:'\"“”‘’()\[\]]")
_SPACE_RE = re.compile(r"\s+")


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

def canonicalize_answer(answer: str, answer_type: str = "TEXT") -> str:
    """Return a compact grouping key without changing the displayed answer.

    COUNT answers such as ``2``, ``two people`` and ``hai người`` should vote
    for the same answer cluster. YES/NO paraphrases are normalized similarly.
    Other answer types keep the lexical normalization used by text_similarity.
    """
    norm = normalize_text(answer)
    kind = str(answer_type or "TEXT").upper()
    if kind == "COUNT":
        if norm.isdigit():
            return str(int(norm))
        tokens = norm.split()
        numeric = [str(_NUMBER_WORDS[t]) for t in tokens if t in _NUMBER_WORDS]
        if numeric:
            return numeric[0]
    elif kind == "YES_NO":
        if norm in {"yes", "yeah", "yep", "true"}:
            return "yes"
        if norm in {"no", "nope", "false"}:
            return "no"
    return norm


def normalize_text(text: str) -> str:
    """Chuẩn hoá: NFC, strip, lowercase, bỏ dấu câu, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text or "").strip().lower()
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    return text


def text_similarity(a: str, b: str) -> float:
    seq_ratio = SequenceMatcher(None, a, b).ratio()

    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a and not tokens_b:
        return seq_ratio
    if not tokens_a or not tokens_b:
        return 0.0
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    return min(seq_ratio, jaccard)


def cluster_by_similarity(
    items: List[T],
    key_func: Callable[[T], str],
    threshold: float,
) -> List[Dict]:
    groups: List[Dict] = []
    for item in items:
        norm = normalize_text(key_func(item))
        placed = False
        for group in groups:
            if text_similarity(norm, group["norm"]) >= threshold:
                group["members"].append(item)
                placed = True
                break
        if not placed:
            groups.append({"norm": norm, "members": [item]})
    return groups
