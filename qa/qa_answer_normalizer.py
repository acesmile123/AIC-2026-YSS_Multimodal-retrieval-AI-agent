# qa_answer_normalizer.py
"""
Làm sạch (normalize) và trích xuất câu trả lời + độ tự tin từ text thô do
VLM sinh ra.

Tách riêng khỏi vlm_engine.py (module này KHÔNG import torch/transformers)
vì hai lý do:

  1. Giữ đúng triết lý test hiện tại của dự án: test_qa_pipeline.py chủ
     động tránh import vlm_engine.py để chạy được trên môi trường không
     có GPU/torch. Logic parse/normalize là logic THUẦN (string in,
     string+float out) nên đáng được test độc lập, không cần torch.
  2. vlm_engine.py có thể tái sử dụng các hàm này mà không kéo thêm phụ
     thuộc nào.

Định dạng mong đợi từ prompt (xem vlm_engine._build_prompt):

    ANSWER: <câu trả lời ngắn gọn>
    CONFIDENCE: <HIGH|MEDIUM|LOW>

Các model instruct nhỏ (vd. Qwen2.5-VL-3B) không phải lúc nào cũng tuân
thủ định dạng tuyệt đối, nên parser ở đây LUÔN có fallback hợp lý thay vì
raise lỗi.
"""
import re
from typing import Optional, Tuple

_ANSWER_FIELD_RE = re.compile(r"ANSWER\s*[:\-]\s*(.+)", re.IGNORECASE)
_CONFIDENCE_FIELD_RE = re.compile(r"CONFIDENCE\s*[:\-]\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)
_LEADING_LABEL_RE = re.compile(r"^(answer|trả lời|final answer)\s*[:\-]\s*", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[\s.。]+$")
_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_CHARS = "\"'“”‘’"

CONFIDENCE_LABEL_TO_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.3}
DEFAULT_CONFIDENCE = 0.6


def clean_answer_text(text: str, max_words: Optional[int] = None) -> str:
    """Làm sạch một chuỗi answer thô: bỏ nhãn lặp, bỏ dấu ngoặc kép bao
    quanh, gộp khoảng trắng, bỏ dấu câu thừa ở cuối, và (tuỳ chọn) cắt
    còn tối đa `max_words` từ - câu trả lời ngắn gọn khớp định dạng
    ground-truth tốt hơn là một câu giải thích dài dòng."""
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = _LEADING_LABEL_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip(_QUOTE_CHARS).strip()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _TRAILING_PUNCT_RE.sub("", cleaned).strip()

    if max_words is not None and max_words > 0:
        words = cleaned.split(" ")
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).strip()

    return cleaned


def parse_vlm_output(
    raw_text: str,
    unknown_token: str = "UNKNOWN",
    max_words: Optional[int] = None,
) -> Tuple[str, float]:
    """
    Parse output thô của VLM thành (answer, confidence).

    - Trả về ("", 0.0) nếu không đủ bằng chứng (raw_text rỗng, hoặc chính
      là unknown_token, hoặc sau khi làm sạch answer vẫn rỗng/là
      unknown_token).
    - Nếu tìm được dòng "ANSWER: ..." thì dùng nó; ngược lại fallback về
      dòng đầu tiên của raw_text (model đôi khi bỏ qua định dạng yêu cầu
      nhưng vẫn trả lời trực tiếp).
    - Nếu tìm được "CONFIDENCE: HIGH|MEDIUM|LOW" thì map sang điểm số;
      ngược lại dùng DEFAULT_CONFIDENCE (trung tính, không thưởng cũng
      không phạt candidate khi model không tuân thủ định dạng).
    """
    if not raw_text:
        return "", 0.0

    text = raw_text.strip()
    if not text:
        return "", 0.0

    if text.upper() == unknown_token.upper():
        return "", 0.0

    match_answer = _ANSWER_FIELD_RE.search(text)
    if match_answer:
        answer_raw = match_answer.group(1).splitlines()[0]
    else:
        answer_raw = text.splitlines()[0]

    match_confidence = _CONFIDENCE_FIELD_RE.search(text)
    if match_confidence:
        confidence = CONFIDENCE_LABEL_TO_SCORE.get(
            match_confidence.group(1).lower(), DEFAULT_CONFIDENCE
        )
    else:
        confidence = DEFAULT_CONFIDENCE

    answer = clean_answer_text(answer_raw, max_words=max_words)

    if not answer or answer.upper() == unknown_token.upper():
        return "", 0.0

    return answer, confidence
