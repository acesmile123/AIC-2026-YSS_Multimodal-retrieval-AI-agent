from __future__ import annotations

import re
from typing import Tuple

from .qa_question_analysis import QuestionAnalysis

_COLOR_WORDS = {
    "red","blue","green","yellow","black","white","brown","gray","grey","orange","purple","pink","red","đen","trắng","đỏ","xanh","vàng","nâu","xám","cam","tím","hồng"
}


def validate_answer(answer: str, analysis: QuestionAnalysis) -> tuple[bool,float,str]:
    text = str(answer or "").strip()
    if not text or text.upper() == "UNKNOWN": return False, 0.0, "empty_or_unknown"
    if analysis.expected_answer_type == "COUNT":
        if re.search(r"\d+", text): return True, 1.0, "count"
        words = {"zero","one","two","three","four","five","six","seven","eight","nine","ten","một","hai","ba","bốn","năm","sáu","bảy","tám","chín","mười","0","1","2","3","4","5","6","7","8","9"}
        if any(w in text.lower().split() for w in words): return True, 0.95, "count"
        return False, 0.1, "not_a_count"
    if analysis.expected_answer_type == "COLOR":
        ok = any(c in text.lower() for c in _COLOR_WORDS)
        return ok, 1.0 if ok else 0.15, "color" if ok else "not_a_color"
    if analysis.expected_answer_type == "YES_NO":
        low = text.lower()
        ok = low in {"yes","no","có","không","yes.","no."} or low.startswith(("yes","no","có","không"))
        return ok, 1.0 if ok else 0.1, "yes_no" if ok else "not_yes_no"
    return True, 0.65, "general"
