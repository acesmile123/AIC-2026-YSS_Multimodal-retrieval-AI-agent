from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).lower().strip()
    return re.sub(r"\s+", " ", text)


# English-only structural patterns. These describe question grammar, not a
# vocabulary of expected objects. Unknown nouns are preserved verbatim and can
# still be resolved by the semantic parser/VLM fallback.
_COUNT_RE = re.compile(r"(?:\b(?:how many|how much|number of)\b|\bbao nhiêu\b|\btất cả bao nhiêu\b|\bmấy\b)", re.I | re.U)
_COLOR_RE = re.compile(r"(?:\b(?:what|which)\s+(?:color|colour)\b|\b(?:màu gì|màu nào)\b)", re.I | re.U)
_YESNO_RE = re.compile(r"^(?:(?:is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b|có phải\b)", re.I | re.U)
_TEMPORAL_RE = re.compile(r"(?:\b(?:what happens next|what happened before|after|before|then|next|finally|first|while|during)\b|\b(?:tiếp theo|trước đó|sau đó|sau khi|trước khi|khi nào)\b)", re.I | re.U)
_LOCATION_RE = re.compile(r"(?:\bwhere\b|\bwhich side\b|\bở đâu\b|\bnằm ở đâu\b|\bđược đặt ở đâu\b|\bvị trí nào\b|\bđứng ở đâu\b)", re.I | re.U)

_RELATION_PATTERNS = [
    (re.compile(r"\bclosest\s+to\b", re.I), "closest_to"),
    (re.compile(r"\bnearest\s+to\b", re.I), "closest_to"),
    (re.compile(r"\bfarthest\s+from\b", re.I), "farthest_from"),
    (re.compile(r"\bin\s+front\s+of\b", re.I), "in_front_of"),
    (re.compile(r"\bbehind\b", re.I), "behind"),
    (re.compile(r"\bleft\s+of\b", re.I), "left_of"),
    (re.compile(r"\bright\s+of\b", re.I), "right_of"),
    (re.compile(r"\bnext\s+to\b", re.I), "next_to"),
    (re.compile(r"\bbeside\b", re.I), "next_to"),
    (re.compile(r"\bnear(?:by)?\b", re.I), "near"),
    (re.compile(r"\babove\b", re.I), "above"),
    (re.compile(r"\bbelow\b", re.I), "below"),
    (re.compile(r"\bphía trước\b", re.I | re.U), "in_front_of"),
    (re.compile(r"\bđằng sau\b", re.I | re.U), "behind"),
    (re.compile(r"\bbên trái\b", re.I | re.U), "left_of"),
    (re.compile(r"\bbên phải\b", re.I | re.U), "right_of"),
    (re.compile(r"\bgần\b", re.I | re.U), "near"),
    (re.compile(r"\bbên cạnh\b", re.I | re.U), "next_to"),
]

_ACTION_RE = re.compile(
    r"(?:\b(?:holding|carrying|wearing|using|eating|drinking|sitting|standing|walking|running|speaking|talking|entering|leaving|opening|closing|performing|operating(?!\s+room)|pointing|looking|riding|driving)\b|\b(?:đang cầm|cầm|mặc|đang mặc|ăn|uống|ngồi|đứng|đi bộ|đang đi|chạy|phát biểu|nói|đọc|giữ)\b)",
    re.I | re.U,
)

_COLOR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "gray", "grey", "pink", "đỏ", "xanh", "xanh dương", "xanh lá", "vàng", "cam", "tím", "đen", "trắng", "nâu", "xám", "hồng"}
_STOP = {
    "what", "which", "who", "how", "many", "much", "number", "of", "is", "are", "was", "were", "do", "does", "did",
    "the", "a", "an", "there", "in", "on", "at", "to", "from", "with", "for", "and", "or", "that", "this", "those", "these",
    "visible", "shown", "seen", "frame", "image", "video", "picture", "there", "any",
}


def _clean_np(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip(" ,.?;:")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    return text


def _find_reference_after(text: str, marker: str) -> str:
    pos = text.lower().find(marker.lower())
    if pos < 0:
        return ""
    tail = text[pos + len(marker):]
    tail = re.split(r"\b(?:and|while|before|after|that|which|who|where)\b", tail, maxsplit=1, flags=re.I)[0]
    return _clean_np(tail)


def _extract_target_for_attribute(text: str) -> str:
    # what color is the roadside pole in the frame with the red fire truck
    m = re.search(r"\b(?:what|which)\s+(?:color|colour)\s+(?:is|was)\s+(.+)", text, re.I)
    if m:
        tail = m.group(1)
        tail = re.split(r"\b(?:in|on|near|beside|next\s+to|with|that|which|who)\b", tail, maxsplit=1, flags=re.I)[0]
        return _clean_np(tail)
    # which color does the sign have
    m = re.search(r"\b(?:what|which)\s+(?:color|colour)\s+does\s+(.+?)\s+(?:have|show)\b", text, re.I)
    if m:
        return _clean_np(m.group(1))
    return ""


def _extract_action_target(text: str) -> str:
    # what is the surgeon holding / what does the surgeon hold
    m = re.search(r"\bwhat\s+(?:is|does|did)\s+(.+?)\s+(?:holding|carrying|wearing|using|eating|drinking|doing|opening|closing|riding|driving|sitting|standing|speaking|talking)\b", text, re.I)
    if m:
        return _clean_np(m.group(1))
    return ""


def extract_count_target(text: str) -> Tuple[str, List[str]]:
    t = normalize_text(text)
    m = re.search(r"(?:\b(?:how\s+many|how\s+much|number\s+of)\b|\bbao nhiêu\b|\bmấy\b)\s+(.+)", t, re.I | re.U)
    if not m:
        return "", []
    span = re.split(
        r"(?:\b(?:are|is|were|was|visible|shown|standing|sitting|in|on|at|near|behind|with|from|who|that|which)\b|\b(?:đang|được|trên|trong|ở|phía trước|phía sau|bên trái|bên phải|gần|với)\b)",
        m.group(1), maxsplit=1, flags=re.I | re.U
    )[0]
    span = _clean_np(span)
    if not span:
        return "", []
    terms = [t for t in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", span, flags=re.UNICODE) if t.lower() not in _STOP]
    return span, list(dict.fromkeys(terms))


def extract_reference(text: str, relation_end: int | None = None) -> Tuple[str, str, Dict[str, str]]:
    t = normalize_text(text)
    attributes: Dict[str, str] = {}
    candidates: List[str] = []
    # Relationship references. Keep descriptor words, including explicit color,
    # because they are useful for selecting the correct detection instance.
    for pat, _ in _RELATION_PATTERNS:
        m = pat.search(t)
        if m and m.end() <= len(t):
            tail = _find_reference_after(t, m.group(0))
            if tail:
                candidates.append(tail)
    for marker in ("with the", "with a", "with an", "next to the", "beside the"):
        tail = _find_reference_after(t, marker)
        if tail:
            candidates.append(tail)
    reference = max(candidates, key=len, default="")
    for token in reference.lower().split():
        if token in _COLOR_WORDS:
            attributes["color"] = token
            break
    return reference, reference, attributes


def extract_relations(text: str, target: str = "") -> List[Dict[str, Any]]:
    t = normalize_text(text)
    out: List[Dict[str, Any]] = []
    # Parse the most specific relation first so "closest to the red suitcase"
    # keeps the complete reference instead of a truncated noun.
    for pat, relation in _RELATION_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        ref = _find_reference_after(t, m.group(0))
        if not ref:
            continue
        attrs = {"color": tok for tok in ref.lower().split() if tok in _COLOR_WORDS}
        # keep at most one explicit color token
        if attrs:
            attrs = {"color": next(iter(attrs.values()))}
        out.append({
            "type": relation,
            "subject": target,
            "reference": ref,
            "reference_entity": ref,
            "reference_attributes": attrs,
            "source": "structural",
        })
        break
    return out


def extract_constraints(text: str, target: str, relations: Sequence[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    t = normalize_text(text)
    out: List[Dict[str, Any]] = []
    if _ACTION_RE.search(t):
        out.append({"type": "action", "entity": target, "value": _ACTION_RE.search(t).group(0).lower(), "source": "structural"})
    # Requested attribute is a question requirement, not a target attribute.
    if _COLOR_RE.search(t):
        out.append({"type": "attribute_request", "entity": target, "value": "color", "source": "structural"})
    if relations:
        for rel in relations:
            if rel.get("reference_attributes"):
                out.append({"type": "reference_attribute", "entity": rel.get("reference"), "attributes": rel["reference_attributes"], "source": "structural"})
    temporal = _TEMPORAL_RE.search(t)
    if temporal:
        value = temporal.group(0).lower()
        direction = "after" if value in {"after", "next", "then"} or "what happens next" in value else "before" if "before" in value else "center"
        out.append({"type": "temporal", "direction": direction, "value": value, "source": "structural"})
    return out


def extract_subqueries(text: str) -> List[Dict[str, Any]]:
    t = normalize_text(text)
    parts = re.split(r"\band among them\b|\bof those\b", t, maxsplit=1)
    if len(parts) != 2 or not all(_COUNT_RE.search(p) for p in parts):
        return []
    return [{**plan_question(parts[0]), "scope": "global"}, {**plan_question(parts[1]), "scope": "previous_result"}]


def _extract_general_target(text: str) -> Tuple[str, List[str]]:
    t = normalize_text(text)
    candidates = []
    for pattern in (
        r"\bwhat\s+is\s+(?:the|a|an)?\s*(.+?)\s+(?:doing|holding|wearing|using|carrying)\b",
        r"\bwhat\s+is\s+(?:the|a|an)?\s*(.+?)\s+(?:color|colour)\b",
        r"\bwhat\s+does\s+(?:the|a|an)?\s*(.+?)\s+(?:have|use|hold|wear)\b",
        r"\bwhich\s+(?:object|person|vehicle|item)\b",
        r"\bwhich\s+(.+?)\s+(?:is|are)\b",
    ):
        m = re.search(pattern, t, re.I)
        if m:
            candidates.append(_clean_np(m.group(1) if m.lastindex else m.group(0)))
    if candidates:
        target = max(candidates, key=len)
        if target.startswith("which "):
            target = target.split(" ", 1)[1]
        return target, [target]
    # Preserve an informative noun phrase instead of canonicalizing to a tiny ontology.
    tokens = [x for x in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", t, flags=re.UNICODE) if x.lower() not in _STOP]
    if tokens:
        return " ".join(tokens[:5]), tokens[:5]
    return "", []


def plan_question(question: str) -> Dict[str, Any]:
    t = normalize_text(question)
    is_count = bool(_COUNT_RE.search(t))
    if _COLOR_RE.search(t):
        target = _extract_target_for_attribute(t) or _extract_action_target(t) or _extract_general_target(t)[0]
    elif _ACTION_RE.search(t) and re.search(r"\bwhat\b", t):
        target = _extract_action_target(t) or _extract_general_target(t)[0]
    elif is_count:
        target = extract_count_target(t)[0]
    elif _TEMPORAL_RE.search(t):
        target = ""
    else:
        target = _extract_general_target(t)[0]
    target_terms = [target] if target else []
    relations = extract_relations(t, target)
    # "with the red fire truck" is a disambiguating reference even when no
    # spatial relation word is present. Keep it separate from the target.
    if not relations and _COLOR_RE.search(t):
        ref = re.search(r"\bwith\s+(?:the|a|an)\s+(.+?)(?:\s+(?:in|on|near|beside|next\s+to)\b|$)", t, re.I)
        if ref:
            ref_text = _clean_np(ref.group(1))
            ref_attrs = {"color": next((w for w in ref_text.lower().split() if w in _COLOR_WORDS), "")}
            ref_attrs = {k:v for k,v in ref_attrs.items() if v}
            relations = [{"type": "reference_context", "subject": target, "reference": ref_text, "reference_entity": ref_text, "reference_attributes": ref_attrs, "source": "structural"}]
    constraints = extract_constraints(t, target, relations)
    temporal = [c for c in constraints if c.get("type") == "temporal"]
    attributes = [c for c in constraints if c.get("type") == "attribute_request"]
    actions = [c for c in constraints if c.get("type") == "action"]
    references = []
    for rel in relations:
        if rel.get("reference"):
            references.append({"text": rel["reference"], "type": rel.get("reference_entity", rel["reference"]), "attributes": rel.get("reference_attributes", {})})
    complexity = "simple"
    if relations or len(constraints) >= 2:
        complexity = "compound"
    if len(relations) >= 2 or temporal or re.search(r"\b(?:and among them|of those|after|before)\b", t):
        complexity = "hard"
    operation = "count" if is_count else "temporal_reasoning" if temporal else "answer"
    expected = "COUNT" if is_count else "COLOR" if _COLOR_RE.search(t) else "LOCATION" if _LOCATION_RE.search(t) else "TEXT"
    if _YESNO_RE.search(t):
        expected = "YES_NO"
    return {
        "normalized": t,
        "subqueries": extract_subqueries(t),
        "is_count": is_count,
        "target": target,
        "target_terms": tuple(dict.fromkeys(target_terms)),
        "relations": relations,
        "constraints": constraints,
        "roles": [],
        "actions": actions,
        "attributes": attributes,
        "temporal": temporal,
        "references": references,
        "complexity": complexity,
        "operation": operation,
        "expected_answer_type": expected,
        "is_multi_hop": bool(relations) or bool(temporal),
    }
