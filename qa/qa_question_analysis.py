from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from . import qa_config
from .qa_query_planner import plan_question

QUESTION_TYPES = (
    "COUNTING", "OBJECT", "ATTRIBUTE", "ACTION", "TEMPORAL", "SPATIAL",
    "RELATIONSHIP", "COMPARISON", "YES_NO", "OCR", "GENERAL"
)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).lower().strip()
    return re.sub(r"\s+", " ", text)


# FAST PATH ONLY. These are high-signal patterns, not an attempt to encode the
# whole language. When the fast path cannot confidently express the intent,
# the classifier falls through to semantic parsing.
_PATTERNS = {
    "COUNTING": [
        r"\bhow many\b", r"\bhow much\b", r"\bnumber of\b",
    ],
    "OBJECT": [
        r"\bwhat (?:animal|object|thing|vehicle|person)s?\b",
    ],
    "ATTRIBUTE": [
        r"\bwhat color\b", r"\bwhat colour\b", r"\bwhat .* wearing\b",
        r"\bwhat type\b", r"\bwhich color\b", r"\bwhat kind\b", r"\bwhat attribute\b",
    ],
    "ACTION": [
        r"\bwhat is .* doing\b", r"\bwhat does .* do\b", r"\bdoing what\b",
    ],
    "TEMPORAL": [
        r"\bwhat happens next\b", r"\bwhat happened before\b", r"\bwhen does\b", r"\bafter\b", r"\bbefore\b",
    ],
    "SPATIAL": [
        r"\bwhere\b", r"\bwhich side\b", r"\bleft\b", r"\bright\b",
        r"\bbehind\b", r"\bin front of\b", r"\bin front\b", r"\bnear\b", r"\bclosest\b",
    ],
    "RELATIONSHIP": [
        r"\bnext to\b", r"\bholding\b", r"\bwith whom\b", r"\bfollowing\b",
    ],
    "COMPARISON": [
        r"\blarger\b", r"\bsmaller\b", r"\bbigger\b", r"\bclosest\b",
        r"\bmost\b", r"\bleast\b", r"\bmore than\b",
    ],
    "YES_NO": [
        r"^is\b", r"^are\b", r"^does\b", r"^do\b", r"^can\b",
    ],
    "OCR": [
        r"\bwhat does .* say\b", r"\bwhat is written\b", r"\btext\b", r"\bsign\b", r"\blogo\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.I | re.U) for p in v] for k, v in _PATTERNS.items()}


@dataclass(frozen=True)
class QuestionAnalysis:
    question: str
    primary_type: str
    types: tuple[str, ...] = field(default_factory=tuple)
    needs_temporal: bool = False
    temporal_direction: str = "center"
    expected_answer_type: str = "TEXT"
    target_terms: tuple[str, ...] = field(default_factory=tuple)
    is_multi_hop: bool = False
    # Semantic layer. Kept optional so all legacy callers remain compatible.
    operation: str = "answer"
    target: str = ""
    relations: tuple[dict, ...] = field(default_factory=tuple)
    attributes: tuple[dict, ...] = field(default_factory=tuple)
    semantic_query: dict = field(default_factory=dict)
    analysis_source: str = "keyword"
    semantic_confidence: float = 0.0
    execution_plan: tuple[dict, ...] = field(default_factory=tuple)
    # High-accuracy query contract. These fields make conditional questions
    # explicit so downstream modules never have to guess what COUNT means.
    count_target: str = ""
    count_constraints: tuple[dict, ...] = field(default_factory=tuple)
    reference_entities: tuple[dict, ...] = field(default_factory=tuple)
    temporal_constraints: tuple[dict, ...] = field(default_factory=tuple)
    query_complexity: str = "simple"
    subqueries: tuple[dict, ...] = field(default_factory=tuple)


class QuestionClassifier:
    """Fast lexical path + strict semantic fallback.

    The lexical path is intentionally small and high-confidence. It is used
    when it can explain the question cheaply. If the question is unknown,
    ambiguous, or requires structure beyond a simple type label, the classifier
    calls an injected semantic parser. The semantic parser is expected to be a
    callable: semantic_parser(question) -> dict.
    """
    def __init__(self, semantic_parser: Optional[Callable[[str], Any]] = None):
        self.semantic_parser = semantic_parser

    def classify(self, question: str) -> QuestionAnalysis:
        q = _norm(question)
        planner = plan_question(q)
        scores = {k: 0 for k in QUESTION_TYPES}
        hits = set()
        for kind, patterns in _COMPILED.items():
            for p in patterns:
                if p.search(q):
                    scores[kind] += 1
                    hits.add(kind)

        # Augment lexical intent with planner-level semantic signals so paraphrases
        # like "who was first", "closest to", or Vietnamese temporal clauses do not
        # fall through as GENERAL merely because no exact keyword pattern matched.
        if planner.get("is_count"):
            hits.add("COUNTING"); scores["COUNTING"] += 1
        if planner.get("relations"):
            hits.add("SPATIAL"); scores["SPATIAL"] += 1
        if planner.get("temporal"):
            hits.add("TEMPORAL"); scores["TEMPORAL"] += 1
        if planner.get("attributes"):
            hits.add("ATTRIBUTE"); scores["ATTRIBUTE"] += 1
        if planner.get("roles") or planner.get("actions"):
            hits.add("ACTION"); scores["ACTION"] += 1
        ordered = tuple(sorted(hits, key=lambda k: (-scores[k], k)))
        lexical_primary = self._primary_from_hits(scores, hits)

        # If lexical signals are missing, or if they only give a partial intent
        # (e.g. COUNT + a relational phrase that needs a graph), prefer semantic
        # parsing. This avoids growing an unbounded keyword dictionary.
        needs_semantic = self._needs_semantic(q, ordered, scores, planner=planner)
        semantic = self._run_semantic(q) if needs_semantic and self.semantic_parser else None

        if semantic:
            return self._from_semantic(q, semantic, lexical_fallback=(ordered, lexical_primary, scores))

        # Dependency-free fast path / backward compatibility.
        if not hits:
            primary = "GENERAL"
        else:
            primary = lexical_primary
        expected = "COUNT" if "COUNTING" in hits else self._expected_type(primary, q)
        planner_target = planner.get("target", "")
        target = planner.get("target_terms") or self._extract_terms(q)
        temporal = "TEMPORAL" in hits or bool(planner.get("temporal"))
        direction = self._temporal_direction(q)
        multi_hop = len(ordered) >= 2 or bool(planner.get("is_multi_hop"))
        relations = tuple(planner.get("relations", ()))
        attributes = tuple(x for x in planner.get("attributes", ()) if isinstance(x, dict))
        ref_entities = tuple(x for x in planner.get("references", ()) if isinstance(x, dict))
        count_constraints = tuple([*planner.get("constraints", ()), *[dict(r, type="relation") for r in relations]]) if expected == "COUNT" else tuple(planner.get("constraints", ()))
        semantic_query = {
            "operation": "count" if expected == "COUNT" else planner.get("operation", "answer"),
            "target": planner_target if planner_target not in {"this", "that", "something", "thing"} else "",
            "target_terms": list(target),
            "relations": list(relations),
            "attributes": list(attributes),
            "reference_entities": list(ref_entities),
            "count_constraints": list(count_constraints),
            "temporal_constraints": list(planner.get("temporal", ())),
            "subqueries": list(planner.get("subqueries", ())),
        }
        execution_plan = self._build_execution_plan(primary, expected, target, relations, attributes, temporal)
        if expected == "COUNT" and any(c.get("type") in {"role", "action", "attribute", "reference_attribute", "temporal"} for c in count_constraints if isinstance(c, dict)):
            base_steps = [s for s in execution_plan if s.get("step") not in {"evidence_consensus", "count"}]
            execution_plan = tuple(base_steps + [{"step": "filter_semantic_constraints", "constraints": list(count_constraints)}, {"step": "evidence_consensus"}, {"step": "count"}])
        complexity = planner.get("complexity", "compound" if len(ordered) >= 2 or relations else "simple")
        return QuestionAnalysis(
            q, primary, ordered, temporal, direction, expected, target,
            multi_hop or bool(relations),
            operation=semantic_query.get("operation", "answer"),
            target=semantic_query.get("target", ""),
            relations=relations,
            semantic_query=semantic_query,
            analysis_source="keyword" if hits else "fallback",
            semantic_confidence=0.0,
            execution_plan=execution_plan,
            count_target=self._normalize_count_target(target[0] if expected == "COUNT" and target else "", target),
            count_constraints=count_constraints,
            reference_entities=ref_entities,
            query_complexity=complexity,
            subqueries=tuple(x for x in planner.get("subqueries", ()) if isinstance(x, dict)),
        )

    @staticmethod
    def _lexical_relations(q: str, target: str = "") -> list[dict]:
        relations = []
        patterns = [
            (r"\bin front of\b", "in_front_of"), (r"\bbehind\b", "behind"),
            (r"\bleft of\b", "left_of"), (r"\bright of\b", "right_of"),
            (r"\bnear(?:by)?\b", "near"), (r"\bnext to\b", "next_to"),
        ]
        for pattern, rel in patterns:
            m = re.search(pattern, q, re.I)
            if not m:
                continue
            ref = re.split(r"\b(?:and|before|after|then|while)\b", q[m.end():], maxsplit=1, flags=re.I)[0].strip(" ,.?!")
            if ref:
                relations.append({"type": rel, "subject": target or "", "reference": ref, "source": "structural"})
            break
        return relations

    @staticmethod
    def _normalize_count_target(target: str, terms=()) -> str:
        text = str(target or "").strip().lower()
        return text

    @staticmethod
    def _primary_from_hits(scores: dict, hits: set[str]) -> str:
        priority = ["COUNTING", "OCR", "TEMPORAL", "ACTION", "ATTRIBUTE", "COMPARISON", "SPATIAL", "RELATIONSHIP", "OBJECT", "YES_NO", "GENERAL"]
        return max(priority, key=lambda x: (scores.get(x, 0), -priority.index(x))) if hits else "GENERAL"

    @staticmethod
    def _needs_semantic(q: str, ordered: tuple[str, ...], scores: dict, planner: dict | None = None) -> bool:
        planner = planner or plan_question(q)
        # No recognizable type -> semantic fallback.
        if not ordered:
            return True
        # Several types can be valid, but relation/reference structure is where
        # lexical rules become brittle. Ask the semantic layer to build a graph.
        if len(ordered) >= 2:
            return True
        # Single-type questions with a relational or comparison phrase often
        # require entity/reference resolution, so semantic parsing is safer.
        if ordered[0] in {"SPATIAL", "RELATIONSHIP", "COMPARISON"}:
            return True
        # Counting questions with explicit relational/object qualifiers need a
        # semantic target instead of blindly using the global object count.
        if ordered[0] == "COUNTING" and (planner.get("relations") or planner.get("constraints") or planner.get("complexity") == "hard"):
            return True
        if planner.get("complexity") == "hard":
            return True
        return False

    def _run_semantic(self, q: str) -> Optional[dict]:
        try:
            raw = self.semantic_parser(q)
            if isinstance(raw, dict):
                return raw
            text = str(raw or "").strip()
            if not text:
                return None
            # Accept JSON fenced or plain JSON.
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            return None
        return None

    @staticmethod
    def _from_semantic(q: str, data: dict, lexical_fallback=None) -> QuestionAnalysis:
        def clean_list(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [x for x in value]
            return [value]

        types = []
        for t in clean_list(data.get("types")):
            t = str(t).upper().strip()
            if t in QUESTION_TYPES and t not in types:
                types.append(t)
        if not types and lexical_fallback:
            types = list(lexical_fallback[0])
        if not types:
            types = ["GENERAL"]
        primary = str(data.get("primary_type") or types[0]).upper()
        if primary not in QUESTION_TYPES:
            primary = types[0]
        operation = str(data.get("operation") or ("count" if "COUNTING" in types else "answer")).strip().lower()
        expected = str(data.get("expected_answer_type") or ("COUNT" if operation == "count" or "COUNTING" in types else "TEXT")).upper()
        temporal_dir = str(data.get("temporal_direction") or "center").lower()
        needs_temporal = bool(data.get("needs_temporal", "TEMPORAL" in types))
        planner = plan_question(q)
        subqueries = tuple(x for x in clean_list(data.get("subqueries")) if isinstance(x, dict)) or tuple(x for x in planner.get("subqueries", ()) if isinstance(x, dict))
        raw_target = str(data.get("target") or "").strip()
        generic_targets = {"this", "that", "something", "thing", "object"}
        target = raw_target if raw_target.lower() not in generic_targets else ""
        planner_target = str(planner.get("target") or "").strip()
        if not target or (planner_target and re.match(r"^(?:the\s+|a\s+|an\s+)?", q) is not None and q.startswith(tuple(str(x) for x in planner.get("target_terms", ()) if x))):
            target = planner_target or target
        target_terms = tuple(dict.fromkeys(str(x).strip() for x in clean_list(data.get("target_terms")) if str(x).strip()))
        planner_terms = planner.get("target_terms") or ()
        if not target_terms:
            target_terms = tuple(str(x).strip() for x in planner_terms if str(x).strip())
        if target and target not in target_terms:
            target_terms = (target, *target_terms)
        relations_sem = tuple(x for x in clean_list(data.get("relations")) if isinstance(x, dict))
        relations_plan = tuple(x for x in planner.get("relations", ()) if isinstance(x, dict))
        relations = relations_plan or relations_sem
        attributes_sem = tuple(x for x in clean_list(data.get("attributes")) if isinstance(x, dict))
        attributes_plan = tuple(x for x in planner.get("attributes", ()) if isinstance(x, dict))
        attributes = attributes_sem or attributes_plan
        semantic_query = dict(data)
        semantic_query.setdefault("subqueries", list(subqueries))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        reference_entities = tuple(x for x in clean_list(data.get("reference_entities")) if isinstance(x, dict))
        if not reference_entities:
            refs = []
            for rel in relations:
                if rel.get("reference_entity") or rel.get("reference"):
                    refs.append({"type": rel.get("reference_entity") or rel.get("reference"), "text": rel.get("reference", "")})
            reference_entities = tuple(refs)
        temporal_constraints = tuple(x for x in clean_list(data.get("temporal_constraints")) if isinstance(x, dict)) or tuple(planner.get("temporal", ()))
        count_constraints = tuple(x for x in clean_list(data.get("count_constraints")) if isinstance(x, dict))
        if not count_constraints:
            count_constraints = tuple([*planner.get("constraints", ()), *[dict(r, type="relation") for r in relations]])
        raw_count_target = str(data.get("count_target") or (target if expected == "COUNT" else "")).strip()
        count_target = QuestionClassifier._normalize_count_target(raw_count_target, target_terms)
        # Derive constraints from attributes/relations when a semantic parser
        # omits the redundant count_constraints field.
        if expected == "COUNT" and not count_constraints:
            count_constraints = tuple([
                {"type": "attribute", **a} for a in attributes
            ] + [
                {"type": "relation", **r} for r in relations
            ])
        complexity = planner.get("complexity", "simple")
        complexity_signals = len(types) + len(relations) + len(attributes) + len(reference_entities) + len(count_constraints)
        if needs_temporal or temporal_constraints:
            complexity_signals += 2
        if operation in {"compare", "temporal_next", "temporal_before", "multi_hop"}:
            complexity_signals += 2
        if complexity_signals >= 6: complexity = "hard"
        elif complexity_signals >= 3: complexity = "compound"
        execution_plan = QuestionClassifier._build_execution_plan(primary, expected, target_terms, relations, attributes, data.get("needs_temporal", False))
        return QuestionAnalysis(
            q, primary, tuple(types), needs_temporal, temporal_dir, expected, target_terms,
            len(types) >= 2 or len(relations) >= 1 or len(attributes) >= 1 or bool(reference_entities),
            operation=operation,
            target=target,
            relations=relations,
            attributes=attributes,
            semantic_query=semantic_query,
            analysis_source="semantic",
            semantic_confidence=max(0.0, min(1.0, confidence)),
            execution_plan=execution_plan,
            count_target=count_target,
            count_constraints=count_constraints,
            reference_entities=reference_entities,
            temporal_constraints=temporal_constraints,
            query_complexity=complexity,
            subqueries=subqueries,
        )

    @staticmethod
    def _build_execution_plan(primary: str, expected: str, target_terms, relations, attributes, needs_temporal: bool=False) -> tuple[dict, ...]:
        """Build a compact deterministic execution plan for downstream QA.

        The plan does not replace visual reasoning; it prevents the VLM from
        being asked to infer the *structure* of the task repeatedly.
        """
        plan = []
        if expected == "COUNT":
            plan.append({"step": "ground", "target": target_terms[0] if target_terms else "object"})
            if attributes:
                plan.append({"step": "filter_attributes", "attributes": list(attributes)})
            if relations:
                plan.append({"step": "filter_relations", "relations": list(relations)})
            if any(c.get("type") in {"role", "action", "temporal"} for c in getattr(target_terms, "constraints", ()) if isinstance(c, dict)):
                pass
            plan.append({"step": "temporal_consensus" if needs_temporal else "evidence_consensus"})
            plan.append({"step": "count"})
        elif primary == "ATTRIBUTE":
            plan.extend([{ "step": "ground", "target": target_terms[0] if target_terms else "entity"}, {"step": "extract_attributes"}])
        elif primary == "ACTION":
            plan.extend([{ "step": "ground", "target": target_terms[0] if target_terms else "entity"}, {"step": "temporal_action_analysis"}])
        elif relations:
            plan.extend([{ "step": "ground_entities"}, {"step": "resolve_relations"}])
        else:
            plan.append({"step": "retrieve_and_reason"})
        if needs_temporal or primary == "TEMPORAL":
            plan.append({"step": "temporal_ordering"})
        return tuple(plan)

    @staticmethod
    def _expected_type(primary: str, q: str) -> str:
        if primary == "COUNTING": return "COUNT"
        if primary == "ATTRIBUTE":
            if "color" in q or "colour" in q: return "COLOR"
            if any(t in q for t in ("type", "kind")): return "CATEGORY"
            return "TEXT"
        if primary == "YES_NO": return "YES_NO"
        if primary == "OCR": return "TEXT"
        if primary == "COMPARISON": return "ENTITY"
        return "TEXT"

    @staticmethod
    def _temporal_direction(q: str) -> str:
        if any(t in q for t in ("next", "then", "after")): return "future"
        if "before" in q: return "past"
        return "center"

    @staticmethod
    def _extract_terms(q: str) -> tuple[str, ...]:
        stop = {"how","many","what","is","are","the","a","an","there","in","on","of","and","does","do","this","that"}
        toks = re.findall(r"[A-Za-z][A-Za-z'-]*", q)
        return tuple(dict.fromkeys(t for t in toks if t not in stop and len(t) >= 2))



def analyze_question(question: str, classifier: Optional[QuestionClassifier] = None,
                     semantic_parser: Optional[Callable[[str], Any]] = None) -> QuestionAnalysis:
    if classifier is not None:
        return classifier.classify(question)
    return QuestionClassifier(semantic_parser=semantic_parser).classify(question)


def is_motion_question(question: str) -> bool:
    return analyze_question(question).needs_temporal


def infer_temporal_window(question: str, analysis: Optional[QuestionAnalysis] = None) -> Tuple[int, int]:
    a = analysis or analyze_question(question)
    if a.primary_type in {"COUNTING", "ATTRIBUTE", "OBJECT", "SPATIAL"} and not a.needs_temporal:
        return 1, 5
    if a.primary_type == "TEMPORAL" or a.temporal_direction != "center":
        return qa_config.TEMPORAL_REASONING_WINDOW_SIZE, qa_config.TEMPORAL_REASONING_STEP
    if a.primary_type in {"ACTION", "RELATIONSHIP"} or a.needs_temporal:
        return qa_config.MOTION_TEMPORAL_WINDOW_SIZE, qa_config.MOTION_FRAME_STEP
    return qa_config.TEMPORAL_WINDOW_SIZE, qa_config.FRAME_STEP
