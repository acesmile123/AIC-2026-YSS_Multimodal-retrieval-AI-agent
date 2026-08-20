from __future__ import annotations

"""Format structured evidence (grounding, OCR, spatial relations, counts,
compound question types) into a short text block that gets injected into the
VLM prompt as supporting context.

This is the piece that makes the QA branch behave like:

    KIS -> question-aware candidate selection -> visual evidence extraction ->
    specialized reasoning -> multi-frame VLM reasoning

instead of:

    KIS -> retrieve caption -> copy caption -> VLM guess

qa_evidence.EvidenceBuilder already computes rich structured evidence
(objects, spatial relations, temporal events, OCR, counts) - this module is
the missing link that surfaces it to the VLM instead of letting it sit
unused in metadata. The VLM is still told to verify everything against the
pixels; this evidence is a hint, not a substitute for looking at the image
(prevents "answer leakage" from stale/wrong grounding).
"""

from typing import Optional, Sequence

from .qa_question_analysis import QuestionAnalysis

MAX_EVIDENCE_CHARS = 600


def build_evidence_context(
    analysis: QuestionAnalysis,
    scene: Optional[dict] = None,
    ocr_text: str = "",
    count_summary: Optional[dict] = None,
    count_value: Optional[int] = None,
) -> str:
    """Build a compact, bulleted evidence block, or "" if there's nothing
    useful to say (a near-empty scene should not add prompt noise)."""
    scene = scene or {}
    lines: list[str] = []

    # Explicitly separate the asked-about target from a disambiguating
    # reference object. This prevents a highly salient reference attribute
    # (e.g. a red fire truck) from becoming the answer to a color question
    # about a different target (e.g. a roadside pole).
    target = getattr(analysis, "target", "") or getattr(analysis, "count_target", "")
    refs = getattr(analysis, "reference_entities", ()) or ()
    if target:
        lines.append(f"QUESTION TARGET: {target}")
    if refs:
        lines.append("QUESTION REFERENCES (for disambiguation only): " + str(list(refs)[:4]))
    if getattr(analysis, "expected_answer_type", "") == "COLOR":
        lines.append("COLOR RULE: answer the color of QUESTION TARGET, never the color of a reference object")

    dominant = scene.get("dominant_objects") or []
    if dominant:
        lines.append("Detected objects (BTC metadata; verify visually): " + ", ".join(dominant[:8]))

    grounded = scene.get("grounded_detections") or []
    if grounded:
        compact = "; ".join(
            f"{d.get('label')}@{float(d.get('score', 0.0)):.2f}"
            for d in grounded[:8] if isinstance(d, dict)
        )
        lines.append("BTC object grounding: " + compact)

    # Never inject a global count when the semantic query contains a spatial
    # or other filtering relation. For direct count questions, or count+attribute
    # questions without a relation filter, the specialized estimate is useful.
    semantic_relations = tuple(analysis.relations or analysis.semantic_query.get("relations", []))
    has_filter_relation = bool(semantic_relations) or any(t in {"SPATIAL", "RELATIONSHIP", "COMPARISON"} for t in analysis.types)
    safe_count = analysis.expected_answer_type == "COUNT" and not has_filter_relation
    if safe_count and count_value is not None:
        lines.append(f"Automated count estimate (verify visually, may be wrong): {count_value}")
    elif safe_count and count_summary:
        raw_total = sum(int(v) for v in count_summary.values() if isinstance(v, (int, float)))
        if raw_total:
            lines.append(f"Object detector saw ~{raw_total} object(s) across frames (raw, not deduplicated).")

    qualified = scene.get("qualified_count") if isinstance(scene, dict) else None
    if analysis.expected_answer_type == "COUNT" and qualified and qualified.get("confidence", 0.0) >= 0.55:
        lines.append(
            "Query-filtered count evidence (NOT final truth; verify visually): "
            f"{qualified.get('count')} {qualified.get('target')} objects; "
            f"confidence={float(qualified.get('confidence', 0.0)):.2f}; "
            f"source={qualified.get('source')}"
        )
    env = scene.get("environment")
    if env and env != "unknown":
        lines.append(f"Scene/environment: {env}")

    relations = scene.get("spatial_relations") or []
    if relations:
        rel_txt = "; ".join(
            f"{r.get('subject')} {r.get('relation')} {r.get('object')}"
            for r in relations[:6]
        )
        lines.append(f"Spatial relations detected: {rel_txt}")

    events = scene.get("temporal_events") or []
    if events:
        lines.append("Temporal sequence: " + " -> ".join(str(e) for e in events[:6]))

    if ocr_text:
        lines.append(f"Text/OCR detected in frame: {ocr_text[:200]!r}")

    if analysis.semantic_query:
        sq = analysis.semantic_query
        # Keep structured intent compact; never let it become a second answer source.
        op = sq.get("operation") or analysis.operation
        target = sq.get("target") or analysis.target
        rels = sq.get("relations") or []
        attrs = sq.get("attributes") or []
        if target or rels or attrs:
            parts = []
            if op: parts.append(f"operation={op}")
            if target: parts.append(f"target={target}")
            if rels: parts.append(f"relations={rels[:4]}")
            if attrs: parts.append(f"attributes={attrs[:4]}")
            lines.append("Semantic question structure (interpretation, verify visually): " + "; ".join(parts))

    count_constraints = getattr(analysis, "count_constraints", ())
    if count_constraints:
        lines.append("Hard counting constraints: " + str(list(count_constraints)[:6]))

    refs = getattr(analysis, "reference_entities", ())
    if refs:
        lines.append("Reference entities: " + str(list(refs)[:4]))
    temporal_constraints = getattr(analysis, "temporal_constraints", ())
    if temporal_constraints:
        lines.append("Temporal constraints: " + str(list(temporal_constraints)[:4]))
    subqueries = getattr(analysis, "subqueries", ())
    if subqueries:
        lines.append("Sub-query decomposition: " + str(list(subqueries)[:3]))
    if getattr(analysis, "query_complexity", "simple") in {"compound", "hard"}:
        lines.append("This is a constrained/compound query. Do not answer from generic scene counts; satisfy every constraint.")

    if getattr(analysis, "execution_plan", None):
        steps = "; ".join(str(step.get("step")) for step in analysis.execution_plan if isinstance(step, dict))
        if steps and steps != "retrieve_and_reason":
            lines.append("Execution plan: " + steps)

    if analysis.is_multi_hop and len(analysis.types) >= 2:
        lines.append(
            "This question has multiple parts (" + " + ".join(analysis.types)
            + "). Answer ALL parts, combined concisely as the question expects."
        )

    if not lines:
        return ""

    text = "\n".join(f"- {line}" for line in lines)
    if len(text) > MAX_EVIDENCE_CHARS:
        text = text[:MAX_EVIDENCE_CHARS].rsplit("\n", 1)[0]
    return text
