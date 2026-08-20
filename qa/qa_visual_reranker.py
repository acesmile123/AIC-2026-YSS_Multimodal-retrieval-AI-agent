from __future__ import annotations

from typing import Iterable, List, Optional

from .qa_question_analysis import QuestionAnalysis, analyze_question
from .qa_text_similarity import text_similarity, normalize_text


class VisualReranker:
    """Question-frame compatibility scorer using structured evidence first.

    It is intentionally model-agnostic. If an external visual scorer is supplied,
    its score is blended with caption/object evidence; otherwise the structured
    evidence path works without loading another model.
    """
    def __init__(self, external_scorer=None):
        self.external_scorer = external_scorer

    def score(self, question: str, caption: str = "", scene: Optional[dict] = None,
              analysis: Optional[QuestionAnalysis] = None, objects: Optional[dict] = None) -> float:
        analysis = analysis or analyze_question(question)
        q = normalize_text(question)
        c = normalize_text(caption)
        caption_score = text_similarity(q, c) if c else 0.0
        object_score = 0.0
        scene = scene or {}
        if analysis.target_terms:
            text = " ".join(scene.get("objects",{}).keys()) + " " + c
            hits = sum(1 for t in analysis.target_terms if t in text)
            object_score = hits / max(1, len(analysis.target_terms))
        type_bonus = 0.0
        # Câu hỏi ghép nên nhận thưởng cho MỌI sub-type đã phát hiện (không
        # chỉ primary_type) - vd. "closest vehicle, what color?" cần cả
        # bonus SPATIAL/COMPARISON lẫn ATTRIBUTE để không thiên vị một nửa
        # câu hỏi.
        active_types = set(analysis.types) | {analysis.primary_type}
        if active_types & {"COUNTING"} and scene.get("objects"): type_bonus += 0.25
        if active_types & {"ATTRIBUTE"} and scene.get("objects"): type_bonus += 0.15
        if active_types & {"ACTION", "TEMPORAL"} and analysis.needs_temporal: type_bonus += 0.10
        if active_types & {"SPATIAL", "RELATIONSHIP", "COMPARISON"} and scene.get("spatial_relations"): type_bonus += 0.10
        if active_types & {"OCR"} and scene.get("ocr_text"): type_bonus += 0.10
        type_bonus = min(type_bonus, 0.35)
        base = min(1.0, 0.55*caption_score + 0.30*object_score + type_bonus)
        if self.external_scorer is not None:
            try:
                ext = float(self.external_scorer.score(question, caption, scene))
                base = 0.7*base + 0.3*max(0.0, min(1.0, ext))
            except Exception:
                pass
        return round(max(0.0, min(1.0, base)), 6)

    def score_metadata(self, question: str, metadata: Optional[dict]) -> float:
        if not metadata:
            return 0.0
        analysis=analyze_question(question)
        labels=[str(x).lower() for x in metadata.get("detection_class_entities", [])]
        if not labels:
            return 0.0
        joined=" ".join(labels)
        score=0.0
        aliases={"cow":["cattle","bull","cow"],"cows":["cattle","bull","cow"],"animal":["animal","cattle","bull","horse","dog","cat"],"animals":["animal","cattle","bull","horse","dog","cat"],"people":["person","man","woman","boy","girl"],"person":["person","man","woman","boy","girl"]}
        for term in analysis.target_terms:
            opts=aliases.get(term,[term])
            if any(o in joined for o in opts):
                score += 1.0
        if analysis.expected_answer_type == "COUNT" and any(x in joined for x in ("cow","cattle","person","vehicle","animal","horse","dog","cat")):
            score += 0.5
        return min(1.0, score/max(1,len(analysis.target_terms)+0.5))

    def rerank(self, question: str, candidates: Iterable[dict]) -> List[dict]:
        out=[]
        analysis = analyze_question(question)
        for c in candidates:
            x=dict(c)
            x["visual_relevance_score"] = self.score(question, x.get("caption",""), x.get("scene"), analysis)
            x["final_retrieval_score"] = 0.6*float(x.get("score",0.0)) + 0.4*x["visual_relevance_score"]
            out.append(x)
        return sorted(out, key=lambda r:r["final_retrieval_score"], reverse=True)
