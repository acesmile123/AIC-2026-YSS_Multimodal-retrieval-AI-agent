from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class QARequest:
    """Exam-facing request: event description + question."""
    event_description: str
    question: str

    def validate(self) -> None:
        if not str(self.event_description or "").strip():
            raise ValueError("event_description must be non-empty")
        if not str(self.question or "").strip():
            raise ValueError("question must be non-empty")

    def to_structured_query(self) -> dict:
        return build_structured_query(self.event_description)


def build_structured_query(event_description: str, query_variants: Optional[List[str]] = None,
                           entities: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Convert the competition's natural-language event description into the
    internal KIS structured-query contract.

    This deliberately keeps the conversion conservative: the event description
    is always the primary query, while extra variants/entities are optional and
    may be supplied by an upstream parser later.
    """
    raw = str(event_description or "").strip()
    variants = [str(v).strip() for v in (query_variants or []) if str(v).strip()]
    if raw and raw not in variants:
        variants.insert(0, raw)
    return {
        "raw_query": raw,
        "query_variants": variants or [raw],
        "entities": list(entities or []),
    }


@dataclass
class QACandidate:
    video_id: str
    frame_id: int
    retrieval_score: float


@dataclass
class QAAnswer:
    video_id: str
    frame_id: int
    answer: str
    score: float
    vlm_confidence: float = 1.0
    error: Optional[str] = None
    visual_relevance_score: float = 0.0
    validation_score: float = 0.0
    validation_reason: str = ""
    answer_type: str = "TEXT"
    evidence_summary: dict = field(default_factory=dict)
    ranking_score: float = 0.0
    evidence_score: float = 0.0
    evidence_notes: str = ""

    @property
    def is_unknown(self) -> bool:
        return not self.answer or self.answer.strip().upper() == "UNKNOWN"

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "answer": self.answer,
            "score": self.score,
            "vlm_confidence": self.vlm_confidence,
            "error": self.error,
            "visual_relevance_score": self.visual_relevance_score,
            "validation_score": self.validation_score,
            "validation_reason": self.validation_reason,
            "answer_type": self.answer_type,
            "evidence_summary": self.evidence_summary,
            "ranking_score": self.ranking_score,
            "evidence_score": self.evidence_score,
            "evidence_notes": self.evidence_notes,
        }

    def to_qa_dict(self) -> dict:
        """Internal Q&A branch contract: video + frame + answer + ranking score."""
        return {
            "video_id": self.video_id,
            "frame_id": int(self.frame_id),
            "answer": self.answer,
            "score": float(self.ranking_score if self.ranking_score else self.score),
        }

    def to_submission_dict(self) -> dict:
        """Competition submission contract: video + frame + answer only."""
        return {
            "video_id": self.video_id,
            "frame_id": int(self.frame_id),
            "answer": self.answer,
        }

    def to_public_dict(self) -> dict:
        # Backward-compatible alias for the competition-facing three-field view.
        return self.to_submission_dict()


@dataclass
class QAResult:
    question: str
    final_answer: Optional[str]
    confidence: float
    evidence: List[QAAnswer] = field(default_factory=list)
    candidates_considered: int = 0
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "candidates_considered": self.candidates_considered,
            "note": self.note,
        }
