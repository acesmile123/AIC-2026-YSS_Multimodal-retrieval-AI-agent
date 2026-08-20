"""AIC 2026 Q&A branch.

Core flow:
    Dùng lại retrieval của module KIS để định vị frame; tích hợp VQA để
    sinh answer từ khung hình.

Public entry points:
    QASystem
    QARequest
    create_exam_request
    QAResult / QACandidate / QAAnswer
    format_submission

The package is intentionally decoupled from the shared KIS implementation:
QASystem accepts a KIS adapter and a VLM answerer, so each component can be
reused or replaced independently for testing, inference, or competition runs.
"""
from .pipeline import QASystem
from .qa_output import format_submission, validate_submission_rows
from .qa_types import QAAnswer, QACandidate, QARequest, QAResult, build_structured_query
from .qa_question_analysis import QuestionAnalysis, QuestionClassifier, analyze_question
from .qa_grounding import Detection, GroundingEngine
from .qa_counting import CountingEngine, CountResult
from .qa_evidence import EvidenceBuilder, EvidenceBundle
from .qa_evidence_memory import EvidenceMemory, EvidenceRecord
from .qa_spatial import SpatialReasoner, Relation
from .qa_temporal import TemporalReasoner, TemporalState
from .qa_visual_reranker import VisualReranker
from .qa_frame_selection import AdaptiveFrameSelector
from .qa_frame_dedup import FrameDeduplicator
from .qa_attributes import AttributeExtractor

__all__ = [
    "QASystem",
    "QAAnswer",
    "QACandidate",
    "QARequest",
    "QAResult",
    "build_structured_query",
    "format_submission",
    "validate_submission_rows",
    "QuestionAnalysis", "QuestionClassifier", "analyze_question",
    "Detection", "GroundingEngine", "CountingEngine", "CountResult",
    "EvidenceBuilder", "EvidenceBundle", "EvidenceMemory", "EvidenceRecord",
    "SpatialReasoner", "Relation", "TemporalReasoner", "TemporalState",
    "VisualReranker", "AdaptiveFrameSelector", "FrameDeduplicator", "AttributeExtractor",
]
