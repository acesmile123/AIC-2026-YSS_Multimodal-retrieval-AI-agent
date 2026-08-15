import pytest
from pydantic import ValidationError

from aic_agent_core.query_models import (
    AnswerType,
    KISQuery,
    LLMStructuredQuery,
    QAQuery,
    TRAKEQuery,
    TaskType,
    TemporalEvent,
)


def common_data() -> dict:
    return {
        "query_id": "q_001",
        "raw_query": "Một người chạy rồi nhảy",
        "query_variants": ["một người chạy rồi nhảy", "a person runs then jumps"],
        "visual_description": "Một người chạy rồi thực hiện cú nhảy",
        "entities": [],
        "needs_ocr": False,
        "needs_asr": False,
    }


def event(index: int) -> TemporalEvent:
    return TemporalEvent(
        index=index,
        description=f"Sự kiện {index}",
        semantic_keyframe=f"Thời điểm bắt đầu sự kiện {index}",
        query_variants=[f"sự kiện {index}"],
        entities=[],
    )


def test_kis_uses_requested_form():
    query = KISQuery(**common_data())
    assert query.question is None
    assert query.events == []
    assert query.temporal_constraints == []
    assert "task_type" not in query.model_dump()


def test_qa_adds_question_and_answer_type():
    query = QAQuery(
        **common_data(),
        question="Có bao nhiêu người?",
        answer_type=AnswerType.COUNT,
    )
    assert query.answer_type is AnswerType.COUNT


def test_flat_llm_model_converts_to_public_kis_without_task_type():
    llm_query = LLMStructuredQuery(**common_data(), task_type=TaskType.KIS)
    query = llm_query.to_public_query()
    assert isinstance(query, KISQuery)
    assert "task_type" not in query.model_dump()


def test_trake_rejects_non_consecutive_events():
    with pytest.raises(ValidationError, match="consecutive"):
        TRAKEQuery(
            **common_data(),
            events=[event(1), event(3)],
            temporal_constraints=["event 1 before event 3"],
        )
