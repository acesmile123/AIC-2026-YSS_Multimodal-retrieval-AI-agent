from __future__ import annotations

from types import SimpleNamespace

from qa.agent_core_adapter import AgentCoreAdapter
from qa.qa_question_analysis import QuestionAnalysis
from qa.pipeline import QASystem


class FakeRouter:
    def route_query(self, raw_text, query_id=None):
        q = raw_text.lower()
        is_question = any(x in q for x in ("how many", "what", "which", "is there", "does"))
        if is_question:
            data = SimpleNamespace(
                model_dump=lambda: {
                    "query_id": query_id or "q_test",
                    "raw_query": raw_text,
                    "query_variants": ["How many people are visible", raw_text],
                    "visual_description": "people visible in the scene",
                    "entities": [{"type": "person", "value": "people", "attribute": None}],
                    "needs_ocr": False,
                    "needs_asr": False,
                    "question": raw_text,
                    "answer_type": "count",
                }
            )
            return SimpleNamespace(value="QA"), data
        data = SimpleNamespace(
            model_dump=lambda: {
                "query_id": query_id or "q_test",
                "raw_query": raw_text,
                "query_variants": ["people on a stage", raw_text],
                "visual_description": "people on a stage",
                "entities": [{"type": "person", "value": "people", "attribute": None}],
                "needs_ocr": False,
                "needs_asr": False,
            }
        )
        return SimpleNamespace(value="KIS"), data


def test_agent_core_event_query_exports_english_only_for_kis():
    adapter = AgentCoreAdapter(router=FakeRouter())
    raw, query = adapter.build_event_query("Một nhóm người trên sân khấu")
    assert raw == "people on a stage"
    assert query["raw_query"] == "people on a stage"
    assert query["query_variants"] == ["people on a stage"]
    assert query["source_raw_query"] == "Một nhóm người trên sân khấu"
    assert query["agent_task_type"] == "KIS"


def test_agent_core_english_variant_is_canonical_first_signal():
    adapter = AgentCoreAdapter(router=FakeRouter())
    data = {
        "raw_query": "Có bao nhiêu người?",
        "query_variants": ["How many people are visible?", "Có bao nhiêu người?"],
    }
    assert adapter.english_variants(data, fallback="Có bao nhiêu người?") == [
        "How many people are visible?"
    ]


def test_agent_core_answer_type_drives_local_analysis():
    qa = QASystem(agent_core=AgentCoreAdapter(router=FakeRouter()))
    analysis = qa.analysis("How many people are visible in the scene?")
    assert isinstance(analysis, QuestionAnalysis)
    assert analysis.expected_answer_type == "COUNT"
    assert "COUNTING" in analysis.types
    assert analysis.analysis_source == "agent_core+planner"
