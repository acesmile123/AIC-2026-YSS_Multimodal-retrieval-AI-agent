"""20-case query-language regression: 10 English + 10 Vietnamese.

The live translation library is intentionally tested separately because it
needs network access. This file verifies that both language paths enter the
same English-only planner contract when a translator adapter is injected.
"""
from __future__ import annotations

from qa.qa_query_planner import plan_question

EN = [
    "How many doctors are visible in the operating room?",
    "What color is the roadside pole in the frame with the red fire truck?",
    "Which object is closest to the red suitcase?",
    "What is the surgeon holding?",
    "Where is the warning sign placed?",
    "Which person is speaking to the camera?",
    "What happens immediately after the woman sits down?",
    "Is there a bicycle near the entrance?",
    "What does the sign say on the wall?",
    "Which vehicle is behind the white van?",
]

VI = [
    "Trong phòng phẫu thuật có bao nhiêu bác sĩ?",
    "Cột bên đường trong khung hình có chiếc xe cứu hỏa màu đỏ có màu gì?",
    "Vật nào gần chiếc vali màu đỏ nhất?",
    "Bác sĩ phẫu thuật đang cầm gì?",
    "Biển cảnh báo nằm ở đâu?",
    "Người nào đang nói với máy quay?",
    "Điều gì xảy ra ngay sau khi người phụ nữ ngồi xuống?",
    "Có xe đạp gần lối vào không?",
    "Biển trên tường ghi gì?",
    "Xe nào ở phía sau chiếc xe tải màu trắng?",
]


def test_english_open_ended_planner():
    for q in EN:
        p = plan_question(q)
        assert p["normalized"]
        assert isinstance(p["target_terms"], tuple)


def test_target_reference_binding():
    p = plan_question(EN[1])
    assert p["target"] == "roadside pole"
    assert p["references"]
    assert p["references"][0]["text"] == "red fire truck"


def test_temporal_does_not_invent_target():
    p = plan_question(EN[6])
    assert p["target"] == ""
    assert p["operation"] == "temporal_reasoning"


def test_vietnamese_cases_have_safe_local_fallback_signals():
    # Agent Core remains the primary bilingual semantic layer, but the local
    # planner must preserve high-signal COUNT/LOCATION intent if Agent Core is unavailable.
    for q in VI:
        p = plan_question(q)
        assert p["normalized"]
    assert plan_question(VI[0])["expected_answer_type"] == "COUNT"
    assert plan_question(VI[4])["expected_answer_type"] == "LOCATION"


class _BilingualRouter:
    def route_query(self, raw_text, query_id=None):
        from types import SimpleNamespace
        english = {
            "Trong phòng phẫu thuật có bao nhiêu bác sĩ?": "How many doctors are visible in the operating room?",
            "Cột bên đường trong khung hình có chiếc xe cứu hỏa màu đỏ có màu gì?": "What color is the roadside pole in the frame with the red fire truck?",
        }.get(raw_text, raw_text)
        data = {
            "query_id": query_id or "q_test",
            "raw_query": raw_text,
            "query_variants": [english, raw_text],
            "visual_description": english,
            "entities": [],
            "relations": [],
            "attributes": [],
            "needs_ocr": False,
            "needs_asr": False,
            "question": english,
            "answer_type": "count" if "bao nhiêu" in raw_text.lower() else "color",
        }
        return SimpleNamespace(value="QA"), SimpleNamespace(model_dump=lambda: data)


def test_vietnamese_inputs_are_normalized_by_agent_core_before_kis():
    from qa.agent_core_adapter import AgentCoreAdapter
    adapter = AgentCoreAdapter(router=_BilingualRouter())
    for q in VI[:2]:
        _, structured = adapter.route(q)
        data = adapter.to_dict(structured)
        english = adapter.english_variant(data, q)
        assert all(ord(ch) < 128 for ch in english)
        assert english != q


def test_fifty_fifty_language_contract():
    assert len(EN) == len(VI) == 10
    assert sum(1 for _ in EN) == sum(1 for _ in VI)
