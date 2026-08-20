from qa.qa_counting import CountingEngine
from qa.qa_grounding import Detection
from qa.qa_spatial import SpatialReasoner
from qa.qa_frame_selection import AdaptiveFrameSelector
from qa.qa_question_analysis import analyze_question


def d(label, score, x1, y1, x2, y2):
    return Detection(label, score, (y1, x1, y2, x2), label)


def test_animal_count_covers_open_domain_labels():
    engine = CountingEngine(min_score=0.2)
    result = engine.count([
        d("cow", .9, .0, .0, .2, .2),
        d("sheep", .9, .2, .0, .4, .2),
        d("bird", .9, .4, .0, .6, .2),
        d("dog", .9, .6, .0, .8, .2),
    ], "animal")
    assert result.count == 4


def test_spatial_depth_relations_are_not_hard_filtered():
    s = SpatialReasoner()
    a = d("person", .9, .2, .5, .4, .9)
    b = d("car", .9, .2, .2, .4, .5)
    assert s.filter_relation([a], [b], "in_front_of", min_confidence=.5) == []


def test_temporal_selector_keeps_start_middle_end():
    records = [{"frame_id": i, "detections": []} for i in range(9)]
    a = analyze_question("What happens after the woman sits down?")
    selected = AdaptiveFrameSelector().select("What happens after the woman sits down?", records, a, limit=3)
    assert [r["frame_id"] for r in selected] == [0, 4, 8]
