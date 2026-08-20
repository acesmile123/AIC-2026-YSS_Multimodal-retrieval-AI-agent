from qa.vlm_engine import QwenVLEngine


def test_build_prompt_allows_json_evidence_braces():
    engine = QwenVLEngine.__new__(QwenVLEngine)
    prompt = engine._build_prompt(
        1,
        "Looking at the TV screen area in the television studio, how many people are standing in front of it?",
        '{"objects": {"person": 2}, "text": "TV"}',
    )
    assert '"objects"' in prompt
    assert '"person": 2' in prompt
    assert 'Question:' in prompt
