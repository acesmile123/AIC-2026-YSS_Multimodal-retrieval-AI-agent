from types import SimpleNamespace

import pytest

from aic_agent_core.config import RouterSettings
from aic_agent_core.exceptions import EmptyQueryError
from aic_agent_core.query_models import Entity, KISQuery, LLMStructuredQuery, TaskType
from aic_agent_core.router import QueryRouter, _gemini_json_schema


RAW = "Tìm một người mặc áo đỏ đang phát biểu ngoài trời"


def kis_payload(raw_query: str = RAW, query_id: str = "q_001") -> LLMStructuredQuery:
    return LLMStructuredQuery(
        query_id=query_id,
        raw_query=raw_query,
        task_type=TaskType.KIS,
        query_variants=[
            "người áo đỏ phát biểu ngoài trời",
            "person in red speaking outdoors",
        ],
        visual_description="Một người mặc áo đỏ phát biểu ngoài trời",
        entities=[Entity(type="person", value="person", attribute="màu áo: đỏ")],
        needs_ocr=False,
        needs_asr=False,
    )


class FakeModels:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def router_with(responses, attempts=2):
    settings = RouterSettings(GOOGLE_API_KEY="test-key", gemini_max_attempts=attempts)
    return QueryRouter(settings=settings, client=FakeClient(responses))


def test_route_query_returns_task_outside_payload(monkeypatch):
    router = router_with([SimpleNamespace(parsed=kis_payload(), text=None)])
    monkeypatch.setattr(router, "_generation_config", lambda: {"schema": "fake"})

    task_type, query = router.route_query(RAW, query_id="q_001")

    assert task_type is TaskType.KIS
    assert isinstance(query, KISQuery)
    assert "task_type" not in query.model_dump()


def test_router_retries_mismatched_raw_query(monkeypatch):
    bad = SimpleNamespace(parsed=kis_payload(raw_query="wrong"), text=None)
    good = SimpleNamespace(parsed=kis_payload(), text=None)
    router = router_with([bad, good])
    monkeypatch.setattr(router, "_generation_config", lambda: {})
    result = router.route(RAW, query_id="q_001")
    assert result.task_type is TaskType.KIS
    assert len(router.client.models.calls) == 2


def test_empty_query_is_rejected_before_api_call(monkeypatch):
    router = router_with([])
    monkeypatch.setattr(router, "_generation_config", lambda: {})
    with pytest.raises(EmptyQueryError):
        router.route("   ")
    assert router.client.models.calls == []


def test_gemini_schema_removes_unsupported_additional_properties():
    schema = _gemini_json_schema()

    def contains_key(value, target):
        if isinstance(value, dict):
            return target in value or any(contains_key(child, target) for child in value.values())
        if isinstance(value, list):
            return any(contains_key(child, target) for child in value)
        return False

    assert not contains_key(schema, "additionalProperties")
    assert "task_type" in schema["properties"]
    assert "kis" not in schema["properties"]
    assert "qa" not in schema["properties"]
    assert "trake" not in schema["properties"]


def test_generation_config_uses_json_schema_not_typed_schema():
    config = router_with([])._generation_config()
    assert config.response_schema is None
    assert config.response_json_schema is not None
