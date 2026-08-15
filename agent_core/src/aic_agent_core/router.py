from __future__ import annotations

from uuid import uuid4
from typing import Any

from pydantic import ValidationError

from .config import RouterSettings
from .exceptions import (
    EmptyQueryError,
    RouterConfigurationError,
    StructuredOutputError,
)
from .query_models import LLMStructuredQuery, RouteResult, StructuredQuery, TaskType
from .prompts import SYSTEM_PROMPT, build_user_prompt


class QueryRouter:
    """Gemini-backed semantic parser and AIC task router.

    ``client`` is injectable to make the core deterministic in unit tests.
    """

    def __init__(self, settings: RouterSettings | None = None, client: Any | None = None):
        self.settings = settings or RouterSettings()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.settings.google_api_key:
                raise RouterConfigurationError(
                    "GOOGLE_API_KEY is missing. Put it in the environment or .env file."
                )
            try:
                from google import genai
            except ImportError as exc:
                raise RouterConfigurationError(
                    "google-genai is not installed; run `pip install -e .`"
                ) from exc
            self._client = genai.Client(api_key=self.settings.google_api_key)
        return self._client

    def route(self, raw_text: str, query_id: str | None = None) -> RouteResult:
        text = _validate_raw_text(raw_text)
        resolved_query_id = _resolve_query_id(query_id)
        feedback: str | None = None
        last_error: Exception | None = None

        for _ in range(self.settings.gemini_max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=build_user_prompt(resolved_query_id, text, feedback),
                    config=self._generation_config(),
                )
                llm_query = self._parse_response(response)
                query = llm_query.to_public_query()
                self._validate_against_input(query, resolved_query_id, text)
                return RouteResult(task_type=llm_query.task_type, structured_query=query)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                feedback = _compact_error(exc)

        raise StructuredOutputError(
            f"Gemini returned invalid structured output after "
            f"{self.settings.gemini_max_attempts} attempt(s): {last_error}"
        ) from last_error

    async def aroute(self, raw_text: str, query_id: str | None = None) -> RouteResult:
        text = _validate_raw_text(raw_text)
        resolved_query_id = _resolve_query_id(query_id)
        feedback: str | None = None
        last_error: Exception | None = None

        for _ in range(self.settings.gemini_max_attempts):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=build_user_prompt(resolved_query_id, text, feedback),
                    config=self._generation_config(),
                )
                llm_query = self._parse_response(response)
                query = llm_query.to_public_query()
                self._validate_against_input(query, resolved_query_id, text)
                return RouteResult(task_type=llm_query.task_type, structured_query=query)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                feedback = _compact_error(exc)

        raise StructuredOutputError(
            f"Gemini returned invalid structured output after "
            f"{self.settings.gemini_max_attempts} attempt(s): {last_error}"
        ) from last_error

    def route_query(
        self, raw_text: str, query_id: str | None = None
    ) -> tuple[TaskType, StructuredQuery]:
        result = self.route(raw_text, query_id)
        return result.task_type, result.structured_query

    async def aroute_query(
        self, raw_text: str, query_id: str | None = None
    ) -> tuple[TaskType, StructuredQuery]:
        result = await self.aroute(raw_text, query_id)
        return result.task_type, result.structured_query

    def _generation_config(self) -> Any:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RouterConfigurationError(
                "google-genai is not installed; run `pip install -e .`"
            ) from exc
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=self.settings.gemini_temperature,
            max_output_tokens=self.settings.gemini_max_output_tokens,
            response_mime_type="application/json",
            # Pass JSON Schema directly instead of ``response_schema=StructuredQuery``.
            # The SDK's typed-schema conversion serializes Pydantic's
            # ``additionalProperties`` as ``additional_properties``, which the
            # Gemini generateContent endpoint rejects with HTTP 400.
            response_json_schema=_gemini_json_schema(),
        )

    @staticmethod
    def _parse_response(response: Any) -> LLMStructuredQuery:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, LLMStructuredQuery):
            return parsed
        if parsed is not None:
            return LLMStructuredQuery.model_validate(parsed)
        response_text = getattr(response, "text", None)
        if not response_text:
            raise ValueError("Gemini response contains neither parsed data nor text")
        return LLMStructuredQuery.model_validate_json(response_text)

    @staticmethod
    def _validate_against_input(
        query: StructuredQuery, query_id: str, raw_text: str
    ) -> None:
        if query.query_id != query_id:
            raise ValueError("query_id must exactly match the supplied query_id")
        if query.raw_query != raw_text:
            raise ValueError("raw_query must exactly match the raw input")


_default_router: QueryRouter | None = None


def _get_default_router() -> QueryRouter:
    global _default_router
    if _default_router is None:
        _default_router = QueryRouter()
    return _default_router


def route_query(
    raw_text: str, query_id: str | None = None
) -> tuple[TaskType, StructuredQuery]:
    """Route one raw query using a lazily initialized default Gemini client."""

    return _get_default_router().route_query(raw_text, query_id)


async def aroute_query(
    raw_text: str, query_id: str | None = None
) -> tuple[TaskType, StructuredQuery]:
    """Async counterpart of :func:`route_query`."""

    return await _get_default_router().aroute_query(raw_text, query_id)


def _validate_raw_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    text = raw_text.strip()
    if not text:
        raise EmptyQueryError("raw_text must not be empty")
    return text


def _resolve_query_id(query_id: str | None) -> str:
    if query_id is None:
        return f"q_{uuid4().hex[:12]}"
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError("query_id must be a non-empty string")
    return query_id.strip()


def _compact_error(error: Exception, limit: int = 1800) -> str:
    value = str(error).replace("\x00", "")
    return value[:limit]


def _gemini_json_schema() -> dict[str, Any]:
    """Return Gemini-compatible JSON Schema without weakening local validation.

    ``extra='forbid'`` remains active in Pydantic when validating Gemini's
    response. Only the unsupported schema keyword is removed from the API
    request payload.
    """

    schema = LLMStructuredQuery.model_json_schema()
    return _remove_schema_key(schema, "additionalProperties")


def _remove_schema_key(value: Any, key_to_remove: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_schema_key(child, key_to_remove)
            for key, child in value.items()
            if key != key_to_remove
        }
    if isinstance(value, list):
        return [_remove_schema_key(child, key_to_remove) for child in value]
    return value
