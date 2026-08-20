from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


QA_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = QA_ROOT.parent
AGENT_CORE_SRC = PROJECT_ROOT / "agent_core" / "src"
if AGENT_CORE_SRC.exists() and str(AGENT_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_CORE_SRC))


class AgentCoreUnavailable(RuntimeError):
    """Agent Core cannot be imported or did not return a usable query."""


class AgentCoreConfigurationError(AgentCoreUnavailable):
    """Agent Core is installed but runtime configuration/credentials are missing."""


class AgentCoreAdapter:
    """Single integration point between QA and the shared AIC Agent Core.

    Agent Core owns language understanding, Vietnamese->English query variants,
    entity extraction, task/answer-type detection, and semantic decomposition.
    QA only converts that public schema into its internal QuestionAnalysis shape.
    """

    def __init__(self, router: Any | None = None):
        self._router = router

    @property
    def router(self):
        if self._router is None:
            try:
                from aic_agent_core import QueryRouter
            except Exception as exc:  # pragma: no cover - environment dependent
                raise AgentCoreUnavailable(
                    "aic-agent-core is unavailable. Install agent_core or provide an injected router."
                ) from exc
            self._router = QueryRouter()
        return self._router

    def route(self, text: str, query_id: str | None = None):
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("text must be non-empty")
        try:
            task_type, structured = self.router.route_query(raw, query_id=query_id)
            return task_type, structured
        except Exception as exc:
            name = type(exc).__name__
            if name == "RouterConfigurationError":
                raise AgentCoreConfigurationError(str(exc)) from exc
            raise

    @staticmethod
    def to_dict(structured: Any) -> dict[str, Any]:
        if hasattr(structured, "model_dump"):
            return structured.model_dump()
        if isinstance(structured, dict):
            return dict(structured)
        raise TypeError(f"Unsupported Agent Core structured query: {type(structured)!r}")

    def build_event_query(self, event_description: str) -> tuple[str, dict[str, Any]]:
        task_type, structured = self.route(event_description)
        data = self.to_dict(structured)
        variants = self.english_variants(data, fallback=event_description)
        if not variants:
            raise AgentCoreUnavailable("Agent Core did not return an English retrieval query")
        english_query = variants[0]
        raw_query = str(data.get("raw_query") or event_description).strip()
        return english_query, {
            "raw_query": english_query,
            "source_raw_query": raw_query,
            "query_variants": variants,
            "entities": list(data.get("entities") or []),
            "relations": list(data.get("relations") or []),
            "attributes": list(data.get("attributes") or []),
            "needs_ocr": bool(data.get("needs_ocr", False)),
            "needs_asr": bool(data.get("needs_asr", False)),
            "agent_task_type": str(getattr(task_type, "value", task_type)),
        }

    def analyze_question(self, question: str):
        _, structured = self.route(question)
        data = self.to_dict(structured)
        return data

    @staticmethod
    def english_variants(data: dict[str, Any], fallback: str) -> list[str]:
        variants = [str(x).strip() for x in data.get("query_variants", []) if str(x).strip()]
        if variants:
            # Agent Core contract: English canonical retrieval query is first.
            # Keep only English retrieval signals downstream; never mix VN into KIS.
            english = variants[0]
            return [english]
        raw = str(data.get("raw_query") or "").strip()
        if raw and all(ord(ch) < 128 for ch in raw):
            return [raw]
        fallback = str(fallback or "").strip()
        if fallback and all(ord(ch) < 128 for ch in fallback):
            return [fallback]
        return []

    @staticmethod
    def english_variant(data: dict[str, Any], fallback: str) -> str:
        variants = AgentCoreAdapter.english_variants(data, fallback)
        if not variants:
            raise AgentCoreUnavailable("Agent Core did not return an English query variant")
        return variants[0]
