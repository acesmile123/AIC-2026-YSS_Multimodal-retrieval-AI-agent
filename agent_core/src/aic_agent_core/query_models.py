from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TaskType(StrEnum):
    KIS = "KIS"
    QA = "QA"
    TRAKE = "TRAKE"


class AnswerType(StrEnum):
    COUNT = "count"
    COLOR = "color"
    TEXT = "text"
    ENTITY = "entity"
    BOOLEAN = "boolean"
    LOCATION = "location"
    TIME = "time"
    ACTION = "action"
    OTHER = "other"


class Entity(StrictModel):
    type: str = Field(min_length=1, description="person, object, action, scene, or text")
    value: str = Field(min_length=1, description="Normalized searchable value")
    attribute: str | None = Field(default=None, description="Color, quantity, clothing, or relation")


class TemporalEvent(StrictModel):
    index: int = Field(ge=1)
    description: str = Field(min_length=3, description="Standalone resolved event")
    semantic_keyframe: str = Field(min_length=3, description="Exact instant to align")
    query_variants: list[str] = Field(min_length=1, max_length=4)
    entities: list[Entity] = Field(default_factory=list)


class CommonQuery(StrictModel):
    query_id: str = Field(min_length=1)
    raw_query: str = Field(min_length=1)
    query_variants: list[str] = Field(min_length=2, max_length=8)
    visual_description: str = Field(min_length=3)
    entities: list[Entity] = Field(default_factory=list)
    needs_ocr: bool
    needs_asr: bool


class KISQuery(CommonQuery):
    question: None = None
    events: list[TemporalEvent] = Field(default_factory=list, max_length=0)
    temporal_constraints: list[str] = Field(default_factory=list, max_length=0)


class QAQuery(CommonQuery):
    question: str = Field(min_length=2)
    answer_type: AnswerType
    events: list[TemporalEvent] = Field(default_factory=list, max_length=0)
    temporal_constraints: list[str] = Field(default_factory=list, max_length=0)


class TRAKEQuery(CommonQuery):
    question: None = None
    events: list[TemporalEvent] = Field(min_length=2, max_length=20)
    temporal_constraints: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_event_order(self) -> "TRAKEQuery":
        indexes = [event.index for event in self.events]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("TRAKE event indexes must be consecutive and chronological")
        return self


StructuredQuery: TypeAlias = KISQuery | QAQuery | TRAKEQuery


class LLMStructuredQuery(CommonQuery):
    """Flat schema sent to Gemini, then converted to a public query class."""

    task_type: TaskType
    question: str | None = None
    answer_type: AnswerType | None = None
    events: list[TemporalEvent] = Field(default_factory=list, max_length=20)
    temporal_constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_fields(self) -> "LLMStructuredQuery":
        if self.task_type is TaskType.KIS:
            if self.question is not None or self.answer_type is not None:
                raise ValueError("KIS requires question=null and answer_type=null")
            if self.events or self.temporal_constraints:
                raise ValueError("KIS requires events=[] and temporal_constraints=[]")
        elif self.task_type is TaskType.QA:
            if not self.question or self.answer_type is None:
                raise ValueError("QA requires question and answer_type")
            if self.events or self.temporal_constraints:
                raise ValueError("QA requires events=[] and temporal_constraints=[]")
        else:
            if self.question is not None or self.answer_type is not None:
                raise ValueError("TRAKE requires question=null and answer_type=null")
            if len(self.events) < 2 or not self.temporal_constraints:
                raise ValueError("TRAKE requires ordered events and temporal_constraints")
            indexes = [event.index for event in self.events]
            if indexes != list(range(1, len(indexes) + 1)):
                raise ValueError("TRAKE event indexes must be consecutive and chronological")
        return self

    def to_public_query(self) -> StructuredQuery:
        common = self.model_dump(
            exclude={"task_type", "question", "answer_type", "events", "temporal_constraints"}
        )
        if self.task_type is TaskType.KIS:
            return KISQuery(**common)
        if self.task_type is TaskType.QA:
            assert self.question is not None and self.answer_type is not None
            return QAQuery(**common, question=self.question, answer_type=self.answer_type)
        return TRAKEQuery(
            **common,
            events=self.events,
            temporal_constraints=self.temporal_constraints,
        )


class RouteResult(StrictModel):
    task_type: TaskType
    structured_query: StructuredQuery
