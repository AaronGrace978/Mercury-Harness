"""Canonical trace, event, and operational-card models."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mercury.tiers import ModelTier, classify_model


class EventType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class OutcomeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class CardKind(str, Enum):
    PLAYBOOK = "playbook"
    RECOVERY = "recovery"
    TOOL_POLICY = "tool_policy"
    HEURISTIC = "heuristic"
    ANTI_PATTERN = "anti_pattern"
    STANDING_ORDER = "standing_order"
    CONTRAST = "contrast"
    DECISION = "decision"


class TaskType(str, Enum):
    BUGFIX = "bugfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    REVIEW = "review"
    GENERAL = "general"


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class TraceEvent(BaseModel):
    type: EventType
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_name: str | None = None
    tool_call_id: str | None = None
    is_error: bool = False
    timestamp: float | None = None

    def tool_names(self) -> list[str]:
        if self.type is EventType.TOOL and self.tool_name:
            return [self.tool_name]
        return [call.name for call in self.tool_calls]


class TraceOutcome(BaseModel):
    status: OutcomeStatus = OutcomeStatus.UNKNOWN
    summary: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status is OutcomeStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status is OutcomeStatus.FAILURE


class AgentTrace(BaseModel):
    id: str
    model: str
    task: str
    events: list[TraceEvent] = Field(default_factory=list)
    outcome: TraceOutcome = Field(default_factory=TraceOutcome)
    started_at: str | None = None
    files_touched: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def tier(self) -> ModelTier:
        return classify_model(self.model)

    @property
    def task_type(self) -> TaskType:
        from mercury.fingerprint import classify_task

        return classify_task(self.task)

    def tool_sequence(self) -> list[str]:
        from_assistant: list[str] = []
        from_results: list[str] = []
        for event in self.events:
            if event.tool_calls:
                from_assistant.extend(call.name for call in event.tool_calls)
            elif event.type is EventType.TOOL and event.tool_name:
                from_results.append(event.tool_name)
        return from_assistant or from_results


class OperationalCard(BaseModel):
    id: str
    kind: CardKind
    title: str
    situation: str
    procedure: str
    rationale: str = ""
    tools: list[str] = Field(default_factory=list)
    # Decision-function fields: what was chosen *and* what was ruled out.
    # Observed-path cards leave these empty; decision/contrast cards fill them.
    chose: str = ""
    rejected: list[str] = Field(default_factory=list)
    task_type: TaskType = TaskType.GENERAL
    source_trace_id: str
    source_model: str
    confidence: float = 0.5
    languages: list[str] = Field(default_factory=list)
    error_signature: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def searchable_text(self) -> str:
        parts = [
            self.kind.value,
            self.title,
            self.situation,
            self.procedure,
            self.rationale,
            self.chose,
            " ".join(self.rejected),
            " ".join(self.tools),
            self.task_type.value,
            " ".join(self.languages),
            self.error_signature or "",
        ]
        return "\n".join(part for part in parts if part)

    def estimated_tokens(self) -> int:
        return max(1, int(len(self.searchable_text().split()) * 1.3))


def card_id(*parts: str) -> str:
    payload = "||".join(parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
    return f"card_{digest}"


def trace_id_for(model: str, task: str, extra: str = "") -> str:
    payload = f"{model}|{task}|{extra}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
    return f"trace_{digest}"
