"""Ingest traces from Mercury, OpenAI, Anthropic, and Cursor-like payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mercury.fingerprint import enrich_trace, looks_like_error
from mercury.models import (
    AgentTrace,
    EventType,
    OutcomeStatus,
    ToolCall,
    TraceEvent,
    TraceOutcome,
    trace_id_for,
)


def load_trace(source: str | Path | dict[str, Any]) -> AgentTrace:
    if isinstance(source, dict):
        payload = source
    else:
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_trace(payload)


def parse_trace(payload: dict[str, Any]) -> AgentTrace:
    if "events" in payload and "task" in payload:
        return enrich_trace(_parse_canonical(payload))
    if "messages" in payload:
        return enrich_trace(_parse_messages(payload))
    raise ValueError("Unrecognized trace format: expected Mercury canonical or chat messages")


def _parse_canonical(payload: dict[str, Any]) -> AgentTrace:
    events = [_parse_event(raw) for raw in payload.get("events", [])]
    outcome_raw = payload.get("outcome") or {}
    if isinstance(outcome_raw, str):
        outcome = TraceOutcome(status=_outcome_status(outcome_raw))
    else:
        outcome = TraceOutcome(
            status=_outcome_status(outcome_raw.get("status", "unknown")),
            summary=str(outcome_raw.get("summary", "")),
        )
    model = str(payload.get("model") or "unknown")
    task = str(payload.get("task") or _task_from_events(events))
    trace_id = str(payload.get("id") or trace_id_for(model, task))
    return AgentTrace(
        id=trace_id,
        model=model,
        task=task,
        events=events,
        outcome=outcome,
        started_at=payload.get("started_at"),
        files_touched=list(payload.get("files_touched") or []),
        languages=list(payload.get("languages") or []),
        metadata=dict(payload.get("metadata") or {}),
    )


def _parse_messages(payload: dict[str, Any]) -> AgentTrace:
    messages = payload["messages"]
    events: list[TraceEvent] = []
    for message in messages:
        role = str(message.get("role") or "assistant").lower()
        if role == "tool":
            content = _content_to_text(message.get("content"))
            events.append(
                TraceEvent(
                    type=EventType.TOOL,
                    content=content,
                    tool_name=message.get("name") or message.get("tool_name"),
                    tool_call_id=message.get("tool_call_id"),
                    is_error=bool(message.get("is_error")) or looks_like_error(content),
                )
            )
            continue
        event_type = {
            "user": EventType.USER,
            "system": EventType.SYSTEM,
            "assistant": EventType.ASSISTANT,
        }.get(role, EventType.ASSISTANT)
        tool_calls = [_parse_tool_call(raw) for raw in message.get("tool_calls") or []]
        content = _content_to_text(message.get("content"))
        events.append(
            TraceEvent(
                type=event_type,
                content=content,
                tool_calls=tool_calls,
            )
        )
    model = str(payload.get("model") or "unknown")
    task = str(payload.get("task") or _task_from_events(events))
    outcome = payload.get("outcome") or {}
    if isinstance(outcome, str):
        parsed_outcome = TraceOutcome(status=_outcome_status(outcome))
    else:
        parsed_outcome = TraceOutcome(
            status=_outcome_status((outcome or {}).get("status", "unknown")),
            summary=str((outcome or {}).get("summary", "")),
        )
    return AgentTrace(
        id=str(payload.get("id") or trace_id_for(model, task)),
        model=model,
        task=task,
        events=events,
        outcome=parsed_outcome,
        files_touched=list(payload.get("files_touched") or []),
        languages=list(payload.get("languages") or []),
        metadata=dict(payload.get("metadata") or {}),
    )


def _parse_event(raw: dict[str, Any]) -> TraceEvent:
    event_type = EventType(str(raw.get("type") or "assistant"))
    content = _content_to_text(raw.get("content"))
    tool_calls = [_parse_tool_call(item) for item in raw.get("tool_calls") or []]
    is_error = bool(raw.get("is_error"))
    if event_type is EventType.TOOL:
        is_error = is_error or looks_like_error(content)
    return TraceEvent(
        type=event_type,
        content=content,
        tool_calls=tool_calls,
        tool_name=raw.get("tool_name") or raw.get("name"),
        tool_call_id=raw.get("tool_call_id"),
        is_error=is_error,
        timestamp=raw.get("timestamp"),
    )


def _parse_tool_call(raw: Any) -> ToolCall:
    if not isinstance(raw, dict):
        return ToolCall(name=str(raw))
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    name = raw.get("name") or fn.get("name") or "unknown"
    arguments = raw.get("arguments") or fn.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return ToolCall(name=str(name), arguments=arguments, call_id=raw.get("id") or raw.get("call_id"))


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    chunks.append(str(part["text"]))
                elif part.get("type") == "tool_use":
                    chunks.append(str(part.get("name") or ""))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _task_from_events(events: list[TraceEvent]) -> str:
    for event in events:
        if event.type is EventType.USER and event.content.strip():
            return event.content.strip().splitlines()[0][:400]
    return "untitled task"


def _outcome_status(raw: str | None) -> OutcomeStatus:
    try:
        return OutcomeStatus(str(raw or "unknown").lower())
    except ValueError:
        return OutcomeStatus.UNKNOWN
