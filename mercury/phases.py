"""Segment an agent trace into operational phases."""

from __future__ import annotations

from dataclasses import dataclass, field

from mercury.models import AgentTrace, EventType, TraceEvent

EXPLORE_TOOLS = {
    "grep",
    "glob",
    "glob_file_search",
    "list_dir",
    "ls",
    "semantic_search",
    "codebase_search",
    "rg",
    "find",
}
LOCALIZE_TOOLS = {"read", "read_file", "open", "cat"}
EDIT_TOOLS = {
    "edit",
    "apply_patch",
    "search_replace",
    "write",
    "write_file",
    "strreplace",
    "str_replace",
}
VERIFY_TOOLS = {"shell", "bash", "run_terminal_cmd", "terminal", "test", "pytest"}
PLAN_TOOLS = {"todo_write", "todo_read", "create_plan"}

_SELF_CORRECT = (
    "actually",
    "wait,",
    "instead",
    "i should have",
    "on second thought",
    "correction",
    "let me redo",
)


@dataclass
class Phase:
    name: str
    events: list[TraceEvent] = field(default_factory=list)

    def tool_names(self) -> list[str]:
        names: list[str] = []
        for event in self.events:
            names.extend(event.tool_names())
        return names

    def summary(self) -> str:
        tools = self.tool_names()
        if not tools and self.events:
            text = next((event.content.strip() for event in self.events if event.content.strip()), "")
            return (text.splitlines()[0] if text else self.name)[:160]
        unique: list[str] = []
        for name in tools:
            if name not in unique:
                unique.append(name)
        return f"{self.name} via {', '.join(unique[:6])}"


def segment_phases(trace: AgentTrace) -> list[Phase]:
    phases: list[Phase] = []
    current: Phase | None = None
    for event in trace.events:
        if event.type is EventType.USER:
            continue
        name = _phase_name(event)
        if current is None or current.name != name:
            current = Phase(name=name)
            phases.append(current)
        current.events.append(event)
    return [phase for phase in phases if phase.events]


def _phase_name(event: TraceEvent) -> str:
    if event.type is EventType.TOOL and event.is_error:
        return "recover"
    names = [name.lower() for name in event.tool_names()]
    if event.type is EventType.TOOL and looks_like_verify_output(event.content) and event.is_error:
        return "recover"
    for name in names:
        if name in EXPLORE_TOOLS:
            return "explore"
        if name in LOCALIZE_TOOLS:
            return "localize"
        if name in EDIT_TOOLS:
            return "edit"
        if name in VERIFY_TOOLS:
            return "verify"
        if name in PLAN_TOOLS:
            return "plan"
    if event.type is EventType.ASSISTANT and event.content and not names:
        lowered = event.content.lower()
        if any(token in lowered for token in _SELF_CORRECT):
            return "recover"
        return "plan"
    if event.type is EventType.TOOL:
        return "act"
    return "plan"


def looks_like_verify_output(content: str) -> bool:
    lowered = content.lower()
    return any(token in lowered for token in ("passed", "failed", "fail", "error", "ok", "pytest"))


def first_tool(trace: AgentTrace) -> str | None:
    sequence = trace.tool_sequence()
    return sequence[0] if sequence else None


def contains_self_correction(trace: AgentTrace) -> list[TraceEvent]:
    hits: list[TraceEvent] = []
    for event in trace.events:
        if event.type is EventType.ASSISTANT and event.content:
            lowered = event.content.lower()
            if any(token in lowered for token in _SELF_CORRECT):
                hits.append(event)
    return hits
