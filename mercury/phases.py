"""Segment an agent trace into operational phases.

Tool names are normalized across harnesses (Cursor, Claude Code, Codex,
OpenAI Assistants, Gemini, etc.) before phase membership is tested. Raw
alias sets still exist for callers that need the expanded vocabulary;
prefer ``normalize_tool`` / ``tool_phase`` / ``is_*_tool`` everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mercury.models import AgentTrace, EventType, TraceEvent

# Canonical tool families used for phase membership and grading.
CANONICAL_EXPLORE = "explore"
CANONICAL_LOCALIZE = "localize"
CANONICAL_EDIT = "edit"
CANONICAL_VERIFY = "verify"
CANONICAL_PLAN = "plan"
CANONICAL_UNKNOWN = "unknown"

# Map every known harness alias → a canonical family name.
# Keys are lowercase; matching strips non-alphanumerics so
# ``run_terminal_cmd``, ``RunTerminalCmd``, and ``run-terminal-cmd`` collide.
_TOOL_FAMILY: dict[str, str] = {
    # explore
    "grep": CANONICAL_EXPLORE,
    "rg": CANONICAL_EXPLORE,
    "find": CANONICAL_EXPLORE,
    "glob": CANONICAL_EXPLORE,
    "globfilesearch": CANONICAL_EXPLORE,
    "file_search": CANONICAL_EXPLORE,
    "filesearch": CANONICAL_EXPLORE,
    "listdir": CANONICAL_EXPLORE,
    "list_dir": CANONICAL_EXPLORE,
    "ls": CANONICAL_EXPLORE,
    "semanticsearch": CANONICAL_EXPLORE,
    "semantic_search": CANONICAL_EXPLORE,
    "codebasesearch": CANONICAL_EXPLORE,
    "codebase_search": CANONICAL_EXPLORE,
    "searchcodebase": CANONICAL_EXPLORE,
    "workspace_symbols": CANONICAL_EXPLORE,
    "workspacesymbols": CANONICAL_EXPLORE,
    "search_files": CANONICAL_EXPLORE,
    "searchfiles": CANONICAL_EXPLORE,
    "websearch": CANONICAL_EXPLORE,
    "web_search": CANONICAL_EXPLORE,
    # localize / read
    "read": CANONICAL_LOCALIZE,
    "readfile": CANONICAL_LOCALIZE,
    "read_file": CANONICAL_LOCALIZE,
    "open": CANONICAL_LOCALIZE,
    "open_file": CANONICAL_LOCALIZE,
    "openfile": CANONICAL_LOCALIZE,
    "cat": CANONICAL_LOCALIZE,
    "view": CANONICAL_LOCALIZE,
    "get_file_contents": CANONICAL_LOCALIZE,
    "getfilecontents": CANONICAL_LOCALIZE,
    "fetch": CANONICAL_LOCALIZE,
    # edit
    "edit": CANONICAL_EDIT,
    "applypatch": CANONICAL_EDIT,
    "apply_patch": CANONICAL_EDIT,
    "searchreplace": CANONICAL_EDIT,
    "search_replace": CANONICAL_EDIT,
    "strreplace": CANONICAL_EDIT,
    "str_replace": CANONICAL_EDIT,
    "write": CANONICAL_EDIT,
    "writefile": CANONICAL_EDIT,
    "write_file": CANONICAL_EDIT,
    "create_file": CANONICAL_EDIT,
    "createfile": CANONICAL_EDIT,
    "multiedit": CANONICAL_EDIT,
    "multi_edit": CANONICAL_EDIT,
    "editnotebook": CANONICAL_EDIT,
    "edit_notebook": CANONICAL_EDIT,
    "delete": CANONICAL_EDIT,
    "delete_file": CANONICAL_EDIT,
    "deletefile": CANONICAL_EDIT,
    "notebookedit": CANONICAL_EDIT,
    # verify / shell
    "shell": CANONICAL_VERIFY,
    "bash": CANONICAL_VERIFY,
    "zsh": CANONICAL_VERIFY,
    "terminal": CANONICAL_VERIFY,
    "runterminalcmd": CANONICAL_VERIFY,
    "run_terminal_cmd": CANONICAL_VERIFY,
    "run_command": CANONICAL_VERIFY,
    "runcommand": CANONICAL_VERIFY,
    "execute": CANONICAL_VERIFY,
    "exec": CANONICAL_VERIFY,
    "test": CANONICAL_VERIFY,
    "pytest": CANONICAL_VERIFY,
    "run_tests": CANONICAL_VERIFY,
    "runtests": CANONICAL_VERIFY,
    # plan
    "todowrite": CANONICAL_PLAN,
    "todo_write": CANONICAL_PLAN,
    "todoread": CANONICAL_PLAN,
    "todo_read": CANONICAL_PLAN,
    "createplan": CANONICAL_PLAN,
    "create_plan": CANONICAL_PLAN,
    "update_plan": CANONICAL_PLAN,
    "updateplan": CANONICAL_PLAN,
}

# Backward-compatible expanded sets (raw aliases, lowercase). Prefer helpers.
EXPLORE_TOOLS = {k for k, v in _TOOL_FAMILY.items() if v == CANONICAL_EXPLORE}
LOCALIZE_TOOLS = {k for k, v in _TOOL_FAMILY.items() if v == CANONICAL_LOCALIZE}
EDIT_TOOLS = {k for k, v in _TOOL_FAMILY.items() if v == CANONICAL_EDIT}
VERIFY_TOOLS = {k for k, v in _TOOL_FAMILY.items() if v == CANONICAL_VERIFY}
PLAN_TOOLS = {k for k, v in _TOOL_FAMILY.items() if v == CANONICAL_PLAN}

_SELF_CORRECT = (
    "actually",
    "wait,",
    "instead",
    "i should have",
    "on second thought",
    "correction",
    "let me redo",
    "rather than",
    "not that",
    "wrong approach",
)


def _compact(name: str) -> str:
    """Lowercase and strip separators so harness naming variants collide."""
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def normalize_tool(name: str | None) -> str:
    """Map a harness-specific tool name to a canonical family.

    Returns one of: explore, localize, edit, verify, plan, or the compacted
    original name when unrecognized (so unknown tools still fingerprint).
    """
    if not name:
        return CANONICAL_UNKNOWN
    compact = _compact(name)
    if compact in _TOOL_FAMILY:
        return _TOOL_FAMILY[compact]
    # Also try without underscores for CamelCase leftovers.
    alnum = "".join(ch for ch in compact if ch.isalnum())
    if alnum in _TOOL_FAMILY:
        return _TOOL_FAMILY[alnum]
    return compact or CANONICAL_UNKNOWN


def tool_phase(name: str | None) -> str:
    """Phase name for a tool call: explore/localize/edit/verify/plan/act."""
    family = normalize_tool(name)
    if family in {
        CANONICAL_EXPLORE,
        CANONICAL_LOCALIZE,
        CANONICAL_EDIT,
        CANONICAL_VERIFY,
        CANONICAL_PLAN,
    }:
        return family
    return "act"


def is_explore_tool(name: str | None) -> bool:
    return normalize_tool(name) == CANONICAL_EXPLORE


def is_localize_tool(name: str | None) -> bool:
    return normalize_tool(name) == CANONICAL_LOCALIZE


def is_edit_tool(name: str | None) -> bool:
    return normalize_tool(name) == CANONICAL_EDIT


def is_verify_tool(name: str | None) -> bool:
    return normalize_tool(name) == CANONICAL_VERIFY


def is_plan_tool(name: str | None) -> bool:
    return normalize_tool(name) == CANONICAL_PLAN


def is_evidence_tool(name: str | None) -> bool:
    return normalize_tool(name) in {CANONICAL_EXPLORE, CANONICAL_LOCALIZE}


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
    names = event.tool_names()
    if event.type is EventType.TOOL and looks_like_verify_output(event.content) and event.is_error:
        return "recover"
    for name in names:
        phase = tool_phase(name)
        if phase != "act":
            return phase
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


def first_tool_family(trace: AgentTrace) -> str | None:
    opener = first_tool(trace)
    return normalize_tool(opener) if opener else None


def contains_self_correction(trace: AgentTrace) -> list[TraceEvent]:
    hits: list[TraceEvent] = []
    for event in trace.events:
        if event.type is EventType.ASSISTANT and event.content:
            lowered = event.content.lower()
            if any(token in lowered for token in _SELF_CORRECT):
                hits.append(event)
    return hits
