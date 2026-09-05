"""Distill HOW a frontier agent operated into operational cards.

This is not answer distillation. Cards capture procedure: tool order,
recovery loops, first-action policy, and self-corrections.
"""

from __future__ import annotations

from collections import Counter

from mercury.fingerprint import error_signature, extract_paths
from mercury.grade import blind_retry
from mercury.models import (
    AgentTrace,
    CardKind,
    EventType,
    OperationalCard,
    TaskType,
    card_id,
)
from mercury.phases import (
    EDIT_TOOLS,
    EXPLORE_TOOLS,
    LOCALIZE_TOOLS,
    VERIFY_TOOLS,
    contains_self_correction,
    first_tool,
    segment_phases,
)

def distill_trace(trace: AgentTrace, *, teacher: bool = True) -> list[OperationalCard]:
    """Extract operational cards from a single agent trace."""
    cards: list[OperationalCard] = []
    phases = segment_phases(trace)
    if teacher:
        playbook = _playbook_card(trace, phases)
        if playbook:
            cards.append(playbook)
        cards.extend(_tool_policy_cards(trace, phases))
        cards.extend(_heuristic_cards(trace, phases))
    cards.extend(_recovery_cards(trace))
    cards.extend(_anti_pattern_cards(trace))
    return cards


def distill_standing_orders(traces: list[AgentTrace]) -> list[OperationalCard]:
    """Majority frontier behaviors become standing orders for lesser models."""
    if len(traces) < 2:
        return []
    successful = [trace for trace in traces if trace.outcome.succeeded]
    if len(successful) < 2:
        return []

    orders: list[tuple[str, str, str, float]] = []
    explore_first = sum(1 for trace in successful if _starts_with(trace, EXPLORE_TOOLS))
    if explore_first / len(successful) >= 0.6:
        orders.append(
            (
                "Search before reading or editing",
                "Starting a coding task in an unfamiliar area",
                "Run a targeted search (grep/glob/semantic) before opening large files or editing.",
                explore_first / len(successful),
            )
        )
    verify_after_edit = sum(1 for trace in successful if _edited_then_verified(trace))
    if verify_after_edit / len(successful) >= 0.5:
        orders.append(
            (
                "Verify after every edit batch",
                "After changing production code",
                "Run the smallest relevant test or command and read the failure before editing again.",
                verify_after_edit / len(successful),
            )
        )
    localize_before_edit = sum(1 for trace in successful if _localized_before_edit(trace))
    if localize_before_edit / len(successful) >= 0.5:
        orders.append(
            (
                "Read the failing site before patching",
                "A bug or failing test has been named",
                "Open the implicated file and the nearest test before writing a patch.",
                localize_before_edit / len(successful),
            )
        )

    cards: list[OperationalCard] = []
    source = successful[0]
    for title, situation, procedure, confidence in orders:
        cards.append(
            OperationalCard(
                id=card_id("standing", title, procedure),
                kind=CardKind.STANDING_ORDER,
                title=title,
                situation=situation,
                procedure=procedure,
                rationale=f"Observed in {int(confidence * 100)}% of successful frontier traces.",
                tools=[],
                task_type=TaskType.GENERAL,
                source_trace_id=source.id,
                source_model=source.model,
                confidence=min(0.95, 0.55 + confidence * 0.4),
            )
        )
    return cards


def _playbook_card(trace: AgentTrace, phases: list) -> OperationalCard | None:
    if not phases:
        return None
    steps: list[str] = []
    seen_names: list[str] = []
    for phase in phases:
        line = _summarize_phase(phase)
        if line:
            steps.append(line)
        if phase.name not in seen_names:
            seen_names.append(phase.name)
    if len(steps) < 2:
        return None
    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    confidence = 0.82 if trace.outcome.succeeded else 0.45
    if trace.outcome.failed:
        confidence = 0.28
    return OperationalCard(
        id=card_id("playbook", trace.id, numbered),
        kind=CardKind.PLAYBOOK,
        title=f"{trace.task_type.value} playbook from {trace.model}",
        situation=trace.task,
        procedure=numbered,
        rationale="Compressed from a frontier agent's actual tool phases, not from its final answer.",
        tools=_unique(trace.tool_sequence()),
        task_type=trace.task_type,
        source_trace_id=trace.id,
        source_model=trace.model,
        confidence=confidence,
        languages=list(trace.languages),
        metadata={"phases": seen_names, "files": trace.files_touched[:12]},
    )


def _summarize_phase(phase) -> str:
    names = phase.tool_names()
    paths: list[str] = []
    for event in phase.events:
        paths.extend(extract_paths(event))
    path_hint = ""
    if paths:
        unique = []
        for path in paths:
            base = path.split("/")[-1]
            if base not in unique:
                unique.append(base)
        path_hint = f" ({', '.join(unique[:4])})"
    if phase.name == "explore":
        queries = _arg_snippets(phase.events, ("query", "pattern", "glob_pattern"))
        extra = f" for {queries[0]}" if queries else ""
        return f"Explore{extra}{path_hint} using {', '.join(_unique(names)[:4]) or 'search'}."
    if phase.name == "localize":
        return f"Read the implicated files{path_hint} before changing them."
    if phase.name == "plan":
        text = next((event.content.strip() for event in phase.events if event.content.strip()), "")
        snippet = text.splitlines()[0][:140] if text else "Form a short plan from evidence."
        return f"Plan: {snippet}"
    if phase.name == "edit":
        return f"Edit{path_hint} with {', '.join(_unique(names)[:3]) or 'the editor'}."
    if phase.name == "verify":
        commands = _arg_snippets(phase.events, ("command", "cmd"))
        extra = f": `{commands[0][:80]}`" if commands else ""
        return f"Verify{extra}."
    if phase.name == "recover":
        return f"Recover from the failure{path_hint} instead of repeating the same edit."
    if names:
        return f"{phase.name}: {', '.join(_unique(names)[:4])}{path_hint}."
    return ""


def _tool_policy_cards(trace: AgentTrace, phases: list) -> list[OperationalCard]:
    cards: list[OperationalCard] = []
    sequence = _unique(trace.tool_sequence())
    if len(sequence) >= 2:
        order = " → ".join(sequence[:8])
        cards.append(
            OperationalCard(
                id=card_id("policy", trace.id, order),
                kind=CardKind.TOOL_POLICY,
                title=f"Tool order for {trace.task_type.value}",
                situation=f"A {trace.task_type.value} task similar to: {trace.task}",
                procedure=f"Prefer this tool order: {order}. Do not skip search/read if the next step would be an edit.",
                rationale="First successful frontier trajectory for this task type.",
                tools=sequence,
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.78 if trace.outcome.succeeded else 0.4,
                languages=list(trace.languages),
            )
        )
    opener = first_tool(trace)
    if opener:
        cards.append(
            OperationalCard(
                id=card_id("first", trace.id, opener),
                kind=CardKind.TOOL_POLICY,
                title=f"First action: {opener}",
                situation=f"Opening a {trace.task_type.value} task like: {trace.task}",
                procedure=f"Start with `{opener}` rather than editing. Gather evidence first.",
                rationale="Frontier models disproportionately open with search/read, not patches.",
                tools=[opener],
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.74 if trace.outcome.succeeded else 0.38,
                languages=list(trace.languages),
            )
        )
    return cards


def _heuristic_cards(trace: AgentTrace, phases: list) -> list[OperationalCard]:
    cards: list[OperationalCard] = []
    phase_names = [phase.name for phase in phases]
    if "verify" in phase_names and "edit" in phase_names:
        if phase_names.index("edit") < _last_index(phase_names, "verify"):
            cards.append(
                _heuristic(
                    trace,
                    "Reproduce or retest after the patch",
                    "You just changed production code",
                    "Run the relevant test or command and only continue if the new output is understood.",
                    trace.tool_sequence(),
                    0.7,
                )
            )
    test_files = [path for path in trace.files_touched if _looks_like_test(path)]
    prod_files = [path for path in trace.files_touched if path not in test_files]
    if test_files and prod_files:
        cards.append(
            _heuristic(
                trace,
                "Read the test that encodes the bug",
                "A failing behavior is described and tests exist",
                f"Open the nearest test ({_basename(test_files[0])}) alongside the production file before patching.",
                ["read"],
                0.68,
            )
        )
    return cards


def _recovery_cards(trace: AgentTrace) -> list[OperationalCard]:
    cards: list[OperationalCard] = []
    events = trace.events
    for index, event in enumerate(events):
        if event.type is not EventType.TOOL or not event.is_error:
            continue
        follow = events[index + 1 : index + 6]
        follow_tools = []
        follow_bits: list[str] = []
        for item in follow:
            follow_tools.extend(item.tool_names())
            if item.type is EventType.ASSISTANT and item.content.strip():
                follow_bits.append(item.content.strip().splitlines()[0][:140])
            elif item.tool_names():
                follow_bits.append("then `" + ", ".join(_unique(item.tool_names())) + "`")
        if not follow_bits:
            continue
        signature = error_signature(event.content) or (event.tool_name or "tool error")
        procedure = " ".join(follow_bits[:4])
        confidence = 0.8 if trace.outcome.succeeded else 0.55
        cards.append(
            OperationalCard(
                id=card_id("recovery", trace.id, signature, procedure),
                kind=CardKind.RECOVERY,
                title=f"Recover from: {signature[:80]}",
                situation=f"A tool result looks like: {signature}",
                procedure=f"Do not retry the same call unchanged. Next: {procedure}",
                rationale="Copied from the frontier agent's actual recovery loop.",
                tools=_unique(follow_tools),
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=confidence,
                languages=list(trace.languages),
                error_signature=signature,
            )
        )
    return cards


def _anti_pattern_cards(trace: AgentTrace) -> list[OperationalCard]:
    cards: list[OperationalCard] = []
    for event in contains_self_correction(trace):
        snippet = event.content.strip().splitlines()[0][:180]
        cards.append(
            OperationalCard(
                id=card_id("anti", trace.id, snippet),
                kind=CardKind.ANTI_PATTERN,
                title="Self-correction: do not keep the first impulse",
                situation=trace.task,
                procedure=f"The frontier agent corrected itself: {snippet}",
                rationale="Self-corrections are negative knowledge: the first impulse was the lesser-model move.",
                tools=event.tool_names(),
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.66,
                languages=list(trace.languages),
            )
        )
    if trace.outcome.failed and first_tool(trace) and first_tool(trace).lower() in {name.lower() for name in EDIT_TOOLS}:
        cards.append(
            OperationalCard(
                id=card_id("anti-edit-first", trace.id),
                kind=CardKind.ANTI_PATTERN,
                title="Do not edit before locating the fault",
                situation=trace.task,
                procedure="Editing as the first action failed. Search and read first, then patch a localized site.",
                rationale="Failed trajectory opened with an edit tool.",
                tools=trace.tool_sequence()[:4],
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.72,
                languages=list(trace.languages),
            )
        )
    retry_hit = blind_retry(trace)
    if retry_hit is not None:
        detail, action = retry_hit
        cards.append(
            OperationalCard(
                id=card_id("anti-blind-retry", trace.id, action.name, action.args_hash),
                kind=CardKind.ANTI_PATTERN,
                title="Never re-run an identical edit after a failure",
                situation=trace.task,
                procedure=(
                    f"This run {detail} (`{action.name}`). "
                    "Change the approach instead: read the failing site, fix the diagnosis, then edit differently."
                ),
                rationale="Repeating a failed edit unchanged is the strongest negative signal in the trace.",
                tools=[action.name],
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.78,
                languages=list(trace.languages),
            )
        )
    return cards


def _heuristic(
    trace: AgentTrace,
    title: str,
    situation: str,
    procedure: str,
    tools: list[str],
    confidence: float,
) -> OperationalCard:
    return OperationalCard(
        id=card_id("heur", trace.id, title, procedure),
        kind=CardKind.HEURISTIC,
        title=title,
        situation=situation,
        procedure=procedure,
        rationale="Behavior present on a successful frontier run.",
        tools=_unique(tools),
        task_type=trace.task_type,
        source_trace_id=trace.id,
        source_model=trace.model,
        confidence=confidence if trace.outcome.succeeded else confidence * 0.5,
        languages=list(trace.languages),
    )


def _arg_snippets(events, keys: tuple[str, ...]) -> list[str]:
    snippets: list[str] = []
    for event in events:
        for call in event.tool_calls:
            for key in keys:
                value = call.arguments.get(key)
                if isinstance(value, str) and value.strip():
                    snippets.append(value.strip())
        if event.content and event.type is EventType.TOOL:
            continue
    return snippets


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _last_index(items: list[str], value: str) -> int:
    index = -1
    for i, item in enumerate(items):
        if item == value:
            index = i
    return index


def _starts_with(trace: AgentTrace, tools: set[str]) -> bool:
    opener = first_tool(trace)
    return bool(opener) and opener.lower() in tools


def _edited_then_verified(trace: AgentTrace) -> bool:
    names = [name.lower() for name in trace.tool_sequence()]
    edit_at = next((i for i, name in enumerate(names) if name in EDIT_TOOLS), None)
    if edit_at is None:
        return False
    return any(name in VERIFY_TOOLS for name in names[edit_at + 1 :])


def _localized_before_edit(trace: AgentTrace) -> bool:
    names = [name.lower() for name in trace.tool_sequence()]
    read_at = next((i for i, name in enumerate(names) if name in LOCALIZE_TOOLS | EXPLORE_TOOLS), None)
    edit_at = next((i for i, name in enumerate(names) if name in EDIT_TOOLS), None)
    if read_at is None or edit_at is None:
        return False
    return read_at < edit_at


def _looks_like_test(path: str) -> bool:
    lower = path.lower()
    return "test" in lower or lower.endswith("_spec.ts") or lower.endswith("_spec.py")


def _basename(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def ngram_tool_policies(traces: list[AgentTrace], n: int = 3) -> list[tuple[tuple[str, ...], int]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for trace in traces:
        sequence = [name.lower() for name in trace.tool_sequence()]
        if len(sequence) < n:
            continue
        for index in range(len(sequence) - n + 1):
            counts[tuple(sequence[index : index + n])] += 1
    return counts.most_common(12)
