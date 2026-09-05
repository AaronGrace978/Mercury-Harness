"""Distill HOW a frontier agent operated into operational cards.

This is not answer distillation. Cards capture procedure: tool order,
recovery loops, first-action policy, self-corrections — and, where the
trace allows, the *decision function*: what was chosen and what was
ruled out, not only the observed path.
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
    contains_self_correction,
    first_tool,
    first_tool_family,
    is_edit_tool,
    is_evidence_tool,
    is_explore_tool,
    is_localize_tool,
    is_verify_tool,
    normalize_tool,
    segment_phases,
)

# Default standing orders used when the store has no successful frontier
# volume yet. Marked provisional so packs are never empty on cold start.
_SEED_ORDERS: list[tuple[str, str, str]] = [
    (
        "Search before reading or editing",
        "Starting a coding task in an unfamiliar area",
        "Run a targeted search (grep/glob/semantic) before opening large files or editing.",
    ),
    (
        "Verify after every edit batch",
        "After changing production code",
        "Run the smallest relevant test or command and read the failure before editing again.",
    ),
    (
        "Read the failing site before patching",
        "A bug or failing test has been named",
        "Open the implicated file and the nearest test before writing a patch.",
    ),
]


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
        cards.extend(_decision_cards(trace, phases))
    cards.extend(_recovery_cards(trace))
    cards.extend(_anti_pattern_cards(trace))
    return cards


def distill_standing_orders(traces: list[AgentTrace]) -> list[OperationalCard]:
    """Standing orders from frontier majority — with cold-start fallbacks.

    - 0 successful traces → seeded defaults (provisional, low confidence)
    - 1 successful trace → that trace's observed policies (provisional)
    - ≥2 successful traces → majority thresholds (confirmed)
    """
    successful = [trace for trace in traces if trace.outcome.succeeded]
    if not successful:
        return _seed_standing_orders(traces)

    n = len(successful)
    provisional = n < 2
    # Cold start: fire on the single observed signal. After volume arrives,
    # require majority agreement so noise from one odd run does not stick.
    explore_threshold = 0.99 if provisional else 0.6
    verify_threshold = 0.99 if provisional else 0.5
    localize_threshold = 0.99 if provisional else 0.5

    orders: list[tuple[str, str, str, float, bool]] = []
    explore_first = sum(1 for trace in successful if _starts_with_explore(trace))
    if explore_first / n >= explore_threshold:
        orders.append(
            (
                "Search before reading or editing",
                "Starting a coding task in an unfamiliar area",
                "Run a targeted search (grep/glob/semantic) before opening large files or editing.",
                explore_first / n,
                provisional,
            )
        )
    verify_after_edit = sum(1 for trace in successful if _edited_then_verified(trace))
    if verify_after_edit / n >= verify_threshold:
        orders.append(
            (
                "Verify after every edit batch",
                "After changing production code",
                "Run the smallest relevant test or command and read the failure before editing again.",
                verify_after_edit / n,
                provisional,
            )
        )
    localize_before_edit = sum(1 for trace in successful if _localized_before_edit(trace))
    if localize_before_edit / n >= localize_threshold:
        orders.append(
            (
                "Read the failing site before patching",
                "A bug or failing test has been named",
                "Open the implicated file and the nearest test before writing a patch.",
                localize_before_edit / n,
                provisional,
            )
        )

    # If the single provisional trace fired nothing, still seed defaults so
    # the pack is never empty waiting for majority volume.
    if provisional and not orders:
        return _seed_standing_orders(successful)

    cards: list[OperationalCard] = []
    source = successful[0]
    for title, situation, procedure, confidence, is_provisional in orders:
        if is_provisional:
            rationale = (
                f"Provisional standing order from 1 successful frontier trace "
                f"({source.model}). Will harden once a majority across traces agrees."
            )
            card_confidence = min(0.7, 0.45 + confidence * 0.2)
        else:
            rationale = f"Observed in {int(confidence * 100)}% of successful frontier traces."
            card_confidence = min(0.95, 0.55 + confidence * 0.4)
        cards.append(
            OperationalCard(
                # Stable id by title so seed → provisional → majority upserts in place.
                id=card_id("standing", title),
                kind=CardKind.STANDING_ORDER,
                title=title,
                situation=situation,
                procedure=procedure,
                rationale=rationale,
                tools=[],
                task_type=TaskType.GENERAL,
                source_trace_id=source.id,
                source_model=source.model,
                confidence=card_confidence,
                metadata={"provisional": is_provisional, "support": n, "rate": confidence},
            )
        )
    return cards


def _seed_standing_orders(traces: list[AgentTrace]) -> list[OperationalCard]:
    """Bootstrap standing orders before any successful frontier volume exists."""
    source_id = traces[0].id if traces else "seed"
    source_model = traces[0].model if traces else "mercury-seed"
    cards: list[OperationalCard] = []
    for title, situation, procedure in _SEED_ORDERS:
        cards.append(
            OperationalCard(
                id=card_id("standing", title),
                kind=CardKind.STANDING_ORDER,
                title=title,
                situation=situation,
                procedure=procedure,
                rationale=(
                    "Seeded cold-start standing order. Replace with majority "
                    "frontier evidence as soon as two successful teacher traces exist."
                ),
                tools=[],
                task_type=TaskType.GENERAL,
                source_trace_id=source_id,
                source_model=source_model,
                confidence=0.42,
                metadata={"provisional": True, "seeded": True, "support": 0},
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
        family = normalize_tool(opener)
        rejected = []
        if is_explore_tool(opener) or is_localize_tool(opener):
            rejected = ["edit-first", "patch without evidence"]
        cards.append(
            OperationalCard(
                id=card_id("first", trace.id, opener),
                kind=CardKind.TOOL_POLICY,
                title=f"First action: {opener}",
                situation=f"Opening a {trace.task_type.value} task like: {trace.task}",
                procedure=f"Start with `{opener}` rather than editing. Gather evidence first.",
                rationale="Frontier models disproportionately open with search/read, not patches.",
                tools=[opener],
                chose=f"{family} via `{opener}`",
                rejected=rejected,
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.74 if trace.outcome.succeeded else 0.38,
                languages=list(trace.languages),
            )
        )
    return cards


def _decision_cards(trace: AgentTrace, phases: list) -> list[OperationalCard]:
    """Emit cards that encode judgment: chose X over Y, not just the path taken.

    Surface signals used:
    - Opening move (explore/localize vs edit)
    - File focus (which of the discovered paths were actually edited)
    - Self-corrections (explicit rejection of the first impulse)
    - Recovery after error (changed approach rather than blind retry)
    """
    cards: list[OperationalCard] = []
    opener = first_tool(trace)
    opener_family = first_tool_family(trace)
    if opener and opener_family in {"explore", "localize"}:
        cards.append(
            OperationalCard(
                id=card_id("decision-open", trace.id, opener),
                kind=CardKind.DECISION,
                title="Chose evidence-gathering over an immediate edit",
                situation=f"Opening a {trace.task_type.value} task like: {trace.task}",
                procedure=(
                    f"Chose `{opener}` ({opener_family}) as the first move. "
                    "Do not open with an edit tool even if a likely file comes to mind."
                ),
                rationale="The decision function is the first move: gather evidence, reject edit-first.",
                tools=[opener],
                chose=f"start with {opener_family}",
                rejected=["start with edit", "patch the first file that sounds related"],
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.8 if trace.outcome.succeeded else 0.5,
                languages=list(trace.languages),
                metadata={"decision": "first_action"},
            )
        )

    # File focus: paths that appeared in explore/read but were never edited.
    seen_paths: list[str] = []
    edited_paths: set[str] = set()
    for event in trace.events:
        for path in extract_paths(event):
            base = path
            if base not in seen_paths:
                seen_paths.append(base)
        if event.type is EventType.ASSISTANT:
            for call in event.tool_calls:
                if is_edit_tool(call.name):
                    for key in ("path", "file", "file_path"):
                        value = call.arguments.get(key)
                        if isinstance(value, str) and value.strip():
                            edited_paths.add(value.strip())
    skipped = [path for path in seen_paths if path not in edited_paths]
    if edited_paths and skipped:
        edited_bases = [_basename(path) for path in list(edited_paths)[:4]]
        skipped_bases = [_basename(path) for path in skipped[:4]]
        cards.append(
            OperationalCard(
                id=card_id("decision-files", trace.id, ",".join(sorted(edited_bases))),
                kind=CardKind.DECISION,
                title="Chose the real fault site over nearby files",
                situation=trace.task,
                procedure=(
                    f"Chose to edit {', '.join(f'`{name}`' for name in edited_bases)}. "
                    f"Left alone: {', '.join(f'`{name}`' for name in skipped_bases)}. "
                    "Touch the implicated site; do not rewrite a neighboring page that merely mentions the symptom."
                ),
                rationale="File selection is a decision, not a side effect of the path taken.",
                tools=["read", "grep"] + list(edited_paths)[:2],
                chose=", ".join(edited_bases),
                rejected=skipped_bases,
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.76 if trace.outcome.succeeded else 0.45,
                languages=list(trace.languages),
                metadata={"decision": "file_focus", "edited": list(edited_paths)[:8]},
            )
        )

    for event in contains_self_correction(trace):
        snippet = event.content.strip().splitlines()[0][:180]
        cards.append(
            OperationalCard(
                id=card_id("decision-correct", trace.id, snippet),
                kind=CardKind.DECISION,
                title="Rejected the first impulse after re-reading evidence",
                situation=trace.task,
                procedure=(
                    f"The frontier agent overruled its own first impulse: {snippet} "
                    "Treat the corrected approach as the decision; discard the impulse."
                ),
                rationale="Self-corrections expose the rejected branch that pure path cards hide.",
                tools=event.tool_names(),
                chose="revised approach after re-evaluation",
                rejected=["keep the first impulse"],
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.72,
                languages=list(trace.languages),
                metadata={"decision": "self_correction"},
            )
        )

    # Recovery-as-decision: after an error, the next action changed.
    events = trace.events
    for index, event in enumerate(events):
        if event.type is not EventType.TOOL or not event.is_error:
            continue
        prev_tools = []
        for prior in events[:index]:
            if prior.type is EventType.ASSISTANT and prior.tool_calls:
                prev_tools = [call.name for call in prior.tool_calls]
        next_tools: list[str] = []
        for later in events[index + 1 : index + 5]:
            if later.type is EventType.ASSISTANT and later.tool_calls:
                next_tools = [call.name for call in later.tool_calls]
                break
        if not prev_tools or not next_tools:
            continue
        if [normalize_tool(name) for name in prev_tools] == [normalize_tool(name) for name in next_tools]:
            # Same family retry — only interesting if args changed; blind_retry
            # already covers identical args. Skip same-family same-move here.
            continue
        signature = error_signature(event.content) or (event.tool_name or "tool error")
        cards.append(
            OperationalCard(
                id=card_id("decision-recover", trace.id, signature, ",".join(next_tools)),
                kind=CardKind.DECISION,
                title="Chose a new approach after the failure",
                situation=f"A tool result looks like: {signature}",
                procedure=(
                    f"After the failure, switched from `{', '.join(prev_tools[:3])}` "
                    f"to `{', '.join(next_tools[:3])}`. Do not repeat the failed move unchanged."
                ),
                rationale="Recovery is a decision against the previous action, not just the next step on the path.",
                tools=_unique(prev_tools + next_tools),
                chose=", ".join(next_tools[:3]),
                rejected=[f"repeat `{name}` unchanged" for name in prev_tools[:3]],
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.78 if trace.outcome.succeeded else 0.55,
                languages=list(trace.languages),
                error_signature=signature,
                metadata={"decision": "recovery_switch"},
            )
        )

    # Phase-order decision: verify after edit when present.
    phase_names = [phase.name for phase in phases]
    if "edit" in phase_names and "verify" in phase_names:
        if phase_names.index("edit") < _last_index(phase_names, "verify"):
            cards.append(
                OperationalCard(
                    id=card_id("decision-verify", trace.id),
                    kind=CardKind.DECISION,
                    title="Chose to verify instead of editing again",
                    situation="You just changed production code",
                    procedure=(
                        "After the edit batch, run the relevant test or command before another patch. "
                        "Reject the urge to keep editing on faith."
                    ),
                    rationale="Verification is an explicit choice against edit-thrash.",
                    tools=[name for name in trace.tool_sequence() if is_verify_tool(name)][:3],
                    chose="verify after edit",
                    rejected=["edit again without evidence", "assume the patch worked"],
                    task_type=trace.task_type,
                    source_trace_id=trace.id,
                    source_model=trace.model,
                    confidence=0.74 if trace.outcome.succeeded else 0.4,
                    languages=list(trace.languages),
                    metadata={"decision": "verify_after_edit"},
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
                chose=procedure[:120],
                rejected=["retry the same call unchanged"],
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
                chose="corrected approach",
                rejected=["keep the first impulse"],
                task_type=trace.task_type,
                source_trace_id=trace.id,
                source_model=trace.model,
                confidence=0.66,
                languages=list(trace.languages),
            )
        )
    opener = first_tool(trace)
    if trace.outcome.failed and opener and is_edit_tool(opener):
        cards.append(
            OperationalCard(
                id=card_id("anti-edit-first", trace.id),
                kind=CardKind.ANTI_PATTERN,
                title="Do not edit before locating the fault",
                situation=trace.task,
                procedure="Editing as the first action failed. Search and read first, then patch a localized site.",
                rationale="Failed trajectory opened with an edit tool.",
                tools=trace.tool_sequence()[:4],
                chose="(should have) search/read first",
                rejected=["edit as first action"],
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
                chose="change approach after failure",
                rejected=[f"repeat `{action.name}` unchanged"],
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


def _starts_with_explore(trace: AgentTrace) -> bool:
    opener = first_tool(trace)
    return bool(opener) and is_explore_tool(opener)


def _edited_then_verified(trace: AgentTrace) -> bool:
    names = trace.tool_sequence()
    edit_at = next((i for i, name in enumerate(names) if is_edit_tool(name)), None)
    if edit_at is None:
        return False
    return any(is_verify_tool(name) for name in names[edit_at + 1 :])


def _localized_before_edit(trace: AgentTrace) -> bool:
    names = trace.tool_sequence()
    read_at = next((i for i, name in enumerate(names) if is_evidence_tool(name)), None)
    edit_at = next((i for i, name in enumerate(names) if is_edit_tool(name)), None)
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
        sequence = [normalize_tool(name) for name in trace.tool_sequence()]
        if len(sequence) < n:
            continue
        for index in range(len(sequence) - n + 1):
            counts[tuple(sequence[index : index + n])] += 1
    return counts.most_common(12)
