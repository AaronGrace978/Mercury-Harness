"""Deterministic behavioral grading for agent traces.

Grades HOW a trace operated — tool order, localization, verification,
recovery — not whether the final answer was right. The point is to prove
a Frontier Operating Pack changes lesser-model behavior without an LLM
judge, without network calls, and without vibes.

Actions are assistant-initiated tool calls (mirroring ``tool_sequence``);
tool results contribute only error markers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from mercury.models import AgentTrace, EventType, OutcomeStatus, ToolCall
from mercury.phases import EDIT_TOOLS, EXPLORE_TOOLS, LOCALIZE_TOOLS, VERIFY_TOOLS

_ERROR = object()


@dataclass(frozen=True)
class Action:
    """One assistant-initiated tool call, fingerprinted by name + arguments."""

    name: str
    args_hash: str
    path: str = ""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class BehaviorReport:
    model: str
    task: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(1.0 for check in self.checks if check.passed) / len(self.checks)

    @property
    def passed_names(self) -> list[str]:
        return [check.name for check in self.checks if check.passed]

    @property
    def failed_names(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]

    def summary(self) -> str:
        pct = f"{self.score:.0%}"
        lines = [
            f"Behavior grade: {pct} ({len(self.passed_names)}/{len(self.checks)} checks) — {self.model}"
        ]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            suffix = f" — {check.detail}" if check.detail else ""
            lines.append(f"  [{mark}] {check.name}{suffix}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "score": round(self.score, 4),
            "passed": self.passed_names,
            "failed": self.failed_names,
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in self.checks
            ],
        }


def grade_trace(trace: AgentTrace) -> BehaviorReport:
    """Deterministically grade operational behavior. Pure function; stores nothing."""
    report = BehaviorReport(model=trace.model, task=trace.task)
    timeline = _timeline(trace)
    actions = [entry for entry in timeline if isinstance(entry, Action)]
    edit_indices = [index for index, action in enumerate(actions) if action.name in EDIT_TOOLS]

    if edit_indices:
        first = actions[0]
        report.checks.append(
            CheckResult(
                "explored_first",
                first.name in EXPLORE_TOOLS,
                f"first tool was `{first.name}`",
            )
        )
        read_at = next(
            (
                index
                for index, action in enumerate(actions)
                if action.name in EXPLORE_TOOLS | LOCALIZE_TOOLS
            ),
            None,
        )
        report.checks.append(
            CheckResult(
                "read_before_edit",
                read_at is not None and read_at < edit_indices[0],
                "" if read_at is not None and read_at < edit_indices[0] else "edited before any search or read",
            )
        )
        verify_at = next(
            (
                index
                for index, action in enumerate(actions)
                if index > edit_indices[0] and action.name in VERIFY_TOOLS
            ),
            None,
        )
        report.checks.append(
            CheckResult(
                "verified_after_edit",
                verify_at is not None,
                "" if verify_at is not None else "never re-ran a test or command after the edit",
            )
        )

    if len(edit_indices) >= 2:
        hit = blind_retry(trace)
        report.checks.append(CheckResult("no_blind_retry", hit is None, hit[0] if hit else ""))

    error_positions = [index for index, entry in enumerate(timeline) if entry is _ERROR]
    if error_positions:
        recovered = True
        detail = ""
        for position in error_positions:
            previous = next(
                (
                    timeline[index]
                    for index in range(position - 1, -1, -1)
                    if isinstance(timeline[index], Action)
                ),
                None,
            )
            if previous is None:
                continue
            following = next(
                (
                    timeline[index]
                    for index in range(position + 1, len(timeline))
                    if isinstance(timeline[index], Action)
                ),
                None,
            )
            if following is None:
                recovered = False
                detail = "stopped operating after an error"
                break
            if following.name == previous.name and following.args_hash == previous.args_hash:
                recovered = False
                detail = f"repeated `{following.name}` unchanged after it failed"
                break
        report.checks.append(CheckResult("recovered_after_error", recovered, detail))

    report.checks.append(
        CheckResult(
            "outcome_recorded",
            trace.outcome.status is not OutcomeStatus.UNKNOWN,
            f"status={trace.outcome.status.value}",
        )
    )
    return report


def blind_retry(trace: AgentTrace) -> tuple[str, Action] | None:
    """Detect an identical edit re-run unchanged (with or without a failure between)."""
    seen: dict[str, bool] = {}
    for entry in _timeline(trace):
        if entry is _ERROR:
            for key in seen:
                seen[key] = True
            continue
        if entry.name not in EDIT_TOOLS:
            continue
        if entry.args_hash in seen:
            if seen[entry.args_hash]:
                return ("re-ran the identical edit after a failure", entry)
            return ("re-ran the identical edit unchanged", entry)
        seen[entry.args_hash] = False
    return None


def _timeline(trace: AgentTrace) -> list[Any]:
    """Interleave assistant tool calls (actions) with error results (markers)."""
    entries: list[Any] = []
    for event in trace.events:
        if event.type is EventType.ASSISTANT and event.tool_calls:
            for call in event.tool_calls:
                entries.append(_action_from_call(call))
        elif event.type is EventType.TOOL and event.is_error:
            entries.append(_ERROR)
    return entries


def _action_from_call(call: ToolCall) -> Action:
    path = (
        call.arguments.get("path")
        or call.arguments.get("file")
        or call.arguments.get("file_path")
        or ""
    )
    payload = json.dumps(
        {"name": call.name, "arguments": call.arguments}, sort_keys=True, default=str
    )
    return Action(
        name=call.name.lower(),
        args_hash=hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest(),
        path=str(path),
    )
