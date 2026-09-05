"""Deterministic behavioral grading for agent traces.

Grades HOW a trace operated — tool order, localization, verification,
recovery — not whether the final answer was right.

Checks are split into two bands:

* **Policy floor** — minimum operating hygiene (explored first, read before
  edit, verified after edit, no blind retry, recovered, outcome recorded).
  A model can clear this floor and still flail.
* **Competence ceiling** — richness / anti-flail signals (enough evidence
  before the first patch, focused edits, phase completeness, no edit thrash).
  This is what separates a disciplined frontier run from "explore once then
  thrash."

The combined ``score`` remains the mean of all fired checks (backward
compatible). Prefer ``policy_score`` / ``competence_score`` / ``grade_delta``
when using grade as an absolute instrument or a before/after pack delta.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from mercury.models import AgentTrace, EventType, OutcomeStatus, ToolCall
from mercury.phases import (
    is_edit_tool,
    is_evidence_tool,
    is_explore_tool,
    is_verify_tool,
    normalize_tool,
    segment_phases,
)

_ERROR = object()

POLICY_CHECKS = {
    "explored_first",
    "read_before_edit",
    "verified_after_edit",
    "no_blind_retry",
    "recovered_after_error",
    "outcome_recorded",
}
COMPETENCE_CHECKS = {
    "evidence_depth",
    "focused_edits",
    "phase_completeness",
    "no_edit_thrash",
}


@dataclass(frozen=True)
class Action:
    """One assistant-initiated tool call, fingerprinted by name + arguments."""

    name: str
    args_hash: str
    path: str = ""
    family: str = ""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    band: str = "policy"  # "policy" | "competence"


@dataclass
class BehaviorReport:
    model: str
    task: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean of all fired checks (floor + ceiling). Backward compatible."""
        if not self.checks:
            return 0.0
        return sum(1.0 for check in self.checks if check.passed) / len(self.checks)

    @property
    def policy_score(self) -> float:
        return self._band_score("policy")

    @property
    def competence_score(self) -> float:
        return self._band_score("competence")

    def _band_score(self, band: str) -> float:
        subset = [check for check in self.checks if check.band == band]
        if not subset:
            return 0.0
        return sum(1.0 for check in subset if check.passed) / len(subset)

    @property
    def passed_names(self) -> list[str]:
        return [check.name for check in self.checks if check.passed]

    @property
    def failed_names(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]

    def summary(self) -> str:
        lines = [
            (
                f"Behavior grade: {self.score:.0%} overall "
                f"(policy floor {self.policy_score:.0%}, "
                f"competence ceiling {self.competence_score:.0%}) — {self.model}"
            )
        ]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            suffix = f" — {check.detail}" if check.detail else ""
            lines.append(f"  [{mark}] ({check.band}) {check.name}{suffix}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "score": round(self.score, 4),
            "policy_score": round(self.policy_score, 4),
            "competence_score": round(self.competence_score, 4),
            "passed": self.passed_names,
            "failed": self.failed_names,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                    "band": check.band,
                }
                for check in self.checks
            ],
        }


@dataclass
class GradeDelta:
    """Before/after pack comparison — the intended use of grade as a delta instrument."""

    before: BehaviorReport
    after: BehaviorReport

    @property
    def score_delta(self) -> float:
        return self.after.score - self.before.score

    @property
    def policy_delta(self) -> float:
        return self.after.policy_score - self.before.policy_score

    @property
    def competence_delta(self) -> float:
        return self.after.competence_score - self.before.competence_score

    @property
    def newly_passed(self) -> list[str]:
        return sorted(set(self.after.passed_names) - set(self.before.passed_names))

    @property
    def newly_failed(self) -> list[str]:
        return sorted(set(self.before.passed_names) - set(self.after.passed_names))

    def summary(self) -> str:
        lines = [
            "Grade delta (after − before):",
            f"  overall     {self.score_delta:+.0%}  ({self.before.score:.0%} → {self.after.score:.0%})",
            f"  policy      {self.policy_delta:+.0%}  ({self.before.policy_score:.0%} → {self.after.policy_score:.0%})",
            f"  competence  {self.competence_delta:+.0%}  ({self.before.competence_score:.0%} → {self.after.competence_score:.0%})",
        ]
        if self.newly_passed:
            lines.append(f"  newly passed: {', '.join(self.newly_passed)}")
        if self.newly_failed:
            lines.append(f"  newly failed: {', '.join(self.newly_failed)}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "score_delta": round(self.score_delta, 4),
            "policy_delta": round(self.policy_delta, 4),
            "competence_delta": round(self.competence_delta, 4),
            "newly_passed": self.newly_passed,
            "newly_failed": self.newly_failed,
        }


def grade_trace(trace: AgentTrace) -> BehaviorReport:
    """Deterministically grade operational behavior. Pure function; stores nothing."""
    report = BehaviorReport(model=trace.model, task=trace.task)
    timeline = _timeline(trace)
    actions = [entry for entry in timeline if isinstance(entry, Action)]
    edit_indices = [index for index, action in enumerate(actions) if is_edit_tool(action.name)]

    if edit_indices:
        first = actions[0]
        report.checks.append(
            CheckResult(
                "explored_first",
                is_explore_tool(first.name),
                f"first tool was `{first.name}` (family={first.family})",
                band="policy",
            )
        )
        read_at = next(
            (index for index, action in enumerate(actions) if is_evidence_tool(action.name)),
            None,
        )
        report.checks.append(
            CheckResult(
                "read_before_edit",
                read_at is not None and read_at < edit_indices[0],
                "" if read_at is not None and read_at < edit_indices[0] else "edited before any search or read",
                band="policy",
            )
        )
        verify_at = next(
            (
                index
                for index, action in enumerate(actions)
                if index > edit_indices[0] and is_verify_tool(action.name)
            ),
            None,
        )
        report.checks.append(
            CheckResult(
                "verified_after_edit",
                verify_at is not None,
                "" if verify_at is not None else "never re-ran a test or command after the edit",
                band="policy",
            )
        )

        # Competence: more than a token explore before the first edit.
        evidence_before = sum(
            1 for action in actions[: edit_indices[0]] if is_evidence_tool(action.name)
        )
        report.checks.append(
            CheckResult(
                "evidence_depth",
                evidence_before >= 2,
                f"{evidence_before} search/read action(s) before first edit",
                band="competence",
            )
        )

        # Competence: edits stay focused on a small path set.
        edit_paths = [actions[i].path for i in edit_indices if actions[i].path]
        unique_paths = {path for path in edit_paths if path}
        focused = len(unique_paths) <= 3 if unique_paths else len(edit_indices) <= 3
        report.checks.append(
            CheckResult(
                "focused_edits",
                focused,
                f"{len(unique_paths) or len(edit_indices)} edit target(s)",
                band="competence",
            )
        )

        # Competence: no edit thrash — edits after the first verify should be rare.
        thrash = False
        thrash_detail = ""
        if verify_at is not None:
            edits_after_verify = sum(1 for i in edit_indices if i > verify_at)
            if edits_after_verify >= 3:
                thrash = True
                thrash_detail = f"{edits_after_verify} edits after first verify"
        if len(edit_indices) >= 5:
            thrash = True
            thrash_detail = thrash_detail or f"{len(edit_indices)} total edits"
        report.checks.append(
            CheckResult(
                "no_edit_thrash",
                not thrash,
                thrash_detail or "edit volume stayed bounded",
                band="competence",
            )
        )

    if len(edit_indices) >= 2:
        hit = blind_retry(trace)
        report.checks.append(
            CheckResult("no_blind_retry", hit is None, hit[0] if hit else "", band="policy")
        )

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
        report.checks.append(CheckResult("recovered_after_error", recovered, detail, band="policy"))

    # Competence: expected coding phase skeleton present.
    phase_names = {phase.name for phase in segment_phases(trace)}
    required = {"explore", "edit", "verify"}
    # localize is preferred but explore alone can satisfy evidence.
    has_evidence_phase = "explore" in phase_names or "localize" in phase_names
    complete = has_evidence_phase and "edit" in phase_names and "verify" in phase_names
    missing = sorted(required - phase_names)
    if not has_evidence_phase:
        missing = sorted(set(missing) | {"explore|localize"})
    report.checks.append(
        CheckResult(
            "phase_completeness",
            complete,
            "phases: " + (", ".join(sorted(phase_names)) or "(none)")
            + (f"; missing {', '.join(missing)}" if missing and not complete else ""),
            band="competence",
        )
    )

    report.checks.append(
        CheckResult(
            "outcome_recorded",
            trace.outcome.status is not OutcomeStatus.UNKNOWN,
            f"status={trace.outcome.status.value}",
            band="policy",
        )
    )
    return report


def grade_delta(before: AgentTrace | BehaviorReport, after: AgentTrace | BehaviorReport) -> GradeDelta:
    """Compare two grades — intended as a pack-effect delta instrument."""
    before_report = before if isinstance(before, BehaviorReport) else grade_trace(before)
    after_report = after if isinstance(after, BehaviorReport) else grade_trace(after)
    return GradeDelta(before=before_report, after=after_report)


def blind_retry(trace: AgentTrace) -> tuple[str, Action] | None:
    """Detect an identical edit re-run unchanged (with or without a failure between)."""
    seen: dict[str, bool] = {}
    for entry in _timeline(trace):
        if entry is _ERROR:
            for key in seen:
                seen[key] = True
            continue
        if not is_edit_tool(entry.name):
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
        family=normalize_tool(call.name),
    )
