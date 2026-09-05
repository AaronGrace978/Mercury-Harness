"""Contrastive distillation: lesser failure vs frontier success on a similar task."""

from __future__ import annotations

from mercury.models import AgentTrace, CardKind, OperationalCard, card_id
from mercury.phases import first_tool, normalize_tool, segment_phases


def contrast_traces(student: AgentTrace, teacher: AgentTrace) -> list[OperationalCard]:
    """Where a lesser model diverged, emit 'don't do X, do Y' cards.

    Contrast is the strongest decision-function signal we have: the student
    documents what was chosen *against*, the teacher documents the replacement.
    """
    cards: list[OperationalCard] = []
    student_first = first_tool(student)
    teacher_first = first_tool(teacher)
    if (
        student_first
        and teacher_first
        and normalize_tool(student_first) != normalize_tool(teacher_first)
    ):
        cards.append(
            OperationalCard(
                id=card_id("contrast-first", student.id, teacher.id, teacher_first),
                kind=CardKind.CONTRAST,
                title="Open like the frontier model, not the lesser one",
                situation=teacher.task,
                procedure=(
                    f"Lesser models often start with `{student_first}`. "
                    f"The frontier model started with `{teacher_first}` and succeeded. "
                    f"Start with `{teacher_first}`."
                ),
                rationale="Contrastive first-action divergence on a matched task.",
                tools=[teacher_first],
                chose=f"start with `{teacher_first}`",
                rejected=[f"start with `{student_first}`"],
                task_type=teacher.task_type,
                source_trace_id=teacher.id,
                source_model=teacher.model,
                confidence=0.88 if teacher.outcome.succeeded and student.outcome.failed else 0.62,
                languages=list(teacher.languages),
                metadata={"student_trace_id": student.id, "student_model": student.model},
            )
        )

    student_phases = [phase.name for phase in segment_phases(student)]
    teacher_phases = [phase.name for phase in segment_phases(teacher)]
    if teacher_phases and student_phases != teacher_phases:
        cards.append(
            OperationalCard(
                id=card_id("contrast-phase", student.id, teacher.id, "|".join(teacher_phases)),
                kind=CardKind.CONTRAST,
                title="Follow the frontier phase order",
                situation=teacher.task,
                procedure=(
                    f"Avoid this lesser-model phase order: {' → '.join(student_phases) or '(none)'}. "
                    f"Use this frontier order: {' → '.join(teacher_phases)}."
                ),
                rationale="Phase-level contrast between a failed student run and a successful teacher run.",
                tools=teacher.tool_sequence()[:8],
                chose=" → ".join(teacher_phases),
                rejected=[" → ".join(student_phases) or "(no phases)"],
                task_type=teacher.task_type,
                source_trace_id=teacher.id,
                source_model=teacher.model,
                confidence=0.84 if teacher.outcome.succeeded else 0.5,
                languages=list(teacher.languages),
                metadata={"student_trace_id": student.id, "student_phases": student_phases},
            )
        )

    student_files = set(student.files_touched)
    teacher_files = set(teacher.files_touched)
    extra = sorted(student_files - teacher_files)
    missing = sorted(teacher_files - student_files)
    if extra or missing:
        bits: list[str] = []
        if extra:
            bits.append("Do not start in " + ", ".join(_base(path) for path in extra[:4]))
        if missing:
            bits.append("The frontier model actually touched " + ", ".join(_base(path) for path in missing[:4]))
        cards.append(
            OperationalCard(
                id=card_id("contrast-files", student.id, teacher.id, ",".join(missing[:6])),
                kind=CardKind.CONTRAST,
                title="Touch the same files the frontier model touched",
                situation=teacher.task,
                procedure=". ".join(bits) + ".",
                rationale="File-set mismatch on a contrastive pair.",
                tools=["read", "grep"],
                chose=", ".join(_base(path) for path in missing[:4]) if missing else "frontier file set",
                rejected=[_base(path) for path in extra[:4]],
                task_type=teacher.task_type,
                source_trace_id=teacher.id,
                source_model=teacher.model,
                confidence=0.7,
                languages=list(teacher.languages),
                metadata={"student_trace_id": student.id},
            )
        )
    return cards


def _base(path: str) -> str:
    return f"`{path.split('/')[-1]}`"
