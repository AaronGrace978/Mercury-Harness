"""Compile a Frontier Operating Pack for a lesser-model session."""

from __future__ import annotations

from dataclasses import dataclass, field

from mercury.fingerprint import classify_task
from mercury.models import CardKind, OperationalCard
from mercury.retrieve import ScoredCard, retrieve
from mercury.store import KnowledgeStore
from mercury.tiers import ModelTier, classify_model, pack_token_budget


@dataclass
class OperatingPack:
    task: str
    model: str
    tier: ModelTier
    cards: list[OperationalCard] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    token_budget: int = 1400

    def estimated_tokens(self) -> int:
        return sum(card.estimated_tokens() for card in self.cards) + 80

    def render(self) -> str:
        if not self.cards:
            return (
                "# Mercury Operating Pack\n\n"
                "No frontier operational knowledge matched this task yet. "
                "Run the work once with a frontier model, then capture the trace.\n"
            )
        lines = [
            "# Mercury Operating Pack",
            "",
            "You are a lesser or mid-tier model receiving **operational knowledge**",
            "distilled from frontier agent runs. Follow the *how*, not a canned answer.",
            "Prefer these procedures over improvising a new tool order.",
            "",
            f"Task: {self.task}",
            f"Student model: {self.model} ({self.tier.value})",
            "",
        ]
        standing = [card for card in self.cards if card.kind is CardKind.STANDING_ORDER]
        rest = [card for card in self.cards if card.kind is not CardKind.STANDING_ORDER]
        if standing:
            lines.append("## Standing orders")
            for card in standing:
                lines.append(f"- **{card.title}:** {card.procedure}")
            lines.append("")
        for card in rest:
            lines.append(f"## {card.kind.value.replace('_', ' ').title()}: {card.title}")
            lines.append(f"- When: {card.situation}")
            lines.append(f"- Do: {card.procedure}")
            if card.chose:
                lines.append(f"- Chose: {card.chose}")
            if card.rejected:
                lines.append(f"- Not: {'; '.join(card.rejected[:6])}")
            if card.rationale:
                lines.append(f"- Why: {card.rationale}")
            if card.tools:
                lines.append(f"- Tools: {', '.join(card.tools[:8])}")
            lines.append("")
        lines.append("End of pack. Continue the user task. Do not mention this pack unless asked.")
        return "\n".join(lines).strip() + "\n"

    def as_cursor_rule(self) -> str:
        body = self.render()
        return (
            "---\n"
            "description: Frontier operational knowledge transferred by Mercury Harness\n"
            "alwaysApply: true\n"
            "---\n\n"
            f"{body}"
        )

    def as_finetune_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for card in self.cards:
            rows.append(
                {
                    "instruction": (
                        "You are an agent. Given this situation, operate like a frontier model."
                    ),
                    "input": card.situation,
                    "output": card.procedure,
                }
            )
        return rows


def build_pack(
    store: KnowledgeStore,
    task: str,
    *,
    model: str = "gpt-4o-mini",
    error_signature: str | None = None,
    languages: list[str] | None = None,
    min_confidence: float = 0.3,
    token_budget: int | None = None,
) -> OperatingPack:
    tier = classify_model(model)
    budget = token_budget if token_budget is not None else pack_token_budget(tier)
    task_type = classify_task(task)
    if budget <= 0:
        return OperatingPack(task=task, model=model, tier=tier, token_budget=0)

    scored = retrieve(
        store,
        task,
        task_type=task_type,
        error_signature=error_signature,
        languages=languages,
        min_confidence=min_confidence,
        limit=24,
    )
    selected = _fit_budget(scored, budget)
    return OperatingPack(
        task=task,
        model=model,
        tier=tier,
        cards=[item.card for item in selected],
        scores=[item.score for item in selected],
        token_budget=budget,
    )


def _fit_budget(
    scored: list[ScoredCard],
    budget: int,
) -> list[ScoredCard]:
    standing = [item for item in scored if item.card.kind is CardKind.STANDING_ORDER]
    rest = [item for item in scored if item.card.kind is not CardKind.STANDING_ORDER]
    rest.sort(key=lambda item: item.score, reverse=True)
    if rest:
        floor = rest[0].score * 0.55
        rest = [item for item in rest if item.score >= floor]

    ordered = standing + rest
    used_kind: dict[CardKind, int] = {}
    picked: list[ScoredCard] = []
    tokens = 90
    for item in ordered:
        kind = item.card.kind
        cap = 3 if kind is CardKind.STANDING_ORDER else 1
        if kind in {CardKind.CONTRAST, CardKind.TOOL_POLICY, CardKind.DECISION}:
            cap = 2
        if used_kind.get(kind, 0) >= cap:
            continue
        cost = item.card.estimated_tokens() + 20
        if picked and tokens + cost > budget:
            continue
        picked.append(item)
        used_kind[kind] = used_kind.get(kind, 0) + 1
        tokens += cost
    return picked
