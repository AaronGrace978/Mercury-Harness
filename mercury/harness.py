"""Facade: capture frontier operations, embed them, inject into lesser models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mercury.contrast import contrast_traces
from mercury.distill import distill_standing_orders, distill_trace
from mercury.embed import HashingEmbedder
from mercury.inject import OperatingPack, build_pack
from mercury.models import AgentTrace, OperationalCard
from mercury.store import KnowledgeStore
from mercury.tiers import ModelTier, is_teacher_tier
from mercury.traceio import load_trace, parse_trace


class MercuryHarness:
    """The flywheel.

    1. Capture a frontier agent trace.
    2. Distill *how it operated* into cards.
    3. Embed those cards in a local store.
    4. At lesser-model session start, inject a Frontier Operating Pack.
    """

    def __init__(
        self,
        path: str | Path = ".mercury",
        *,
        teacher_tiers: tuple[ModelTier, ...] = (ModelTier.FRONTIER,),
        student_tiers: tuple[ModelTier, ...] = (ModelTier.LESSER, ModelTier.CAPABLE),
        allow_student_success: bool = False,
        min_confidence: float = 0.3,
    ) -> None:
        self.store = KnowledgeStore(path)
        self.embedder = HashingEmbedder()
        self.teacher_tiers = teacher_tiers
        self.student_tiers = student_tiers
        self.allow_student_success = allow_student_success
        self.min_confidence = min_confidence

    @classmethod
    def init(cls, path: str | Path = ".mercury", **kwargs: Any) -> "MercuryHarness":
        Path(path).mkdir(parents=True, exist_ok=True)
        return cls(path, **kwargs)

    def capture(
        self,
        source: str | Path | dict[str, Any] | AgentTrace,
        *,
        force_teacher: bool = False,
    ) -> AgentTrace:
        trace = source if isinstance(source, AgentTrace) else load_trace(source) if not isinstance(source, dict) else parse_trace(source)
        self.store.put_trace(trace)
        if self._should_teach(trace, force_teacher=force_teacher):
            self._embed_cards(distill_trace(trace, teacher=True))
        elif trace.outcome.failed:
            # Failures still donate recovery / anti-pattern knowledge.
            self._embed_cards(distill_trace(trace, teacher=False))
        self._refresh_standing_orders()
        return trace

    def contrast(
        self,
        student_source: str | Path | dict[str, Any] | AgentTrace,
        teacher_source: str | Path | dict[str, Any] | AgentTrace,
    ) -> list[OperationalCard]:
        student = student_source if isinstance(student_source, AgentTrace) else (
            parse_trace(student_source) if isinstance(student_source, dict) else load_trace(student_source)
        )
        teacher = teacher_source if isinstance(teacher_source, AgentTrace) else (
            parse_trace(teacher_source) if isinstance(teacher_source, dict) else load_trace(teacher_source)
        )
        self.store.put_trace(student)
        self.store.put_trace(teacher)
        cards = contrast_traces(student, teacher)
        if self._should_teach(teacher, force_teacher=True):
            cards = list(cards) + distill_trace(teacher, teacher=True)
        self._embed_cards(cards)
        self._refresh_standing_orders()
        return cards

    def pack(
        self,
        task: str,
        *,
        model: str = "gpt-4o-mini",
        error_signature: str | None = None,
        languages: list[str] | None = None,
    ) -> OperatingPack:
        return build_pack(
            self.store,
            task,
            model=model,
            error_signature=error_signature,
            languages=languages,
            min_confidence=self.min_confidence,
        )

    def distill_all(self) -> int:
        cards: list[OperationalCard] = []
        traces = self.store.traces()
        for trace in traces:
            teach = self._should_teach(trace)
            cards.extend(distill_trace(trace, teacher=teach))
        cards.extend(distill_standing_orders([t for t in traces if is_teacher_tier(t.tier, self.teacher_tiers)]))
        self._embed_cards(cards)
        return len(cards)

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def close(self) -> None:
        self.store.close()

    def _should_teach(self, trace: AgentTrace, *, force_teacher: bool = False) -> bool:
        if force_teacher:
            return True
        if is_teacher_tier(trace.tier, self.teacher_tiers) and (trace.outcome.succeeded or trace.outcome.status.value == "partial"):
            return True
        if is_teacher_tier(trace.tier, self.teacher_tiers) and not trace.outcome.failed:
            # Unknown outcome on a frontier run still teaches procedure.
            return True
        if self.allow_student_success and trace.outcome.succeeded:
            return True
        return False

    def _embed_cards(self, cards: list[OperationalCard]) -> None:
        for card in cards:
            if card.confidence < self.min_confidence:
                continue
            embedding = self.embedder.embed(card.searchable_text())
            self.store.put_card(card, embedding)

    def _refresh_standing_orders(self) -> None:
        teachers = [
            trace
            for trace in self.store.traces()
            if is_teacher_tier(trace.tier, self.teacher_tiers)
        ]
        for card in distill_standing_orders(teachers):
            embedding = self.embedder.embed(card.searchable_text())
            self.store.put_card(card, embedding)
