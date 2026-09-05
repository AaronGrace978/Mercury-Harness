"""Hybrid retrieval: hashing cosine + BM25 with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from mercury.embed import BM25Index, HashingEmbedder, cosine, tokenize
from mercury.models import CardKind, OperationalCard, TaskType
from mercury.store import KnowledgeStore


@dataclass
class ScoredCard:
    card: OperationalCard
    score: float
    dense: float
    lexical: float


def retrieve(
    store: KnowledgeStore,
    query: str,
    *,
    embedder: HashingEmbedder | None = None,
    task_type: TaskType | None = None,
    kinds: tuple[CardKind, ...] | None = None,
    error_signature: str | None = None,
    languages: list[str] | None = None,
    min_confidence: float = 0.3,
    limit: int = 8,
    dense_weight: float = 0.6,
) -> list[ScoredCard]:
    rows = store.cards_with_embeddings()
    if not rows:
        return []
    embedder = embedder or HashingEmbedder()
    query_vec = embedder.embed(_query_text(query, error_signature, languages))
    texts = [card.searchable_text() for card, _ in rows]
    bm25 = BM25Index()
    bm25.fit(texts)
    lexical_scores = bm25.score(_query_text(query, error_signature, languages))

    dense_list: list[float] = []
    for card, embedding in rows:
        vector = embedding if embedding is not None else embedder.embed(card.searchable_text())
        dense_list.append(cosine(query_vec, vector))

    fused = _rrf(dense_list, lexical_scores, dense_weight=dense_weight)
    scored: list[ScoredCard] = []
    for index, ((card, _), score) in enumerate(zip(rows, fused)):
        if card.confidence < min_confidence:
            continue
        if kinds and card.kind not in kinds:
            continue
        if not _task_overlap(_query_text(query, error_signature, languages), card) and card.kind is not CardKind.STANDING_ORDER:
            continue
        bonus = 0.0
        if task_type and card.task_type is task_type:
            bonus += 0.01
        if languages and any(lang in card.languages for lang in languages):
            bonus += 0.008
        if error_signature and card.error_signature:
            if error_signature[:40] in card.error_signature or card.error_signature[:40] in error_signature:
                bonus += 0.045
        scored.append(
            ScoredCard(
                card=card,
                score=(score * 10.0) + bonus + (card.confidence * 0.02),
                dense=dense_list[index],
                lexical=lexical_scores[index],
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


def _task_overlap(query: str, card: OperationalCard) -> bool:
    """Keep cards whose situation shares mass with the student task."""
    query_prefixes = _prefixes(query)
    if not query_prefixes:
        return True
    blob = " ".join(
        [
            card.situation,
            card.title,
            " ".join(card.languages),
            card.error_signature or "",
            card.task_type.value,
        ]
    )
    card_prefixes = _prefixes(blob)
    return len(query_prefixes & card_prefixes) >= 2


def _prefixes(text: str) -> set[str]:
    return {token[:5] for token in tokenize(text) if len(token) >= 4}


def _query_text(query: str, error_signature: str | None, languages: list[str] | None) -> str:
    parts = [query]
    if error_signature:
        parts.append(error_signature)
    if languages:
        parts.extend(languages)
    return " ".join(parts)


def _rrf(dense: list[float], lexical: list[float], *, dense_weight: float, k: int = 20) -> list[float]:
    dense_rank = _ranks(dense)
    lex_rank = _ranks(lexical)
    fused: list[float] = []
    lexical_weight = 1.0 - dense_weight
    for index in range(len(dense)):
        fused.append(
            dense_weight * (1.0 / (k + dense_rank[index]))
            + lexical_weight * (1.0 / (k + lex_rank[index]))
        )
    return fused


def _ranks(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks
