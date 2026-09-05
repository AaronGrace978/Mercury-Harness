"""SQLite-backed operational knowledge store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mercury.models import AgentTrace, OperationalCard
from mercury.traceio import parse_trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    tier TEXT NOT NULL,
    task TEXT NOT NULL,
    outcome TEXT NOT NULL,
    json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    situation TEXT NOT NULL,
    procedure TEXT NOT NULL,
    rationale TEXT NOT NULL,
    source_trace_id TEXT NOT NULL,
    source_model TEXT NOT NULL,
    confidence REAL NOT NULL,
    json TEXT NOT NULL,
    embedding TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cards_kind ON cards(kind);
CREATE INDEX IF NOT EXISTS idx_cards_confidence ON cards(confidence);
"""


class KnowledgeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "mercury.sqlite"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def put_trace(self, trace: AgentTrace) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO traces (id, model, tier, task, outcome, json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model = excluded.model,
                tier = excluded.tier,
                task = excluded.task,
                outcome = excluded.outcome,
                json = excluded.json
            """,
            (
                trace.id,
                trace.model,
                trace.tier.value,
                trace.task,
                trace.outcome.status.value,
                trace.model_dump_json(),
                now,
            ),
        )
        self._conn.commit()

    def get_trace(self, trace_id: str) -> AgentTrace | None:
        row = self._conn.execute("SELECT json FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        return parse_trace(json.loads(row["json"]))

    def traces(self) -> list[AgentTrace]:
        rows = self._conn.execute("SELECT json FROM traces ORDER BY created_at").fetchall()
        return [parse_trace(json.loads(row["json"])) for row in rows]

    def put_card(self, card: OperationalCard, embedding: list[float] | None = None) -> None:
        now = card.created_at
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cards (
                id, kind, title, situation, procedure, rationale,
                source_trace_id, source_model, confidence, json, embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card.id,
                card.kind.value,
                card.title,
                card.situation,
                card.procedure,
                card.rationale,
                card.source_trace_id,
                card.source_model,
                card.confidence,
                card.model_dump_json(),
                json.dumps(embedding) if embedding is not None else None,
                now,
            ),
        )
        self._conn.commit()

    def put_cards(self, cards: Iterable[OperationalCard], embeddings: dict[str, list[float]] | None = None) -> int:
        count = 0
        embeddings = embeddings or {}
        for card in cards:
            self.put_card(card, embeddings.get(card.id))
            count += 1
        return count

    def cards(self) -> list[OperationalCard]:
        rows = self._conn.execute("SELECT json FROM cards ORDER BY confidence DESC").fetchall()
        return [OperationalCard.model_validate_json(row["json"]) for row in rows]

    def cards_with_embeddings(self) -> list[tuple[OperationalCard, list[float] | None]]:
        rows = self._conn.execute("SELECT json, embedding FROM cards ORDER BY confidence DESC").fetchall()
        out: list[tuple[OperationalCard, list[float] | None]] = []
        for row in rows:
            card = OperationalCard.model_validate_json(row["json"])
            embedding = json.loads(row["embedding"]) if row["embedding"] else None
            out.append((card, embedding))
        return out

    def stats(self) -> dict[str, int | float]:
        trace_count = self._conn.execute("SELECT COUNT(*) AS n FROM traces").fetchone()["n"]
        card_count = self._conn.execute("SELECT COUNT(*) AS n FROM cards").fetchone()["n"]
        teachers = self._conn.execute(
            "SELECT COUNT(*) AS n FROM traces WHERE tier = 'frontier'"
        ).fetchone()["n"]
        by_kind = {
            row["kind"]: row["n"]
            for row in self._conn.execute(
                "SELECT kind, COUNT(*) AS n FROM cards GROUP BY kind"
            )
        }
        return {
            "traces": trace_count,
            "cards": card_count,
            "frontier_traces": teachers,
            "by_kind": by_kind,
        }
