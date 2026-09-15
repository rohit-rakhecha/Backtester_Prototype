"""Stage 08 -- Strategy library + deployment rail.

A persistent, queryable research graph (SQLite): every Card, code hash, dataset snapshot,
result, decision, and failed idea, plus the promotion ladder state machine required by the
board memo:

    REPLICATED -> INDIA_VALIDATED -> ROBUST -> ORTHOGONAL -> PORTFOLIO_USEFUL
    -> PAPER_TRADED -> LIVE

Economic intuition: this is what turns "20 papers screened/day" into "1 portfolio-useful
signal / few weeks" instead of twenty forgotten spreadsheets (memo's North Star). Rejected
and failed ideas are stored with the same rigor as promoted ones -- "treat negative results
and failed replications as reusable knowledge" -- so the next researcher with the same idea
can find out in one query, not by re-running a week of backtests.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class Rung(str, Enum):
    replicated = "REPLICATED"
    india_validated = "INDIA_VALIDATED"
    robust = "ROBUST"
    orthogonal = "ORTHOGONAL"
    portfolio_useful = "PORTFOLIO_USEFUL"
    paper_traded = "PAPER_TRADED"
    live = "LIVE"
    rejected = "REJECTED"

    @classmethod
    def order(cls) -> list["Rung"]:
        return [
            cls.replicated, cls.india_validated, cls.robust, cls.orthogonal,
            cls.portfolio_useful, cls.paper_traded, cls.live,
        ]


SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    card_id TEXT PRIMARY KEY,
    title TEXT,
    card_hash TEXT,
    current_rung TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id TEXT,
    event_type TEXT,          -- 'promotion' | 'rejection' | 'result'
    from_rung TEXT,
    to_rung TEXT,
    dataset_snapshot_hash TEXT,
    code_hash TEXT,
    model_version TEXT,
    overrides TEXT,           -- JSON
    evidence TEXT,            -- JSON
    reviewer TEXT,
    note TEXT,
    timestamp TEXT,
    FOREIGN KEY(card_id) REFERENCES strategies(card_id)
);
"""


class StrategyLibrary:
    def __init__(self, db_path: str = "data/lineage/strategy_library.db"):
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def register(self, card_id: str, title: str, card_hash: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO strategies (card_id, title, card_hash, current_rung, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (card_id, title, card_hash, None, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def current_rung(self, card_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT current_rung FROM strategies WHERE card_id = ?", (card_id,)
        ).fetchone()
        return row[0] if row else None

    def promote(
        self,
        card_id: str,
        to_rung: Rung,
        dataset_snapshot_hash: str,
        code_hash: str,
        model_version: str,
        reviewer: str,
        note: str,
        evidence: dict,
        overrides: dict | None = None,
        allow_skip: bool = False,
    ) -> None:
        """Advance a strategy on the promotion ladder. Refuses to skip rungs unless
        `allow_skip=True` is explicitly passed with a reason in `note` -- Stage 08 red flag:
        'more than one rung skipped in a single promotion' (docs/PIPELINE.md)."""
        current = self.current_rung(card_id)
        order = Rung.order()
        current_idx = -1 if current is None else order.index(Rung(current))
        target_idx = order.index(to_rung)
        if not allow_skip and target_idx > current_idx + 1:
            raise ValueError(
                f"Refusing to promote {card_id} from '{current}' directly to '{to_rung.value}' "
                f"-- skips {target_idx - current_idx - 1} rung(s). Pass allow_skip=True with a "
                "documented reason if this is intentional."
            )
        self.conn.execute(
            "INSERT INTO events (card_id, event_type, from_rung, to_rung, dataset_snapshot_hash, "
            "code_hash, model_version, overrides, evidence, reviewer, note, timestamp) "
            "VALUES (?, 'promotion', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                card_id, current, to_rung.value, dataset_snapshot_hash, code_hash, model_version,
                json.dumps(overrides or {}), json.dumps(evidence, default=str), reviewer, note,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.execute(
            "UPDATE strategies SET current_rung = ? WHERE card_id = ?", (to_rung.value, card_id)
        )
        self.conn.commit()

    def reject(self, card_id: str, reviewer: str, note: str, evidence: dict) -> None:
        """Records a rejection as reusable knowledge -- NOT a deletion (Stage 08 red flag:
        'a rejected/failed idea is deleted rather than archived with its reasoning')."""
        if not note.strip():
            raise ValueError("A rejection must record its reasoning (Gate B red flags).")
        current = self.current_rung(card_id)
        self.conn.execute(
            "INSERT INTO events (card_id, event_type, from_rung, to_rung, reviewer, note, evidence, timestamp) "
            "VALUES (?, 'rejection', ?, ?, ?, ?, ?, ?)",
            (card_id, current, Rung.rejected.value, reviewer, note, json.dumps(evidence, default=str),
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.execute(
            "UPDATE strategies SET current_rung = ? WHERE card_id = ?", (Rung.rejected.value, card_id)
        )
        self.conn.commit()

    def history(self, card_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT event_type, from_rung, to_rung, reviewer, note, timestamp FROM events "
            "WHERE card_id = ? ORDER BY id", (card_id,)
        ).fetchall()
        return [
            {"event_type": r[0], "from_rung": r[1], "to_rung": r[2], "reviewer": r[3],
             "note": r[4], "timestamp": r[5]}
            for r in rows
        ]

    def all_strategies(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT card_id, title, current_rung, created_at FROM strategies ORDER BY created_at"
        ).fetchall()
        return [
            {"card_id": r[0], "title": r[1], "current_rung": r[2], "created_at": r[3]}
            for r in rows
        ]
