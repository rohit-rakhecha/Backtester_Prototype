"""Gate A / Gate B -- human sign-off, logged immutably.

Both gates write to the same append-only JSONL audit log. Nothing in this module lets a
prior entry be edited or deleted -- the log is the institutional memory the promotion
ladder (Stage 08) is built on, and an editable log would defeat the "No result without
lineage" control.

Gate A (owner: researcher) approves the ECONOMIC DEFINITION in a Strategy Card.
Gate B (owner: PM/IC) makes the INVESTMENT DECISION on validated results.
Keeping them as two distinct gate types with two distinct owner roles is deliberate --
see docs/PIPELINE.md, Gate A/B sections, for why conflating them creates a bottleneck.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum


class GateBDecision(str, Enum):
    reject = "reject"
    fix = "fix"
    observe = "observe"
    promote = "promote"


@dataclass
class GateAEntry:
    gate: str
    card_id: str
    card_hash: str
    reviewer: str
    decision: str  # approved | rejected
    note: str
    ambiguities_addressed: list[str]
    timestamp: str


@dataclass
class GateBEntry:
    gate: str
    card_id: str
    result_id: str
    reviewer: str
    decision: str  # GateBDecision
    note: str
    dataset_snapshot_hash: str
    code_hash: str
    model_version: str
    overrides: dict
    evidence_summary: dict
    timestamp: str


class AuditLog:
    def __init__(self, path: str = "data/lineage/audit_log.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _append(self, entry: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def gate_a(self, card, reviewer: str, decision: str, note: str) -> GateAEntry:
        if decision == "approved" and card.unresolved_ambiguities():
            raise ValueError(
                "Gate A cannot approve a Card with unresolved ambiguities -- "
                f"{len(card.unresolved_ambiguities())} still open. "
                "Resolve card.ambiguities[i].resolution first (see docs/PIPELINE.md, "
                "Gate A red flags: 'approves without reading the ambiguities list')."
            )
        entry = GateAEntry(
            gate="A",
            card_id=card.card_id,
            card_hash=card.content_hash(),
            reviewer=reviewer,
            decision=decision,
            note=note,
            ambiguities_addressed=[a.field for a in card.ambiguities],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        card.gate_a_status = decision
        card.gate_a_reviewer = reviewer
        card.gate_a_note = note
        self._append(asdict(entry))
        return entry

    def gate_b(
        self,
        card_id: str,
        result_id: str,
        reviewer: str,
        decision: GateBDecision,
        note: str,
        dataset_snapshot_hash: str,
        code_hash: str,
        model_version: str,
        evidence_summary: dict,
        overrides: dict | None = None,
    ) -> GateBEntry:
        if not note.strip():
            raise ValueError(
                "Gate B decision requires a non-empty note -- unreasoned rejections/"
                "promotions throw away the 'reusable knowledge' the memo requires "
                "(docs/PIPELINE.md, Gate B red flags)."
            )
        entry = GateBEntry(
            gate="B",
            card_id=card_id,
            result_id=result_id,
            reviewer=reviewer,
            decision=decision.value if isinstance(decision, GateBDecision) else decision,
            note=note,
            dataset_snapshot_hash=dataset_snapshot_hash,
            code_hash=code_hash,
            model_version=model_version,
            overrides=overrides or {},
            evidence_summary=evidence_summary,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._append(asdict(entry))
        return entry

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
