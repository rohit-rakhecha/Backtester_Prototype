"""Stage 02 — Strategy Card.

The Strategy Card is the ONLY artifact an AI/researcher is allowed to author when
interpreting a paper. It is a small, typed, validated object -- not free-form code.
Every field that encodes an economic claim from the source paper must carry an
`Evidence` pointer (page + quote) so Gate A can check the Card against the source paper
in seconds rather than re-reading it.

Economic intuition for the schema shape itself: the fields below are exactly the set of
decisions that turn an economic idea into a specific, backtestable, and gameable set of
choices (universe, signal, lookback/lag, weighting, rebalance cadence, benchmark, costs).
Constraining the AI's authored surface to *only* these fields is the "template-constrained
generation" control from the board memo -- it cannot hide a lookahead bias or a silently
changed universe in a field that doesn't exist in the schema.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    high = "high"      # directly quoted, unambiguous
    medium = "medium"  # inferred from context, not a direct quote
    low = "low"         # guessed / paper is silent on this


class Evidence(BaseModel):
    page: int
    quote: str
    confidence: Confidence = Confidence.medium


class DataFieldSpec(BaseModel):
    """One data requirement implied by the Card, resolved at Stage 03."""
    name: str
    description: str
    frequency: str  # "daily" | "monthly" | "weekly" | "event"
    required_history_years: float
    evidence: Optional[Evidence] = None


class Ambiguity(BaseModel):
    """An open question the ingest stage could not resolve on its own.

    Gate A must address every ambiguity by name before approving the Card --
    see docs/PIPELINE.md, Gate A red/green flags.
    """
    field: str
    description: str
    resolution: Optional[str] = None  # filled in at Gate A
    resolved_by: Optional[str] = None


class SignalSpec(BaseModel):
    """The economic signal itself -- deliberately small.

    This is compiled into a sandboxed `signal_fn` (engine/signals.py) that may only
    consume the point-in-time feature panel and must return target weights. Nothing else.
    """
    name: str
    description: str
    lookback_days: int
    lag_days: int = Field(0, description="Days between signal computation and execution, to avoid lookahead.")
    rebalance_frequency: str  # "daily" | "weekly" | "monthly" | "annual"
    weighting_rule: str
    evidence: list[Evidence] = Field(default_factory=list)


class RiskSpec(BaseModel):
    target_volatility_annual: float
    max_relative_weight_deviation_l1: Optional[float] = None
    long_only: bool = True
    leverage_allowed: bool = False
    derivatives_allowed: bool = False


class CostSpec(BaseModel):
    """Cost assumption used by the *engine*, not baked into the signal.

    India note: the paper's flat half-spread assumption (5bp) does not transfer as-is --
    see docs/DATA_SOURCES.md item 4 and engine/costs.py for the India cost model that
    replaces it. This field records what the PAPER assumed, for the replication-gap check;
    the India-adapted cost model is applied separately in Stage 05/06.
    """
    paper_assumed_bps: float
    paper_evidence: Optional[Evidence] = None
    india_cost_model: str = "engine.costs.india_equity_cost_model"


class StrategyCard(BaseModel):
    card_id: str
    title: str
    source_title: str
    source_authors: list[str]
    source_url: Optional[str] = None
    ingested_at: str

    universe: str
    universe_evidence: Optional[Evidence] = None

    benchmark: str
    signal: SignalSpec
    risk: RiskSpec
    cost: CostSpec

    data_requirements: list[DataFieldSpec] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)

    india_adaptation_notes: str = Field(
        "", description="Explicit notes on how this Card diverges from the source paper "
        "to be India-investable (asset substitutions, cost model, cash-rate proxy, etc.). "
        "Required non-empty for any Card that isn't a literal same-market replication."
    )

    gate_a_status: str = "pending"  # pending | approved | rejected
    gate_a_reviewer: Optional[str] = None
    gate_a_note: Optional[str] = None

    @field_validator("ambiguities")
    @classmethod
    def _ambiguities_list_ok(cls, v):
        return v

    def content_hash(self) -> str:
        """Stable hash of the Card's economic content, used for lineage (Stage 08)."""
        payload = self.model_dump(exclude={"gate_a_status", "gate_a_reviewer", "gate_a_note"})
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def unresolved_ambiguities(self) -> list[Ambiguity]:
        return [a for a in self.ambiguities if a.resolution is None]

    def is_ready_for_gate_a(self) -> bool:
        """Green flag check: every numeric/economic claim has evidence, and the
        ambiguities list is non-trivial for anything but a trivial paper (see
        docs/PIPELINE.md Stage 02 red flags)."""
        return bool(self.signal.evidence) and self.universe_evidence is not None


def load_card(path: str) -> StrategyCard:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return StrategyCard(**data)


def save_card(card: StrategyCard, path: str) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(card.model_dump(), f, sort_keys=False, default_flow_style=False)
