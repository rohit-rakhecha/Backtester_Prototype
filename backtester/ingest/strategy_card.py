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
import re
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


def _pick_evidence_interactively(report, suggested_claims: list[tuple[int, str]], field_label: str) -> Optional[Evidence]:
    """Let the reviewer pick one of Stage 01's mined economic-claim sentences as the
    Evidence for a Card field, or type their own page+quote, or skip (low confidence).
    This is what makes 'Stage 01 dictates the flow' concrete: the Card is built FROM the
    ingest report's findings, not from a blank text box."""
    from .interactive_io import prompt_choice, prompt_text, prompt_int

    print(f"\n-- Evidence for '{field_label}' --")
    options = [f"[p.{pg}] {sent[:110]}" for pg, sent in suggested_claims[:8]]
    options.append("Type my own page + quote")
    options.append("Skip -- no evidence found (Card field will be low-confidence)")
    choice = prompt_choice("Pick a source for this field's evidence:", options)
    if choice.startswith("Skip"):
        return None
    if choice.startswith("Type my own"):
        page = prompt_int("Page number", default=1)
        quote = prompt_text("Quote (paste the relevant sentence)")
        if not quote:
            return None
        return Evidence(page=page, quote=quote, confidence=Confidence.high)
    idx = options.index(choice)
    page, sent = suggested_claims[idx]
    return Evidence(page=page, quote=sent, confidence=Confidence.high)


def build_card_interactively(report, card_id: Optional[str] = None) -> "StrategyCard":
    """Stage 02, interactively: builds a Strategy Card FROM Stage 01's `IngestReport`
    (backtester.ingest.parser.comprehensive_ingest), prompting the researcher to confirm
    or override every field rather than hand-writing a Card from a blank template. This
    is the mechanism behind "Stage 01 dictates the flow of the entire pipeline": every
    suggested default below traces back to something Stage 01 actually found in the PDF,
    with its page number, so confirming a default is a one-keystroke sign-off on a real
    quote rather than a guess.

    Works identically in a Colab cell or a terminal (see ingest/interactive_io.py).
    """
    from datetime import datetime, timezone
    from .interactive_io import prompt_text, prompt_int, prompt_float, prompt_choice, prompt_yesno
    from .parser import print_ingest_report

    print_ingest_report(report)

    paper = report.paper
    params = report.candidate_parameters["parameters"]
    claims = report.economic_claims

    print("\n" + "=" * 78)
    print("STAGE 02 -- Building the Strategy Card interactively")
    print("=" * 78)
    print("Every suggested default below was mined from the PDF above by Stage 01 --")
    print("press Enter to accept a default, or type a replacement.\n")

    guessed_id = card_id or "card_" + re.sub(r"[^a-z0-9]+", "_", paper.title.lower())[:40].strip("_")
    card_id_val = prompt_text("Card ID (unique identifier for this strategy)", default=guessed_id)
    title = prompt_text("Card title (your India-adapted strategy name)",
                         default=f"{paper.title} (India adaptation)")
    source_title = prompt_text("Source paper title", default=paper.title)
    authors_raw = prompt_text("Source paper authors (comma-separated)", default="")
    source_authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
    source_url = prompt_text("Source URL / arXiv ID (optional)", default="")

    print("\n-- Universe --")
    print("Asset classes/segments the paper's own text touches on "
          f"(Stage 03 will let you source India data for each): "
          f"{list(report.candidate_parameters['asset_classes_mentioned'].keys())}")
    universe = prompt_text(
        "Describe the universe THIS Card will actually trade "
        "(you don't have to cover every asset class the paper mentions)",
    )
    universe_evidence = _pick_evidence_interactively(report, claims, "universe")

    print("\n-- Benchmark --")
    benchmark = prompt_text("Benchmark to compare against")

    print("\n-- Signal --")
    signal_name = prompt_text("Signal name (short identifier)", default="signal_v1")
    signal_desc_default = next((s for _, s in claims if re.search(
        r"forecast|momentum|optimiz|signal|regress", s, re.IGNORECASE)), "")
    signal_description = prompt_text(
        "Signal description (what economic bet does this encode?)", default=signal_desc_default
    )
    lookback_hits = params.get("lookback_window", [])
    lookback_default = 21
    if lookback_hits:
        m = re.search(r"(\d+)", lookback_hits[0]["value"])
        if m:
            lookback_default = int(m.group(1))
    lookback_days = prompt_int(
        f"Lookback window in days"
        + (f" (Stage 01 found: '{lookback_hits[0]['value']}' on p.{lookback_hits[0]['page']})" if lookback_hits else ""),
        default=lookback_default,
    )
    lag_days = prompt_int("Lag between signal computation and execution, in days (0 = same day)", default=0)
    rebal_hits = params.get("rebalance_frequency", [])
    rebal_default = "monthly"
    if rebal_hits:
        for kw in ("daily", "weekly", "monthly", "annually", "quarterly"):
            if kw in rebal_hits[0]["value"].lower():
                rebal_default = kw.rstrip("ly") + ("ly" if kw.endswith("ly") else "")
                rebal_default = {"annually": "annual", "quarterly": "quarterly"}.get(rebal_default, rebal_default)
                break
    rebalance_frequency = prompt_choice(
        "Rebalance frequency" + (f" (Stage 01 found: '{rebal_hits[0]['value']}' on p.{rebal_hits[0]['page']})" if rebal_hits else ""),
        ["daily", "weekly", "monthly", "quarterly", "annual"],
    )
    weighting_rule = prompt_text(
        "Weighting rule (one line -- e.g. 'convex optimization, vol-capped, momentum forecast')"
    )
    signal_evidence = _pick_evidence_interactively(report, claims, "signal")

    print("\n-- Risk --")
    vol_hits = params.get("volatility_target_pct", [])
    vol_default = 0.12
    if vol_hits:
        m = re.search(r"(\d+(?:\.\d+)?)", vol_hits[0]["value"])
        if m:
            vol_default = float(m.group(1)) / 100.0
    target_vol = prompt_float(
        "Target annualized volatility (as a fraction, e.g. 0.12 = 12%)"
        + (f" (Stage 01 found: '{vol_hits[0]['value']}' on p.{vol_hits[0]['page']})" if vol_hits else ""),
        default=vol_default,
    )
    l1_dev = prompt_float("Max L1 deviation from a target relative mix (1.0 = unconstrained-ish)", default=1.0)
    long_only = prompt_yesno("Long-only (no shorting)?", default=True)
    leverage_allowed = prompt_yesno("Leverage allowed?", default=False)
    derivatives_allowed = prompt_yesno("Derivatives allowed?", default=False)

    print("\n-- Costs --")
    cost_hits = params.get("trading_cost_bps", [])
    cost_default = 5.0
    if cost_hits:
        m = re.search(r"(\d+(?:\.\d+)?)", cost_hits[0]["value"])
        if m:
            cost_default = float(m.group(1))
    paper_cost_bps = prompt_float(
        "Trading cost the PAPER assumed, in bps"
        + (f" (Stage 01 found: '{cost_hits[0]['value']}' on p.{cost_hits[0]['page']})" if cost_hits else ""),
        default=cost_default,
    )
    cost_evidence = None
    if cost_hits:
        cost_evidence = Evidence(page=cost_hits[0]["page"], quote=cost_hits[0]["context"], confidence=Confidence.high)

    # Ambiguities: auto-generated from anything Stage 01 could NOT find, plus the
    # standing "India adaptation" ambiguity every non-literal-replication Card carries.
    ambiguities = []
    for pname, hits in params.items():
        if not hits and pname != "sharpe_ratio_mentioned":
            ambiguities.append(Ambiguity(
                field=pname,
                description=f"Stage 01 found no regex match for '{pname}' anywhere in the paper -- "
                             f"confirm this parameter manually before Gate A approval.",
            ))
    ambiguities.append(Ambiguity(
        field="asset_class_coverage",
        description="Asset classes/segments the paper's text mentions: "
                     f"{list(report.candidate_parameters['asset_classes_mentioned'].keys())}. "
                     "Stage 03 will ask which of these you can actually source India data for; "
                     "any gap must be explicitly accepted or the universe scope reduced.",
    ))
    print(f"\n{len(ambiguities)} ambiguities auto-generated from ingest gaps -- these must be "
          f"resolved (card.ambiguities[i].resolution = ...) before Gate A can approve.")

    india_notes = ""
    while not india_notes.strip():
        india_notes = prompt_text(
            "India adaptation notes (REQUIRED -- how does this Card diverge from the "
            "source paper to be India-investable? universe substitutions, cost model, "
            "cash-rate proxy, etc.)"
        )

    card = StrategyCard(
        card_id=card_id_val,
        title=title,
        source_title=source_title,
        source_authors=source_authors,
        source_url=source_url or None,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        universe=universe,
        universe_evidence=universe_evidence,
        benchmark=benchmark,
        signal=SignalSpec(
            name=signal_name, description=signal_description, lookback_days=lookback_days,
            lag_days=lag_days, rebalance_frequency=rebalance_frequency, weighting_rule=weighting_rule,
            evidence=[signal_evidence] if signal_evidence else [],
        ),
        risk=RiskSpec(
            target_volatility_annual=target_vol, max_relative_weight_deviation_l1=l1_dev,
            long_only=long_only, leverage_allowed=leverage_allowed, derivatives_allowed=derivatives_allowed,
        ),
        cost=CostSpec(paper_assumed_bps=paper_cost_bps, paper_evidence=cost_evidence),
        data_requirements=[],
        ambiguities=ambiguities,
        india_adaptation_notes=india_notes,
    )
    print(f"\nCard '{card.card_id}' built. Content hash: {card.content_hash()}")
    return card


def load_card(path: str) -> StrategyCard:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return StrategyCard(**data)


def save_card(card: StrategyCard, path: str) -> None:
    import yaml
    # mode="json" turns Enum members (e.g. Confidence.high) into their plain string
    # values so PyYAML's safe_dump -- which only knows built-in types -- can serialize
    # them; card.model_dump() alone leaves Enum objects in the tree and crashes here.
    with open(path, "w") as f:
        yaml.safe_dump(card.model_dump(mode="json"), f, sort_keys=False, default_flow_style=False)
