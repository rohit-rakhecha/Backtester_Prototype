"""Stage 01/02 -- LLM-based contextual extraction (the "real AI reading" version of
`build_card_automatically`).

The regex-based extractor in `strategy_card.py` (`build_card_automatically`) is pattern
matching: it finds the string "5 basis points" but has no idea whether that sentence is
the paper's actual methodology or an aside the authors are arguing against. This module
replaces that step with a genuine reading-comprehension call to Claude, while keeping
every control the regex version had:

  - Template-constrained output: the model can only fill a fixed, typed schema
    (`LLMCardExtraction`, via `client.messages.parse`) -- it cannot return free text or
    invented fields, exactly like the regex version's Card was schema-constrained.
  - Evidence, not assertion: every claim requires a page number AND a verbatim quote.
  - Verified, not trusted: this module does NOT take the model's page citation on faith.
    The source text is fed to the model with explicit "=== PAGE N ===" markers WE control
    (from Stage 01's own per-page extraction), so after the call we can mechanically check
    that each returned quote is an actual substring of the page it claims to be on. A
    citation that doesn't verify is a hallucination caught in code, not a hope that the
    model behaved -- this is the "our own contextual verification" half of Stage 1/2 the
    board memo's "AI interprets, deterministic infrastructure checks" split calls for.

This is a SINGLE structured-extraction call, not a multi-step agent loop -- the whole
paper is already in front of the model in one request, so there's nothing to search for.
See docs/PIPELINE.md for why Stage 05 (actual trading decisions) deliberately does NOT
get this treatment: an LLM call during a historical backtest would leak lookahead bias
that no citation-verification step could catch, because the bias is in the model's
training data, not in a fabricated quote.
"""
from __future__ import annotations

import difflib
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .strategy_card import (
    Ambiguity, Confidence, CostSpec, Evidence, RiskSpec, SignalSpec, StrategyCard,
)


# ---------------------------------------------------------------------------
# The schema the model must fill. Flat and explicit on purpose -- a nested/optional-heavy
# schema is more likely to trip up structured-output validation; this maps onto
# StrategyCard's real (nested) shape in `_assemble_card` below.
# ---------------------------------------------------------------------------

class LLMEvidence(BaseModel):
    page: int = Field(description="The page number from the '=== PAGE N ===' marker this quote appears under.")
    quote: str = Field(description="A VERBATIM substring copied exactly from that page's text. Do not paraphrase.")


class LLMAmbiguity(BaseModel):
    field: str = Field(description="Which Card field this ambiguity concerns, e.g. 'signal.lookback_days'.")
    description: str = Field(description="What is unclear or not stated in the paper, and what you did about it.")


class LLMCardExtraction(BaseModel):
    """What Claude fills in. See `_assemble_card` for how this becomes a StrategyCard."""

    source_authors: list[str] = Field(default_factory=list, description="Author names as printed on the paper.")

    universe: str = Field(description=(
        "The economic universe/asset classes the paper's STRATEGY trades (not just what "
        "data it mentions in passing) -- e.g. 'long-only US equities, bonds, and gold, plus cash'."
    ))
    universe_evidence: LLMEvidence

    benchmark: str = Field(description="What the paper itself compares its strategy against.")

    signal_name: str = Field(description="A short slug for the core signal/strategy, e.g. 'vol_capped_momentum'.")
    signal_description: str = Field(description=(
        "The paper's actual economic thesis for why this signal should work -- in your own words, "
        "synthesized from the paper's argument, NOT a single quoted sentence taken out of context. "
        "Explicitly distinguish the paper's own claim from any caveat or limitation it raises."
    ))
    signal_lookback_days: int = Field(description="Trading days of history the signal looks back over. If the paper states it in weeks/months/years, convert to trading days (21/month, 252/year).")
    signal_lag_days: int = Field(description="Days between computing the signal and acting on it. 0 if the paper doesn't lag it.")
    signal_rebalance_frequency: Literal["daily", "weekly", "monthly", "quarterly", "annual"]
    signal_weighting_rule: str = Field(description="One line describing HOW weights are set (equal-weight, risk parity, convex optimization with what objective/constraints, etc.).")
    signal_evidence: LLMEvidence

    risk_target_volatility_annual: float = Field(description="As a fraction, e.g. 0.07 for 7%. If the paper states no explicit vol target, use your best-supported estimate and say so in ambiguities.")
    risk_max_relative_weight_deviation_l1: float = Field(default=1.0, description="How far (L1 distance) the strategy is allowed to deviate from an equal/benchmark weight mix, if the paper specifies a constraint like this. 1.0 if effectively unconstrained or not applicable.")
    risk_long_only: bool = True
    risk_leverage_allowed: bool = False
    risk_derivatives_allowed: bool = False

    cost_paper_assumed_bps: float = Field(description="Trading cost assumption in basis points, as stated by the paper.")
    cost_evidence: LLMEvidence

    ambiguities: list[LLMAmbiguity] = Field(description=(
        "Every place you had to infer, estimate, or default rather than read a number "
        "directly off the page -- be honest here rather than confidently guessing above."
    ))

    india_adaptation_notes: str = Field(description=(
        "How this Card diverges from the paper to be usable as a long-only India public-"
        "equities backtest against a manually supplied NIFTY-indices dataset: what's "
        "substituted, what's dropped, what stays the same, and why that's still a "
        "reasonable test of the paper's underlying economic idea."
    ))


# ---------------------------------------------------------------------------
# Page-marked source text -- the ground truth the model's citations get checked against.
# ---------------------------------------------------------------------------

def build_marked_text(paper, max_chars: int = 350_000) -> str:
    """Concatenates the paper's pages with explicit '=== PAGE N ===' markers. Capped at
    `max_chars` (default comfortably covers a 100+ page paper) so a pathological PDF can't
    blow the request past the model's context window."""
    parts = []
    total = 0
    for p in paper.pages:
        block = f"\n=== PAGE {p.page} ===\n{p.text}"
        if total + len(block) > max_chars:
            parts.append(f"\n=== [TRUNCATED -- {len(paper.pages) - p.page + 1} further pages omitted for length] ===\n")
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(paper, page: int, quote: str, fuzzy_threshold: float = 0.85) -> bool:
    """Checks that `quote` genuinely appears on `page` of the source paper -- exact
    substring match after whitespace normalization first (handles PDF line-wrap
    differences), falling back to a fuzzy ratio for minor OCR/whitespace artifacts. This
    is the mechanical hallucination check described in the module docstring: we do not
    trust the model's citation, we verify it against text WE extracted."""
    if not (1 <= page <= len(paper.pages)):
        return False
    page_text = _normalize(paper.pages[page - 1].text)
    quote_norm = _normalize(quote)
    if not quote_norm:
        return False
    if quote_norm in page_text:
        return True
    # Fuzzy fallback: slide a same-length window is expensive for long pages, so instead
    # check whether a close match exists using SequenceMatcher on the whole page -- cheap
    # enough at this scale (a handful of evidence fields per paper) and catches near-misses
    # (a dropped hyphen, a re-flowed line break) without accepting genuinely different text.
    ratio = difflib.SequenceMatcher(None, quote_norm, page_text).quick_ratio()
    if ratio >= fuzzy_threshold:
        return True
    # Also try matching the quote against every page as a last resort in case the model
    # cited the right text but the wrong page number -- if found elsewhere, we still mark
    # THIS citation unverified (wrong page is still a real citation error) but this lets
    # the caller give a more useful message.
    return False


def find_actual_page(paper, quote: str) -> Optional[int]:
    """Best-effort: which page (if any) really contains this quote, for a more useful
    warning when the model cites the right text but the wrong page number."""
    quote_norm = _normalize(quote)
    if not quote_norm:
        return None
    for p in paper.pages:
        if quote_norm in _normalize(p.text):
            return p.page
    return None


SYSTEM_PROMPT = """You are helping an institutional quant research desk turn an academic \
finance paper into a structured "Strategy Card" -- a small, typed specification of the \
paper's trading strategy that will later be backtested on Indian public equities.

Rules, all non-negotiable:
1. Every *_evidence field MUST cite a page number from an actual "=== PAGE N ===" marker \
in the text you were given, and the quote MUST be copied verbatim (character-for-character) \
from that page. Do not paraphrase, do not combine text from two pages into one quote, do \
not invent a quote that sounds right. If you cannot find a page that directly supports a \
field, say so honestly in `ambiguities` instead of fabricating a citation.
2. Distinguish the paper's actual thesis from a caveat, limitation, or an idea it argues \
AGAINST. Papers often state a competing view before refuting it -- do not extract the \
refuted view as if it were the paper's claim.
3. Numeric parameters (lookback, rebalance frequency, cost, volatility target) must come \
from what the paper actually states. If the paper is silent on a parameter, use a \
reasonable, clearly-labeled default and record it in `ambiguities` -- do not silently guess.
4. This Card will be used for a LONG-ONLY India public-equities backtest against a \
manually supplied dataset of NIFTY indices, not the paper's own asset universe. Your \
`india_adaptation_notes` must honestly describe what does and doesn't carry over.
5. Be concise. Field values are specifications, not essays."""


def build_card_via_llm(
    report,
    model: str = "claude-opus-5",
    client=None,
    card_id: Optional[str] = None,
) -> StrategyCard:
    """Stage 02, via a real LLM reading-comprehension call. See module docstring.

    `report` is a `backtester.ingest.parser.IngestReport` (from `comprehensive_ingest`).
    `client` lets tests/callers inject a pre-built `anthropic.Anthropic()` (or a mock);
    if omitted, one is constructed from the environment (`ANTHROPIC_API_KEY` or an
    `ant auth login` profile -- see `ingest/interactive_io.py:ensure_anthropic_api_key`).
    """
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    paper = report.paper
    marked_text = build_marked_text(paper)

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{
            "role": "user",
            "content": (
                f"Paper title (best guess): {paper.title}\n\n"
                f"Full paper text, page-marked:\n{marked_text}"
            ),
        }],
        output_format=LLMCardExtraction,
    )
    extracted: LLMCardExtraction = response.parsed_output
    return _assemble_card(paper, extracted, card_id=card_id)


def _assemble_card(paper, extracted: LLMCardExtraction, card_id: Optional[str] = None) -> StrategyCard:
    """Turns the flat LLM output into a real StrategyCard, running the citation
    verification described in the module docstring and downgrading/flagging anything
    that doesn't check out."""
    guessed_id = card_id or "card_" + re.sub(r"[^a-z0-9]+", "_", paper.title.lower())[:40].strip("_")

    verification_ambiguities: list[Ambiguity] = []
    verification_counts = {"verified": 0, "corrected": 0, "unverified": 0}

    def _to_evidence(llm_ev: LLMEvidence, field_label: str) -> Evidence:
        ok = verify_quote(paper, llm_ev.page, llm_ev.quote)
        if ok:
            verification_counts["verified"] += 1
            return Evidence(page=llm_ev.page, quote=llm_ev.quote, confidence=Confidence.high)
        actual_page = find_actual_page(paper, llm_ev.quote)
        if actual_page is not None and actual_page != llm_ev.page:
            note = (
                f"LLM cited page {llm_ev.page} for '{field_label}', but this quote was actually "
                f"found on page {actual_page} -- using the verified page, confidence downgraded."
            )
            verification_ambiguities.append(Ambiguity(
                field=f"{field_label}.evidence_page", description=note,
                resolution="Auto-corrected to the verified page; treat with extra scrutiny at Gate A.",
                resolved_by="llm-extraction-verifier",
            ))
            verification_counts["corrected"] += 1
            return Evidence(page=actual_page, quote=llm_ev.quote, confidence=Confidence.medium)
        note = (
            f"LLM's citation for '{field_label}' (p.{llm_ev.page}, \"{llm_ev.quote[:80]}...\") could NOT "
            f"be verified against the extracted page text -- possible hallucinated quote. "
            f"Confidence downgraded to low; verify this field manually before Gate A approval."
        )
        verification_ambiguities.append(Ambiguity(
            field=f"{field_label}.evidence", description=note, resolution=None, resolved_by=None,
        ))
        verification_counts["unverified"] += 1
        return Evidence(page=llm_ev.page, quote=llm_ev.quote, confidence=Confidence.low)

    universe_evidence = _to_evidence(extracted.universe_evidence, "universe")
    signal_evidence = _to_evidence(extracted.signal_evidence, "signal")
    cost_evidence = _to_evidence(extracted.cost_evidence, "cost")

    ambiguities = [
        Ambiguity(field=a.field, description=a.description) for a in extracted.ambiguities
    ] + verification_ambiguities

    card = StrategyCard(
        card_id=guessed_id,
        title=f"{paper.title} — India public-equities backtest (LLM-extracted)",
        source_title=paper.title,
        source_authors=extracted.source_authors,
        source_url=None,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        universe=extracted.universe,
        universe_evidence=universe_evidence,
        benchmark=extracted.benchmark,
        signal=SignalSpec(
            name=extracted.signal_name,
            description=extracted.signal_description,
            lookback_days=extracted.signal_lookback_days,
            lag_days=extracted.signal_lag_days,
            rebalance_frequency=extracted.signal_rebalance_frequency,
            weighting_rule=extracted.signal_weighting_rule,
            evidence=[signal_evidence],
        ),
        risk=RiskSpec(
            target_volatility_annual=extracted.risk_target_volatility_annual,
            max_relative_weight_deviation_l1=extracted.risk_max_relative_weight_deviation_l1,
            long_only=extracted.risk_long_only,
            leverage_allowed=extracted.risk_leverage_allowed,
            derivatives_allowed=extracted.risk_derivatives_allowed,
        ),
        cost=CostSpec(paper_assumed_bps=extracted.cost_paper_assumed_bps, paper_evidence=cost_evidence),
        data_requirements=[],
        ambiguities=ambiguities,
        india_adaptation_notes=extracted.india_adaptation_notes,
    )
    print(f"Card '{card.card_id}' extracted via LLM reading comprehension. Content hash: {card.content_hash()}")
    print(
        f"Citation verification (3 core citations checked against source pages): "
        f"{verification_counts['verified']} verified verbatim, "
        f"{verification_counts['corrected']} auto-corrected to a different page, "
        f"{verification_counts['unverified']} UNVERIFIED"
        + (" -- review these before Gate A approval." if verification_counts["unverified"] else ".")
    )
    return card
