"""Tests for the LLM-based Stage 02 extraction (backtester/ingest/llm_extract.py).

No real API calls here -- these test the parts that don't need one: the page-marked
text builder, and (most importantly) the citation-verification logic that is the whole
point of doing this in code rather than trusting the model's citation. `build_card_via_llm`
itself is exercised with a mocked `anthropic` client so the request/response wiring is
checked without spending money or requiring network access.
"""
import re
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtester.ingest.llm_extract import (
    LLMAmbiguity, LLMCardExtraction, LLMEvidence, build_card_via_llm, build_marked_text,
    find_actual_page, verify_quote,
)
from backtester.ingest.parser import PageEvidence, ParsedPaper
from backtester.ingest.strategy_card import Confidence


def _fake_paper() -> ParsedPaper:
    return ParsedPaper(
        source_path="fake.pdf",
        title="Fake Paper Title",
        pages=[
            PageEvidence(1, "This is the abstract. We propose a momentum strategy for equities."),
            PageEvidence(2, "The universe is long-only equities, rebalanced monthly."),
            PageEvidence(3, "We assume a trading cost of 5 basis points per trade, a conservative estimate."),
        ],
    )


def test_build_marked_text_includes_page_markers_for_every_page():
    paper = _fake_paper()
    marked = build_marked_text(paper)
    for p in paper.pages:
        assert f"=== PAGE {p.page} ===" in marked
        assert p.text in marked


def test_build_marked_text_truncates_past_max_chars():
    paper = _fake_paper()
    marked = build_marked_text(paper, max_chars=10)
    assert "TRUNCATED" in marked


def test_verify_quote_true_for_real_substring():
    paper = _fake_paper()
    assert verify_quote(paper, 3, "trading cost of 5 basis points") is True


def test_verify_quote_false_for_fabricated_text():
    paper = _fake_paper()
    assert verify_quote(paper, 3, "this sentence does not appear anywhere in the paper") is False


def test_verify_quote_false_for_wrong_page():
    paper = _fake_paper()
    # The text is real, but lives on page 3, not page 1.
    assert verify_quote(paper, 1, "trading cost of 5 basis points") is False


def test_verify_quote_tolerant_of_whitespace_reflow():
    paper = _fake_paper()
    # PDF extraction often reflows whitespace/line breaks -- verification should still pass.
    assert verify_quote(paper, 3, "trading   cost  of 5\nbasis points") is True


def test_find_actual_page_locates_quote_on_a_different_page_than_claimed():
    paper = _fake_paper()
    assert find_actual_page(paper, "trading cost of 5 basis points") == 3
    assert find_actual_page(paper, "nonexistent text xyz") is None


def _fake_extraction(cost_page: int, cost_quote: str) -> LLMCardExtraction:
    return LLMCardExtraction(
        source_authors=["A. Author"],
        universe="Long-only equities",
        universe_evidence=LLMEvidence(page=2, quote="long-only equities, rebalanced monthly"),
        benchmark="Broad market index",
        signal_name="momentum_v1",
        signal_description="Momentum-based signal.",
        signal_lookback_days=63,
        signal_lag_days=0,
        signal_rebalance_frequency="monthly",
        signal_weighting_rule="Equal-weight among top momentum names.",
        signal_evidence=LLMEvidence(page=1, quote="We propose a momentum strategy for equities."),
        risk_target_volatility_annual=0.12,
        cost_paper_assumed_bps=5.0,
        cost_evidence=LLMEvidence(page=cost_page, quote=cost_quote),
        ambiguities=[LLMAmbiguity(field="risk.target_volatility_annual", description="Not stated; defaulted to 12%.")],
        india_adaptation_notes="Adapted for India public equities via a manually uploaded NIFTY dataset.",
    )


def test_build_card_via_llm_verified_citations_get_high_confidence():
    paper = _fake_paper()
    report = MagicMock(paper=paper)
    extraction = _fake_extraction(cost_page=3, cost_quote="trading cost of 5 basis points")

    fake_response = MagicMock(parsed_output=extraction)
    fake_client = MagicMock()
    fake_client.messages.parse.return_value = fake_response

    card = build_card_via_llm(report, client=fake_client)

    assert card.cost.paper_evidence.confidence == Confidence.high
    assert card.cost.paper_evidence.page == 3
    assert card.universe_evidence.confidence == Confidence.high
    assert card.signal.evidence[0].confidence == Confidence.high
    # The model's own honest ambiguity ("not stated, defaulted") correctly stays
    # unresolved -- it's a genuine open question for a human, not a verification failure.
    unresolved_fields = [a.field for a in card.unresolved_ambiguities()]
    assert unresolved_fields == ["risk.target_volatility_annual"]
    assert card.source_authors == ["A. Author"]


def test_build_card_via_llm_wrong_page_gets_auto_corrected_and_flagged():
    paper = _fake_paper()
    report = MagicMock(paper=paper)
    # Real text, but claims page 1 instead of the true page 3.
    extraction = _fake_extraction(cost_page=1, cost_quote="trading cost of 5 basis points")

    fake_client = MagicMock()
    fake_client.messages.parse.return_value = MagicMock(parsed_output=extraction)

    card = build_card_via_llm(report, client=fake_client)

    assert card.cost.paper_evidence.page == 3  # corrected
    assert card.cost.paper_evidence.confidence == Confidence.medium
    corrected_ambiguities = [a for a in card.ambiguities if "evidence_page" in a.field]
    assert len(corrected_ambiguities) == 1
    assert corrected_ambiguities[0].resolution is not None  # auto-resolved, not blocking


def test_build_card_via_llm_fabricated_quote_stays_unresolved_and_low_confidence():
    paper = _fake_paper()
    report = MagicMock(paper=paper)
    extraction = _fake_extraction(cost_page=3, cost_quote="this text was never in the paper at all")

    fake_client = MagicMock()
    fake_client.messages.parse.return_value = MagicMock(parsed_output=extraction)

    card = build_card_via_llm(report, client=fake_client)

    assert card.cost.paper_evidence.confidence == Confidence.low
    unresolved = card.unresolved_ambiguities()
    assert any("cost.evidence" == a.field for a in unresolved)


def test_build_card_via_llm_calls_api_with_expected_shape():
    paper = _fake_paper()
    report = MagicMock(paper=paper)
    extraction = _fake_extraction(cost_page=3, cost_quote="trading cost of 5 basis points")

    fake_client = MagicMock()
    fake_client.messages.parse.return_value = MagicMock(parsed_output=extraction)

    build_card_via_llm(report, client=fake_client, model="claude-opus-5")

    kwargs = fake_client.messages.parse.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["output_format"] is LLMCardExtraction
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "system" in kwargs and len(kwargs["system"]) > 0
    assert "=== PAGE 1 ===" in kwargs["messages"][0]["content"]
