"""Tests for the non-interactive pieces of the new comprehensive-ingest and Stage 06
transparency features -- the interactive builder/discovery functions are exercised by
hand (scripted `input()` mocking) rather than in this suite, since their value is in the
prompting flow itself, not in pure logic; see PR history for those manual runs."""
import numpy as np
import pandas as pd

from backtester.ingest.parser import (
    ParsedPaper, PageEvidence, extract_abstract, extract_economic_claims,
    extract_candidate_parameters,
)
from backtester.validation.research import signal_diagnostics, decompose_vs_equal_weight
from backtester.engine.portfolio import PortfolioSimulator
from backtester.engine import signals as sig
from backtester.engine.costs import IndiaEquityCostModel


def _fake_paper() -> ParsedPaper:
    p1 = (
        "Some Paper Title\n\n"
        "Abstract\n"
        "We propose a simple momentum strategy. We find that trailing 63-day returns "
        "predict future performance because winners tend to keep winning. The portfolio "
        "is rebalanced monthly using a trading cost of 8 basis points.\n"
        "1 Introduction\n"
        "This section has nothing interesting in it for the abstract extractor.\n"
    )
    p2 = (
        "We target a volatility of 10% annualized. Bonds and gold are not considered here; "
        "this is an equity-only momentum strategy using ETFs.\n"
    )
    return ParsedPaper(source_path="fake.pdf", title="Some Paper Title",
                        pages=[PageEvidence(1, p1), PageEvidence(2, p2)])


def test_extract_abstract_stops_before_next_section():
    paper = _fake_paper()
    abstract = extract_abstract(paper)
    assert abstract is not None
    assert "momentum strategy" in abstract
    assert "nothing interesting" not in abstract


def test_extract_economic_claims_finds_reasoning_sentences():
    paper = _fake_paper()
    claims = extract_economic_claims(paper)
    assert any("We find that" in s for _, s in claims)
    assert any(page == 1 for page, _ in claims)


def test_extract_candidate_parameters_finds_cost_and_vol_and_rebalance():
    paper = _fake_paper()
    result = extract_candidate_parameters(paper)
    params = result["parameters"]
    assert params["trading_cost_bps"], "should find '8 basis points'"
    assert "8" in params["trading_cost_bps"][0]["value"]
    assert params["rebalance_frequency"], "should find 'rebalanced monthly'"
    assert params["volatility_target_pct"], "should find '10% annualized volatility'"
    assert "equity" in result["asset_classes_mentioned"]


def test_signal_diagnostics_empty_log_reports_gracefully():
    out = signal_diagnostics([], pd.DataFrame(), horizon_days=21)
    assert out["n_rebalances_evaluated"] == 0
    assert "No diagnostics" in out["interpretation"]


def test_signal_diagnostics_detects_perfectly_predictive_alpha():
    """Construct a synthetic case where alpha at each rebalance is EXACTLY the realized
    forward return -- the Information Coefficient should be ~1.0 (perfect rank agreement),
    proving the IC calculation itself is correct, independent of any real strategy run."""
    dates = pd.bdate_range("2020-01-01", periods=120)
    n_assets = 4
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(len(dates), n_assets)),
                            index=dates, columns=["A", "B", "C", "D"])

    diagnostics_log = []
    horizon = 10
    for rebal_pos in [20, 40, 60, 80]:
        rebal_date = dates[rebal_pos]
        fwd = (1 + returns.iloc[rebal_pos + 1: rebal_pos + 1 + horizon]).prod() - 1
        diagnostics_log.append({"date": rebal_date, "alpha": fwd.to_dict()})

    out = signal_diagnostics(diagnostics_log, returns, horizon_days=horizon)
    assert out["n_rebalances_evaluated"] == 4
    assert out["mean_ic"] > 0.99, f"expected ~1.0 IC for perfectly predictive alpha, got {out['mean_ic']}"
    assert "Positive mean IC" in out["interpretation"]


def test_decompose_vs_equal_weight_matches_manual_sharpe_arithmetic():
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, size=(len(dates), 3)), index=dates, columns=["A", "B", "C"])
    equal_weight = np.array([1 / 3, 1 / 3, 1 / 3])
    cost = IndiaEquityCostModel()

    eq_result = PortfolioSimulator(asset_returns=returns, cost_model=cost, rebalance_frequency="ME").run(
        sig.fixed_weight_benchmark(equal_weight)
    )
    vc_result = PortfolioSimulator(asset_returns=returns, cost_model=cost, rebalance_frequency="ME").run(
        sig.vol_controlled(equal_weight, target_vol_annual=0.10)
    )
    mw_result = PortfolioSimulator(asset_returns=returns, cost_model=cost, rebalance_frequency="ME").run(
        sig.markowitz(equal_weight, target_vol_annual=0.10, asset_names=["A", "B", "C"])
    )

    out = decompose_vs_equal_weight(eq_result, vc_result, mw_result)
    assert abs(out["vol_cap_effect"] - (out["vol_controlled_sharpe"] - out["equal_weight_sharpe"])) < 1e-9
    assert abs(out["alpha_tilt_effect"] - (out["markowitz_sharpe"] - out["vol_controlled_sharpe"])) < 1e-9
    assert "narrative" in out and len(out["narrative"]) > 0
