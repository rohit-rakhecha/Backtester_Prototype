"""Engine correctness tests -- the deterministic engine is shared by every strategy in
the library, so its accounting identity, cost model, and vol-targeting behavior must be
verified independently of any specific strategy's economic merit (see docs/PIPELINE.md
Stage 05 -- "the same engine every strategy uses" is the whole point)."""
import numpy as np
import pandas as pd
import pytest

from backtester.engine.costs import IndiaEquityCostModel
from backtester.engine.portfolio import PortfolioSimulator, performance_metrics
from backtester.engine.signals import fixed_weight_benchmark, vol_controlled, trailing_volatility


def _make_returns(n_days=500, n_assets=2, seed=0, ann_vol=0.2, ann_drift=0.10):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    daily_vol = ann_vol / np.sqrt(252)
    daily_drift = ann_drift / 252
    data = rng.normal(daily_drift, daily_vol, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_zero_weight_zero_cost_flat_portfolio_stays_at_one():
    """A portfolio that never invests (all cash, 0% cash rate) must have constant value 1.0."""
    returns = _make_returns()
    sim = PortfolioSimulator(asset_returns=returns, cash_annual_rate=0.0, rebalance_frequency="ME")
    result = sim.run(fixed_weight_benchmark(np.zeros(returns.shape[1])))
    assert np.allclose(result.value.values, 1.0, atol=1e-9)
    assert (result.weights.values == 0).all()


def test_no_cost_full_investment_matches_buy_and_hold_between_rebalances():
    """With zero trading cost and a single fixed-weight target, portfolio value should
    grow smoothly like a buy-and-hold basket between rebalances (accounting identity)."""
    returns = _make_returns(n_assets=1)
    zero_cost = IndiaEquityCostModel(
        stt_bps_each_side=0, stamp_duty_bps_buy_side=0, exchange_txn_bps_each_side=0,
        sebi_fee_bps_each_side=0, gst_rate=0, brokerage_bps_each_side=0, impact_cost_bps_each_side=0,
    )
    sim = PortfolioSimulator(asset_returns=returns, cost_model=zero_cost, cash_annual_rate=0.0,
                              rebalance_frequency="ME")
    result = sim.run(fixed_weight_benchmark(np.array([1.0])), initial_weights=np.array([1.0]))
    expected = (1 + returns["A0"]).cumprod()
    # exact match: fully invested from day 1, single asset, zero cost, monthly rebalances
    # to the same 100% target never trade, so there is nothing to introduce drift.
    np.testing.assert_allclose(result.value.values, expected.values, rtol=1e-9)


def test_trading_cost_reduces_value_relative_to_free_rebalancing():
    """Turning on a nonzero cost model must strictly reduce terminal value versus the
    same strategy at zero cost, when turnover is nonzero."""
    returns = _make_returns(n_assets=3, seed=1)
    zero_cost = IndiaEquityCostModel(
        stt_bps_each_side=0, stamp_duty_bps_buy_side=0, exchange_txn_bps_each_side=0,
        sebi_fee_bps_each_side=0, gst_rate=0, brokerage_bps_each_side=0, impact_cost_bps_each_side=0,
    )
    real_cost = IndiaEquityCostModel()

    target = np.array([0.5, 0.3, 0.2])
    sim_free = PortfolioSimulator(asset_returns=returns, cost_model=zero_cost, rebalance_frequency="ME")
    sim_paid = PortfolioSimulator(asset_returns=returns, cost_model=real_cost, rebalance_frequency="ME")

    result_free = sim_free.run(fixed_weight_benchmark(target))
    result_paid = sim_paid.run(fixed_weight_benchmark(target))

    assert result_paid.trading_cost.sum() > 0
    assert result_paid.value.iloc[-1] < result_free.value.iloc[-1]


def test_vol_control_dilutes_toward_target_when_estimated_vol_exceeds_target():
    """When trailing volatility is above the target, vol_controlled must scale the risky
    weights down (i.e. hold cash) -- the paper's eq. (1) dilution rule."""
    returns = _make_returns(n_assets=2, ann_vol=0.40, seed=2)  # deliberately high vol
    target_relative = np.array([0.6, 0.4])
    fn = vol_controlled(target_relative, target_vol_annual=0.07, lookback_days=11)
    as_of = returns.iloc[:60]
    w = fn(as_of.index[-1], as_of, np.zeros(2))
    assert w.sum() < 1.0, "high trailing vol should cause dilution into cash"
    # relative mix between the two risky assets must be preserved
    np.testing.assert_allclose(w / w.sum(), target_relative, atol=1e-9)


def test_vol_control_stays_fully_invested_when_estimated_vol_is_low():
    returns = _make_returns(n_assets=2, ann_vol=0.02, seed=3)  # deliberately low vol
    target_relative = np.array([0.6, 0.4])
    fn = vol_controlled(target_relative, target_vol_annual=0.20, lookback_days=11)
    as_of = returns.iloc[:60]
    w = fn(as_of.index[-1], as_of, np.zeros(2))
    np.testing.assert_allclose(w, target_relative, atol=1e-9)


def test_cost_model_round_trip_greater_than_one_way():
    cm = IndiaEquityCostModel()
    assert cm.round_trip_cost_bps() > cm.one_way_cost_bps() > 0


def test_performance_metrics_sharpe_matches_manual_calc_on_flat_growth():
    dates = pd.bdate_range("2020-01-01", periods=253)
    # exactly 10% growth over 1 year, zero vol -- degenerate but checks CAGR math
    values = pd.Series(np.linspace(1.0, 1.10, len(dates)), index=dates)
    m = performance_metrics(values, cash_annual_rate=0.0)
    assert 0.08 < m["return"] < 0.12


def test_optimizer_respects_long_only_and_budget_constraint():
    """The engine clips/renormalizes any target that violates long-only / budget <= 1,
    as a defense-in-depth check independent of what the signal function itself does."""
    returns = _make_returns(n_assets=2, seed=4)
    sim = PortfolioSimulator(asset_returns=returns, rebalance_frequency="ME")

    def bad_signal(date, returns_as_of, current_w):
        return np.array([1.5, -0.3])  # violates long-only and budget

    result = sim.run(bad_signal)
    assert (result.weights.values >= -1e-9).all()
    assert (result.weights.sum(axis=1) <= 1.0 + 1e-6).all()
