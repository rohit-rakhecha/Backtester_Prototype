"""Stage 05 -- deterministic portfolio accounting engine.

This is the FIXED, REUSABLE engine every strategy in the library runs through -- the
direct implementation of "deterministic infrastructure should own... execution, costs,
portfolio construction, and measurement" from the board memo. No AI-authored code runs
here; the only pluggable piece is `weight_fn`, a small function that maps a point-in-time
feature snapshot to target weights (the compiled Strategy Card signal).

Accounting follows the source paper's Appendix A exactly, generalized from 3 assets to N
and adapted long-only-India (assets are Indian equity factor sleeves; cash accrues at a
configurable rate, defaulting to 0% until an India risk-free series is sourced -- see
docs/DATA_SOURCES.md item 2, and the explicit `cash_annual_rate=0.0` default below, which
is a disclosed placeholder, not a silent assumption of a risk-free India cash return).

Identity enforced every trading day, no exceptions:
    V_t = V_{t-1} * ( sum_i w_i * (1 + r_i) + c * (1 + r_cash) )
On a rebalance day, weights are moved to the target and a trading cost is deducted from
cash BEFORE recomputing post-trade weights -- mirrors the paper's eq. in Appendix A.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .costs import IndiaEquityCostModel


WeightFn = Callable[[pd.Timestamp, pd.DataFrame, np.ndarray], np.ndarray]
# signature: (rebalance_date, feature_panel_as_of_date, current_drifted_weights) -> target weights (len n_assets)


@dataclass
class SimulationResult:
    value: pd.Series
    weights: pd.DataFrame       # drifted weights each day (post any rebalance that day)
    cash_weight: pd.Series
    turnover: pd.Series         # per-day turnover (0 on non-rebalance days)
    trading_cost: pd.Series     # currency cost deducted each rebalance day
    rebalance_dates: list


@dataclass
class PortfolioSimulator:
    """Long-only, unlevered, cash-holding portfolio simulator.

    asset_returns: DataFrame of daily simple returns, columns = asset names, index = dates.
    cost_model: charged on the traded fraction of NOTIONAL at each rebalance (both legs
        already netted into `one_way_cost_bps` -- see engine/costs.py).
    cash_annual_rate: annualized cash accrual rate (simple), converted to a daily rate.
        Defaults to 0.0 -- see docstring; do not treat this default as a real assumption
        about India cash returns, it is a placeholder pending Stage 03 data sourcing.
    """

    asset_returns: pd.DataFrame
    cost_model: IndiaEquityCostModel = field(default_factory=IndiaEquityCostModel)
    cash_annual_rate: float = 0.0
    rebalance_frequency: str = "ME"  # pandas offset alias: "ME" month-end, "W" week, "YE" year

    def __post_init__(self):
        self.assets = list(self.asset_returns.columns)
        self.n = len(self.assets)
        self._daily_cash_rate = self.cash_annual_rate / 252.0

    def _rebalance_dates(self) -> list:
        idx = self.asset_returns.index
        grouped = pd.Series(idx, index=idx).groupby(
            pd.Grouper(freq=self.rebalance_frequency)
        ).last().dropna()
        return list(grouped.values)

    def run(self, weight_fn: WeightFn, initial_weights: Optional[np.ndarray] = None) -> SimulationResult:
        idx = self.asset_returns.index
        rebal_dates = set(self._rebalance_dates())

        n = self.n
        w = np.zeros(n) if initial_weights is None else initial_weights.copy()
        cash = 1.0 - w.sum()
        V = 1.0

        values = []
        weight_rows = []
        cash_rows = []
        turnover_rows = []
        cost_rows = []
        rebalanced_on = []

        r_mat = self.asset_returns.fillna(0.0).values

        for t, date in enumerate(idx):
            r = r_mat[t]
            # valid (non-launched) assets return 0 contribution, weight frozen at 0 --
            # handled upstream by feature panel construction (Stage 04 launch_dates check)
            asset_growth = w * (1.0 + r)
            cash_growth = cash * (1.0 + self._daily_cash_rate)
            V_new = V * (asset_growth.sum() + cash_growth)
            if V_new <= 0:
                raise FloatingPointError(f"Portfolio value went non-positive at {date}")
            w = (V * asset_growth) / V_new
            cash = (V * cash_growth) / V_new
            V = V_new

            turnover = 0.0
            cost = 0.0
            if date in rebal_dates:
                target = weight_fn(date, self.asset_returns.loc[:date], w)
                target = np.clip(target, 0.0, None)
                if target.sum() > 1.0 + 1e-9:
                    target = target / target.sum()  # long-only, fully-invested-or-less cap
                traded = np.abs(target - w)
                turnover = 0.5 * traded.sum()
                cost = self.cost_model.trade_cost(traded.sum() * V)
                V_post = V - cost
                if V_post <= 0:
                    raise FloatingPointError(f"Trading cost exceeded portfolio value at {date}")
                # post-trade weights are fractions of the post-cost value (paper, Appendix A)
                w = target
                cash = 1.0 - target.sum()
                V = V_post
                rebalanced_on.append(date)

            values.append(V)
            weight_rows.append(w.copy())
            cash_rows.append(cash)
            turnover_rows.append(turnover)
            cost_rows.append(cost)

        value = pd.Series(values, index=idx, name="value")
        weights = pd.DataFrame(weight_rows, index=idx, columns=self.assets)
        cash_weight = pd.Series(cash_rows, index=idx, name="cash")
        turnover = pd.Series(turnover_rows, index=idx, name="turnover")
        trading_cost = pd.Series(cost_rows, index=idx, name="trading_cost")

        return SimulationResult(
            value=value,
            weights=weights,
            cash_weight=cash_weight,
            turnover=turnover,
            trading_cost=trading_cost,
            rebalance_dates=rebalanced_on,
        )


def annualized_turnover(turnover: pd.Series) -> float:
    return 252.0 * turnover.mean()


def performance_metrics(value: pd.Series, cash_annual_rate: float = 0.0) -> dict:
    """Return, volatility, Sharpe (vs cash), max/mean drawdown -- the same six metrics
    the source paper reports (§2.6), computed the same way (CAGR-based Sharpe, not the
    conventional arithmetic-mean Sharpe -- see their §2.6 note on the two agreeing closely)."""
    rets = value.pct_change().dropna()
    n_years = (value.index[-1] - value.index[0]).days / 365.25
    cagr = (value.iloc[-1] / value.iloc[0]) ** (1.0 / n_years) - 1.0
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - cash_annual_rate) / ann_vol if ann_vol > 0 else float("nan")
    running_max = value.cummax()
    drawdown = value / running_max - 1.0
    max_dd = drawdown.min()
    mean_dd = drawdown.mean()
    return {
        "return": cagr,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "mean_drawdown": mean_dd,
    }
