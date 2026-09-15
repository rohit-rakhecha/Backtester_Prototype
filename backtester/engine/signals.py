"""Stage 05 -- signal functions (the ONLY place an AI-authored economic idea runs).

Every function here has the exact signature `WeightFn` from engine/portfolio.py:
    (rebalance_date, returns_as_of_date, current_drifted_weights) -> target_weights

This is the "template-constrained generation" control in code: a signal function can only
read the point-in-time return history (no other I/O) and must return a weight vector of
the right length. It cannot reach into the accounting engine, cannot see future data (the
`returns_as_of_date` frame is already point-in-time truncated by the simulator), and
cannot change the cost model. Swap the closure's economic logic; the engine around it never
changes -- that is what makes results across strategies comparable (Stage 07) and auditable
(Stage 08).

Three signal families, mirroring the source paper's §2.2-2.4 structure, adapted long-only
India / N-asset factor sleeves instead of 3-asset stock/bond/gold:
  - fixed_weight_benchmark   : paper's §2.2 (e.g. equal-weight across factor sleeves)
  - vol_controlled           : paper's §2.3 eq. (1), diluting a benchmark with cash to a vol target
  - markowitz                : paper's §2.4 eq. (3), convex optimization with a vol cap and
                                 an L1 trust region around a target mix, using a simple trailing
                                 momentum forecast for alpha_t (paper's "simple Markowitz", §2.4)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def fixed_weight_benchmark(target: np.ndarray):
    """Static long-only benchmark (paper §2.2). Returns a WeightFn that always targets
    the same relative weights -- the null hypothesis every dynamic strategy here must beat.
    """
    target = np.asarray(target, dtype=float)

    def _fn(date, returns_as_of, current_w):
        return target.copy()

    return _fn


def trailing_volatility(returns_as_of: pd.DataFrame, weights: np.ndarray, lookback_days: int = 11) -> float:
    """Empirical annualized vol of a weighted combination, trailing window (paper §2.3:
    an 11-day window is short but reactive; see paper's justification for why noisier-but-
    reactive beats smoother-but-stale here -- empirically validated in Stage 06)."""
    r = returns_as_of.tail(lookback_days).fillna(0.0).values @ weights
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(252))


def vol_controlled(target_relative: np.ndarray, target_vol_annual: float, lookback_days: int = 11):
    """Dilutes a fixed-weight benchmark with cash to hit a target ex-ante volatility
    (paper eq. (1), §2.3). Economic intuition: de-risk into cash when estimated volatility
    is high, without changing the RELATIVE mix of risky sleeves -- this is risk *sizing*,
    not risk *timing* (no view on which sleeve will outperform)."""
    target_relative = np.asarray(target_relative, dtype=float)
    target_relative = target_relative / target_relative.sum()

    def _fn(date, returns_as_of, current_w):
        sigma = trailing_volatility(returns_as_of, target_relative, lookback_days)
        if sigma <= target_vol_annual or sigma == 0.0:
            return target_relative.copy()
        scale = target_vol_annual / sigma
        return target_relative * scale

    return _fn


def momentum_alpha(returns_as_of: pd.DataFrame, halflife_days: int = 252, horizon_days: int = 21) -> np.ndarray:
    """Simple causal momentum forecast: EWMA of past daily returns, scaled to a
    `horizon_days`-ahead estimate (paper's "simple Markowitz" alpha, §2.4). Purely a
    function of return history available as-of the rebalance date -- no lookahead."""
    r = returns_as_of.fillna(0.0)
    lam = np.log(2) / halflife_days
    weights = np.exp(-lam * np.arange(len(r))[::-1])
    weights = weights / weights.sum()
    ewma_daily = (r.values * weights[:, None]).sum(axis=0)
    return ewma_daily * horizon_days


def markowitz(
    target_relative: np.ndarray,
    target_vol_annual: float,
    l1_trust_region: float = 1.0,
    cov_lookback_days: int = 11,
    alpha_halflife_days: int = 252,
    horizon_days: int = 21,
    one_way_cost_bps: float = 15.0,
    cash_annual_rate: float = 0.0,
):
    """Convex, long-only optimization with a hard volatility cap and an L1 trust region
    around a target relative mix -- direct generalization of the paper's eq. (3) from 3
    assets to N factor sleeves. Solved with cvxpy/Clarabel, same solver family the paper
    uses (their §2.4). Objective: maximize next-period return net of trading cost, subject
    to long-only, a volatility cap, and staying within `l1_trust_region` of the target mix
    in relative-weight space -- this last constraint is what keeps the optimizer from
    concentrating entirely into whichever sleeve had the best trailing momentum, i.e. it
    bounds how much active bet the optimizer is allowed to take away from the benchmark mix.
    """
    import cvxpy as cp

    target_relative = np.asarray(target_relative, dtype=float)
    target_relative = target_relative / target_relative.sum()
    n = len(target_relative)
    daily_cash = cash_annual_rate / 252.0

    def _fn(date, returns_as_of, current_w):
        hist = returns_as_of.fillna(0.0)
        if len(hist) < max(cov_lookback_days, 30):
            return target_relative.copy()

        cov_daily = hist.tail(cov_lookback_days).cov().values
        Sigma = cov_daily * 252.0
        # numerical floor for stability on short windows
        Sigma = Sigma + np.eye(n) * 1e-8

        alpha = momentum_alpha(hist, halflife_days=alpha_halflife_days, horizon_days=horizon_days)
        r_cash_h = (1 + daily_cash) ** horizon_days - 1.0

        w = cp.Variable(n)
        s = one_way_cost_bps / 10_000.0
        obj = alpha @ w + r_cash_h * (1 - cp.sum(w)) - s * cp.norm1(w - current_w)

        constraints = [
            w >= 0,
            cp.sum(w) <= 1,
            cp.norm1(w - cp.sum(w) * target_relative) <= l1_trust_region * cp.sum(w),
            cp.quad_form(w, Sigma) <= target_vol_annual ** 2,
        ]
        prob = cp.Problem(cp.Maximize(obj), constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS)
            except Exception:
                return target_relative.copy() * min(1.0, target_vol_annual / max(trailing_volatility(hist, target_relative, cov_lookback_days), 1e-6))

        if w.value is None:
            return target_relative.copy()
        out = np.clip(np.asarray(w.value).flatten(), 0.0, None)
        if out.sum() > 1.0:
            out = out / out.sum()
        return out

    return _fn
