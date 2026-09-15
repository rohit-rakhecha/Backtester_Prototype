"""Stage 06 -- Research validation.

A single full-sample Sharpe ratio overstates confidence by construction (docs/PIPELINE.md
Stage 06). This module makes three checks mandatory computation rather than something a
researcher has to remember: (1) a stationary block bootstrap confidence interval on the
headline metrics, mirroring the paper's Appendix C; (2) a sub-period breakdown, so a lucky
regime isn't mistaken for skill (paper's Table 2); (3) a Deflated Sharpe Ratio that corrects
for the number of specifications actually searched (paper's Appendix E).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..engine.portfolio import performance_metrics


def stationary_bootstrap_indices(n: int, mean_block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sampler (paper Appendix C)."""
    p = 1.0 / mean_block_len
    idx = np.empty(n, dtype=int)
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return idx


def bootstrap_metrics(
    value: pd.Series,
    cash_annual_rate: float = 0.0,
    n_replications: int = 2000,
    mean_block_len: int = 21,
    seed: int = 0,
) -> dict:
    """95% bootstrap intervals for return, Sharpe, max drawdown -- paper's Table 8."""
    rets = value.pct_change().dropna().values
    n = len(rets)
    rng = np.random.default_rng(seed)

    boot_return, boot_sharpe, boot_maxdd = [], [], []
    for _ in range(n_replications):
        idx = stationary_bootstrap_indices(n, mean_block_len, rng)
        sample_rets = rets[idx]
        sample_val = np.cumprod(1 + sample_rets)
        n_years = n / 252.0
        cagr = sample_val[-1] ** (1.0 / n_years) - 1.0
        vol = np.std(sample_rets, ddof=1) * np.sqrt(252)
        sharpe = (cagr - cash_annual_rate) / vol if vol > 0 else np.nan
        running_max = np.maximum.accumulate(sample_val)
        max_dd = np.min(sample_val / running_max - 1.0)
        boot_return.append(cagr)
        boot_sharpe.append(sharpe)
        boot_maxdd.append(max_dd)

    def ci(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        return float(np.mean(arr)), (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    r_mean, r_ci = ci(boot_return)
    s_mean, s_ci = ci(boot_sharpe)
    d_mean, d_ci = ci(boot_maxdd)
    return {
        "return": {"mean": r_mean, "ci95": r_ci},
        "sharpe": {"mean": s_mean, "ci95": s_ci},
        "max_drawdown": {"mean": d_mean, "ci95": d_ci},
        "n_replications": n_replications,
        "mean_block_len": mean_block_len,
    }


def paired_sharpe_difference(
    value_a: pd.Series,
    value_b: pd.Series,
    cash_annual_rate: float = 0.0,
    n_replications: int = 2000,
    mean_block_len: int = 21,
    seed: int = 0,
) -> dict:
    """Bootstrap the DIFFERENCE in Sharpe ratio between two portfolios, paired across
    replications (paper Table 9) -- answers "is A significantly better than B", not just
    "does A look better than B" on the point estimate."""
    ra = value_a.pct_change().dropna()
    rb = value_b.pct_change().dropna()
    common = ra.index.intersection(rb.index)
    ra, rb = ra.loc[common].values, rb.loc[common].values
    n = len(common)
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_replications):
        idx = stationary_bootstrap_indices(n, mean_block_len, rng)
        sa, sb = ra[idx], rb[idx]

        def sharpe_of(r):
            val = np.cumprod(1 + r)
            n_years = n / 252.0
            cagr = val[-1] ** (1.0 / n_years) - 1.0
            vol = np.std(r, ddof=1) * np.sqrt(252)
            return (cagr - cash_annual_rate) / vol if vol > 0 else np.nan

        diffs.append(sharpe_of(sa) - sharpe_of(sb))
    diffs = np.array(diffs)
    diffs = diffs[~np.isnan(diffs)]
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p_diff_le_0": float(np.mean(diffs <= 0)),
    }


def subperiod_sharpe(value: pd.Series, cash_annual_rate: float = 0.0, n_periods: int = 4) -> pd.DataFrame:
    """Sharpe ratio by equal-length subperiods (paper Table 2) -- exposes regime-dependence
    a single full-sample number hides."""
    idx = value.index
    edges = pd.date_range(idx[0], idx[-1], periods=n_periods + 1)
    rows = []
    for i in range(n_periods):
        seg = value.loc[edges[i]:edges[i + 1]]
        if len(seg) < 10:
            continue
        m = performance_metrics(seg, cash_annual_rate)
        rows.append({"start": edges[i].date(), "end": edges[i + 1].date(), **m})
    return pd.DataFrame(rows)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_sharpes: list[float],
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), used exactly as the paper
    uses it in Appendix E to correct the headline Sharpe for the hyper-parameter sweep that
    produced it. `trial_sharpes` must be the ACTUAL set of specifications explored --
    understating this count is the red flag called out in docs/PIPELINE.md Stage 06.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    N = len(trial_sharpes)
    sr_std = float(np.std(trial_sharpes, ddof=1)) if N > 1 else 0.0
    euler_gamma = 0.5772156649
    if N > 1 and sr_std > 0:
        expected_max_sr = sr_std * (
            (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / N)
            + euler_gamma * stats.norm.ppf(1 - 1.0 / (N * np.e))
        )
    else:
        expected_max_sr = 0.0

    denom = np.sqrt(max(1e-12, 1 - skew * observed_sharpe + (kurt - 1) / 4.0 * observed_sharpe ** 2))
    z = (observed_sharpe - expected_max_sr) * np.sqrt(n_obs - 1) / denom
    dsr = float(stats.norm.cdf(z))
    return {
        "deflated_sharpe_ratio": dsr,
        "expected_max_sharpe_under_null": float(expected_max_sr),
        "n_trials": int(N),
        "trial_sharpe_std": sr_std,
    }


def cost_sensitivity(
    simulate_fn,
    cost_bps_grid: list[float] = (5.0, 15.0, 30.0, 50.0),
) -> pd.DataFrame:
    """Re-runs a strategy at multiple one-way cost assumptions -- paper's Appendix G,
    Table 18. `simulate_fn(one_way_cost_bps) -> SimulationResult` must be supplied by the
    caller (keeps this module independent of the specific engine wiring)."""
    rows = []
    for bps in cost_bps_grid:
        result = simulate_fn(bps)
        m = performance_metrics(result.value)
        rows.append({"one_way_cost_bps": bps, **m})
    return pd.DataFrame(rows)
