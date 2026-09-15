"""Stage 07 -- Portfolio validation.

A strategy can be statistically robust (survives Stage 06) and still be useless to the
book -- either because it's relabeled beta, or because it duplicates something already
promoted (docs/PIPELINE.md Stage 07). This module computes: factor exposure (regression
against the benchmark and against each library sleeve, LAGGED to avoid look-ahead in the
exposure check itself -- see docs/DATA_SOURCES.md's note on circularity), turnover, CVaR,
and incremental information ratio versus an existing book/benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def factor_exposure(strategy_returns: pd.Series, factor_returns: pd.DataFrame, lag: int = 1) -> dict:
    """OLS beta of strategy returns on LAGGED factor returns. Lagging by `lag` days avoids
    the circularity flagged in docs/DATA_SOURCES.md: if the strategy's own signal is built
    from these same factor indices, a contemporaneous regression would just recover the
    portfolio weights, not genuine exposure."""
    X = factor_returns.shift(lag).loc[strategy_returns.index].fillna(0.0)
    y = strategy_returns.reindex(X.index).fillna(0.0)
    X_mat = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(X_mat, y.values, rcond=None)
    fitted = X_mat @ beta
    resid = y.values - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y.values - y.values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    betas = {"alpha_daily": float(beta[0])}
    betas.update({col: float(b) for col, b in zip(X.columns, beta[1:])})
    betas["r_squared"] = r2
    return betas


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Conditional Value at Risk (Expected Shortfall) at level alpha, daily."""
    r = returns.dropna().values
    if len(r) == 0:
        return float("nan")
    var_threshold = np.percentile(r, alpha * 100)
    tail = r[r <= var_threshold]
    return float(tail.mean()) if len(tail) > 0 else float(var_threshold)


def incremental_information_ratio(
    strategy_returns: pd.Series,
    existing_book_returns: pd.Series,
    weight_in_book: float = 0.10,
) -> dict:
    """Information ratio of adding the strategy to an existing book at a given weight,
    versus the book alone -- answers "does this make the portfolio better", not just
    "is this strategy good in isolation" (docs/PIPELINE.md Stage 07 green flags)."""
    common = strategy_returns.index.intersection(existing_book_returns.index)
    s = strategy_returns.loc[common].fillna(0.0)
    b = existing_book_returns.loc[common].fillna(0.0)
    combined = (1 - weight_in_book) * b + weight_in_book * s

    def ann_sharpe(r):
        vol = r.std() * np.sqrt(252)
        return (r.mean() * 252) / vol if vol > 0 else float("nan")

    active = combined - b
    te = active.std() * np.sqrt(252)
    ir = (active.mean() * 252) / te if te > 0 else float("nan")
    return {
        "book_alone_sharpe": ann_sharpe(b),
        "book_plus_strategy_sharpe": ann_sharpe(combined),
        "information_ratio_of_addition": ir,
        "tracking_error_of_addition": te,
        "weight_in_book": weight_in_book,
    }


def turnover_and_cvar_report(
    strategy_returns: pd.Series,
    turnover_series: pd.Series,
    alpha: float = 0.05,
) -> dict:
    return {
        "annualized_turnover": float(252.0 * turnover_series.mean()),
        "daily_cvar_5pct": cvar(strategy_returns, alpha=alpha),
        "worst_single_day": float(strategy_returns.min()),
    }


def similarity_to_library(new_returns: pd.Series, library_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation of a candidate strategy's returns to every strategy already in the
    library -- the "fingerprint" / similarity map check (Stage 08's research graph exists
    partly to make this lookup possible instead of re-discovering duplication by memory)."""
    rows = []
    for name, r in library_returns.items():
        common = new_returns.index.intersection(r.index)
        if len(common) < 30:
            continue
        corr = float(np.corrcoef(new_returns.loc[common], r.loc[common])[0, 1])
        rows.append({"library_strategy": name, "correlation": corr})
    return pd.DataFrame(rows).sort_values("correlation", ascending=False) if rows else pd.DataFrame(
        columns=["library_strategy", "correlation"]
    )
