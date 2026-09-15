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
        if len(arr) == 0:
            # Every replication produced NaN (e.g. a strategy that never traded and
            # had zero realized volatility) -- surface this as an explicit gap rather
            # than crashing with an opaque IndexError deep inside numpy.percentile.
            return float("nan"), (float("nan"), float("nan"))
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


def signal_diagnostics(
    diagnostics_log: list[dict],
    asset_returns: pd.DataFrame,
    horizon_days: int = 21,
    card_signal_description: str | None = None,
) -> dict:
    """Evaluate whether the alpha the Markowitz optimizer actually acted on
    (engine/signals.py `markowitz(..., diagnostics=log)`) predicted forward returns, on
    this exact dataset. This is Stage 06's direct, mechanism-level answer to "why did
    equal-weight outperform the optimizer": if the Information Coefficient computed here
    is near zero, the tilt away from equal-weight was -- empirically, in this sample --
    close to noise, so an optimizer trusting it does no better (and net of turnover cost,
    often worse) than doing nothing. This is also the explicit link back to Stage 02: the
    Card's `signal.description` is a CLAIM about what should predict returns; this is the
    evidence for or against that claim holding in the data actually used, not a restatement
    of what the source paper claimed for ITS market and period.

    For every rebalance in the log: cross-sectional Spearman rank correlation between the
    alpha vector used AT that rebalance and each asset's realized forward `horizon_days`
    return (looked up strictly AFTER the rebalance date -- this is a pure post-hoc
    diagnostic, never fed back into the strategy itself). This is the standard quant-
    research "Information Coefficient" (IC): IC > 0 means the signal ranked winners above
    losers more often than chance; IC ~ 0 means it didn't; IC < 0 means it was
    anti-predictive.
    """
    if not diagnostics_log:
        return {
            "mean_ic": float("nan"), "hit_rate": float("nan"), "n_rebalances_evaluated": 0,
            "interpretation": "No diagnostics recorded -- pass a `diagnostics=[]` list into "
                               "engine.signals.markowitz() to enable this check.",
            "card_signal_claim": card_signal_description,
        }

    dates = asset_returns.index
    records = []
    for entry in diagnostics_log:
        d = pd.Timestamp(entry["date"])
        pos = dates.searchsorted(d)
        if pos + 1 + horizon_days > len(dates):
            continue
        fwd_ret = (1 + asset_returns.iloc[pos + 1: pos + 1 + horizon_days]).prod() - 1
        alpha_series = pd.Series(entry["alpha"])
        common = alpha_series.index.intersection(fwd_ret.index)
        if len(common) < 3:
            continue
        ic = alpha_series.loc[common].corr(fwd_ret.loc[common], method="spearman")
        if ic == ic:  # not NaN
            records.append({"date": d, "ic": ic})

    if not records:
        return {
            "mean_ic": float("nan"), "hit_rate": float("nan"), "n_rebalances_evaluated": 0,
            "interpretation": "Diagnostics were recorded but none had enough forward-return "
                               "history to evaluate (e.g. every rebalance was too close to "
                               "the end of the sample).",
            "card_signal_claim": card_signal_description,
        }

    ic_df = pd.DataFrame(records)
    mean_ic = float(ic_df["ic"].mean())
    hit_rate = float((ic_df["ic"] > 0).mean())

    if mean_ic > 0.05:
        interp = (
            "Positive mean IC: the alpha signal DID rank future winners above losers more "
            "often than chance in this dataset -- an optimizer tilting toward it has a real "
            "(if modest) edge to exploit, net of whether trading costs eat it."
        )
    elif mean_ic > -0.05:
        interp = (
            "Mean IC is close to zero: the alpha signal's cross-sectional ranking of assets "
            "was essentially UNCORRELATED with what actually outperformed over the next "
            "period. This is direct evidence for why deviating from equal-weight did not "
            "clearly help here -- the optimizer was tilting toward a signal that, "
            "empirically, was not predictive in this sample. It does not mean the Strategy "
            "Card's signal claim (Stage 02) is wrong in general -- only that this specific "
            "dataset/period did not bear it out; a different universe, period, or horizon "
            "could show a different IC."
        )
    else:
        interp = (
            "Negative mean IC: the alpha signal ranked future LOSERS above winners more "
            "often than chance -- tilting toward it was actively counterproductive here."
        )

    return {
        "mean_ic": mean_ic,
        "hit_rate": hit_rate,
        "n_rebalances_evaluated": len(ic_df),
        "interpretation": interp,
        "card_signal_claim": card_signal_description,
    }


def decompose_vs_equal_weight(
    equal_weight_result, vol_controlled_result, markowitz_result, cash_annual_rate: float = 0.0
) -> dict:
    """Splits the Markowitz strategy's Sharpe difference from static equal-weight into two
    effects, valid because all three legs share the same equal-weight target relative mix
    (see examples -- vol_controlled() and markowitz() are both called with
    target_relative=equal_weight):
      - vol-cap effect    : Sharpe(vol-controlled equal-weight) - Sharpe(static equal-weight)
      - alpha-tilt effect : Sharpe(Markowitz) - Sharpe(vol-controlled equal-weight)
    This answers "why did equal-weight outperform the optimizer" at the MECHANISM level
    (which of the two things the optimizer does -- cap risk, or tilt toward alpha -- is
    responsible) rather than only at the outcome level (one Sharpe number).
    """
    from ..engine.portfolio import performance_metrics

    m_eq = performance_metrics(equal_weight_result.value, cash_annual_rate)
    m_vc = performance_metrics(vol_controlled_result.value, cash_annual_rate)
    m_mw = performance_metrics(markowitz_result.value, cash_annual_rate)

    vol_cap_effect = m_vc["sharpe"] - m_eq["sharpe"]
    alpha_tilt_effect = m_mw["sharpe"] - m_vc["sharpe"]

    lines = [
        f"Static equal-weight Sharpe:            {m_eq['sharpe']:.3f}",
        f"+ vol-cap effect:                      {vol_cap_effect:+.3f}  -> vol-controlled equal-weight Sharpe: {m_vc['sharpe']:.3f}",
        f"+ alpha-tilt effect (the optimizer):   {alpha_tilt_effect:+.3f}  -> Markowitz Sharpe: {m_mw['sharpe']:.3f}",
    ]
    if vol_cap_effect < 0:
        lines.append(
            "-> Vol-capping ALONE hurt Sharpe here: it trades return for a smoother ride, "
            "and a lower-volatility strategy does not automatically have a higher Sharpe."
        )
    if alpha_tilt_effect > 0:
        lines.append(
            "-> The alpha tilt ADDED Sharpe on top of the vol-cap baseline -- check "
            "signal_diagnostics() to see whether that recovery is attributable to real "
            "predictive skill (positive Information Coefficient) or to the tilt happening "
            "to reduce concentration/drawdown risk rather than to picking winners."
        )
    else:
        lines.append(
            "-> The alpha tilt did NOT add Sharpe on top of the vol-cap baseline -- check "
            "signal_diagnostics() for the Information Coefficient; a near-zero or negative "
            "IC would directly explain this."
        )

    return {
        "equal_weight_sharpe": m_eq["sharpe"],
        "vol_controlled_sharpe": m_vc["sharpe"],
        "markowitz_sharpe": m_mw["sharpe"],
        "vol_cap_effect": vol_cap_effect,
        "alpha_tilt_effect": alpha_tilt_effect,
        "narrative": "\n".join(lines),
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
