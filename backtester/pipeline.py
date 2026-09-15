"""Orchestrates Stages 01-08 with both gates enforced.

This module wires the other modules together in the order the board memo specifies. It
does not contain economic logic itself -- that lives in the Strategy Card (Stage 02) and
the signal function (Stage 05). See `examples/run_pipeline.py` for a concrete, runnable
walk-through against the attached NSE factor-index data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .data.pit_loader import PointInTimeDataset
from .data.sources import DataFeasibilityRegistry
from .engine.portfolio import PortfolioSimulator, SimulationResult, performance_metrics
from .gates.audit import AuditLog, GateBDecision
from .ingest.strategy_card import StrategyCard
from .library.ledger import Rung, StrategyLibrary
from .validation import research as research_validation
from .validation import portfolio as portfolio_validation


@dataclass
class PipelineContext:
    card: StrategyCard
    dataset: PointInTimeDataset
    registry: DataFeasibilityRegistry
    audit: AuditLog
    library: StrategyLibrary


def stage03_check_feasibility(registry: DataFeasibilityRegistry) -> dict:
    result = registry.check()
    print(registry.report())
    if not result["ok_to_proceed"]:
        print("\nSTAGE 03 BLOCKED:", result)
    return result


def stage05_build_and_run(
    dataset: PointInTimeDataset,
    asset_cols: list[str],
    weight_fn,
    cash_annual_rate: float,
    cost_model,
    rebalance_frequency: str = "M",
) -> SimulationResult:
    returns = dataset.df[asset_cols].pct_change()
    sim = PortfolioSimulator(
        asset_returns=returns,
        cost_model=cost_model,
        cash_annual_rate=cash_annual_rate,
        rebalance_frequency=rebalance_frequency,
    )
    return sim.run(weight_fn)


def stage06_research_validation(result: SimulationResult, cash_annual_rate: float, trial_sharpes: list[float]) -> dict:
    metrics = performance_metrics(result.value, cash_annual_rate)
    boot = research_validation.bootstrap_metrics(result.value, cash_annual_rate, n_replications=500)
    subperiods = research_validation.subperiod_sharpe(result.value, cash_annual_rate)
    n_obs = len(result.value)
    dsr = research_validation.deflated_sharpe_ratio(metrics["sharpe"], trial_sharpes, n_obs)
    return {
        "point_estimate": metrics,
        "bootstrap": boot,
        "subperiods": subperiods.to_dict(orient="records"),
        "deflated_sharpe_ratio": dsr,
    }


def stage07_portfolio_validation(
    result: SimulationResult,
    benchmark_returns: pd.DataFrame,
    existing_book_returns: Optional[pd.Series] = None,
) -> dict:
    strat_returns = result.value.pct_change().dropna()
    exposure = portfolio_validation.factor_exposure(strat_returns, benchmark_returns)
    tv = portfolio_validation.turnover_and_cvar_report(strat_returns, result.turnover)
    out = {"factor_exposure": exposure, "turnover_and_cvar": tv}
    if existing_book_returns is not None:
        out["incremental_ir"] = portfolio_validation.incremental_information_ratio(
            strat_returns, existing_book_returns
        )
    return out


def summarize_gate_b_evidence(research: dict, portfolio: dict) -> dict:
    return {
        "sharpe_point_estimate": research["point_estimate"]["sharpe"],
        "sharpe_ci95": research["bootstrap"]["sharpe"]["ci95"],
        "deflated_sharpe_ratio": research["deflated_sharpe_ratio"]["deflated_sharpe_ratio"],
        "annualized_turnover": portfolio["turnover_and_cvar"]["annualized_turnover"],
        "daily_cvar_5pct": portfolio["turnover_and_cvar"]["daily_cvar_5pct"],
        "factor_r_squared": portfolio["factor_exposure"]["r_squared"],
    }
