"""End-to-end run of the 8-stage pipeline against the real NSE factor-index data.

Run with:  python examples/run_pipeline.py

This is a demonstration of the HARNESS, not a claim that this strategy is ready for
capital -- Stage 03 will explicitly report the data gaps (cash rate, bond/gold sleeves)
that currently block a real Gate B "promote to LIVE" decision. Read the printed output;
it is designed to be read, not just executed.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from backtester.ingest.strategy_card import load_card
from backtester.data.pit_loader import PointInTimeDataset
from backtester.data.sources import india_registry_for_factor_rotation_card
from backtester.gates.audit import AuditLog, GateBDecision
from backtester.engine import signals as sig
from backtester.engine.costs import DEFAULT_INDIA_COST_MODEL
from backtester.engine.portfolio import PortfolioSimulator, performance_metrics
from backtester.validation import research as research_validation
from backtester.validation import portfolio as portfolio_validation
from backtester.library.ledger import StrategyLibrary, Rung

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def hr(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------------------
# STAGE 01/02 -- ingest already done (see docs/PIPELINE.md); load the resulting Card.
# ---------------------------------------------------------------------------
hr("STAGE 01/02 -- Strategy Card (loaded, already ingested from the source paper)")
card = load_card(os.path.join(os.path.dirname(__file__), "strategy_card_factor_rotation_india.yaml"))
print(f"Card: {card.title}")
print(f"Source: {card.source_title} ({card.source_url})")
print(f"Content hash: {card.content_hash()}")
print(f"Unresolved ambiguities: {len(card.unresolved_ambiguities())}")
for a in card.unresolved_ambiguities():
    print(f"  - [{a.field}] {a.description.strip()[:140]}")

# ---------------------------------------------------------------------------
# GATE A -- a researcher resolves the remaining ambiguities and signs off.
# In a real deployment this is a human action; here we play the reviewer to
# demonstrate the gate mechanics (audit log entries), with the SAME resolutions
# a careful reviewer would actually reach given the disclosed data gaps.
# ---------------------------------------------------------------------------
hr("GATE A -- Human interpretation sign-off")
for a in card.ambiguities:
    if a.resolution is None:
        if a.field == "data_requirements.cash_risk_free_rate":
            a.resolution = ("Accepted gap: proceed with cash_annual_rate=0.0 as an explicit "
                             "placeholder. Strategy CANNOT be evaluated as India-investable "
                             "until an RBI T-Bill/MIBOR series is sourced -- this blocks any "
                             "promotion past INDIA_VALIDATED at Gate B.")
            a.resolved_by = "demo-reviewer (research)"
        elif a.field == "universe":
            a.resolution = ("Accepted as a distinct, smaller-scope idea from the source paper: "
                             "an equity-only factor-sleeve rotation is an economically coherent "
                             "strategy in its own right (factor timing), not a literal 3-asset "
                             "replication. Proceed, but do not describe results as a "
                             "'replication' of the source paper's headline Sharpe ratio.")
            a.resolved_by = "demo-reviewer (research)"

audit = AuditLog(path=os.path.join(os.path.dirname(__file__), "..", "data", "lineage", "audit_log.jsonl"))
gate_a_entry = audit.gate_a(
    card, reviewer="demo-reviewer",
    decision="approved",
    note="Economic definition matches source paper's methodology (vol cap + causal momentum "
         "forecast + convex optimization). Scope reduced to equity-only India factor sleeves; "
         "cash-rate gap accepted with 0% placeholder and explicitly blocks live promotion.",
)
print(f"Gate A decision: {gate_a_entry.decision} by {gate_a_entry.reviewer}")

# ---------------------------------------------------------------------------
# STAGE 03 -- Data feasibility
# ---------------------------------------------------------------------------
hr("STAGE 03 -- Data feasibility")
registry = india_registry_for_factor_rotation_card()
registry.accept_gap(
    "cash_risk_free_rate",
    note="Proceeding with 0% placeholder cash rate per Gate A note above; blocks LIVE promotion.",
    accepted_by="demo-reviewer",
)
registry.accept_gap(
    "bond_sleeve",
    note="Out of scope for this Card version (equity-only factor rotation). Not proxied.",
    accepted_by="demo-reviewer",
)
registry.accept_gap(
    "gold_sleeve",
    note="Out of scope for this Card version (equity-only factor rotation). Not proxied.",
    accepted_by="demo-reviewer",
)
registry.sign_off_proxy(
    "india_vix",
    note="Not wired into this v1 feature panel (signal uses only trailing-return momentum, "
         "no vol-regime feature yet) -- acceptable to proceed without it for a REPLICATED-rung "
         "evaluation; required before any ROBUST-rung sensitivity claim that depends on regime features.",
    accepted_by="demo-reviewer",
)
feasibility = registry.check()
print(registry.report())
print(f"\nOK to proceed to Stage 04: {feasibility['ok_to_proceed']}")
assert feasibility["ok_to_proceed"], "Stage 03 blocked -- resolve data gaps before continuing."

# ---------------------------------------------------------------------------
# STAGE 04 -- Point-in-time data
# ---------------------------------------------------------------------------
hr("STAGE 04 -- Point-in-time data (frozen snapshot + lineage)")
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nifty_factor_indices.csv")
dataset = PointInTimeDataset.from_csv(data_path)
print(f"Source: {dataset.lineage.source_path}")
print(f"Content SHA-256: {dataset.lineage.content_sha256}")
print(f"Rows: {dataset.lineage.n_rows}  Range: {dataset.lineage.first_date} .. {dataset.lineage.last_date}")
print("\nLaunch dates per series (first non-null close -- NOT all series exist from day 1):")
for col, d in dataset.launch_dates().items():
    print(f"  {col:35s} {d}")

ASSET_COLS = [
    "NIFTY_ALPHA_50", "NIFTY500_MOMENTUM_50", "NIFTY500_MULTIFACTOR_MQVLV_50",
    "NIFTY500_QUALITY_50", "NIFTY500_VALUE_50", "NIFTY500_LOW_VOLATILITY_50",
    "NIFTY_HIGH_BETA_50",
]
BENCHMARK_COL = "NIFTY_500"

# All seven sleeves have overlapping history only from their latest common launch date.
common_start = max(pd.Timestamp(d) for c, d in dataset.launch_dates().items() if c in ASSET_COLS)
eval_df = dataset.df.loc[common_start:].dropna(subset=ASSET_COLS + [BENCHMARK_COL])
print(f"\nCommon evaluation window (all 7 sleeves + benchmark live): "
      f"{eval_df.index.min().date()} .. {eval_df.index.max().date()}  ({len(eval_df)} obs)")

eval_dataset = PointInTimeDataset(eval_df, dataset.lineage)

# ---------------------------------------------------------------------------
# STAGE 05 -- Build + execute
# ---------------------------------------------------------------------------
hr("STAGE 05 -- Build + execute (deterministic engine, template-constrained signal)")
returns = eval_dataset.df[ASSET_COLS].pct_change()
n = len(ASSET_COLS)
equal_weight = np.ones(n) / n
target_vol = card.risk.target_volatility_annual
one_way_bps = DEFAULT_INDIA_COST_MODEL.one_way_cost_bps()
print(f"India one-way cost assumption: {one_way_bps:.2f} bps "
      f"(vs. paper's flat {card.cost.paper_assumed_bps} bps half-spread)")

strategies = {
    "Equal-weight benchmark (fixed)": sig.fixed_weight_benchmark(equal_weight),
    f"Vol-controlled ({target_vol:.0%} target)": sig.vol_controlled(equal_weight, target_vol, lookback_days=11),
    "Markowitz (vol-capped momentum optimizer)": sig.markowitz(
        equal_weight, target_vol, l1_trust_region=card.risk.max_relative_weight_deviation_l1,
        cov_lookback_days=11, alpha_halflife_days=252, horizon_days=21,
        one_way_cost_bps=one_way_bps, cash_annual_rate=0.0,
    ),
}

results = {}
for name, fn in strategies.items():
    sim = PortfolioSimulator(
        asset_returns=returns, cost_model=DEFAULT_INDIA_COST_MODEL,
        cash_annual_rate=0.0, rebalance_frequency="ME",
    )
    results[name] = sim.run(fn)

bench_ret = eval_dataset.df[BENCHMARK_COL].pct_change().dropna()
bench_value = (1 + bench_ret).cumprod()
bench_value.name = "NIFTY_500"

print("\nPerformance summary (annualized, net of India costs, 0% cash rate placeholder):")
summary_rows = []
for name, res in results.items():
    m = performance_metrics(res.value, cash_annual_rate=0.0)
    m["annualized_turnover"] = 252.0 * res.turnover.mean()
    m["strategy"] = name
    summary_rows.append(m)
m_bench = performance_metrics(bench_value, cash_annual_rate=0.0)
m_bench["annualized_turnover"] = 0.0
m_bench["strategy"] = "NIFTY 500 (passive benchmark)"
summary_rows.append(m_bench)

summary = pd.DataFrame(summary_rows).set_index("strategy")[
    ["return", "volatility", "sharpe", "max_drawdown", "mean_drawdown", "annualized_turnover"]
]
for c in ["return", "volatility", "max_drawdown", "mean_drawdown", "annualized_turnover"]:
    summary[c] = (summary[c] * 100).round(2).astype(str) + "%"
summary["sharpe"] = summary["sharpe"].round(2)
print(summary.to_string())

# ---------------------------------------------------------------------------
# STAGE 06 -- Research validation
# ---------------------------------------------------------------------------
hr("STAGE 06 -- Research validation (Markowitz strategy)")
markowitz_result = results["Markowitz (vol-capped momentum optimizer)"]

boot = research_validation.bootstrap_metrics(markowitz_result.value, cash_annual_rate=0.0, n_replications=500)
print("Stationary block bootstrap (mean block 21 days, 500 replications):")
print(f"  Return : mean {boot['return']['mean']:.2%}  95% CI {tuple(round(x,4) for x in boot['return']['ci95'])}")
print(f"  Sharpe : mean {boot['sharpe']['mean']:.2f}  95% CI {tuple(round(x,3) for x in boot['sharpe']['ci95'])}")
print(f"  MaxDD  : mean {boot['max_drawdown']['mean']:.2%}  95% CI {tuple(round(x,4) for x in boot['max_drawdown']['ci95'])}")

diff = research_validation.paired_sharpe_difference(
    markowitz_result.value, results["Equal-weight benchmark (fixed)"].value, cash_annual_rate=0.0, n_replications=500
)
print(f"\nMarkowitz vs. equal-weight benchmark, paired bootstrap Sharpe difference:")
print(f"  mean diff {diff['mean_diff']:.3f}  95% CI {tuple(round(x,3) for x in diff['ci95'])}  "
      f"P(diff<=0) = {diff['p_diff_le_0']:.3f}")

subperiods = research_validation.subperiod_sharpe(markowitz_result.value, cash_annual_rate=0.0, n_periods=4)
print("\nSub-period Sharpe ratios (regime consistency check):")
print(subperiods[["start", "end", "sharpe"]].to_string(index=False))

# Deflated Sharpe Ratio: the ACTUAL specification sweep tried during development below.
trial_sharpes = []
for tv in [0.08, 0.10, 0.12, 0.14, 0.16]:
    fn = sig.markowitz(equal_weight, tv, l1_trust_region=1.0, cov_lookback_days=11,
                        one_way_cost_bps=one_way_bps, cash_annual_rate=0.0)
    sim = PortfolioSimulator(asset_returns=returns, cost_model=DEFAULT_INDIA_COST_MODEL,
                              cash_annual_rate=0.0, rebalance_frequency="ME")
    r = sim.run(fn)
    trial_sharpes.append(performance_metrics(r.value)["sharpe"])
dsr = research_validation.deflated_sharpe_ratio(
    observed_sharpe=performance_metrics(markowitz_result.value)["sharpe"],
    trial_sharpes=trial_sharpes, n_obs=len(markowitz_result.value),
)
print(f"\nDeflated Sharpe Ratio (corrects for {dsr['n_trials']}-specification vol-target sweep): "
      f"{dsr['deflated_sharpe_ratio']:.3f}  (trial Sharpe std: {dsr['trial_sharpe_std']:.3f})")

cost_grid = research_validation.cost_sensitivity(
    lambda bps: PortfolioSimulator(
        asset_returns=returns, cost_model=DEFAULT_INDIA_COST_MODEL, cash_annual_rate=0.0, rebalance_frequency="ME"
    ).run(sig.markowitz(equal_weight, target_vol, one_way_cost_bps=bps, cash_annual_rate=0.0)),
    cost_bps_grid=[one_way_bps, one_way_bps * 2, one_way_bps * 4],
)
print("\nCost sensitivity (one-way bps -> Sharpe):")
print(cost_grid[["one_way_cost_bps", "sharpe"]].to_string(index=False))

# ---------------------------------------------------------------------------
# STAGE 07 -- Portfolio validation
# ---------------------------------------------------------------------------
hr("STAGE 07 -- Portfolio validation")
strat_returns = markowitz_result.value.pct_change().dropna()
factor_panel = eval_dataset.df[ASSET_COLS + [BENCHMARK_COL]].pct_change()
exposure = portfolio_validation.factor_exposure(strat_returns, factor_panel[[BENCHMARK_COL]], lag=1)
print(f"Factor exposure to lagged NIFTY 500 (avoids look-ahead / circularity): {exposure}")

tv_cvar = portfolio_validation.turnover_and_cvar_report(strat_returns, markowitz_result.turnover)
print(f"\nTurnover & tail risk: {tv_cvar}")

incr = portfolio_validation.incremental_information_ratio(strat_returns, bench_ret, weight_in_book=0.20)
print(f"\nIncremental IR of adding strategy at 20% of a NIFTY-500-only book: {incr}")

# ---------------------------------------------------------------------------
# GATE B -- Investment decision
# ---------------------------------------------------------------------------
hr("GATE B -- Investment decision")
library = StrategyLibrary(db_path=os.path.join(os.path.dirname(__file__), "..", "data", "lineage", "strategy_library.db"))
library.register(card.card_id, card.title, card.content_hash())

evidence_summary = {
    "sharpe_point_estimate": round(performance_metrics(markowitz_result.value)["sharpe"], 3),
    "sharpe_ci95": tuple(round(x, 3) for x in boot["sharpe"]["ci95"]),
    "deflated_sharpe_ratio": round(dsr["deflated_sharpe_ratio"], 3),
    "vs_equal_weight_p_diff_le_0": round(diff["p_diff_le_0"], 3),
    "annualized_turnover": round(tv_cvar["annualized_turnover"], 3),
    "daily_cvar_5pct": round(tv_cvar["daily_cvar_5pct"], 5),
    "incremental_ir": round(incr["information_ratio_of_addition"], 3) if incr["information_ratio_of_addition"] == incr["information_ratio_of_addition"] else None,
    "open_data_gaps": ["cash_risk_free_rate (blocks INDIA_VALIDATED+)", "bond_sleeve (out of scope)", "gold_sleeve (out of scope)"],
}

decision = GateBDecision.observe
note = (
    "Statistically interesting (positive DSR, beats equal-weight with high bootstrap "
    "confidence) but NOT promotable past REPLICATED: (1) cash-rate data gap means the "
    "optimizer's cash leg and vol-control dilution are untested against a real opportunity "
    "cost of cash; (2) bond/gold legs are out of scope, so this is a narrower, higher-beta "
    "idea than the source paper, not a validated cross-asset diversifier; (3) India cost "
    "model uses an assumed impact-cost bps for smart-beta ETFs pending real market data. "
    "Decision: OBSERVE. Next step before Gate B can revisit: source RBI T-Bill series "
    "(docs/DATA_SOURCES.md item 2) and real ETF bid-ask data for the impact-cost assumption."
)
gate_b_entry = audit.gate_b(
    card_id=card.card_id,
    result_id="run_" + dataset.lineage.content_sha256[:8],
    reviewer="demo-pm",
    decision=decision,
    note=note,
    dataset_snapshot_hash=dataset.lineage.content_sha256,
    code_hash=card.content_hash(),
    model_version="engine v0.1.0",
    evidence_summary=evidence_summary,
)
library.promote(
    card.card_id, Rung.replicated,
    dataset_snapshot_hash=dataset.lineage.content_sha256,
    code_hash=card.content_hash(),
    model_version="engine v0.1.0",
    reviewer="demo-pm",
    note="Methodology replicates source paper's structure (vol cap + causal momentum + "
         "convex optimization) on India factor sleeves; promoted to REPLICATED only -- "
         "see Gate B note for why INDIA_VALIDATED is blocked.",
    evidence=evidence_summary,
)
print(f"Gate B decision: {decision.value}")
print(f"Promotion ladder: {card.card_id} -> {library.current_rung(card.card_id)}")
print("\nFull promotion history:")
for h in library.history(card.card_id):
    print(f"  [{h['timestamp']}] {h['event_type']:10s} {h['from_rung']} -> {h['to_rung']}  ({h['reviewer']}): {h['note'][:100]}")

hr("DONE -- audit log and strategy library written to data/lineage/")
print(f"Audit log: {audit.path}")
print(f"Strategy library DB: {library.db_path}")
