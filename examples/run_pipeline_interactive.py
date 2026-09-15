"""Interactive end-to-end run: upload ANY paper, choose/upload ANY data, run all 8 stages.

Unlike `examples/run_pipeline.py` (a fixed, non-interactive worked example pinned to the
attached factor-rotation Card and the attached NSE CSV -- kept as a known-good regression
demo), this script is the general entry point: it asks YOU to upload a research paper,
builds the Strategy Card interactively from what Stage 01 mines out of it, asks YOU which
India data sources to use or upload for whatever asset classes the paper touches on (not
just equities -- bonds, mutual funds, commodities, rates, anything), and only then runs
the same deterministic engine and validation stages every Card in this repo runs through.

Run with:  python examples/run_pipeline_interactive.py
(or via the `%run` cell in examples/run_in_jupyter.ipynb, which is the intended way to
run this in Colab -- prompts render as inline text boxes there, and PDF/CSV uploads use
Colab's native file-upload widget.)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from backtester.ingest.parser import comprehensive_ingest
from backtester.ingest.strategy_card import build_card_interactively, save_card
from backtester.ingest.interactive_io import upload_file, prompt_text, prompt_choice, in_colab
from backtester.data.pit_loader import merge_universe_components
from backtester.data.sources import discover_datasets_interactively
from backtester.gates.audit import AuditLog, GateBDecision
from backtester.engine import signals as sig
from backtester.engine.costs import DEFAULT_INDIA_COST_MODEL
from backtester.engine.portfolio import PortfolioSimulator, performance_metrics
from backtester.validation import research as research_validation
from backtester.validation import portfolio as portfolio_validation
from backtester.library.ledger import StrategyLibrary, Rung

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hr(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------------------
# STAGE 01 -- Ingest: YOU upload the paper.
# ---------------------------------------------------------------------------
hr("STAGE 01 -- Ingest (upload your research paper)")
print(f"Running in Colab: {in_colab()}")
paper_path = upload_file("Upload the research paper PDF you want to turn into a strategy", save_dir="data/uploads/papers")
if not paper_path:
    raise SystemExit("No paper uploaded -- nothing to ingest. Re-run this cell and upload a PDF.")

report = comprehensive_ingest(paper_path)
# print_ingest_report(report) is called inside build_card_interactively below, so the
# abstract/claims/candidate-parameters are shown right before the field they inform.

# ---------------------------------------------------------------------------
# STAGE 02 -- Strategy Card, built interactively FROM what Stage 01 found.
# ---------------------------------------------------------------------------
hr("STAGE 02 -- Strategy Card (built interactively from the ingest report above)")
card = build_card_interactively(report)
card_yaml_path = os.path.join(REPO_ROOT, "data", "lineage", f"{card.card_id}.yaml")
os.makedirs(os.path.dirname(card_yaml_path), exist_ok=True)
save_card(card, card_yaml_path)
print(f"Card saved to {card_yaml_path}")

# ---------------------------------------------------------------------------
# GATE A -- resolve every ambiguity, then approve/reject.
# ---------------------------------------------------------------------------
hr("GATE A -- Human interpretation sign-off")
if card.unresolved_ambiguities():
    print(f"{len(card.unresolved_ambiguities())} ambiguities need resolving before this Card can be approved:\n")
    for a in card.unresolved_ambiguities():
        print(f"[{a.field}] {a.description}")
        a.resolution = prompt_text("  Resolution (how is this being handled?)")
        a.resolved_by = prompt_text("  Resolved by (your name/role)", default="reviewer")

reviewer = prompt_text("Gate A reviewer name", default="reviewer")
decision = prompt_choice("Gate A decision", ["approved", "rejected"])
note = prompt_text("Gate A note (why -- required)")
audit = AuditLog(path=os.path.join(REPO_ROOT, "data", "lineage", "audit_log.jsonl"))
gate_a_entry = audit.gate_a(card, reviewer=reviewer, decision=decision, note=note)
print(f"Gate A decision: {gate_a_entry.decision} by {gate_a_entry.reviewer}")
if decision != "approved":
    raise SystemExit("Card rejected at Gate A -- stopping here. Re-run and address the ambiguities to proceed.")

# ---------------------------------------------------------------------------
# STAGE 03 -- Data feasibility & source discovery, interactive, any asset class.
# ---------------------------------------------------------------------------
hr("STAGE 03 -- Data feasibility & source discovery")
registry, components = discover_datasets_interactively(
    card, report.candidate_parameters["asset_classes_mentioned"], data_dir=os.path.join(REPO_ROOT, "data", "raw"),
)
print("\n" + registry.report())
feasibility = registry.check()
print(f"\nOK to proceed to Stage 04: {feasibility['ok_to_proceed']}")
if not feasibility["ok_to_proceed"]:
    raise SystemExit(f"Stage 03 blocked: {feasibility}")

tradable = [c for c in components if c.asset_class != "cash_rate"]
cash_components = [c for c in components if c.asset_class == "cash_rate"]
if len(tradable) < 2:
    raise SystemExit(
        f"Only {len(tradable)} tradable instrument(s) selected -- need at least 2 to build a "
        "portfolio. Re-run Stage 03 and select/upload more."
    )

# ---------------------------------------------------------------------------
# STAGE 04 -- Point-in-time data: merge whatever was selected/uploaded, whichever files
# they came from, into one date-indexed panel.
# ---------------------------------------------------------------------------
hr("STAGE 04 -- Point-in-time data (merged from your Stage 03 selections)")
dataset = merge_universe_components(tradable)
print(f"Merged {len(tradable)} tradable series from: {dataset.lineage.source_path}")
print(f"Rows: {dataset.lineage.n_rows}  Range: {dataset.lineage.first_date} .. {dataset.lineage.last_date}")
print("\nLaunch dates per series (first non-null value):")
for col, d in dataset.launch_dates().items():
    print(f"  {col:30s} {d}")

ASSET_COLS = list(dataset.df.columns)
common_start = max(pd.Timestamp(d) for d in dataset.launch_dates().values())
eval_df = dataset.df.loc[common_start:].dropna(subset=ASSET_COLS)
print(f"\nCommon evaluation window (all {len(ASSET_COLS)} series live): "
      f"{eval_df.index.min().date()} .. {eval_df.index.max().date()}  ({len(eval_df)} obs)")

cash_annual_rate = 0.0
if cash_components:
    cash_dataset = merge_universe_components(cash_components)
    cash_col = cash_components[0].label
    cash_series = cash_dataset.df[cash_col].reindex(eval_df.index).ffill()
    # Simplifying assumption, disclosed rather than silent: treat the selected cash-rate
    # column as an ANNUALIZED PERCENTAGE YIELD (e.g. a T-Bill yield printed as "6.5"), and
    # use its full-sample mean as a constant annual cash rate. A real deployment should
    # instead feed a time-varying DAILY rate into the engine -- flagged here, not hidden.
    cash_annual_rate = float(cash_series.mean()) / 100.0
    print(f"\nCash-rate column '{cash_col}' selected -- using its mean ({cash_annual_rate:.2%} "
          "annualized) as a CONSTANT rate for this run. This is a simplification: a real "
          "deployment needs a time-varying daily rate, not a full-sample average.")
else:
    print("\nNo cash-rate data source selected in Stage 03 -- cash_annual_rate stays at the "
          "0% placeholder (see the accepted gap logged in the Stage 03 report above).")

# ---------------------------------------------------------------------------
# STAGE 05 -- Build + execute.
# ---------------------------------------------------------------------------
hr("STAGE 05 -- Build + execute (deterministic engine, template-constrained signal)")
returns = eval_df[ASSET_COLS].pct_change()
n = len(ASSET_COLS)
equal_weight = np.ones(n) / n
target_vol = card.risk.target_volatility_annual
one_way_bps = DEFAULT_INDIA_COST_MODEL.one_way_cost_bps()
print(f"Universe ({n} instruments): {ASSET_COLS}")
print(f"India one-way cost assumption: {one_way_bps:.2f} bps "
      f"(vs. paper's flat {card.cost.paper_assumed_bps} bps assumption)")
print(f"Target annualized volatility (from Card): {target_vol:.1%}")

markowitz_diagnostics: list[dict] = []
strategies = {
    "Equal-weight benchmark (fixed)": sig.fixed_weight_benchmark(equal_weight),
    f"Vol-controlled ({target_vol:.0%} target)": sig.vol_controlled(equal_weight, target_vol, lookback_days=11),
    "Markowitz (vol-capped momentum optimizer)": sig.markowitz(
        equal_weight, target_vol, l1_trust_region=card.risk.max_relative_weight_deviation_l1 or 1.0,
        cov_lookback_days=11, alpha_halflife_days=252, horizon_days=21,
        one_way_cost_bps=one_way_bps, cash_annual_rate=cash_annual_rate,
        asset_names=ASSET_COLS, diagnostics=markowitz_diagnostics,
    ),
}

results = {}
for name, fn in strategies.items():
    simr = PortfolioSimulator(
        asset_returns=returns, cost_model=DEFAULT_INDIA_COST_MODEL,
        cash_annual_rate=cash_annual_rate, rebalance_frequency="ME",
    )
    results[name] = simr.run(fn)

print("\nPerformance summary (annualized, net of India costs):")
summary_rows = []
for name, res in results.items():
    m = performance_metrics(res.value, cash_annual_rate=cash_annual_rate)
    m["annualized_turnover"] = 252.0 * res.turnover.mean()
    m["strategy"] = name
    summary_rows.append(m)
summary = pd.DataFrame(summary_rows).set_index("strategy")[
    ["return", "volatility", "sharpe", "max_drawdown", "mean_drawdown", "annualized_turnover"]
]
for c in ["return", "volatility", "max_drawdown", "mean_drawdown", "annualized_turnover"]:
    summary[c] = (summary[c] * 100).round(2).astype(str) + "%"
summary["sharpe"] = summary["sharpe"].round(2)
print(summary.to_string())

# ---------------------------------------------------------------------------
# STAGE 06 -- Research validation.
# ---------------------------------------------------------------------------
hr("STAGE 06 -- Research validation (Markowitz strategy)")
markowitz_result = results["Markowitz (vol-capped momentum optimizer)"]

boot = research_validation.bootstrap_metrics(markowitz_result.value, cash_annual_rate=cash_annual_rate, n_replications=500)
print("Stationary block bootstrap (mean block 21 days, 500 replications):")
print(f"  Return : mean {boot['return']['mean']:.2%}  95% CI {tuple(round(x,4) for x in boot['return']['ci95'])}")
print(f"  Sharpe : mean {boot['sharpe']['mean']:.2f}  95% CI {tuple(round(x,3) for x in boot['sharpe']['ci95'])}")
print(f"  MaxDD  : mean {boot['max_drawdown']['mean']:.2%}  95% CI {tuple(round(x,4) for x in boot['max_drawdown']['ci95'])}")

diff = research_validation.paired_sharpe_difference(
    markowitz_result.value, results["Equal-weight benchmark (fixed)"].value,
    cash_annual_rate=cash_annual_rate, n_replications=500,
)
print("\nMarkowitz vs. equal-weight benchmark, paired bootstrap Sharpe difference:")
print(f"  mean diff {diff['mean_diff']:.3f}  95% CI {tuple(round(x,3) for x in diff['ci95'])}  "
      f"P(diff<=0) = {diff['p_diff_le_0']:.3f}")

subperiods = research_validation.subperiod_sharpe(markowitz_result.value, cash_annual_rate=cash_annual_rate, n_periods=4)
print("\nSub-period Sharpe ratios (regime consistency check):")
print(subperiods[["start", "end", "sharpe"]].to_string(index=False))

trial_sharpes = []
for tv in [target_vol * f for f in (0.67, 0.83, 1.0, 1.17, 1.33)]:
    fn = sig.markowitz(equal_weight, tv, l1_trust_region=1.0, cov_lookback_days=11,
                        one_way_cost_bps=one_way_bps, cash_annual_rate=cash_annual_rate)
    r = PortfolioSimulator(asset_returns=returns, cost_model=DEFAULT_INDIA_COST_MODEL,
                            cash_annual_rate=cash_annual_rate, rebalance_frequency="ME").run(fn)
    trial_sharpes.append(performance_metrics(r.value)["sharpe"])
dsr = research_validation.deflated_sharpe_ratio(
    observed_sharpe=performance_metrics(markowitz_result.value)["sharpe"],
    trial_sharpes=trial_sharpes, n_obs=len(markowitz_result.value),
)
print(f"\nDeflated Sharpe Ratio (corrects for {dsr['n_trials']}-specification vol-target sweep): "
      f"{dsr['deflated_sharpe_ratio']:.3f}  (trial Sharpe std: {dsr['trial_sharpe_std']:.3f})")

hr("STAGE 06b -- Signal transparency: is the alpha actually predictive?")
print(f"Strategy Card's signal claim (Stage 02): \"{card.signal.description}\"")
sig_diag = research_validation.signal_diagnostics(
    markowitz_diagnostics, returns, horizon_days=21, card_signal_description=card.signal.description,
)
print("\nInformation Coefficient (rank correlation of alpha vs. realized forward return):")
print(f"  mean IC: {sig_diag['mean_ic']:.4f}   hit rate: {sig_diag['hit_rate']:.1%}   "
      f"rebalances evaluated: {sig_diag['n_rebalances_evaluated']}")
print(f"  {sig_diag['interpretation']}")

print("\nDecomposition of Markowitz vs. equal-weight Sharpe (same equal-weight target mix in all three legs):")
decomp = research_validation.decompose_vs_equal_weight(
    results["Equal-weight benchmark (fixed)"], results[f"Vol-controlled ({target_vol:.0%} target)"], markowitz_result,
    cash_annual_rate=cash_annual_rate,
)
print(decomp["narrative"])

# ---------------------------------------------------------------------------
# STAGE 07 -- Portfolio validation.
# ---------------------------------------------------------------------------
hr("STAGE 07 -- Portfolio validation")
strat_returns = markowitz_result.value.pct_change().dropna()
benchmark_col = prompt_choice("Pick a column to use as the benchmark for factor-exposure checks", ASSET_COLS)
factor_panel = eval_df[[benchmark_col]].pct_change()
exposure = portfolio_validation.factor_exposure(strat_returns, factor_panel, lag=1)
print(f"\nFactor exposure to lagged '{benchmark_col}': {exposure}")

tv_cvar = portfolio_validation.turnover_and_cvar_report(strat_returns, markowitz_result.turnover)
print(f"\nTurnover & tail risk: {tv_cvar}")

bench_ret = eval_df[benchmark_col].pct_change().dropna()
incr = portfolio_validation.incremental_information_ratio(strat_returns, bench_ret, weight_in_book=0.20)
print(f"\nIncremental IR of adding strategy at 20% of a '{benchmark_col}'-only book: {incr}")

# ---------------------------------------------------------------------------
# GATE B -- Investment decision.
# ---------------------------------------------------------------------------
hr("GATE B -- Investment decision")
library = StrategyLibrary(db_path=os.path.join(REPO_ROOT, "data", "lineage", "strategy_library.db"))
library.register(card.card_id, card.title, card.content_hash())

gate_b_reviewer = prompt_text("Gate B reviewer name", default="reviewer")
gate_b_decision_str = prompt_choice("Gate B decision", [d.value for d in GateBDecision])
gate_b_decision = GateBDecision(gate_b_decision_str)
gate_b_note = prompt_text("Gate B note (why -- required)")

evidence_summary = {
    "sharpe_point_estimate": round(performance_metrics(markowitz_result.value, cash_annual_rate)["sharpe"], 3),
    "sharpe_ci95": tuple(round(x, 3) for x in boot["sharpe"]["ci95"]),
    "deflated_sharpe_ratio": round(dsr["deflated_sharpe_ratio"], 3),
    "mean_information_coefficient": round(sig_diag["mean_ic"], 4) if sig_diag["mean_ic"] == sig_diag["mean_ic"] else None,
    "vs_equal_weight_p_diff_le_0": round(diff["p_diff_le_0"], 3),
    "annualized_turnover": round(tv_cvar["annualized_turnover"], 3),
}
audit.gate_b(
    card_id=card.card_id, result_id="run_" + dataset.lineage.content_sha256[:8],
    reviewer=gate_b_reviewer, decision=gate_b_decision, note=gate_b_note,
    dataset_snapshot_hash=dataset.lineage.content_sha256, code_hash=card.content_hash(),
    model_version="engine v0.1.0", evidence_summary=evidence_summary,
)

if gate_b_decision == GateBDecision.reject:
    library.reject(card.card_id, reviewer=gate_b_reviewer, note=gate_b_note, evidence=evidence_summary)
else:
    library.promote(
        card.card_id, Rung.replicated, dataset_snapshot_hash=dataset.lineage.content_sha256,
        code_hash=card.content_hash(), model_version="engine v0.1.0", reviewer=gate_b_reviewer,
        note=f"Gate B decision: {gate_b_decision.value}. {gate_b_note}", evidence=evidence_summary,
    )

print(f"\nGate B decision: {gate_b_decision.value}")
print(f"Promotion ladder: {card.card_id} -> {library.current_rung(card.card_id)}")
print("\nFull promotion history:")
for h in library.history(card.card_id):
    print(f"  [{h['timestamp']}] {h['event_type']:10s} {h['from_rung']} -> {h['to_rung']}  ({h['reviewer']}): {h['note'][:100]}")

hr("DONE")
print(f"Card:          {card_yaml_path}")
print(f"Audit log:     {audit.path}")
print(f"Strategy library: {library.db_path}")
