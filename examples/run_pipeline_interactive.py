"""Interactive end-to-end run: upload YOUR research paper + YOUR NIFTY-indices dataset.

Two manual uploads drive this run, nothing else:
  1. The research paper (PDF) -- Stage 01 parses it comprehensively (abstract, every
     economic-reasoning sentence, candidate parameters, asset classes discussed), and
     Stage 02 auto-interprets a Strategy Card directly from that parsing -- no field-by-
     field interview. Swap in a different paper and a different Card comes out, because
     every default traces back to what THAT paper's text actually said.
  2. The NIFTY-indices dataset (CSV **or** the raw Excel file, e.g. the attached
     `Factor_Indices_Historical_Price_Data.xlsx` / `nifty_factor_indices.csv`, or your own
     file in the same shape) -- used AS-IS as the tradable universe for a simple, long-only
     India public-equities backtest. No per-asset-class questionnaire: this pipeline's
     scope for this run is fixed to Indian public equities via whatever columns are in the
     file you upload. Both formats are read through the same point-in-time loader
     (`backtester/data/pit_loader.py:PointInTimeDataset.from_file`), so you don't need to
     pre-convert an Excel file to CSV yourself.

Unlike `examples/run_pipeline.py` (a fixed, non-interactive worked example pinned to the
attached factor-rotation Card and the attached CSV -- kept as a known-good regression
demo), this script is the general entry point for trying a NEW paper against the SAME
kind of dataset. The broader multi-asset-class discovery flow (bonds, mutual funds,
commodities, ...) still exists in `backtester/data/sources.py:discover_datasets_interactively`
for later use; it's just not part of this simplified default path.

Run with:  python examples/run_pipeline_interactive.py
(or via the `%run` cell in examples/run_in_jupyter.ipynb -- prompts render as inline text
boxes in Colab, and PDF/CSV uploads use Colab's native file-upload widget.)
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from backtester.ingest.parser import comprehensive_ingest, print_ingest_report
from backtester.ingest.strategy_card import build_card_automatically, save_card
from backtester.ingest.interactive_io import upload_file, prompt_text, prompt_choice, in_colab
from backtester.data.pit_loader import PointInTimeDataset
from backtester.data.sources import DataFeasibilityRegistry, DataSourceEntry, Feasibility
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
DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "data", "raw", "nifty_factor_indices.csv")


def hr(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------------------
# STAGE 01 -- Ingest: YOU upload the paper. This stage's output drives everything below --
# comprehensive on purpose, since a different paper should produce a different Card.
# ---------------------------------------------------------------------------
hr("STAGE 01 -- Ingest (upload your research paper)")
print(f"Running in Colab: {in_colab()}")
paper_path = upload_file("Upload the research paper PDF you want to turn into a strategy", save_dir="data/uploads/papers")
if not paper_path:
    raise SystemExit("No paper uploaded -- nothing to ingest. Re-run this cell and upload a PDF.")

report = comprehensive_ingest(paper_path)
print_ingest_report(report)

# ---------------------------------------------------------------------------
# STAGE 02 -- Strategy Card, AUTO-INTERPRETED from Stage 01's report. No questions here --
# every field below traces to something Stage 01 found (or a documented default), logged
# as an auto-resolved ambiguity rather than asked interactively.
# ---------------------------------------------------------------------------
hr("STAGE 02 -- Strategy Card (auto-interpreted from the Stage 01 ingest report above)")
card = build_card_automatically(report)
print(f"\nUniverse (fixed scope for this run): {card.universe}")
print(f"Signal (auto-interpreted): {card.signal.description[:200]}")
print("\nAuto-resolved fields (what was found in the paper vs. what was defaulted):")
for a in card.ambiguities:
    print(f"  [{a.field}] {a.resolution}")

card_yaml_path = os.path.join(REPO_ROOT, "data", "lineage", f"{card.card_id}.yaml")
os.makedirs(os.path.dirname(card_yaml_path), exist_ok=True)
save_card(card, card_yaml_path)
print(f"\nCard saved to {card_yaml_path}")

# ---------------------------------------------------------------------------
# GATE A -- one human decision on the whole auto-interpreted Card, not a per-field
# interview (the fields were already auto-resolved above, with reasoning attached).
# ---------------------------------------------------------------------------
hr("GATE A -- Human interpretation sign-off")
reviewer = prompt_text("Gate A reviewer name", default="reviewer")
decision = prompt_choice("Gate A decision -- approve this auto-interpreted Card?", ["approved", "rejected"])
note = prompt_text(
    "Gate A note (required -- e.g. confirm the auto-interpretation looks right, or say what's off)"
)
audit = AuditLog(path=os.path.join(REPO_ROOT, "data", "lineage", "audit_log.jsonl"))
gate_a_entry = audit.gate_a(card, reviewer=reviewer, decision=decision, note=note)
print(f"Gate A decision: {gate_a_entry.decision} by {gate_a_entry.reviewer}")
if decision != "approved":
    raise SystemExit("Card rejected at Gate A -- stopping here.")

# ---------------------------------------------------------------------------
# STAGE 03 -- Data: YOU upload the NIFTY-indices dataset (or accept the bundled one).
# Fixed scope for this run: India public equities, whatever columns are in the file.
# No per-asset-class questionnaire.
# ---------------------------------------------------------------------------
hr("STAGE 03 -- Data feasibility (India public equities, single uploaded dataset)")
print("This run's scope is fixed to India public equities via one NIFTY-indices dataset --")
print("no per-asset-class questions. Upload your own CSV or Excel file, or accept the bundled example.\n")
dataset_path = upload_file(
    "Upload your NIFTY-indices dataset (.csv or .xlsx -- leave blank / cancel to use the "
    "bundled data/raw/nifty_factor_indices.csv)",
    save_dir=os.path.join(REPO_ROOT, "data", "raw"),
)
if not dataset_path:
    dataset_path = DEFAULT_DATASET_PATH
    print(f"No file uploaded -- using the bundled dataset: {dataset_path}")
if not os.path.exists(dataset_path):
    raise SystemExit(f"Dataset not found at {dataset_path} -- upload a .csv or .xlsx file and re-run.")
print(f"Using dataset: {dataset_path}")

registry = DataFeasibilityRegistry()
registry.register(DataSourceEntry(
    requirement="india_equity_universe",
    feasibility=Feasibility.available,
    source=f"Manually uploaded dataset: {dataset_path}",
))
print(registry.report())

# ---------------------------------------------------------------------------
# STAGE 04 -- Point-in-time data: load the uploaded dataset as-is, auto-detect the
# broad-market benchmark column, and trim to the common evaluation window.
# ---------------------------------------------------------------------------
hr("STAGE 04 -- Point-in-time data")
dataset = PointInTimeDataset.from_file(dataset_path)
print(f"Source: {dataset.lineage.source_path}")
print(f"Content SHA-256: {dataset.lineage.content_sha256}")
print(f"Rows: {dataset.lineage.n_rows}  Range: {dataset.lineage.first_date} .. {dataset.lineage.last_date}")
print("\nLaunch dates per series (first non-null close):")
for col, d in dataset.launch_dates().items():
    print(f"  {col:30s} {d}")

ALL_COLS = list(dataset.df.columns)
# Auto-detect the broad-market benchmark. Prefer an EXACT "NIFTY 500"-style column name
# (e.g. "NIFTY_500", "NIFTY500") over a merely-substring match, because a factor sleeve
# like "NIFTY500_MOMENTUM_50" also contains "500" but is emphatically NOT the broad-market
# benchmark -- picking it by loose substring match would silently mislabel a sleeve as the
# benchmark. Falls back to the first column only if no exact-style match exists at all.
# This is a documented heuristic, not a silent guess -- printed below either way.
_exact_rx = re.compile(r"^nifty[_\s]?500$", re.IGNORECASE)
exact_matches = [c for c in ALL_COLS if _exact_rx.match(c.strip())]
if exact_matches:
    BENCHMARK_COL = exact_matches[0]
    detection_note = f"exact match for a 'NIFTY 500'-style column name ('{BENCHMARK_COL}')"
else:
    substring_matches = [c for c in ALL_COLS if "500" in c]
    BENCHMARK_COL = substring_matches[0] if substring_matches else ALL_COLS[0]
    detection_note = (
        f"no exact 'NIFTY 500' column found -- fell back to a loose '500' substring match "
        f"('{BENCHMARK_COL}'); VERIFY this is actually the broad-market index, not a factor sleeve"
        if substring_matches else "no '500'-style column found at all -- used the first column"
    )
ASSET_COLS = [c for c in ALL_COLS if c != BENCHMARK_COL]
print(f"\nAuto-detected benchmark column: '{BENCHMARK_COL}' ({detection_note})")
print(f"Tradable universe ({len(ASSET_COLS)} instruments): {ASSET_COLS}")

card.benchmark = f"{BENCHMARK_COL} (auto-detected from the uploaded dataset)"
save_card(card, card_yaml_path)

common_start = max(pd.Timestamp(d) for c, d in dataset.launch_dates().items() if c in ASSET_COLS)
eval_df = dataset.df.loc[common_start:].dropna(subset=ASSET_COLS + [BENCHMARK_COL])
print(f"\nCommon evaluation window (all series live): "
      f"{eval_df.index.min().date()} .. {eval_df.index.max().date()}  ({len(eval_df)} obs)")

cash_annual_rate = 0.0
print(f"\nCash rate: fixed at the {cash_annual_rate:.0%} placeholder for this simplified "
      "equities-only flow (see docs/DATA_SOURCES.md -- an India risk-free series is not "
      "part of this run's scope).")

# ---------------------------------------------------------------------------
# STAGE 05 -- Build + execute.
# ---------------------------------------------------------------------------
hr("STAGE 05 -- Build + execute (deterministic engine, template-constrained signal)")
returns = eval_df[ASSET_COLS].pct_change()
n = len(ASSET_COLS)
equal_weight = np.ones(n) / n
target_vol = card.risk.target_volatility_annual
one_way_bps = DEFAULT_INDIA_COST_MODEL.one_way_cost_bps()
print(f"India one-way cost assumption: {one_way_bps:.2f} bps "
      f"(vs. paper's {card.cost.paper_assumed_bps} bps assumption)")
print(f"Target annualized volatility (auto-interpreted from Card): {target_vol:.1%}")

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

bench_ret = eval_df[BENCHMARK_COL].pct_change().dropna()
bench_value = (1 + bench_ret).cumprod()
bench_value.name = BENCHMARK_COL

print("\nPerformance summary (annualized, net of India costs):")
summary_rows = []
for name, res in results.items():
    m = performance_metrics(res.value, cash_annual_rate=cash_annual_rate)
    m["annualized_turnover"] = 252.0 * res.turnover.mean()
    m["strategy"] = name
    summary_rows.append(m)
m_bench = performance_metrics(bench_value, cash_annual_rate=cash_annual_rate)
m_bench["annualized_turnover"] = 0.0
m_bench["strategy"] = f"{BENCHMARK_COL} (passive benchmark)"
summary_rows.append(m_bench)
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
print(f"Strategy Card's signal claim (Stage 02, auto-interpreted): \"{card.signal.description}\"")
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
# STAGE 07 -- Portfolio validation (against the auto-detected benchmark).
# ---------------------------------------------------------------------------
hr("STAGE 07 -- Portfolio validation")
strat_returns = markowitz_result.value.pct_change().dropna()
factor_panel = eval_df[[BENCHMARK_COL]].pct_change()
exposure = portfolio_validation.factor_exposure(strat_returns, factor_panel, lag=1)
print(f"Factor exposure to lagged '{BENCHMARK_COL}' (avoids look-ahead): {exposure}")

tv_cvar = portfolio_validation.turnover_and_cvar_report(strat_returns, markowitz_result.turnover)
print(f"\nTurnover & tail risk: {tv_cvar}")

incr = portfolio_validation.incremental_information_ratio(strat_returns, bench_ret, weight_in_book=0.20)
print(f"\nIncremental IR of adding strategy at 20% of a '{BENCHMARK_COL}'-only book: {incr}")

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
