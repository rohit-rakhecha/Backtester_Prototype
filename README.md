# Backtester Prototype — AI Research Operating System (India, Long-Only)

A **universal backtester** that turns any quant research paper into a governed, auditable,
India-investable long-only strategy. It implements the 8-stage pipeline + 2 human gates
from the attached board memo (`AI Research Operating System`), with the philosophy stated
there verbatim:

> AI interprets. Deterministic infrastructure computes. Humans govern. The research graph
> compounds knowledge.

The AI (an LLM, or a human researcher) never writes free-form backtest code. It only fills
in a small, structured **Strategy Card** and a small, template-constrained **signal function**.
Everything else — data, execution, costs, portfolio construction, measurement — is owned by
deterministic, tested, reusable engine code in this repo.

## Why this design (economic intuition, not just plumbing)

Backtests are cheap to run and easy to overfit. The paper attached as the worked example
(Devanathan, Tzikas, Boyd, *"Simple Dynamic Stock/Bond/Gold Portfolios"*, 2026) is itself an
argument for this: it shows that (a) a hard risk constraint plus a *simple, causal* return
forecast beats seven standard risk-based allocators and a tuned Black-Litterman model, (b)
the edge survives costs, taxes, inflation, and a stationary-block-bootstrap significance
test, and (c) the authors *still* report a Deflated Sharpe Ratio to correct for their own
specification search (Appendix E). That is exactly the discipline institutional promotion
requires — and exactly what free-form, one-off backtest scripts fail to enforce. This repo
makes those checks mandatory infrastructure, not something a researcher has to remember.

## The pipeline

```
01 INGEST → 02 STRATEGY CARD → GATE A (human) → 03 DATA FEASIBILITY → 04 POINT-IN-TIME DATA
→ 05 BUILD + EXECUTE → 06 RESEARCH VALIDATION → 07 PORTFOLIO VALIDATION → GATE B (human)
→ 08 STRATEGY LIBRARY + DEPLOYMENT RAIL (promotion ladder)
```

Full write-up of every stage — what it does, the economic intuition for why it exists, and
the **Red flags / Green flags** a reviewer should look for at that stage — is in
[`docs/PIPELINE.md`](docs/PIPELINE.md).

Required Indian data sources, derived by actually parsing the attached research paper and
mapping its US data (SPY/AGG/GLD/DFF/CPILFESL/Fama-French/FRED yields) onto India-investable
equivalents, are in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Run it on your own paper (recommended)

`examples/run_pipeline_interactive.py` is the general entry point: it prompts you to
**upload any research paper**, builds the Strategy Card interactively from what Stage 01
mines out of it (abstract, economic-reasoning sentences, candidate parameters, asset
classes mentioned), then walks Stage 03 as a conversation — for every asset class the
paper touches on (equity, bond, mutual fund, commodity, cash rate, derivatives, ...) you
either pick from data already in `data/raw/` or upload a new file, tagged by asset class.
Only once that's settled does it run Stages 05–08 through the same fixed engine every
Card in this repo uses.

```bash
pip install -r requirements.txt
python examples/run_pipeline_interactive.py
```

In Colab, run this via the `%run` cell in `examples/run_in_jupyter.ipynb` (section 3A) —
prompts render as inline text boxes, and uploads use Colab's native file-upload widget.

## Worked example included (fixed, non-interactive)

`data/raw/nifty_factor_indices.csv` is the attached NSE factor-index price history (daily
closes, 2003–2026, for NIFTY Alpha 50, NIFTY500 Momentum 50, NIFTY500 Multifactor MQVLV 50,
NIFTY500 Quality 50, NIFTY500 Value 50, NIFTY500 Low Volatility 50, NIFTY High Beta 50, and
the NIFTY 500 benchmark). `examples/strategy_card_factor_rotation_india.yaml` is a Strategy
Card that replicates the *methodology* of the attached paper — hard volatility constraint +
simple causal return forecast + convex optimization, long-only, monthly rebalance — but
re-targets it at an **India factor-sleeve rotation** (long-only, no leverage, no derivatives)
instead of stock/bond/gold. `examples/run_pipeline.py` runs all 8 stages end to end against
this real data, with no prompts, and prints the promotion-ladder verdict. It also includes
Stage 06's signal-transparency diagnostics (Information Coefficient + a vol-cap-vs-alpha-tilt
decomposition), which on this worked example show *why* the optimizer roughly matches, but
doesn't clearly beat, a naive equal-weight benchmark — see `docs/PIPELINE.md` Stage 06.
Kept as a fixed regression check independent of the interactive flow above.

```bash
pip install -r requirements.txt
python examples/run_pipeline.py
```

This demonstrates the harness on real NSE data. It is **not** a live-ready strategy: capacity,
point-in-time index-reconstitution data, and a real INR cash-rate series (see
`docs/DATA_SOURCES.md`) still need to be wired in before this could pass Gate B for real
capital — the pipeline will tell you that itself (see the Data Feasibility and India
Validation stage outputs).

## Repository layout

```
backtester/
  ingest/        01 — PDF/paper parsing → structured evidence
  ingest/strategy_card.py   02 — Strategy Card schema (the ONLY thing AI is allowed to author)
  gates/          Gate A / Gate B — human sign-off + immutable audit log
  data/           03 — data feasibility registry, 04 — point-in-time loader + lineage hashing
  engine/         05 — deterministic backtest engine (accounting, costs, optimizer) — no AI code runs here
  validation/     06 — research validation, 07 — portfolio validation
  library/        08 — strategy library / promotion ladder (SQLite research graph)
  pipeline.py     orchestrates 01→08 with the two gates enforced
examples/         worked Strategy Card + runnable end-to-end script on real NSE data
docs/             PIPELINE.md (steps + red/green flags), DATA_SOURCES.md
tests/            engine correctness tests (accounting identity, cost model, vol targeting)
```

## Non-negotiable controls (from the board memo, enforced in code, not policy)

- **No live API in backtests** — `engine/` only reads frozen, hashed snapshots (`data/pit_loader.py`).
- **No free-form engine code** — the only "AI-authored" artifact is the Strategy Card
  (`ingest/strategy_card.py`) and the small `signal_fn` it is compiled to; the accounting,
  costs, and optimizer are fixed, tested library code (`engine/portfolio.py`, `engine/costs.py`).
- **No silent proxy use** — every data field is tagged `available | proxy | unavailable` in
  the Strategy Card, and a `proxy` tag requires an explicit sign-off note (`data/sources.py`).
- **No result without lineage** — every backtest run is hashed against its Strategy Card,
  its code, and its dataset snapshot (`library/ledger.py`).
- **No paper goes straight to capital** — the promotion ladder
  (`REPLICATED → INDIA VALIDATED → ROBUST → ORTHOGONAL → PORTFOLIO USEFUL → PAPER TRADED → LIVE`)
  is enforced as an explicit state machine in `library/ledger.py`; Gate B can only move a
  strategy one rung at a time, and every promotion call records dataset snapshot id, code
  hash, model version, overrides, evidence, and reviewer.
