# Pipeline stages — economic intuition, and Red/Green flags

Each stage below maps 1:1 to a module in `backtester/`. For every stage: **what it does**,
**why it exists (economic intuition)**, and the **Red flags / Green flags** a reviewer
(researcher at Gate A, PM/IC at Gate B) should use to judge output at that stage.

---

## 01 — Ingest  (`backtester/ingest/parser.py`)

**What.** Parse the uploaded paper (PDF/arXiv/SSRN/upload) into layout-aware text, tables,
equations, and metadata (title, authors, date, arXiv id), each evidence unit tagged with a
page number so every later claim is traceable back to a specific page.

**Why.** A Strategy Card built from memory or paraphrase silently drops constraints. The
Boyd et al. paper, for instance, states its universe constraint in one sentence on p.2 ("only
the three broad assets... long only, no leverage, no derivatives, and no proprietary data")
— miss that sentence and you build a levered multi-asset strategy that looks like a
replication but isn't. Page-anchored evidence lets Gate A check the Card against the source
in seconds instead of re-reading the paper.

**Red flags**
- No page citation for a claimed parameter (lookback window, cost assumption, universe).
- Numbers pulled from an abstract/press summary rather than the methods section.
- Tables/equations that didn't parse (e.g. the optimization problem, eq. 3) are silently
  skipped rather than flagged as "needs manual entry."

**Green flags**
- Every field in the resulting Strategy Card carries a `(page, quote)` evidence pointer.
- Ambiguous or missing details (e.g. "conservative estimate of trading costs" without a
  number until §2.5) are surfaced as open questions for Gate A, not silently guessed.

**Comprehensive, not just field-by-field.** `comprehensive_ingest()` mines three things
from every paper, not just whatever a specific Card field happens to search for:
(1) the Abstract, verbatim — the authors' own one-paragraph economic thesis; (2) every
sentence anywhere in the paper containing economic-reasoning language ("we find",
"because", "consistent with", "driven by", ...), page-anchored, regardless of which
section it's in; (3) regex-mined candidate parameters (lookback window, rebalance
frequency, cost bps, volatility target) AND which broad asset classes/segments the
paper's own vocabulary touches on (equity, bond, gold/commodity, mutual fund, cash rate,
factor style, derivatives) — this last one is what drives Stage 03's interactive data
discovery below, so a bond-heavy paper actually gets asked about bond data, not just
whatever the Card author remembered to type. This is the concrete meaning of "Stage 01
dictates the flow of the entire pipeline": `ingest/strategy_card.py:build_card_interactively`
is driven entirely off this report, never off a blank form.

---

## 02 — Strategy Card  (`backtester/ingest/strategy_card.py`)

**What.** The *only* artifact the AI is allowed to author. A small, structured, validated
object: universe, signal definition, lookback/lag, weighting rule, rebalance frequency,
benchmark, cost assumptions, explicit ambiguities, and a confidence score per field.

**Why.** This is the "template-constrained generation" control. An AI asked to "write a
backtest" can hide a lookahead bias in fifty lines of pandas. An AI asked to fill twelve
typed fields in a schema cannot hide anything — the schema has no field for "use future
data." Constraining the AI's output surface is what makes the AI's economic error a *code
review* problem instead of a *needle in a haystack* problem.

**Red flags**
- A signal defined only in prose ("momentum") with no lookback/skip/lag actually pinned down.
- Ambiguities list is empty for a paper that plainly has ambiguities (real papers almost
  always do — see ours: "conservative estimate of trading costs" needed a page-3 lookup to
  resolve to 5bp).
- Any field with `confidence: high` that isn't backed by a direct quote.

**Green flags**
- Every numeric parameter traces to an equation or table in the source (e.g. our Card cites
  eq. (1) for the vol-control dilution rule, eq. (3) for the optimization problem, Table 7
  for the 42-feature forecast).
- Ambiguities are resolved explicitly at Gate A, not silently defaulted.

**Three builders, same underlying report.** All three are driven entirely off Stage 01's
`IngestReport` — the difference is how the interpretation is done and how much a human is
asked, not what the AI is allowed to author (the constraint is still the schema, either
way):
- `build_card_via_llm()` (`backtester/ingest/llm_extract.py`) — the recommended path in
  `examples/run_pipeline_interactive.py` when an API key is available. A real Claude call
  reads the full paper (page-marked, so it can cite precisely) and fills the same typed
  Card via `client.messages.parse(..., output_format=LLMCardExtraction)` — structured
  output, so it cannot return anything outside the schema. This is genuine reading
  comprehension rather than pattern matching: it can tell a paper's actual thesis apart
  from a caveat it's arguing against, which the regex version cannot (see the worked
  example in this repo's history, where the regex path picked *"fixed-weight portfolios
  can be difficult to beat... because forecasting is challenging"* — the authors' hedge,
  not their claim — as the "signal description"). Critically, **the model's citations are
  not trusted** — every `Evidence.quote` it returns is checked in code
  (`llm_extract.verify_quote`) against the actual text of the page it claims, using the
  same `=== PAGE N ===` markers fed to the model as ground truth. A verified quote gets
  `confidence: high`; a real quote on the wrong page gets auto-corrected with a logged,
  pre-resolved `Ambiguity`; a quote that cannot be found anywhere in the paper (a
  hallucination) gets `confidence: low` and an **unresolved** `Ambiguity` that blocks Gate
  A approval until a human checks it. This is the concrete implementation of "AI
  interprets, deterministic infrastructure verifies" for Stage 02 specifically — the
  verification step is plain string matching, not another model call, so it cannot itself
  hallucinate.
- `build_card_automatically()` — the no-API-key fallback. Same schema, same
  page-anchored-default-or-documented-fallback pattern, but via the regex/keyword mining
  in `parser.py` instead of a model call — faster, free, and available offline, at the
  cost of the comprehension a real model brings (it cannot distinguish a thesis from a
  caveat, only find keyword matches).
- `build_card_interactively()` — stops and asks at each field instead of auto-selecting;
  useful when a Card needs closer human authorship than either automatic pass gives it.
  Still available, just not the default entry point.

In every case, Gate A reviews and approves/rejects the *whole* interpreted Card as one
decision (plus resolving any items the builder itself left open), with the full reasoning
already visible, instead of authoring it field by field.

---

## GATE A — Human interpretation  (`backtester/gates/audit.py`, `gate="A"`)

**What.** A researcher approves the *economic definition* in the Strategy Card: is this
actually what the paper claims, are the assumptions right, are proxies acceptable. One
sign-off, logged with reviewer identity, timestamp, and a hash of the Card version approved.

**Why.** This is the cheapest point in the whole pipeline to catch a misinterpretation —
before any data is pulled or any code is run. It is also the *only* point where domain
judgment about "does this economic idea even make sense for India" is exercised, before
sunk cost accumulates.

**Red flags:** Gate A approves without reading the ambiguities list; approval note is empty
or generic ("looks fine"); the same reviewer approves their own Card.

**Green flags:** Approval note addresses every open ambiguity by name; a proxy substitution
(e.g. "gold" → a specific India gold ETF) is justified in one sentence of economic reasoning,
not just accepted.

---

## 03 — Data feasibility  (`backtester/data/sources.py`)

**What.** For every data requirement in the (approved) Card, classify it
`available | proxy | unavailable`, record history depth, licence, frequency, and coverage,
and either fail fast (unavailable, no proxy) or require explicit sign-off (proxy).

**Why.** This is where a US paper's "just a handful of widely available public economic
data" (their §2.4) meets the fact that India does not have a FRED, GLD is not cross-listed,
and Fama-French factors are a US construct. Silent substitution here is the single biggest
source of a false "replication" — the strategy "replicates" the paper's *code* but is
actually testing a different economic bet because the inputs quietly changed meaning.

**Red flags:** A proxy is used without a sign-off note; "unavailable" data is silently
dropped from the feature set instead of failing the Card back to Gate A; point-in-time
availability (was this data actually published on this date, in this form) isn't checked
for macro series.

**Green flags:** Every proxy sign-off states *why* it's an acceptable substitute and *how*
it might change the result (see `docs/DATA_SOURCES.md` for every proxy used in the worked
example, each with this reasoning spelled out).

**Two ways to run Stage 03, matching the two Card builders above.**
- **Default (`examples/run_pipeline_interactive.py`):** a single, fixed-scope step for
  this repo's current use case — India public equities via one manually uploaded
  NIFTY-indices dataset. You upload a CSV (or accept the bundled example), it's registered
  as `AVAILABLE` with no per-asset-class questions, and every column in it becomes the
  tradable universe. Simple by design: no bond/mutual-fund/derivatives discovery clutter
  when the run's scope doesn't need it.
- **`discover_datasets_interactively()`** — the general, any-asset-class version, kept
  available for when a paper's signal actually needs something this repo's current data
  doesn't have: it walks through every asset class Stage 01 found the paper talking about
  (falling back to a fixed checklist — equity, bond, gold/commodity, mutual fund, cash
  rate, derivatives), and for each one asks whether to use a column already in
  `data/raw/`, upload a new file (tagged by asset class), or explicitly decline with a
  reason. Not the default path, but the mechanism is there the moment a bond- or
  mutual-fund-driven paper needs it, without changing a line of engine code.

---

## 04 — Point-in-time data  (`backtester/data/pit_loader.py`)

**What.** Load prices, fundamentals, index membership, corporate actions, delistings, and
publication dates as a **frozen snapshot** with a content hash and a lineage record (source,
retrieval date, transformations applied). Backtests never read a live feed.

**Why.** Point-in-time correctness is the difference between a backtest and a fiction. Two
concrete failure modes this stage exists to prevent, specific to the India factor-index
setting used in the worked example: (1) **index reconstitution lookahead** — NIFTY500
Momentum 50's constituents in 2015 are not today's constituents; the price series is
survivorship-adjusted by NSE, but a strategy that trades the *constituents* rather than the
index itself must use the membership list *as published on that date*, not today's; (2)
**launch-date truncation** — several factor indices in the attached dataset have no printed
value before their live-calculation start date (e.g. the multifactor and momentum indices
begin years after NIFTY 500); backfilling them with a "back-tested" value the index provider
computed after the fact, and then presenting it as historical performance, silently mixes
live and back-tested regimes with different biases. This loader keeps both dates visible.

**Red flags:** A snapshot has no hash / no retrieval date; a series's pre-launch history is
used without flagging it as index-provider back-tested (not live) data; membership data is
"today's constituents applied to yesterday's prices."

**Green flags:** Every load call returns data plus a lineage record; the loader refuses to
serve data past a strategy's configured `as_of` date (hard point-in-time cutoff enforced in
code, not by convention).

---

## 05 — Build + execute  (`backtester/engine/*.py`)

**What.** The Strategy Card's signal is compiled into a small, sandboxed `signal_fn`
(template-constrained: it may only consume the point-in-time feature panel and must return
target weights, nothing else). This function is plugged into the **fixed, reusable**
deterministic engine — accounting, cost model, optimizer — the same engine every strategy
uses. Synthetic sanity tests (e.g. "does the signal reduce risk in a hand-built high-vol
scenario") run before the real backtest.

**Why.** This is the load-bearing separation of the whole design: *AI writes the 20 lines
that encode an economic idea; deterministic code owns the 500 lines that turn weights into
audited P&L.* It is the direct implementation of the memo's "No live API in backtests / No
free-form engine code" controls, and it is why the accounting identity in
`engine/portfolio.py` (portfolio value evolves by asset return plus cash return, net of
trading cost, every day, no exceptions) is unit-tested rather than re-derived per strategy.

**Red flags:** A signal function does anything other than read the feature panel and return
weights (file I/O, network calls, mutable global state); the cost model is bypassed "just
for this backtest"; the optimizer's constraints (long-only, no leverage, volatility cap) are
loosened without a Gate A amendment.

**Green flags:** The same `engine/portfolio.py` accounting path is used for every strategy in
the library (so results are comparable); trading costs are charged at every rebalance,
including in the optimizer's own objective (mirroring the paper's eq. (3) — the optimizer
*anticipates* the cost of moving to a new weight, it doesn't get told about it after the fact).

---

## 06 — Research validation  (`backtester/validation/research.py`)

**What.** Replication gap (does our number match the paper's, if replicating), out-of-sample
/ walk-forward performance, parameter sensitivity, sub-period and regime breakdowns, cost
sensitivity, and a **Deflated Sharpe Ratio** that corrects the headline Sharpe for how many
specifications were actually tried.

**Why.** A single full-sample Sharpe ratio is close to meaningless on its own — the paper
itself makes this point (Appendix C/E): its 20-year Markowitz Sharpe of 1.08 has a bootstrap
95% interval of [0.62, 1.57], its advantage over the 60/40 benchmark is significant at 5%
but its advantage over its *own* volatility-controlled variant is not, and its Deflated
Sharpe Ratio calculation explicitly accounts for a 17-specification hyperparameter sweep.
Reporting only the point estimate would overstate confidence by construction — the more
free parameters searched, the more a purely lucky specification looks skilled. This stage
makes that check a mandatory computation, not a footnote a researcher has to remember to add.

**Red flags:** A single backtest run is reported as "the result" with no sensitivity sweep
behind it; walk-forward and in-sample windows overlap; the confidence interval is omitted
because it "doesn't look as good"; the DSR trial count is deliberately understated.

**Green flags:** Every reported Sharpe ratio carries a bootstrap interval; the DSR trial
count is the *actual* number of specifications explored, including ones that were dropped;
performance is broken out by sub-period/regime so a lucky decade isn't mistaken for skill
(our engine reports both 5-year subperiods and ex-2008-style crisis-exclusion splits, mirroring
Table 2 and the paragraph beneath it in the source paper).

**Signal transparency — mechanism, not just outcome.** A Sharpe ratio alone can't tell you
*why* a strategy did or didn't beat a naive baseline. Two functions close that gap, and
both tie explicitly back to Stage 02's `signal.description` claim:
- `signal_diagnostics()` computes the realized **Information Coefficient** — the
  cross-sectional rank correlation between the alpha the optimizer actually used at each
  rebalance (logged via `engine/signals.py:markowitz(..., diagnostics=log)`) and each
  asset's realized forward return. An IC near zero is direct, quantitative evidence that
  the signal wasn't predictive in this sample — not a restatement of what the source paper
  claimed for a different market/period.
- `decompose_vs_equal_weight()` splits the optimizer's Sharpe difference from static
  equal-weight into a **vol-cap effect** and an **alpha-tilt effect**, because a Markowitz-
  style strategy does two different things (cap risk, and tilt toward a forecast) and a
  single Sharpe number conflates them. On the attached worked example this decomposition
  showed the vol-cap effect was *negative* (−0.095 Sharpe) and the alpha-tilt effect
  *positive* (+0.094), roughly cancelling — which the Information Coefficient (≈0.02,
  essentially noise) directly explains: the optimizer's tilt wasn't picking real winners,
  it just happened to claw back what the vol cap gave up.

**Red flags (signal transparency):** Reporting only that "the optimizer beat/matched
equal-weight" without checking IC or running the decomposition; treating a near-zero IC as
proof the *paper's* signal claim is wrong in general, rather than as evidence about this
specific dataset/period.

**Green flags:** Every Markowitz-family result in the library carries its IC and
decomposition alongside the Sharpe ratio; a reviewer can distinguish "this signal has real
predictive power here" from "this strategy happens to have a good Sharpe for reasons
unrelated to the signal" (e.g. incidental risk reduction).

---

## 07 — Portfolio validation  (`backtester/validation/portfolio.py`)

**What.** Exposure to market beta, size, sector, and liquidity; a "factor fingerprint" and
similarity score against everything already in the strategy library; incremental information
ratio versus the existing book; turnover; and CVaR.

**Why.** A backtest can be statistically "robust" (survives stage 06) and still be
**useless to the portfolio** — either because it's just relabeled market beta (the paper's
own Table 3/§3.3 point: the Markowitz portfolio's *average* weights, held statically, only
earn a 0.73 Sharpe versus 1.08 dynamically — so a naive fingerprint check on average
exposure alone would understate what's actually driving the return) or because it's highly
correlated with a strategy already promoted. This stage is what keeps the research library
from filling up with twenty variations on the same bet.

**Red flags:** Incremental IR is computed against an empty comparison set (i.e., never
checked against the existing book); CVaR/liquidity checks use today's liquidity for a
signal computed on 2015 prices (survivorship in the *risk* check, not just the *return*
check); a "new" signal's factor fingerprint is never compared to prior library entries.

**Green flags:** The signal's incremental IR is computed net of the nearest existing library
entry; turnover and CVaR are reported net of the same cost model used in stage 05, not
recomputed with a friendlier assumption; liquidity checks flag names/sleeves that could not
absorb the strategy's own implied capacity.

---

## GATE B — Investment decision  (`backtester/gates/audit.py`, `gate="B"`)

**What.** PM/IC reviews stages 06–07 output and records one of: `reject | fix | observe |
promote`. A `promote` decision advances exactly one rung on the promotion ladder and is
logged with the dataset snapshot id, code hash, model version, any manual overrides, the
evidence bundle, and the reviewer's identity.

**Why.** This is the second and last human checkpoint, and it is deliberately narrow: by
the time a Card reaches Gate B, the *interpretation* question (Gate A) is already settled —
this gate is purely an investment judgment (risk budget, capacity, correlation to the
existing book), which is a PM/IC call, not a researcher call. Keeping the two gates separate
is what lets the pipeline scale to "screen 20 papers a day" without either gate becoming a
bottleneck that reviews things outside its competence.

**Red flags:** A `promote` decision with no evidence bundle attached; a rejection with no
reason recorded (rejected ideas are supposed to be reusable knowledge, per the memo's North
Star — an unreasoned rejection throws that away); more than one rung skipped in a single
promotion.

**Green flags:** Every decision, including `reject`, is written to the library with its
reasoning (so it can be found again when a similar idea is investigated later); `observe`
is used for genuinely borderline cases rather than as a way to avoid deciding.

---

## 08 — Strategy library + deployment rail  (`backtester/library/ledger.py`)

**What.** A persistent, queryable research graph: every Card, code hash, dataset snapshot,
result, decision, and failed idea, plus the promotion ladder state machine:

```
REPLICATED → INDIA VALIDATED → ROBUST → ORTHOGONAL → PORTFOLIO USEFUL → PAPER TRADED → LIVE
```

**Why.** This is what turns "20 papers screened / day" into "1 portfolio-useful signal /
few weeks" instead of into twenty forgotten spreadsheets. Negative results are stored with
the same rigor as positive ones (the memo: "treat negative results and failed replications
as reusable knowledge") so the next researcher who has the same idea doesn't re-spend the
same week finding out it doesn't survive India costs.

**Red flags:** A rejected/failed idea is deleted rather than archived with its reasoning;
a promotion skips recording the dataset snapshot or code hash (breaks reproducibility);
duplicate ideas aren't detected because the similarity map isn't checked before starting new
work.

**Green flags:** Every ladder transition is a single, logged, hash-anchored event; a
researcher can query "what have we already tried that looks like this" before spending a day
building it again.
