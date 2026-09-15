# Data sources required — derived from parsing the attached paper

This is the **output of Stage 03 (Data Feasibility)** run against the Strategy Card built
from *"Simple Dynamic Stock/Bond/Gold Portfolios"* (Devanathan, Tzikas, Boyd, 2026), after
re-targeting the Card at India long-only equities. Every row below traces to a specific
data requirement stated in the paper (§2.5, §2.4, Appendix B/F) and is classified
`AVAILABLE | PROXY | UNAVAILABLE` for an India-based long-only desk, per the memo's control
"No silent proxy use."

## Core asset / sleeve prices

| Paper's requirement | Page/§ | India mapping | Status | Notes |
|---|---|---|---|---|
| SPY (US equity) daily adjusted close | §2.5 | **NIFTY 500 / NIFTY 50 TRI** (broad equity beta) | AVAILABLE | Provided in `data/raw/nifty_factor_indices.csv` (NSE Indices Ltd, 2003–2026 daily). Use Total Return Index (TRI), not price index, to avoid dividend-drag bias — flag: the attached file's "NIFTY 500" column must be confirmed as price or TRI before use (see Open Issue below). |
| Factor/momentum tilt used for `α_t` | §2.4, Table 7 | **NIFTY Alpha 50, NIFTY500 Momentum 50, NIFTY500 Multifactor MQVLV 50, NIFTY500 Quality 50, NIFTY500 Value 50, NIFTY500 Low Volatility 50, NIFTY High Beta 50** | AVAILABLE | Same file. These replace the paper's single-asset trailing-return momentum signal with an explicit long-only *factor-sleeve rotation* — see Strategy Card. |
| AGG (US investment-grade bonds) | §2.5 | **Indian G-Sec / constant-maturity gilt index, or a gilt ETF (e.g. Nippon India ETF Nifty 8-13yr G-Sec / an AMFI gilt fund NAV series)** | PROXY | No bond ETF/index price series attached. Needed before Gate A can sign off a bond sleeve; until sourced, the India version of this strategy must run equity-factor-sleeves-only (as built) and flag "bond sleeve unavailable" rather than silently dropping duration exposure. |
| GLD (gold) | §2.5 | **MCX Gold spot/futures continuous series, or a domestic gold ETF NAV (e.g. Nippon India ETF Gold BeES / SBI Gold ETF)** | PROXY | Not attached. Gold-as-inflation-hedge economic logic (paper §1.1, refs [35,5,6,24]) transfers to India reasonably (INR gold price also reflects a currency-depreciation hedge specific to India, which is a *stronger* argument locally than the paper's US framing — worth noting for Gate A). |
| Cash / risk-free rate (federal funds rate, `DFF`) | §2.5, §2.4 | **India: MIBOR / T-Bill (91-day) yield, or a liquid/overnight mutual fund NAV series as the tradable cash-equivalent proxy** | PROXY / UNAVAILABLE (not attached) | Critical gap — the volatility-control dilution (eq. 1) and the Markowitz optimizer's cash leg (eq. 3) both require this series. Must be sourced from RBI (91-day T-Bill cutoff yield, published weekly at auction) or NSE MIBOR before the optimizer's cash-return term can be anything but zero. Current worked example runs with cash return **hardcoded to 0%** and this is surfaced as an explicit Gate A open item, not silently assumed. |

## Macro-financial features (Table 7 of the paper — feeds `α_t` in the "Markowitz" forecast)

| Paper's feature group | India mapping | Status | Notes |
|---|---|---|---|
| US Treasury yields (3mo, 2/5/10/30yr), 10yr real yield, 10yr breakeven inflation | **RBI G-Sec yield curve (CCIL/FBIL daily), India 10yr real yield needs CPI-linked bond or a computed real yield (10yr nominal − expected CPI)**; India has no liquid inflation-linked bond market comparable to US TIPS | PROXY (partial) | Breakeven-inflation equivalent is effectively UNAVAILABLE at the needed liquidity/history in India; recommend proxying with RBI's own inflation survey/expectations series or omitting this feature with a documented rationale. |
| VIX (equity vol) | **India VIX (NSE, ticker INDIAVIX)** | AVAILABLE (not attached — sourced separately from NSE) | Direct, liquid, well-established analogue. |
| Broad dollar index | **RBI's 36-currency NEER/REER index, or USDINR itself** | PROXY | Economic role differs: paper uses USD strength as a risk-off/carry signal; for an India book, USDINR / REER plays a structurally different role (imported inflation, FPI flow proxy) — needs its own justification at Gate A, not a mechanical substitution. |
| Core CPI (`CPILFESL`, FRED) | **MOSPI CPI (Combined), or CPI ex food & fuel as the India "core" analogue** | AVAILABLE | Published monthly by MOSPI; note the *publication lag* (India CPI has ~12-day lag vs. release date) must be respected in the point-in-time loader — using the print's *release date*, not its *reference month end*, in the feature panel. |
| Fama-French factors (market, size, value, profitability, investment, momentum, reversal) | **India Fama-French-style factor returns are not an official published series.** Nearest available: NSE factor indices themselves (attached file already **is** most of this — Alpha, Momentum, Quality, Value, Low-Vol act as investable factor-return proxies) | PROXY (already used as the core signal, not a forecasting feature — see Strategy Card) | Using the NSE indices as both the *tradable sleeves* and the *factor-exposure features* is intentional and disclosed here to avoid circularity: Stage 07 (portfolio validation) must fingerprint the strategy against these same indices' *lagged* returns, not contemporaneous, to avoid look-ahead in the factor-exposure check. |
| Trading volume (asset liquidity feature) | **NSE bhavcopy daily volume/delivery data** | AVAILABLE (not attached — sourced separately, freely downloadable from nseindia.com) | Needed for the liquidity-feature group in Table 7 and for Stage 07's capacity/liquidity check. |

## Structural / point-in-time data (needed regardless of which features are used)

| Requirement | Why (maps to memo's "No result without lineage" control) | Status |
|---|---|---|
| Index methodology documents (NIFTY Alpha 50, Momentum 50, MQVLV 50, etc.) | To confirm rebalance frequency, capping rules, and — critically — each index's **live calculation start date** vs. back-tested history, per Stage 04's point-in-time discussion. | Must be sourced from NSE Indices factsheets; not attached. |
| Point-in-time index constituent membership (if trading constituents rather than an index/ETF wrapper) | Prevents reconstitution lookahead (Stage 04). | UNAVAILABLE in this bundle — if the deployed strategy trades single stocks rather than an index-tracking product, this is a hard blocker for Gate B, not just a nice-to-have. |
| Corporate actions / delisting record | Needed only if trading constituents directly; not needed if implemented via index-tracking ETFs/funds, which is the default assumption in the worked Strategy Card. | N/A for current Card (uses index-level implementation) |
| A daily NAV series for whatever ETF/index fund is actually used to implement each sleeve (tracking error, expense ratio) | The paper trades ETFs (SPY/AGG/GLD), not the indices themselves, and charges a realistic bid-ask spread (§2.4: 5bp) — an India implementation should likewise cost the *investable product* (ETF/index fund), not the index, since index returns overstate what's actually capturable. | PROXY — worked example costs the index return directly and documents this as a known optimistic bias (see `docs/PIPELINE.md`, Stage 05 red flags) until real ETF NAV series are sourced. |

## Open issues carried to Gate A (must be resolved before this Card can pass Gate A)

1. **Confirm whether the attached "NIFTY 500" series is the Price Index or Total Return
   Index.** This materially changes the long-run return number (TRI runs ~1.5–2%/yr above
   PRI in India due to dividend reinvestment) and must not be silently assumed either way.
2. **Cash-rate / risk-free series is entirely missing from the attached data.** Until an
   RBI T-Bill or MIBOR series is sourced, the volatility-control and Markowitz optimizer
   legs of this strategy cannot be honestly evaluated — the worked example runs with this
   gap explicitly flagged rather than assuming a 0% or arbitrary rate.
3. **Bond and gold sleeves are unavailable in the attached bundle.** The worked example is
   therefore an **equity-factor-sleeve rotation**, not a full stock/bond/gold replication —
   this is a deliberate, disclosed scope reduction, not a silent one.
4. **Transaction cost assumption.** The paper uses 5bp half-spread on liquid US ETFs (§2.4,
   §2.5). Indian factor-sleeve products (smart-beta index funds) are materially less liquid
   and carry India-specific costs the paper's model doesn't have: **STT (0.001% on delivery
   equity sell side), stamp duty (0.015% on buy side), exchange transaction charges, GST on
   brokerage, and a wider bid-ask/impact cost for smart-beta ETFs versus SPY/AGG/GLD.** See
   `backtester/engine/costs.py` for the India cost model built to replace the paper's flat
   5bp assumption — this is a **Green flag** item (explicit adaptation, not a copy-paste of
   a US cost assumption into an India context) that Gate A should specifically check.
