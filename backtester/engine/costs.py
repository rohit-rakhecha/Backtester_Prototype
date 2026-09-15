"""Stage 05 -- India transaction cost model.

The source paper charges a flat 5bp half-spread on SPY/AGG/GLD (§2.4/§2.5) -- reasonable
for three of the most liquid ETFs on earth. That assumption does NOT transfer to Indian
smart-beta index products, which trade with wider spreads and carry India-specific
statutory costs the paper's model has no line item for. Copy-pasting the paper's 5bp
into an India backtest would be exactly the "silent proxy" failure mode this whole
pipeline exists to prevent (see docs/DATA_SOURCES.md, item 4) -- so this is a distinct,
disclosed, India-specific cost model, not a parameter tweak.

Cost components modeled (delivery/CNC equity trades, current India statutory rates):
  - STT (Securities Transaction Tax): 0.1% on BOTH buy and sell for delivery equity
    (₹0.001 per ₹1 traded, each side) -- this is the single largest fixed cost component
    and, unlike US markets, is NOT symmetric-optional; it is charged on every trade.
  - Stamp duty: 0.015% on the buy side only (state-mandated, SEBI-uniform since 2020).
  - Exchange transaction charges: ~0.00297% (NSE) on turnover, each side.
  - SEBI turnover fee: ~0.0001% each side.
  - GST: 18% on (brokerage + exchange transaction charges), each side.
  - Brokerage: assumed 0 (discount broker, direct/index-fund execution) -- flagged as an
    explicit assumption, not hidden.
  - Impact cost / bid-ask spread: NOT a statutory rate, must be estimated per instrument;
    smart-beta index funds are materially less liquid than SPY/AGG/GLD, so a wider
    half-spread than the paper's 5bp is used by default (see `impact_cost_bps` below) and
    is exactly the kind of number that needs a Gate A sign-off with real market data
    before this strategy could pass to PORTFOLIO USEFUL on the promotion ladder.

This model deliberately keeps every component as a separate, named, overridable field --
a single blended "India cost is roughly 2x US cost" number would hide which component is
actually doing the work, and would be a red flag under Stage 05 ("cost model bypassed").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndiaEquityCostModel:
    stt_bps_each_side: float = 10.0          # 0.1% = 10 bps, delivery equity, each side
    stamp_duty_bps_buy_side: float = 1.5      # 0.015% buy side only
    exchange_txn_bps_each_side: float = 0.297  # NSE ~0.00297%
    sebi_fee_bps_each_side: float = 0.01
    gst_rate: float = 0.18                    # applied to (brokerage + exchange charges)
    brokerage_bps_each_side: float = 0.0      # assume discount/direct index-fund execution
    impact_cost_bps_each_side: float = 15.0   # wider than paper's 5bp flat spread -- see docstring

    def round_trip_cost_bps(self) -> float:
        """Total cost, in bps of notional, for a full buy+sell round trip."""
        buy = (
            self.stt_bps_each_side
            + self.stamp_duty_bps_buy_side
            + self.exchange_txn_bps_each_side
            + self.sebi_fee_bps_each_side
            + self.brokerage_bps_each_side
            + self.impact_cost_bps_each_side
        )
        buy += self.gst_rate * (self.brokerage_bps_each_side + self.exchange_txn_bps_each_side)
        sell = (
            self.stt_bps_each_side
            + self.exchange_txn_bps_each_side
            + self.sebi_fee_bps_each_side
            + self.brokerage_bps_each_side
            + self.impact_cost_bps_each_side
        )
        sell += self.gst_rate * (self.brokerage_bps_each_side + self.exchange_txn_bps_each_side)
        return buy + sell

    def one_way_cost_bps(self) -> float:
        """Approximate one-way (single trade) cost -- used the same way the paper uses
        its half-spread `s/2` in the rebalance cost term (their eq. in §2.4/Appendix A)."""
        return self.round_trip_cost_bps() / 2.0

    def trade_cost(self, traded_notional: float) -> float:
        """Cost in currency units for a single trade of given (absolute) notional."""
        return traded_notional * (self.one_way_cost_bps() / 10_000.0)


PAPER_COST_MODEL_BPS = 5.0  # the source paper's flat half-spread assumption, for replication-gap comparisons

DEFAULT_INDIA_COST_MODEL = IndiaEquityCostModel()
