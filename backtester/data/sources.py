"""Stage 03 -- Data feasibility.

For every data requirement implied by an (approved) Strategy Card, classify it
AVAILABLE | PROXY | UNAVAILABLE and either fail fast or require an explicit sign-off note.

Economic intuition: this is where a US paper's data universe meets what an India desk can
actually get, point-in-time, without a silent substitution changing the economic bet being
tested. See docs/DATA_SOURCES.md for the full worked classification for the attached paper.
The registry below is the machine-checkable version of that document -- so a Card cannot
silently proceed past Stage 03 with an unresolved PROXY or UNAVAILABLE requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Feasibility(str, Enum):
    available = "available"
    proxy = "proxy"
    unavailable = "unavailable"


@dataclass
class DataSourceEntry:
    requirement: str
    feasibility: Feasibility
    source: str
    proxy_rationale: str | None = None  # REQUIRED if feasibility == proxy
    signoff_by: str | None = None
    signoff_note: str | None = None

    def __post_init__(self):
        if self.feasibility == Feasibility.proxy and not self.proxy_rationale:
            raise ValueError(
                f"Data requirement '{self.requirement}' is marked PROXY but has no "
                "proxy_rationale. This violates the 'No silent proxy use' control."
            )


class DataFeasibilityRegistry:
    """Registry of data source classifications for a given Strategy Card.

    Usage: register every requirement implied by the Card, then call `check()` -- it
    raises if any UNAVAILABLE requirement lacks an explicit `accept_gap` sign-off, and
    returns the list of PROXY entries that still need a Gate A sign-off note.
    """

    def __init__(self):
        self.entries: dict[str, DataSourceEntry] = {}
        self._accepted_gaps: set[str] = set()

    def register(self, entry: DataSourceEntry) -> None:
        self.entries[entry.requirement] = entry

    def accept_gap(self, requirement: str, note: str, accepted_by: str) -> None:
        """Explicitly accept that a requirement is unavailable and the Card will proceed
        with reduced scope (e.g. 'equity-sleeves-only, no bond/gold leg'). This is the
        only way an UNAVAILABLE requirement is allowed to pass Stage 03 -- see
        docs/PIPELINE.md Stage 03 red/green flags."""
        e = self.entries.get(requirement)
        if e is None:
            raise KeyError(requirement)
        e.signoff_note = note
        e.signoff_by = accepted_by
        self._accepted_gaps.add(requirement)

    def sign_off_proxy(self, requirement: str, note: str, accepted_by: str) -> None:
        e = self.entries.get(requirement)
        if e is None:
            raise KeyError(requirement)
        e.signoff_note = note
        e.signoff_by = accepted_by

    def check(self) -> dict:
        blocking = []
        needs_signoff = []
        for req, e in self.entries.items():
            if e.feasibility == Feasibility.unavailable and req not in self._accepted_gaps:
                blocking.append(req)
            if e.feasibility == Feasibility.proxy and not e.signoff_by:
                needs_signoff.append(req)
        return {
            "ok_to_proceed": not blocking and not needs_signoff,
            "blocking_unavailable": blocking,
            "proxies_needing_signoff": needs_signoff,
        }

    def report(self) -> str:
        lines = ["Stage 03 -- Data Feasibility Report", "=" * 40]
        for req, e in self.entries.items():
            lines.append(f"[{e.feasibility.value.upper():11s}] {req}")
            lines.append(f"    source: {e.source}")
            if e.proxy_rationale:
                lines.append(f"    proxy rationale: {e.proxy_rationale}")
            if e.signoff_by:
                lines.append(f"    signed off by {e.signoff_by}: {e.signoff_note}")
            elif e.feasibility != Feasibility.available:
                lines.append("    ** NEEDS SIGN-OFF **")
        return "\n".join(lines)


def india_registry_for_factor_rotation_card() -> DataFeasibilityRegistry:
    """The Stage 03 registry for the worked example Strategy Card
    (`examples/strategy_card_factor_rotation_india.yaml`), matching docs/DATA_SOURCES.md.
    """
    reg = DataFeasibilityRegistry()
    reg.register(DataSourceEntry(
        requirement="equity_factor_sleeves_daily_close",
        feasibility=Feasibility.available,
        source="NSE Indices Ltd -- data/raw/nifty_factor_indices.csv "
               "(Alpha 50, Momentum 50, Multifactor MQVLV 50, Quality 50, Value 50, "
               "Low Volatility 50, High Beta 50), daily close, 2003-2026.",
    ))
    reg.register(DataSourceEntry(
        requirement="benchmark_daily_close",
        feasibility=Feasibility.available,
        source="NSE Indices Ltd -- NIFTY 500, data/raw/nifty_factor_indices.csv",
    ))
    reg.register(DataSourceEntry(
        requirement="cash_risk_free_rate",
        feasibility=Feasibility.unavailable,
        source="RBI 91-day T-Bill cutoff yield / MIBOR (not attached, not yet sourced)",
        proxy_rationale=None,
    ))
    reg.register(DataSourceEntry(
        requirement="bond_sleeve",
        feasibility=Feasibility.unavailable,
        source="India G-Sec index / gilt ETF NAV (not attached, not yet sourced)",
    ))
    reg.register(DataSourceEntry(
        requirement="gold_sleeve",
        feasibility=Feasibility.unavailable,
        source="MCX gold / domestic gold ETF NAV (not attached, not yet sourced)",
    ))
    reg.register(DataSourceEntry(
        requirement="india_vix",
        feasibility=Feasibility.proxy,
        source="NSE INDIAVIX (not attached in this bundle)",
        proxy_rationale="Direct India analogue of the paper's VIX feature; not yet wired "
                          "into the feature panel for this worked example.",
    ))
    return reg
