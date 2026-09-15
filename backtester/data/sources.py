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

import glob
import os
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


# ---------------------------------------------------------------------------
# Generic, interactive Stage 03 -- works for ANY paper and ANY asset class, not just
# the hardcoded factor-rotation worked example above.
# ---------------------------------------------------------------------------

@dataclass
class UniverseComponent:
    """One column, from one file, tagged with what kind of instrument it is. This is
    the atomic unit Stage 04/05 actually load -- deliberately asset-class agnostic
    (equity, bond, mutual fund, commodity, rate, ...) because a Card built from a
    different paper may need a completely different mix (per the board memo: "we might
    shift to other asset classes eg bonds/mutual funds too")."""
    label: str
    asset_class: str
    source_path: str
    column: str


def list_local_csv_datasets(data_dir: str = "data/raw") -> dict[str, list[str]]:
    """Every CSV or Excel file already on disk and its non-date columns -- what Stage 03
    can offer the user to pick from before asking them to upload anything new. Name kept
    as `_csv_` for backward compatibility with existing callers; it covers .xlsx/.xls too
    (see `backtester/data/pit_loader.py:_read_tabular_file`, used here for both formats)."""
    from .pit_loader import _read_tabular_file

    out: dict[str, list[str]] = {}
    paths = sorted(
        glob.glob(os.path.join(data_dir, "*.csv"))
        + glob.glob(os.path.join(data_dir, "*.xlsx"))
        + glob.glob(os.path.join(data_dir, "*.xls"))
    )
    for path in paths:
        try:
            cols = list(_read_tabular_file(path).columns)
        except Exception:
            cols = []
        out[path] = [c for c in cols if str(c).strip().lower() != "date"]
    return out


# The same broad vocabulary parser.py uses to scan the paper's own text -- kept as the
# default asset-class checklist so Stage 03 always asks about the full range even when a
# paper's wording happens to use different terms for the same thing.
DEFAULT_ASSET_CLASSES = ["equity", "bond", "gold_commodity", "mutual_fund", "cash_rate", "derivatives"]


def discover_datasets_interactively(
    card,
    asset_classes_mentioned: dict[str, int] | None = None,
    data_dir: str = "data/raw",
) -> tuple[DataFeasibilityRegistry, list[UniverseComponent]]:
    """Stage 03, interactively, generalized to any asset class. For every asset class the
    source paper touches on (from Stage 01's `asset_classes_mentioned`, falling back to
    `DEFAULT_ASSET_CLASSES` so nothing is skipped just because the regex missed a keyword),
    ask the user to:
      (a) pick from columns already available in `data_dir` (e.g. the attached NSE factor
          indices), and/or
      (b) upload a new file for it -- explicitly supporting non-equity data (a bond index,
          a mutual fund NAV series, a commodity price series, an RBI rate series, ...),
          tagged by asset class so later stages know what they're holding, and/or
      (c) explicitly decline with a reason, which is recorded as an accepted gap (never a
          silent drop -- the "No silent proxy use" control from the board memo).
    Returns a populated DataFeasibilityRegistry (for the Stage 03 report/gate) plus the
    list of UniverseComponents Stage 04 will actually load.
    """
    from ..ingest.interactive_io import prompt_multiselect, prompt_yesno, prompt_text, upload_file

    registry = DataFeasibilityRegistry()
    components: list[UniverseComponent] = []

    local = list_local_csv_datasets(data_dir)
    all_local_choices = [(path, col) for path, cols in local.items() for col in cols]

    asset_classes = list((asset_classes_mentioned or {}).keys()) or list(DEFAULT_ASSET_CLASSES)
    for c in DEFAULT_ASSET_CLASSES:
        if c not in asset_classes:
            asset_classes.append(c)

    print("\n" + "=" * 78)
    print("STAGE 03 -- Data feasibility & source discovery (interactive)")
    print("=" * 78)
    print(f"Strategy Card universe (from Stage 02): {card.universe}")
    print("For EACH asset class/segment below: pick existing local data, upload a new")
    print("file (any asset class -- equity, bond, mutual fund, commodity, rate, ...), or")
    print("explicitly decline with a reason. Nothing is silently skipped.\n")

    for cls in asset_classes:
        mention_note = f" (mentioned {asset_classes_mentioned[cls]}x in the paper)" if asset_classes_mentioned and cls in asset_classes_mentioned else ""
        print(f"\n-- Asset class / segment: {cls}{mention_note} --")
        options = [f"{col}  (in {os.path.basename(path)})" for path, col in all_local_choices]
        picks = prompt_multiselect(f"Use existing local column(s) for '{cls}'?", options)
        for label in picks:
            idx = options.index(label)
            path, col = all_local_choices[idx]
            components.append(UniverseComponent(label=col, asset_class=cls, source_path=path, column=col))
            registry.register(DataSourceEntry(
                requirement=f"{cls}:{col}", feasibility=Feasibility.available,
                source=f"local file {path}, column '{col}'",
            ))

        while prompt_yesno(f"Upload an additional file for '{cls}'?", default=False):
            path = upload_file(
                f"Upload a .csv or .xlsx file for '{cls}' (must have a 'date' column)", save_dir=data_dir
            )
            if not path or not os.path.exists(path):
                print("  No file received -- skipping upload.")
                break
            try:
                from .pit_loader import _read_tabular_file
                cols = [c for c in _read_tabular_file(path).columns if str(c).strip().lower() != "date"]
            except Exception as e:
                print(f"  Could not read {path}: {e}")
                cols = []
            for col in cols:
                if prompt_yesno(f"  Use column '{col}' from {os.path.basename(path)}?", default=True):
                    components.append(UniverseComponent(label=col, asset_class=cls, source_path=path, column=col))
                    registry.register(DataSourceEntry(
                        requirement=f"{cls}:{col}", feasibility=Feasibility.available,
                        source=f"uploaded file {path}, column '{col}'",
                    ))

        if not any(c.asset_class == cls for c in components):
            reason = prompt_text(
                f"No data sourced for '{cls}' -- reason (required, e.g. 'out of scope for v1', "
                "'not available in India', 'will proxy later')"
            )
            req_name = f"{cls}:unsourced"
            registry.register(DataSourceEntry(
                requirement=req_name, feasibility=Feasibility.unavailable,
                source="not sourced in this session",
            ))
            registry.accept_gap(req_name, note=reason, accepted_by="interactive-session")

    return registry, components
