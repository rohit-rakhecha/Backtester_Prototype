"""Stage 04 -- Point-in-time data.

Loads prices as a FROZEN SNAPSHOT: a content-hashed, lineage-recorded copy of a source
file. The engine (Stage 05) never reads a live feed and never reads past a strategy's
configured `as_of` cutoff -- both are enforced here in code, not left to convention.

Economic intuition / failure modes this prevents (see docs/PIPELINE.md Stage 04):
  1. Index reconstitution lookahead -- avoided here by working at the index-return level
     (we consume NSE's own point-in-time-correct index series) rather than reconstructing
     constituent weights ourselves; if a future Card trades constituents directly, a
     membership-as-of-date loader must be added and this module documents that gap.
  2. Launch-date truncation -- each factor index in the attached file has a first date
     with a non-null value; dates before that are NOT "zero return", they are "this index
     did not exist yet". This loader distinguishes NaN-before-launch from a real gap and
     will not silently forward-fill across a launch boundary.

Accepts both CSV and Excel (.xlsx/.xls) source files -- a manually uploaded dataset is
just as likely to be the raw spreadsheet someone downloaded from NSE/a data vendor as a
pre-cleaned CSV, and this stage should not fail on the file format alone (see
`_read_tabular_file` / `from_file`).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


@dataclass
class Lineage:
    source_path: str
    content_sha256: str
    retrieved_at: str
    n_rows: int
    columns: list[str]
    first_date: str
    last_date: str

    def to_dict(self) -> dict:
        return self.__dict__


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_excel_multi_index_layout(path: str) -> "pd.DataFrame | None":
    """Parses the specific layout of the attached `Factor_Indices_Historical_Price_Data.xlsx`
    (and any file shaped the same way): row 2 holds an index/series name every other
    column, row 3 holds 'Date'/'Close' sub-headers under each, and data starts row 4.
    Returns None (not an error) if this specific shape isn't detected, so the caller can
    fall back to a generic single-header read instead of failing outright.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 4:
        return None

    names_row = rows[1]
    subheader_row = rows[2]
    date_col_idx = None
    close_col_by_name: dict[str, int] = {}
    current_name = None
    for idx, val in enumerate(names_row):
        if val:
            current_name = str(val).strip()
        sub = subheader_row[idx] if idx < len(subheader_row) else None
        if sub and str(sub).strip().lower() == "date" and date_col_idx is None:
            date_col_idx = idx
        elif sub and str(sub).strip().lower() == "close" and current_name:
            close_col_by_name[current_name] = idx

    if date_col_idx is None or not close_col_by_name:
        return None  # doesn't match this layout -- let the generic reader try instead

    records = []
    for r in rows[3:]:
        date = r[date_col_idx] if date_col_idx < len(r) else None
        if date is None:
            continue
        row = {"date": date}
        for name, idx in close_col_by_name.items():
            row[name.replace(" ", "_")] = r[idx] if idx < len(r) else None
        records.append(row)
    return pd.DataFrame(records)


def _read_tabular_file(path: str, date_col: str = "date") -> pd.DataFrame:
    """Reads a CSV or Excel file into a date-indexed DataFrame, normalizing the date
    column name to `date_col` regardless of how it was capitalized/spaced in the source.
    This is the single place file-format handling lives -- every loader in this module
    goes through it, so a manually uploaded .xlsx works everywhere a .csv does."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls", ".xlsm"):
        df = _read_excel_multi_index_layout(path)
        if df is None:
            df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}' for {path} -- expected .csv, .xlsx, .xls, or .xlsm."
        )

    if date_col not in df.columns:
        candidates = [c for c in df.columns if str(c).strip().lower() == date_col.lower()]
        if not candidates:
            raise ValueError(
                f"{path}: no '{date_col}' column found (columns present: {list(df.columns)}). "
                "Point-in-time loading requires an explicit date column -- Stage 04 refuses "
                "to guess which column is the date."
            )
        df = df.rename(columns={candidates[0]: date_col})

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True).set_index(date_col)
    return df


class PointInTimeDataset:
    """A frozen, hashed, lineage-recorded price panel.

    `as_of(date)` is the hard point-in-time cutoff: any code that needs "what did we know
    on date X" must go through it, never through raw `.df` access, to keep the engine from
    accidentally leaking future data into a signal computed at an earlier date.
    """

    def __init__(self, df: pd.DataFrame, lineage: Lineage):
        self.df = df
        self.lineage = lineage

    @classmethod
    def from_file(cls, path: str, date_col: str = "date") -> "PointInTimeDataset":
        """Loads a CSV or Excel (.xlsx/.xls/.xlsm) file -- use this for any manually
        uploaded dataset, since you can't assume which format the user provided."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- Stage 04 refuses to fabricate a snapshot. "
                "Source the file per docs/DATA_SOURCES.md first."
            )
        df = _read_tabular_file(path, date_col=date_col)
        lineage = Lineage(
            source_path=path,
            content_sha256=_hash_file(path),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            n_rows=len(df),
            columns=list(df.columns),
            first_date=str(df.index.min().date()),
            last_date=str(df.index.max().date()),
        )
        return cls(df, lineage)

    @classmethod
    def from_csv(cls, path: str, date_col: str = "date") -> "PointInTimeDataset":
        """Kept as an explicit alias for callers that only ever pass CSVs -- delegates to
        `from_file`, which also handles Excel, so existing CSV-only call sites keep working
        unchanged while new/updated call sites can move to `from_file` for format-agnostic
        loading."""
        return cls.from_file(path, date_col=date_col)

    def launch_dates(self) -> dict:
        """First non-null date per column -- the point at which each series actually
        starts existing, distinct from a data gap. See module docstring, failure mode 2."""
        return {c: str(self.df[c].dropna().index.min().date()) for c in self.df.columns}

    def as_of(self, date) -> pd.DataFrame:
        """Everything known up to and including `date`. Hard cutoff: no lookahead."""
        return self.df.loc[: pd.Timestamp(date)]

    def save_snapshot(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.df.to_csv(out_path)
        lineage_path = out_path + ".lineage.json"
        with open(lineage_path, "w") as f:
            json.dump(self.lineage.to_dict(), f, indent=2)


def merge_universe_components(components: list, date_col: str = "date") -> "PointInTimeDataset":
    """Merge Stage 03's interactively-selected `UniverseComponent`s -- each naming a
    source file (CSV or Excel) and a column, possibly across several different files
    (e.g. an equity index from one upload, a bond index from another) -- into a single
    date-indexed price panel, labeled by each component's `label`. Loads each source file
    once even when multiple components share it.

    Duck-types on `comp.source_path` / `comp.column` / `comp.label` rather than importing
    `data.sources.UniverseComponent`, so this stays a one-way dependency (sources.py may
    depend on pit_loader concepts later; pit_loader never needs to know about Stage 03's
    interactive-discovery types).
    """
    cache: dict[str, pd.DataFrame] = {}
    series: dict[str, pd.Series] = {}
    source_files: list[str] = []
    for comp in components:
        if comp.source_path not in cache:
            raw = _read_tabular_file(comp.source_path, date_col=date_col)
            cache[comp.source_path] = raw
            source_files.append(comp.source_path)
        series[comp.label] = cache[comp.source_path][comp.column]
    merged = pd.DataFrame(series).sort_index()

    lineage = Lineage(
        source_path="; ".join(source_files),
        content_sha256="; ".join(_hash_file(p) for p in source_files),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        n_rows=len(merged),
        columns=list(merged.columns),
        first_date=str(merged.index.min().date()) if len(merged) else "n/a",
        last_date=str(merged.index.max().date()) if len(merged) else "n/a",
    )
    return PointInTimeDataset(merged, lineage)
