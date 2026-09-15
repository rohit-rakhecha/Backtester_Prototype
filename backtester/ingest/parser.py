"""Stage 01 -- Ingest.

Parses a research paper (PDF) into page-anchored evidence units: layout text, tables
(best-effort), and metadata. Every extracted unit keeps its page number so a Strategy
Card built from it can cite `Evidence(page=..., quote=...)` -- see docs/PIPELINE.md,
Stage 01 for why this matters (a Card built from paraphrase silently drops constraints).

This module is deliberately dumb about INTERPRETATION: it extracts and tags evidence, it
does not decide what the strategy IS. That judgment call happens at Stage 02, checked by
a human at Gate A -- keeping this stage mechanical is itself a control (an ingest bug is
easy to catch; an ingest stage that also "interprets" hides interpretation errors inside
a parsing library).

It is NOT dumb about COVERAGE, though -- the whole pipeline is only as good as what this
stage surfaces, since Stage 02 (the Strategy Card) is built from whatever evidence this
stage hands it. So beyond raw page text, this module also mines three things every paper
has and every Strategy Card needs, comprehensively rather than on a field-by-field basis:
  1. the ABSTRACT (the paper's own one-paragraph economic thesis),
  2. ECONOMIC CLAIM sentences ("we find", "because", "consistent with", "driven by", ...)
     wherever they appear, not just near a specific parameter,
  3. CANDIDATE PARAMETERS (lookback windows, rebalance frequency, cost assumptions,
     volatility targets, and universe/asset-class keywords) via regex, each still
     page-anchored so a human can verify it in seconds rather than re-reading the paper.
This is what "Stage 01 dictates the flow of the entire pipeline" means in code: Stage 02's
interactive Card builder (ingest/strategy_card.py:build_card_interactively) is driven
entirely off the `IngestReport` this module produces, not off a human re-reading the PDF.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PageEvidence:
    page: int
    text: str


@dataclass
class ParsedPaper:
    source_path: str
    title: str
    pages: list[PageEvidence] = field(default_factory=list)

    def find(self, pattern: str, flags=re.IGNORECASE) -> list[PageEvidence]:
        """Return pages whose text matches a regex -- used to locate evidence for a
        specific Strategy Card field (e.g. cost assumption, lookback window)."""
        rx = re.compile(pattern, flags)
        return [p for p in self.pages if rx.search(p.text)]

    def quote_near(self, pattern: str, context_chars: int = 200) -> list[tuple[int, str]]:
        rx = re.compile(pattern, re.IGNORECASE)
        out = []
        for p in self.pages:
            m = rx.search(p.text)
            if m:
                start = max(0, m.start() - context_chars // 2)
                end = min(len(p.text), m.end() + context_chars // 2)
                out.append((p.page, p.text[start:end].strip()))
        return out

    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def parse_pdf(path: str) -> ParsedPaper:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageEvidence(page=i, text=text))
    title = pages[0].text.strip().splitlines()[0] if pages and pages[0].text.strip() else path
    return ParsedPaper(source_path=path, title=title, pages=pages)


# ---------------------------------------------------------------------------
# Comprehensive extraction -- abstract, economic claims, candidate parameters
# ---------------------------------------------------------------------------

def extract_abstract(paper: ParsedPaper, max_chars: int = 2500) -> str | None:
    """Pull the Abstract block off the first couple of pages. Heuristic: text between
    the word 'Abstract' and the next section heading ('1 Introduction', '1. Introduction',
    or a bare digit-led heading), capped at `max_chars` as a safety valve for papers with
    an unusual layout."""
    text = "\n".join(p.text for p in paper.pages[:2])
    m = re.search(r"\bAbstract\b\s*", text, re.IGNORECASE)
    if not m:
        return None
    rest = text[m.end():]
    end_m = re.search(r"\n\s*1[\.\s]+[A-Z][a-zA-Z ]{2,40}\n", rest)
    abstract = rest[: end_m.start()] if end_m else rest[:max_chars]
    return abstract.strip()[:max_chars]


# Sentence-level economic-reasoning markers. Deliberately broad: the point of Stage 01 is
# recall (surface everything that might matter for a human to read at Gate A), not
# precision -- false positives cost a human a half-second skim; false negatives cost a
# silently dropped assumption downstream.
_CLAIM_MARKERS = [
    r"\bwe find\b", r"\bwe show\b", r"\bwe demonstrate\b", r"\bwe argue\b",
    r"\bconsistent with\b", r"\bthe intuition\b", r"\bbecause\b", r"\bas a result\b",
    r"\bsuggests? that\b", r"\bevidence that\b", r"\boutperform", r"\bdriven by\b",
    r"\bthis (?:is|indicates|implies)\b", r"\bwe (?:believe|expect)\b",
    r"\bthe (?:reason|rationale)\b", r"\battribut\w+ to\b",
]
_CLAIM_RX = re.compile("|".join(_CLAIM_MARKERS), re.IGNORECASE)
_SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def extract_economic_claims(paper: ParsedPaper, max_claims: int = 15) -> list[tuple[int, str]]:
    """Every sentence, anywhere in the paper, that contains economic-reasoning language --
    not just sentences near a parameter. This is the raw material for the 'why' a
    researcher needs at Gate A: what does the PAPER say is driving its result, which the
    Strategy Card's `india_adaptation_notes` must then argue still applies (or doesn't)
    once the universe/data is substituted."""
    out: list[tuple[int, str]] = []
    for p in paper.pages:
        for sent in _SENTENCE_SPLIT_RX.split(p.text):
            sent = sent.strip().replace("\n", " ")
            if len(sent) < 25 or len(sent) > 400:
                continue
            if _CLAIM_RX.search(sent):
                out.append((p.page, sent))
            if len(out) >= max_claims:
                return out
    return out


# Candidate-parameter regexes. Each entry: (field_name, pattern, group_index_for_value).
_PARAM_PATTERNS = {
    "lookback_window": r"(trailing|lookback|rolling)\s+(?:window\s+of\s+)?(\d+)[\s-]*(day|month|year)s?",
    "rebalance_frequency": r"rebalanc\w*\s+(?:it\s+)?(annually|monthly|weekly|daily|quarterly)",
    "trading_cost_bps": r"(\d+(?:\.\d+)?)\s*(?:basis points|bps|bp\b)",
    "volatility_target_pct": r"(?:target(?:ed)?\s+(?:a\s+)?volatility|volatility\s+target)[^\d]{0,20}(\d+(?:\.\d+)?)\s*%",
    "sharpe_ratio_mentioned": r"sharpe\s+ratio\s+(?:of\s+)?(\d+\.\d+)",
}

# Broad asset-class / universe vocabulary -- deliberately spans equities, bonds, mutual
# funds, commodities, and derivatives so Stage 02/03 know to ask about ALL of them, not
# just equities, even on a run that currently only has equity data available (per the
# board memo: "the nature of both the research paper and the signal we want generated"
# might require other asset classes later).
_ASSET_CLASS_VOCAB = {
    "equity": [r"\bstock", r"\bequit", r"\bshares?\b", r"\bETF\b"],
    "bond": [r"\bbond", r"\bfixed[- ]income\b", r"\btreasur", r"\bgilt", r"\byield curve\b"],
    "gold_commodity": [r"\bgold\b", r"\bcommodit", r"\bprecious metal"],
    "mutual_fund": [r"\bmutual fund", r"\bNAV\b", r"\bAMFI\b"],
    "cash_rate": [r"\bfederal funds rate\b", r"\brisk-free rate\b", r"\bT-bill\b", r"\bMIBOR\b", r"\bcash\b"],
    "factor_style": [r"\bmomentum\b", r"\bvalue\b", r"\bquality\b", r"\blow volatility\b", r"\bfactor\b"],
    "derivatives": [r"\bfutures?\b", r"\boptions?\b", r"\bderivative"],
}


def extract_candidate_parameters(paper: ParsedPaper) -> dict:
    """Regex-mine candidate Strategy Card parameters, each with page citations, plus
    which broad asset classes the paper's vocabulary touches (feeds Stage 03's
    interactive data-source discovery -- see docs/PIPELINE.md and
    data/sources.discover_asset_classes_mentioned)."""
    results: dict[str, list[dict]] = {k: [] for k in _PARAM_PATTERNS}
    for name, pattern in _PARAM_PATTERNS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        for p in paper.pages:
            for m in rx.finditer(p.text):
                results[name].append({
                    "page": p.page,
                    "value": m.group(0),
                    "context": p.text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ").strip(),
                })

    asset_classes_mentioned: dict[str, int] = {}
    full_text = paper.full_text()
    for cls, patterns in _ASSET_CLASS_VOCAB.items():
        count = sum(len(re.findall(pat, full_text, re.IGNORECASE)) for pat in patterns)
        if count > 0:
            asset_classes_mentioned[cls] = count

    return {
        "parameters": results,
        "asset_classes_mentioned": asset_classes_mentioned,
    }


@dataclass
class IngestReport:
    """The single artifact Stage 01 hands to Stage 02. Everything the interactive
    Strategy Card builder shows a researcher comes from here -- see module docstring."""
    paper: ParsedPaper
    abstract: str | None
    economic_claims: list[tuple[int, str]]
    candidate_parameters: dict


def comprehensive_ingest(path: str) -> IngestReport:
    """Stage 01 entry point: parse the PDF and mine everything Stage 02 needs from it."""
    paper = parse_pdf(path)
    abstract = extract_abstract(paper)
    claims = extract_economic_claims(paper)
    params = extract_candidate_parameters(paper)
    return IngestReport(paper=paper, abstract=abstract, economic_claims=claims, candidate_parameters=params)


def print_ingest_report(report: IngestReport) -> None:
    print(f"Title (guess, page 1 first line): {report.paper.title}")
    print(f"Pages parsed: {len(report.paper.pages)}")
    print()
    print("=" * 78)
    print("ABSTRACT (economic thesis, as stated by the authors)")
    print("=" * 78)
    print(report.abstract or "(no Abstract heading found -- read the paper's intro manually)")
    print()
    print("=" * 78)
    print(f"ECONOMIC CLAIM SENTENCES ({len(report.economic_claims)} found, page-anchored)")
    print("=" * 78)
    for page, sent in report.economic_claims:
        print(f"  [p.{page}] {sent}")
    print()
    print("=" * 78)
    print("CANDIDATE PARAMETERS (regex-mined, verify each against its page before trusting it)")
    print("=" * 78)
    for name, hits in report.candidate_parameters["parameters"].items():
        if not hits:
            print(f"  {name}: not found")
            continue
        print(f"  {name}:")
        for h in hits[:5]:
            print(f"    [p.{h['page']}] \"{h['value']}\"  ...{h['context']}...")
    print()
    print("=" * 78)
    print("ASSET CLASSES / MARKET SEGMENTS THE PAPER TALKS ABOUT (mention counts)")
    print("=" * 78)
    for cls, count in sorted(report.candidate_parameters["asset_classes_mentioned"].items(),
                              key=lambda kv: -kv[1]):
        print(f"  {cls:16s} {count} mentions")
    print("  -> Stage 03 will ask you, for EACH of these, whether you have India data for it,")
    print("     want to use what's already in data/raw/, or need to upload a new file.")
