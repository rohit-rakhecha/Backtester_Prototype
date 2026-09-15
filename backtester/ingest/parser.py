"""Stage 01 -- Ingest.

Parses a research paper (PDF) into page-anchored evidence units: layout text, tables
(best-effort), and metadata. Every extracted unit keeps its page number so a Strategy
Card built from it can cite `Evidence(page=..., quote=...)` -- see docs/PIPELINE.md,
Stage 01 for why this matters (a Card built from paraphrase silently drops constraints).

This module is deliberately dumb: it extracts and tags evidence, it does not interpret.
Interpretation (turning evidence into a Strategy Card) is a human/AI judgment call made
at Stage 02, checked by a human at Gate A -- keeping this stage mechanical is itself a
control (an ingest bug is easy to catch; an ingest stage that also "interprets" hides
interpretation errors inside a parsing library).
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


def parse_pdf(path: str) -> ParsedPaper:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageEvidence(page=i, text=text))
    title = pages[0].text.strip().splitlines()[0] if pages and pages[0].text.strip() else path
    return ParsedPaper(source_path=path, title=title, pages=pages)
