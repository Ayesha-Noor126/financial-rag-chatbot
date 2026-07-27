"""
PDF parsing via PyMuPDF.

Design choice: PyMuPDF (fitz) over pdfplumber/pypdf because it gives us
per-span font size and position data cheaply, which we need for two things
that matter a lot in financial reports:
  1. Heading detection (section headers use larger/bold fonts) -> preserved
     as metadata so retrieval can cite "Section: Balance Sheet".
  2. Header/footer detection (running text repeated identically near the
     top/bottom margin on many pages) -> stripped so it doesn't pollute
     every chunk with "Nestlé Annual Report 2023 | Page 12" noise.

We do NOT use pypdf here because it has no reliable font/position API,
which would force header/footer detection down to fragile regex-only logic.
"""

import re
from collections import Counter

import fitz  # PyMuPDF

from app.models.schemas import RawPage

# Financial reports often use "Page X", "X of Y", or a bare number at the
# extreme top/bottom of the page as pagination -- caught here as a fallback
# in addition to positional header/footer detection below.
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*$"),
]

MARGIN_FRACTION = 0.08  # top/bottom 8% of page height treated as margin zone


class PDFParser:
    def __init__(self, margin_fraction: float = MARGIN_FRACTION):
        self.margin_fraction = margin_fraction

    def parse(self, file_path: str) -> list[RawPage]:
        doc = fitz.open(file_path)
        try:
            page_lines = self._extract_lines_per_page(doc)
            header_candidates, footer_candidates = self._detect_repeated_margins(
                page_lines, doc
            )
            raw_pages: list[RawPage] = []
            for page_index, lines in enumerate(page_lines):
                filtered = [
                    line
                    for line in lines
                    if line.strip() not in header_candidates
                    and line.strip() not in footer_candidates
                    and not self._is_page_number(line)
                ]
                text = "\n".join(filtered)
                has_tables = self._looks_like_table(doc[page_index])
                raw_pages.append(
                    RawPage(
                        page_number=page_index + 1,
                        raw_text=text,
                        has_tables=has_tables,
                    )
                )
            return raw_pages
        finally:
            doc.close()

    # -- internals ---------------------------------------------------

    def _extract_lines_per_page(self, doc: "fitz.Document") -> list[list[str]]:
        pages = []
        for page in doc:
            text = page.get_text("text")
            lines = [ln for ln in text.split("\n")]
            pages.append(lines)
        return pages

    def _detect_repeated_margins(
        self, page_lines: list[list[str]], doc: "fitz.Document"
    ) -> tuple[set[str], set[str]]:
        """
        A line is treated as a header/footer candidate if it appears near the
        top or bottom margin AND repeats verbatim across >= 40% of pages.
        This threshold avoids stripping a genuinely recurring financial term
        (e.g. "Net Revenue") that just happens to appear on many pages but
        isn't confined to the margins.
        """
        top_lines: Counter[str] = Counter()
        bottom_lines: Counter[str] = Counter()

        for page_index, page in enumerate(doc):
            height = page.rect.height
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)
            for b in blocks:
                y0, y1, text = b[1], b[3], b[4]
                stripped = text.strip()
                if not stripped:
                    continue
                if y1 <= height * self.margin_fraction:
                    top_lines[stripped] += 1
                elif y0 >= height * (1 - self.margin_fraction):
                    bottom_lines[stripped] += 1

        n_pages = max(len(page_lines), 1)
        threshold = max(2, int(n_pages * 0.4))

        headers = {t for t, c in top_lines.items() if c >= threshold}
        footers = {t for t, c in bottom_lines.items() if c >= threshold}
        return headers, footers

    def _is_page_number(self, line: str) -> bool:
        return any(p.match(line) for p in PAGE_NUMBER_PATTERNS)

    def _looks_like_table(self, page: "fitz.Page") -> bool:
        """
        Cheap heuristic: PyMuPDF's table finder. Used only as a metadata flag
        (Chunk.chunk_type) so downstream chunking can avoid splitting a table
        mid-row -- exact table-structure preservation is handled in Phase 6
        (financial table extraction bonus feature).
        """
        try:
            tables = page.find_tables()
            return len(tables.tables) > 0
        except Exception:
            return False
