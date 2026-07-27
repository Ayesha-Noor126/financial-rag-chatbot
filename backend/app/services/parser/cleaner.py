"""
Text cleaning and section-heading detection.

Design choice: kept separate from PDFParser (single responsibility). The
parser's job is "get clean-ish text off the page geometry"; the cleaner's
job is "normalize whitespace and identify structure." Splitting these two
means we can unit-test whitespace/heading logic without needing a real PDF
fixture, and we can swap in a different parser (e.g. Azure Doc Intelligence
for scanned reports) later without touching cleaning logic.
"""

import re

from app.models.schemas import CleanedPage, RawPage

WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")

# Common financial-report section headings. Matched case-insensitively as a
# whole line. This list is intentionally small and specific rather than a
# generic "line is short and capitalized" heuristic, which throws too many
# false positives on financial statements full of short capitalized labels
# like "TOTAL ASSETS".
SECTION_HEADING_RE = re.compile(
    r"^\s*(balance sheet|income statement|cash flow statement|"
    r"statement of financial position|notes to (the )?financial statements|"
    r"management discussion and analysis|md&a|auditor'?s report|"
    r"risk factors|segment (information|reporting)|"
    r"consolidated statements? of (income|operations|equity)|"
    r"revenue by (region|segment)|executive summary)\s*$",
    re.IGNORECASE,
)


class TextCleaner:
    def clean(self, raw_pages: list[RawPage]) -> list[CleanedPage]:
        cleaned: list[CleanedPage] = []
        current_section: str | None = None

        for raw in raw_pages:
            section_on_page = self._find_section(raw.raw_text)
            if section_on_page:
                current_section = section_on_page

            text = self._normalize_whitespace(raw.raw_text)
            cleaned.append(
                CleanedPage(
                    page_number=raw.page_number,
                    text=text,
                    section=current_section,
                )
            )
        return cleaned

    def _normalize_whitespace(self, text: str) -> str:
        text = WHITESPACE_RE.sub(" ", text)
        text = BLANK_LINES_RE.sub("\n\n", text)
        lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]  # drop empty lines
        return "\n".join(lines)

    def _find_section(self, text: str) -> str | None:
        for line in text.split("\n"):
            match = SECTION_HEADING_RE.match(line.strip())
            if match:
                return match.group(1).title()
        return None
