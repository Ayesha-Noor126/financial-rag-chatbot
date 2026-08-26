"""
Unit tests for TextCleaner (app/services/parser/cleaner.py).

What we test
────────────
• _normalize_whitespace collapses multiple spaces and blank lines.
• _find_section correctly identifies known headings (case-insensitive).
• _find_section returns None for lines that look like headings but aren't
  in the allow-list (e.g. "TOTAL ASSETS").
• clean() wires parse-and-clean correctly for a list of RawPages.

No external dependencies — TextCleaner is pure Python.
"""

import pytest

from app.models.schemas import RawPage
from app.services.parser.cleaner import SECTION_HEADING_RE, TextCleaner


# ── _normalize_whitespace ─────────────────────────────────────────────────────

class TestNormalizeWhitespace:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_collapses_multiple_spaces(self):
        result = self.cleaner._normalize_whitespace("hello   world")
        assert "  " not in result
        assert "hello world" in result

    def test_collapses_multiple_tabs(self):
        result = self.cleaner._normalize_whitespace("col1\t\tcol2")
        assert "\t" not in result

    def test_collapses_excess_blank_lines(self):
        text = "paragraph one\n\n\n\nparagraph two"
        result = self.cleaner._normalize_whitespace(text)
        # Three or more consecutive newlines → two (one blank line)
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_whitespace_per_line(self):
        text = "  line one  \n  line two  "
        result = self.cleaner._normalize_whitespace(text)
        for line in result.split("\n"):
            assert line == line.strip()

    def test_empty_string_returns_empty(self):
        assert self.cleaner._normalize_whitespace("") == ""

    def test_preserves_single_blank_line(self):
        text = "paragraph one\n\nparagraph two"
        result = self.cleaner._normalize_whitespace(text)
        assert "paragraph one" in result
        assert "paragraph two" in result


# ── _find_section ─────────────────────────────────────────────────────────────

class TestFindSection:
    def setup_method(self):
        self.cleaner = TextCleaner()

    @pytest.mark.parametrize("heading", [
        "Balance Sheet",
        "BALANCE SHEET",
        "balance sheet",
        "Income Statement",
        "Cash Flow Statement",
        "Statement of Financial Position",
        "Notes to the Financial Statements",
        "Notes to Financial Statements",
        "Management Discussion and Analysis",
        "MD&A",
        "Auditor's Report",
        "Auditors Report",
        "Risk Factors",
        "Segment Information",
        "Segment Reporting",
        "Executive Summary",
    ])
    def test_known_headings_detected(self, heading):
        text = f"Some preamble text\n{heading}\nSome body text"
        result = self.cleaner._find_section(text)
        assert result is not None, f"Expected '{heading}' to be detected as a section heading"

    def test_returns_title_cased_string(self):
        result = self.cleaner._find_section("balance sheet\nsome text")
        assert result == result.title()

    def test_unknown_heading_like_phrase_returns_none(self):
        # "TOTAL ASSETS" looks like a heading but is not in the allow-list
        result = self.cleaner._find_section("TOTAL ASSETS\n1,000,000")
        assert result is None

    def test_returns_none_when_no_heading_present(self):
        text = "The company reported revenue of Rs 832 million in Q3 2017."
        assert self.cleaner._find_section(text) is None

    def test_heading_with_surrounding_whitespace_detected(self):
        # SECTION_HEADING_RE uses ^\s* and \s*$ so leading/trailing spaces are OK
        text = "  Balance Sheet  \nAssets..."
        result = self.cleaner._find_section(text)
        assert result is not None


# ── clean() (full pipeline) ───────────────────────────────────────────────────

class TestClean:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_returns_same_number_of_pages(self):
        pages = [
            RawPage(page_number=1, raw_text="Hello world", has_tables=False),
            RawPage(page_number=2, raw_text="Goodbye world", has_tables=False),
        ]
        cleaned = self.cleaner.clean(pages)
        assert len(cleaned) == 2

    def test_page_numbers_preserved(self):
        pages = [
            RawPage(page_number=7, raw_text="Some text", has_tables=False),
        ]
        cleaned = self.cleaner.clean(pages)
        assert cleaned[0].page_number == 7

    def test_section_propagates_across_pages(self):
        """
        A heading found on page 1 should carry forward to page 2
        if page 2 has no heading of its own.
        """
        pages = [
            RawPage(page_number=1, raw_text="Balance Sheet\nAssets 1000", has_tables=False),
            RawPage(page_number=2, raw_text="Liabilities 500", has_tables=False),
        ]
        cleaned = self.cleaner.clean(pages)
        assert cleaned[0].section is not None
        assert cleaned[1].section == cleaned[0].section  # carried forward

    def test_section_updates_on_new_heading(self):
        pages = [
            RawPage(page_number=1, raw_text="Balance Sheet\nAssets 1000", has_tables=False),
            RawPage(page_number=2, raw_text="Income Statement\nRevenue 500", has_tables=False),
        ]
        cleaned = self.cleaner.clean(pages)
        assert cleaned[0].section != cleaned[1].section

    def test_no_section_when_no_heading(self):
        pages = [RawPage(page_number=1, raw_text="Random financial text", has_tables=False)]
        cleaned = self.cleaner.clean(pages)
        assert cleaned[0].section is None

    def test_empty_pages_list(self):
        assert self.cleaner.clean([]) == []
