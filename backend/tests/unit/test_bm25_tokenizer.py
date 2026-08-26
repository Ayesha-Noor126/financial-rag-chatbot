"""
Unit tests for the BM25 tokenizer (app/services/rag/retriever/bm25_index.py).

`tokenize()` is a pure function with no external dependencies — the fastest
possible test category.

What we test
────────────
• Output is always lowercase.
• Currency symbols ($, €, £, ₹) are preserved as part of tokens.
• Alphanumeric tokens are kept; punctuation-only tokens are dropped.
• Empty / whitespace-only input returns an empty list.
• Financial figures like "$2.4bn" and "₹125,682" tokenize correctly.
• The function is deterministic (same input → same output).
"""

import pytest

from app.services.rag.retriever.bm25_index import tokenize


class TestTokenize:
    # ── Basic lowercasing ────────────────────────────────────────────────────

    def test_output_is_lowercase(self):
        tokens = tokenize("Revenue ASSETS Profit")
        assert all(t == t.lower() for t in tokens), "All tokens must be lowercase"

    def test_mixed_case_word_lowercased(self):
        assert "netsol" in tokenize("NetSol Technologies Limited")

    # ── Currency symbols preserved ───────────────────────────────────────────

    def test_dollar_symbol_in_token(self):
        tokens = tokenize("Revenue was $2.4bn")
        assert any("$" in t for t in tokens), "$ should be kept inside the token"

    def test_euro_symbol_in_token(self):
        tokens = tokenize("Amount €50,000")
        assert any("€" in t for t in tokens)

    def test_pound_symbol_in_token(self):
        tokens = tokenize("GBP £1,200")
        assert any("£" in t for t in tokens)

    def test_rupee_symbol_in_token(self):
        tokens = tokenize("Profit ₹125,682 thousand")
        assert any("₹" in t for t in tokens), "₹ should be kept — Pakistani/Indian docs use it"

    def test_percent_symbol_in_token(self):
        tokens = tokenize("Margin improved 37%")
        assert any("%" in t for t in tokens)

    # ── Numeric tokens ───────────────────────────────────────────────────────

    def test_plain_numbers_kept(self):
        tokens = tokenize("FY2023 Q3 2017")
        assert "fy2023" in tokens or "2023" in tokens
        assert "2017" in tokens

    # ── Empty / whitespace ───────────────────────────────────────────────────

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert tokenize("   \t\n  ") == []

    # ── Punctuation-only strings ─────────────────────────────────────────────

    def test_punctuation_only_not_included(self):
        tokens = tokenize("!!! ??? ...")
        assert tokens == [], f"Expected no tokens, got {tokens}"

    # ── Determinism ──────────────────────────────────────────────────────────

    def test_deterministic(self):
        text = "Net profit Rs 125,682 thousand for Q3 FY2017"
        assert tokenize(text) == tokenize(text)

    # ── Real financial sentence ───────────────────────────────────────────────

    def test_financial_sentence_tokens(self):
        text = "Profit before taxation for the period 128,569"
        tokens = tokenize(text)
        assert "profit" in tokens
        assert "taxation" in tokens
        assert "period" in tokens
        assert "128" in tokens or any("128" in t for t in tokens)
