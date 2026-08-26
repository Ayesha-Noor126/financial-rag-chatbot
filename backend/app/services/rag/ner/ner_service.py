"""
Named Entity Recognition.

Design choice: spaCy's default en_core_web_sm catches ORG (company) and
sometimes GPE (region) well, but it has no concept of "financial metric" as
an entity type and is unreliable on bare years/currency symbols in short
queries ("revenue 2024" is too sparse a context for statistical NER). So we
run spaCy first, then layer a small domain lexicon + regex pass on top for
the categories the spec explicitly asks for (Financial Metric, Year,
Currency) rather than trying to train a custom NER model, which would be
overkill for a bounded, well-known vocabulary of ~20 financial terms.

Output feeds two places: (1) the retriever can boost BM25 queries with
extracted entities, (2) the confidence scorer (Phase 3) can check whether
entities in the question actually appear in the retrieved chunks, as a
cheap hallucination signal.
"""

import re

import spacy

from app.core.config import Settings

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
CURRENCY_RE = re.compile(r"\b(USD|EUR|GBP|CHF|PKR|INR|RS|RUPEES?|\$|€|£|₹)\b", re.IGNORECASE)

FINANCIAL_METRIC_TERMS = [
    "revenue", "revenues", "net income", "net profit", "net profits", "operating profit", "operating profits",
    "gross profit", "gross profits", "ebitda", "ebit", "total assets", "total liabilities", "cash flow",
    "operating cash flow", "free cash flow", "equity", "dividend", "dividends",
    "earnings per share", "earning per share", "eps", "gross margin", "operating margin",
    "cost of goods sold", "cogs", "capital expenditure", "capex",
]

REGION_TERMS = [
    "asia", "europe", "north america", "south america", "africa", "oceania",
    "middle east", "latin america", "apac", "emea", "americas",
]


class ExtractedEntities:
    def __init__(self):
        self.companies: list[str] = []
        self.metrics: list[str] = []
        self.years: list[str] = []
        self.regions: list[str] = []
        self.currencies: list[str] = []

    def to_dict(self) -> dict:
        return {
            "companies": self.companies,
            "metrics": self.metrics,
            "years": self.years,
            "regions": self.regions,
            "currencies": self.currencies,
        }

    def is_empty(self) -> bool:
        return not any(
            [self.companies, self.metrics, self.years, self.regions, self.currencies]
        )


class NERService:
    def __init__(self, settings: Settings):
        try:
            self.nlp = spacy.load(settings.spacy_model_name)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{settings.spacy_model_name}' not found. "
                f"Run: python -m spacy download {settings.spacy_model_name}"
            ) from e

    def extract(self, text: str) -> ExtractedEntities:
        entities = ExtractedEntities()
        
        # spaCy NER on title-cased version as well as raw text, so lowercased queries ("netsol technologies")
        # get recognized as ORG entities even if uncapitalized.
        doc = self.nlp(text)
        doc_title = self.nlp(text.title())

        for ent in list(doc.ents) + list(doc_title.ents):
            if ent.label_ == "ORG":
                entities.companies.append(ent.text)
            elif ent.label_ == "GPE" and ent.text.lower() in REGION_TERMS:
                entities.regions.append(ent.text)

        lowered = text.lower()

        # Extract 4-digit years
        entities.years = list({m.group(0) for m in YEAR_RE.finditer(text)})

        entities.currencies = list({m.group(0).upper() for m in CURRENCY_RE.finditer(text)})

        for term in FINANCIAL_METRIC_TERMS:
            if term in lowered:
                entities.metrics.append(term)

        for term in REGION_TERMS:
            if term in lowered and term not in [r.lower() for r in entities.regions]:
                entities.regions.append(term)

        entities.companies = list(dict.fromkeys(entities.companies))
        entities.regions = list(dict.fromkeys(entities.regions))
        entities.metrics = list(dict.fromkeys(entities.metrics))

        return entities