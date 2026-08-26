import pytest
from app.core.config import get_settings
from app.services.rag.ner.ner_service import NERService

def test_ner_currency_extraction():
    settings = get_settings()
    ner = NERService(settings)
    
    text = "The profit was ₹125,682 thousand or Rs 125,682 thousand compared to 100 USD and 50 EUR."
    entities = ner.extract(text)
    
    assert any(c in entities.currencies for c in ["₹", "RS", "USD", "EUR"])
