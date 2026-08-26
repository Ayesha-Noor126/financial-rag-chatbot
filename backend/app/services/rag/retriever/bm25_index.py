"""
BM25 keyword index.

Design choice: this is what saves us when a user asks "What was the FY2023
Nestlé Waters segment revenue?" -- dense embeddings alone are notoriously
weak at exact-match retrieval for entity names, years, and specific figures
because they encode semantic similarity, not lexical precision. BM25 catches
the exact tokens ("2023", "Nestlé Waters") that a paraphrase-trained
embedding model can blur together with similar-sounding segments. This is
combined with FAISS results via Reciprocal Rank Fusion in the retriever
(Phase 2), not used standalone.

rank_bm25 (pure Python) is intentionally simple/in-memory rather than
Elasticsearch -- at the corpus size this app targets (single/few uploaded
reports), spinning up a search cluster is unjustified operational overhead.
If corpus size grows into the hundreds of thousands of chunks, this is the
component to swap for OpenSearch/Elasticsearch BM25.
"""

import json
import pickle
import re

from rank_bm25 import BM25Okapi

from app.core.config import Settings

TOKEN_RE = re.compile(r"[a-zA-Z0-9%$€£₹]+")


def tokenize(text: str) -> list[str]:
    # Lowercased alphanumeric + currency symbols kept intact, since "$", "%"
    # and currency codes are semantically load-bearing in financial text
    # ("$2.4bn" vs "2.4bn" vs "€2.4bn" are different facts).
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store_path = settings.faiss_dir / "bm25_store.pkl"
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

        if self.store_path.exists():
            self._load()

    def add(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids.extend(chunk_ids)
        self.corpus_tokens.extend(tokenize(t) for t in texts)
        self._rebuild()
        self._save()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def remove_document(self, chunk_ids_to_remove: set[str]) -> None:
        keep = [
            (cid, toks)
            for cid, toks in zip(self.chunk_ids, self.corpus_tokens)
            if cid not in chunk_ids_to_remove
        ]
        self.chunk_ids = [c for c, _ in keep]
        self.corpus_tokens = [t for _, t in keep]
        self._rebuild()
        self._save()

    def _rebuild(self) -> None:
        # BM25Okapi has no incremental-update API, so we rebuild the index
        # in memory on every add. Cheap in pure Python up to tens of
        # thousands of chunks; if this becomes a bottleneck, batch uploads
        # before rebuilding instead of rebuilding per-document.
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def _save(self) -> None:
        with open(self.store_path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "corpus_tokens": self.corpus_tokens}, f)

    def _load(self) -> None:
        with open(self.store_path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.corpus_tokens = data["corpus_tokens"]
        self._rebuild()
