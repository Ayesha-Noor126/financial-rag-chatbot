"""
Embedding generation.

Design choice: BAAI/bge-large-en-v1.5 is loaded once as a module-level
singleton (expensive to load, ~1.3GB) rather than per-request. We wrap
encode() with an LRU cache keyed on text hash -- financial documents often
have repeated boilerplate ("as reported under IFRS...") across chunks, and
users frequently re-ask similar questions, so caching embeddings measurably
cuts latency on both ingestion and query paths without any accuracy cost
(the cache key is exact text, not semantic, so there's zero risk of cache
poisoning across different content).

bge models require a query instruction prefix for asymmetric search
(different prefix for queries vs. documents) -- this is easy to forget and
silently degrades retrieval quality by ~5-10% if skipped, so it's enforced
here via separate embed_documents / embed_query methods rather than one
generic embed() the caller could misuse.
"""

import hashlib
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import Settings

# bge models expect this exact instruction prefix for the QUERY side only.
# Document/passage side is embedded with no prefix. See BAAI/bge-large-en-v1.5
# model card -- mismatching this is the single most common bge integration bug.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    _model: SentenceTransformer | None = None

    def __init__(self, settings: Settings):
        self.settings = settings
        if Embedder._model is None:
            Embedder._model = SentenceTransformer(settings.embedding_model_name)
        self.model = Embedder._model
        self._cache: dict[str, list[float]] = {}
        self._cache_size = settings.embedding_cache_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed passages (no instruction prefix) for indexing."""
        uncached_idx, uncached_texts = [], []
        results: list[list[float] | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_idx.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            vectors = self.model.encode(
                uncached_texts,
                batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=True,  # required for cosine sim via inner product
                show_progress_bar=False,
            )
            for i, vec in zip(uncached_idx, vectors):
                vec_list = vec.tolist()
                results[i] = vec_list
                self._store_cache(self._cache_key(texts[i]), vec_list)

        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with the required bge instruction prefix."""
        prefixed = BGE_QUERY_INSTRUCTION + text
        key = self._cache_key(prefixed)
        if key in self._cache:
            return self._cache[key]
        vector = self.model.encode(
            prefixed, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        self._store_cache(key, vector)
        return vector

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _store_cache(self, key: str, vector: list[float]) -> None:
        if len(self._cache) >= self._cache_size:
            # Evict oldest inserted key (simple FIFO -- swap for a proper LRU
            # structure like cachetools.LRUCache if profiling shows this
            # matters; at cache_size~512 the dict scan cost is negligible).
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = vector
