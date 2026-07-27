"""
FAISS vector index (HNSW).

Design choice: IndexHNSWFlat over IndexFlatL2 or IVF. For a single/few-
document financial chatbot (thousands, not millions, of chunks), HNSW gives
near-exact recall with sub-millisecond query time and, critically, doesn't
require a training step (unlike IVF, which needs enough vectors to train
cluster centroids -- awkward when a user uploads just one 40-page PDF).
Flat would be exact but O(n) per query; at this corpus size the difference
is small today but HNSW costs nothing extra and scales if multi-document
support (Phase 6) grows the corpus.

We store chunk_id alongside each vector in a parallel list rather than
relying on FAISS's internal integer IDs alone, because metadata (page,
section, filename) lives in the separate SQL metadata store keyed by
chunk_id -- FAISS only ever needs to hand back "which chunk_id matched."

Index is persisted to disk (write_index/read_index) so re-ingesting on
every server restart isn't required.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.core.config import Settings

# HNSW construction parameter: higher = better recall, slower build & more
# memory. 32 is the standard default recommended by the FAISS wiki for
# text-embedding use cases up to a few hundred thousand vectors.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64  # higher = better recall at query time, still sub-ms at this scale


class FaissIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dim = settings.embedding_dim
        self.index_path = settings.faiss_dir / "index.hnsw"
        self.ids_path = settings.faiss_dir / "chunk_ids.json"

        self.index: faiss.IndexHNSWFlat
        self.chunk_ids: list[str] = []

        if self.index_path.exists() and self.ids_path.exists():
            self._load()
        else:
            self._init_empty()

    def _init_empty(self) -> None:
    # --- TEMPORARY DIAGNOSTIC SWAP — testing if HNSW itself is the hang ---
        self.index = faiss.IndexFlatIP(self.dim)
    # self.index = faiss.IndexHNSWFlat(self.dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    # self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    # self.index.hnsw.efSearch = HNSW_EF_SEARCH
        faiss.omp_set_num_threads(1)
        self.chunk_ids = []
        print("FAISS thread cap set")

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        print("FAISS: converting vectors")
        arr = np.array(vectors, dtype="float32")

        print(f"FAISS: array shape = {arr.shape}")
        print(f"FAISS: expected dimension = {self.dim}")

        print("FAISS: adding vectors")
        self.index.add(arr)

        print("FAISS: vectors added")

        self.chunk_ids.extend(chunk_ids)
        print(f"FAISS: chunk ids = {len(self.chunk_ids)}")

        print("FAISS: saving index")
        self._save()

        print("FAISS: index saved")

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, similarity_score), ...] sorted descending."""
        if self.index.ntotal == 0:
            return []
        arr = np.array([query_vector], dtype="float32")
        scores, indices = self.index.search(arr, min(top_k, self.index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def remove_document(self, chunk_ids_to_remove: set[str]) -> None:
        """
        FAISS HNSW doesn't support efficient single-vector deletion, so we
        rebuild the index excluding the removed chunk_ids. Acceptable here
        because deletions (DELETE /document) are rare relative to queries --
        rebuilding trades a slow delete for fast, simple reads, which is the
        right tradeoff for this access pattern.
        """
        keep_mask = [cid not in chunk_ids_to_remove for cid in self.chunk_ids]
        if all(keep_mask):
            return

        kept_ids = [cid for cid, keep in zip(self.chunk_ids, keep_mask) if keep]
        kept_vectors = [
            self.index.reconstruct(i) for i, keep in enumerate(keep_mask) if keep
        ]
        self._init_empty()
        if kept_vectors:
            self.add(kept_ids, [v.tolist() for v in kept_vectors])
        else:
            self._save()


    def _save(self) -> None:
        print(f"Writing FAISS index to: {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))

        print(f"Writing chunk ids to: {self.ids_path}")
        self.ids_path.write_text(json.dumps(self.chunk_ids))

        print("FAISS save complete")

   
    def _load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.chunk_ids = json.loads(self.ids_path.read_text())
