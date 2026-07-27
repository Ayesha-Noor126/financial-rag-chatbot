"""
Ingestion orchestrator.

Design choice: this is the only place that knows the *order* of pipeline
stages (parse -> clean -> chunk -> embed -> index -> persist metadata).
Every stage above is independently testable in isolation; this service just
wires them together and handles document-level failure (mark_failed) so a
bad PDF doesn't leave the indexes in a half-written state for that document.

Dependency injection: all five collaborators (parser, cleaner, chunker,
embedder, faiss_index, bm25_index, metadata_store) are constructor-injected
rather than instantiated inline. This is what makes it possible to unit-test
ingest_document() with fakes/mocks instead of spinning up a real embedding
model and FAISS index in every test.
"""

import uuid

from loguru import logger

from app.models.schemas import UploadResponse
from app.services.parser.cleaner import TextCleaner
from app.services.parser.pdf_parser import PDFParser
from app.services.rag.chunker import SemanticChunker
from app.services.rag.embeddings.embedder import Embedder
from app.services.rag.metadata.store import MetadataStore
from app.services.rag.retriever.bm25_index import BM25Index
from app.services.rag.retriever.faiss_index import FaissIndex


class IngestionService:
    def __init__(
        self,
        parser: PDFParser,
        cleaner: TextCleaner,
        chunker: SemanticChunker,
        embedder: Embedder,
        faiss_index: FaissIndex,
        bm25_index: BM25Index,
        metadata_store: MetadataStore,
    ):
        self.parser = parser
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.metadata_store = metadata_store

    def ingest_document(self, file_path: str, filename: str) -> UploadResponse:
        document_id = str(uuid.uuid4())
        self.metadata_store.create_document(document_id, filename)

        try:
            logger.info(f"[{document_id}] Parsing {filename}")
            raw_pages = self.parser.parse(file_path)

            logger.info(f"[{document_id}] Cleaning {len(raw_pages)} pages")
            cleaned_pages = self.cleaner.clean(raw_pages)

            logger.info(f"[{document_id}] Chunking")
            chunks = self.chunker.chunk_document(document_id, filename, cleaned_pages)
            if not chunks:
                raise ValueError("No extractable text found in document")

            logger.info(f"[{document_id}] Embedding {len(chunks)} chunks")
            vectors = self.embedder.embed_documents([c.text for c in chunks])

            logger.info(f"[{document_id}] Indexing (FAISS + BM25)")
            chunk_ids = [c.chunk_id for c in chunks]
            self.faiss_index.add(chunk_ids, vectors)
            self.bm25_index.add(chunk_ids, [c.text for c in chunks])

            self.metadata_store.save_chunks(chunks)
            self.metadata_store.mark_ready(
                document_id, num_pages=len(raw_pages), num_chunks=len(chunks)
            )

            return UploadResponse(
                document_id=document_id,
                filename=filename,
                num_pages=len(raw_pages),
                num_chunks=len(chunks),
                status="ready",
            )

        except Exception:
            logger.exception(f"[{document_id}] Ingestion failed")
            self.metadata_store.mark_failed(document_id)
            raise
