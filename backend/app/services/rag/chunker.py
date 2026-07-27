"""
Semantic chunking.

Design choice: chunk boundaries follow paragraph/sentence structure first,
and only fall back to a hard token cut when a single paragraph exceeds the
max chunk size (common with dense financial tables). Naive fixed-size
character chunking is what causes most hallucination in financial RAG --
it slices a sentence like "net income increased by 12%" away from the
number it refers to. Token-aware, boundary-respecting chunking with 100-
token overlap keeps semantically related sentences together and gives the
reranker enough shared context between adjacent chunks to score continuity
correctly.

We use tiktoken for token counting (fast, no model download needed) even
though the embedding model isn't a tiktoken-tokenized model -- it's used
purely as a consistent length proxy, not for actual model tokenization.
"""

import re
import uuid

import tiktoken

from app.core.config import Settings
from app.models.schemas import Chunk, ChunkType, CleanedPage

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class SemanticChunker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk_document(
        self, document_id: str, filename: str, pages: list[CleanedPage]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer_text = ""
        buffer_tokens = 0
        buffer_page = pages[0].page_number if pages else 1
        buffer_section = pages[0].section if pages else None
        chunk_index = 0

        def flush(overlap_text: str = "") -> str:
            nonlocal buffer_text, buffer_tokens, chunk_index
            text = buffer_text.strip()
            if self._count_tokens(text) >= self.settings.min_chunk_tokens:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        filename=filename,
                        page_number=buffer_page,
                        section=buffer_section,
                        text=text,
                        token_count=self._count_tokens(text),
                        chunk_type=ChunkType.TEXT,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
            buffer_text = overlap_text
            buffer_tokens = self._count_tokens(overlap_text)
            return overlap_text

        for page in pages:
            for paragraph in PARAGRAPH_SPLIT_RE.split(page.text):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                para_tokens = self._count_tokens(paragraph)

                # A single paragraph bigger than the whole target chunk size
                # (dense table dumps) gets sentence-split as a fallback.
                if para_tokens > self.settings.chunk_size_tokens:
                    for sentence in SENTENCE_SPLIT_RE.split(paragraph):
                        buffer_text, buffer_tokens, buffer_page, buffer_section = (
                            self._add_unit(
                                sentence,
                                buffer_text,
                                buffer_page,
                                buffer_section,
                                page,
                                flush,
                            )
                        )
                else:
                    buffer_text, buffer_tokens, buffer_page, buffer_section = (
                        self._add_unit(
                            paragraph,
                            buffer_text,
                            buffer_page,
                            buffer_section,
                            page,
                            flush,
                        )
                    )

        if buffer_text.strip():
            flush()

        return chunks

    # -- internals ---------------------------------------------------

    def _add_unit(
        self,
        unit: str,
        buffer_text: str,
        buffer_page: int,
        buffer_section: str | None,
        page: CleanedPage,
        flush,
    ) -> tuple[str, int, int, str | None]:
        candidate = (buffer_text + "\n\n" + unit).strip() if buffer_text else unit
        candidate_tokens = self._count_tokens(candidate)

        if candidate_tokens <= self.settings.chunk_size_tokens:
            return candidate, candidate_tokens, buffer_page or page.page_number, (
                buffer_section or page.section
            )

        # Over budget: flush current buffer, carry the overlap tail forward,
        # then start the new buffer with the current unit appended.
        overlap = self._tail_overlap(buffer_text)
        flush(overlap)
        new_buffer = (overlap + "\n\n" + unit).strip() if overlap else unit
        return new_buffer, self._count_tokens(new_buffer), page.page_number, page.section

    def _tail_overlap(self, text: str) -> str:
        """Return the last ~chunk_overlap_tokens worth of text to seed the next chunk."""
        if not text:
            return ""
        tokens = self.encoder.encode(text)
        overlap_tokens = tokens[-self.settings.chunk_overlap_tokens :]
        return self.encoder.decode(overlap_tokens)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text)) if text else 0
