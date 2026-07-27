"""
Pydantic models shared across the ingestion pipeline.

Design choice: define these once and import everywhere (parser -> chunker ->
embedder -> index -> metadata store) instead of passing raw dicts. This gives
type safety across service boundaries, which matters a lot once the pipeline
has 6+ stages -- a typo in a dict key should fail at import time via mypy/IDE,
not silently at query time three services later.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"


class RawPage(BaseModel):
    """Output of PyMuPDF extraction for a single page, pre-cleaning."""
    page_number: int
    raw_text: str
    has_tables: bool = False


class CleanedPage(BaseModel):
    """A page after header/footer/page-number stripping and whitespace cleanup."""
    page_number: int
    text: str
    section: Optional[str] = None


class Chunk(BaseModel):
    """A single semantic chunk ready for embedding."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section: Optional[str] = None
    text: str
    token_count: int
    chunk_type: ChunkType = ChunkType.TEXT
    chunk_index: int  # position within the document, used for overlap/merge logic


class EmbeddedChunk(BaseModel):
    """A chunk plus its dense vector, ready for FAISS insertion."""
    chunk: Chunk
    vector: list[float]


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    num_pages: int
    num_chunks: int
    status: str = "processing"  # processing | ready | failed


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
    status: str



class Citation(BaseModel):
    """One grounded source reference, per the spec's 'always cite page/section/chunk id' rule."""
    chunk_id: str
    page_number: int
    section: Optional[str] = None
    filename: str


class WebSource(BaseModel):
    """A Tavily result, kept structurally separate from document Citations
    so the frontend can render 'From your document' vs 'From the web' as
    visually distinct sections, per the spec's separation requirement."""
    title: str
    url: str
    snippet: str





class ChatResponse(BaseModel):
    answer: str
    confidence_score: float
    citations: list[Citation] = Field(default_factory=list)
    used_web_fallback: bool = False
    web_sources: list[WebSource] = Field(default_factory=list)
    rewritten_query: Optional[str] = None
    session_id: Optional[str] = None
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # omit to start a new session
    message: str
    stream: bool = False


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class HistoryMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime
    citations: list[Citation] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]