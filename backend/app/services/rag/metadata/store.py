"""
Metadata store (SQLite via SQLModel).

Design choice: FAISS and BM25 only know "chunk_id -> vector/tokens". Every
other fact we need to cite -- page number, section, filename, chunk text
itself, document status -- lives here, keyed by chunk_id/document_id. SQLite
is enough for this workload (single-writer ingestion, read-heavy queries)
and needs zero infra; the interface is narrow enough (a handful of methods)
that swapping to Postgres later for concurrent multi-user writes is a
drop-in change, not a rewrite.

We store the full chunk text here (not just in FAISS/BM25) because FAISS
returns only vector IDs and BM25 returns only token-matched IDs -- neither
carries the original text back. This table is the single source of truth
the LLM prompt is ultimately built from.
"""

from datetime import datetime

from sqlmodel import Field, Session, SQLModel, select

from app.core.config import Settings
from app.core.db import get_engine
from app.models.schemas import Chunk, ChunkType, DocumentMetadata


class DocumentRow(SQLModel, table=True):
    __tablename__ = "documents"

    document_id: str = Field(primary_key=True)
    filename: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    num_pages: int = 0
    num_chunks: int = 0
    status: str = "processing"


class ChunkRow(SQLModel, table=True):
    __tablename__ = "chunks"

    chunk_id: str = Field(primary_key=True)
    document_id: str = Field(index=True)
    filename: str
    page_number: int
    section: str | None = None
    text: str
    token_count: int
    chunk_type: str = ChunkType.TEXT.value
    chunk_index: int


class MetadataStore:
    def __init__(self, settings: Settings):
        self.engine = get_engine()
        SQLModel.metadata.create_all(self.engine)

    def create_document(self, document_id: str, filename: str) -> None:
        with Session(self.engine) as session:
            session.add(
                DocumentRow(
                    document_id=document_id,
                    filename=filename,
                    status="processing",
                )
            )
            session.commit()

    def save_chunks(self, chunks: list[Chunk]) -> None:
        with Session(self.engine) as session:
            for c in chunks:
                session.add(
                    ChunkRow(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        filename=c.filename,
                        page_number=c.page_number,
                        section=c.section,
                        text=c.text,
                        token_count=c.token_count,
                        chunk_type=c.chunk_type.value,
                        chunk_index=c.chunk_index,
                    )
                )
            session.commit()

    def mark_ready(self, document_id: str, num_pages: int, num_chunks: int) -> None:
        with Session(self.engine) as session:
            doc = session.get(DocumentRow, document_id)
            if doc:
                doc.num_pages = num_pages
                doc.num_chunks = num_chunks
                doc.status = "ready"
                session.add(doc)
                session.commit()

    def mark_failed(self, document_id: str) -> None:
        with Session(self.engine) as session:
            doc = session.get(DocumentRow, document_id)
            if doc:
                doc.status = "failed"
                session.add(doc)
                session.commit()

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkRow]:
        if not chunk_ids:
            return []

        with Session(self.engine) as session:
            statement = select(ChunkRow).where(
                ChunkRow.chunk_id.in_(chunk_ids)
            )
            return list(session.exec(statement))

    def get_chunk_ids_for_document(self, document_id: str) -> list[str]:
        with Session(self.engine) as session:
            statement = select(ChunkRow.chunk_id).where(
                ChunkRow.document_id == document_id
            )
            return list(session.exec(statement))

    def delete_document(self, document_id: str) -> None:
        with Session(self.engine) as session:
            chunks = session.exec(
                select(ChunkRow).where(
                    ChunkRow.document_id == document_id
                )
            ).all()

            for chunk in chunks:
                session.delete(chunk)

            doc = session.get(DocumentRow, document_id)
            if doc:
                session.delete(doc)

            session.commit()

    def list_documents(self) -> list[DocumentRow]:
        with Session(self.engine) as session:
            return list(session.exec(select(DocumentRow)))