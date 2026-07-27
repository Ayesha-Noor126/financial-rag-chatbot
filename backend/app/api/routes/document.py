"""
DELETE /document/{document_id}

Design choice on ordering: index removal (FAISS, BM25) happens BEFORE the
metadata row is deleted. If something fails partway through, a leftover
metadata row is a recoverable inconsistency (the delete can just be
retried) -- whereas deleting metadata first and then failing to clean up
the vector indexes would leave orphaned vectors with no document_id left
to find them by, which is much harder to clean up later.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_bm25_index, get_faiss_index, get_metadata_store
from app.services.rag.metadata.store import MetadataStore
from app.services.rag.retriever.bm25_index import BM25Index
from app.services.rag.retriever.faiss_index import FaissIndex

router = APIRouter()


@router.delete("/document/{document_id}")
async def delete_document(
    document_id: str,
    metadata_store: MetadataStore = Depends(get_metadata_store),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    bm25_index: BM25Index = Depends(get_bm25_index),
) -> dict:
    chunk_ids = metadata_store.get_chunk_ids_for_document(document_id)
    if not chunk_ids:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_id_set = set(chunk_ids)
    faiss_index.remove_document(chunk_id_set)
    bm25_index.remove_document(chunk_id_set)
    metadata_store.delete_document(document_id)

    return {"document_id": document_id, "status": "deleted", "chunks_removed": len(chunk_ids)}