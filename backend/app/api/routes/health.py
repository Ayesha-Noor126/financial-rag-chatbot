from fastapi import APIRouter, Depends

from app.api.dependencies import get_bm25_index, get_faiss_index, get_metadata_store
from app.services.rag.metadata.store import MetadataStore
from app.services.rag.retriever.bm25_index import BM25Index
from app.services.rag.retriever.faiss_index import FaissIndex

router = APIRouter()


@router.get("/health")
async def health_check(
    faiss_index: FaissIndex = Depends(get_faiss_index),
    bm25_index: BM25Index = Depends(get_bm25_index),
    metadata_store: MetadataStore = Depends(get_metadata_store),
) -> dict:
    return {
        "status": "ok",
        "faiss_vectors": faiss_index.index.ntotal,
        "bm25_documents": len(bm25_index.chunk_ids),
        "documents_indexed": len(metadata_store.list_documents()),
    }
