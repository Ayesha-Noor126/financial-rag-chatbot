"""
GET /debug/query — development test endpoint for the query pipeline
(Phase 2, updated for the Phase 4 async pipeline). Delete once you're
satisfied /chat covers this.
"""

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_query_pipeline_service
from app.services.rag.query_pipeline_service import QueryPipelineService

router = APIRouter()


@router.get("/debug/query")
async def debug_query(
    q: str = Query(..., description="Raw user question"),
    pipeline: QueryPipelineService = Depends(get_query_pipeline_service),
) -> dict:
    result = await pipeline.run(q)
    return {
        "original_query": result.original_query,
        "rewritten_query": result.rewritten_query,
        "entities": result.entities.to_dict(),
        "reranked_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "relevance_score": round(score, 4),
                "text_preview": chunk.text[:200],
            }
            for chunk, score in result.reranked_chunks
        ],
    }