"""
GET /debug/answer — development test endpoint chaining the query pipeline
and answer service (Phase 3, updated for the Phase 4 async pipeline).
Delete once you're satisfied /chat covers this.
"""

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_answer_service, get_query_pipeline_service
from app.models.schemas import ChatResponse
from app.services.rag.generation.answer_service import AnswerService
from app.services.rag.query_pipeline_service import QueryPipelineService

router = APIRouter()


@router.get("/debug/answer", response_model=ChatResponse)
async def debug_answer(
    q: str = Query(..., description="Raw user question"),
    pipeline: QueryPipelineService = Depends(get_query_pipeline_service),
    answer_service: AnswerService = Depends(get_answer_service),
) -> ChatResponse:
    pipeline_result = await pipeline.run(q)
    return await answer_service.generate(pipeline_result)