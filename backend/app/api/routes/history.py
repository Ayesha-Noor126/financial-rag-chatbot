"""
GET /history/{session_id}
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chat_history_store
from app.models.schemas import HistoryResponse
from app.services.chat.session_store import ChatHistoryStore

router = APIRouter()


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(
    session_id: str,
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
) -> HistoryResponse:
    messages = history_store.get_history(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found or has no messages")
    return HistoryResponse(session_id=session_id, messages=messages)