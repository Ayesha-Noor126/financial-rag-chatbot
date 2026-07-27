"""
POST /chat

Design choice: when request.stream is True, this returns a StreamingResponse
(text/event-stream) instead of the ChatResponse model -- FastAPI allows a
route to return different response types at runtime, but that means we
can't declare response_model=ChatResponse on the decorator (it would only
describe the non-streaming branch). This is a deliberate tradeoff: slightly
less complete OpenAPI docs in exchange for one endpoint serving both modes,
which matches how the frontend actually wants to call it (one form, a
toggle). The X-Session-Id response header lets the streaming client learn
the session_id without waiting for the final metadata event.
"""
"""
POST /chat
...
"""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.core.langfuse_client import get_langfuse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat.chat_service import ChatService

router = APIRouter()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    lf = get_langfuse()
    print("Langfuse client:", lf)

    # Newer Langfuse SDK exposes `trace()` / `span()` and `update()` APIs.
    root_trace = None
    if lf:
        # create a trace (previous code used start_span)
        root_trace = lf.trace(
            name="chat-turn",
            input=request.message,
            metadata={"stream": request.stream},
        )
        if request.session_id:
            # attach session id to the trace so it's grouped correctly
            root_trace.update(session_id=request.session_id)

    if request.stream:
        session_id, token_stream = await chat_service.handle_message_stream(
            request.session_id, request.message, trace=root_trace
        )
        # NOTE: for streaming we can't end/flush here -- see step 6
        return StreamingResponse(
            token_stream,
            media_type="text/event-stream",
            headers={"X-Session-Id": session_id},
        )

    response: ChatResponse = await chat_service.handle_message(
        request.session_id, request.message, trace=root_trace
    )
    if root_trace:
        # update the trace with output; no explicit `end()` on traces
        root_trace.update(output=response.answer)
        lf.flush()  # force send now instead of waiting for batch interval
    return response