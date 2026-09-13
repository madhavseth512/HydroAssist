"""The consultation.

The conversation belongs to the session and to nobody else. Each route returns
the whole chat state — messages, context, the suggestion card, the retrieved
sources — because that is what the panel redraws after any of them.
"""

from fastapi import APIRouter, Depends, HTTPException, Response

from src.api import presenters, schemas
from src.api.deps import current_session, get_limiter
from src.api.sessions import SessionState

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("", response_model=schemas.ChatStateModel)
def read_chat(session: SessionState = Depends(current_session)):
    return presenters.chat_state(session)


@router.post("/messages", response_model=schemas.ChatStateModel)
def send_message(payload: schemas.ChatRequest,
                 session: SessionState = Depends(current_session)):
    """Ask the consultant. The reply lands in the message list, so the browser
    redraws from one response.

    Rate-limited per session and per day, to stay inside the model's free tier.
    """
    decision = get_limiter().check(session.id)
    if not decision.allowed:
        raise HTTPException(status_code=429, detail=decision.detail,
                            headers={"Retry-After": str(decision.retry_after_s)})

    try:
        session.chat.send_message(payload.message)
    except Exception as exc:
        # The model is a free-tier service: out of quota or offline is a normal
        # condition the user should see explained, not a crash.
        raise HTTPException(status_code=502,
                            detail=f"The consultant is unavailable: {exc}")
    return presenters.chat_state(session)


@router.post("/confirm", response_model=schemas.ChatStateModel)
def confirm_method(payload: schemas.ConfirmMethodRequest,
                   session: SessionState = Depends(current_session)):
    """Accept the suggested method, or override it. This is the bridge: the
    workspace form follows what the conversation settles on."""
    session.chat.confirm_method(payload.method)
    return presenters.chat_state(session)


@router.post("/reset", response_model=schemas.ChatStateModel)
def reset_chat(session: SessionState = Depends(current_session)):
    """Start the conversation again. The project and its readings are kept —
    only the discussion is cleared."""
    session.chat.reset()
    session.result = None
    session.record = {}
    session.saved_result = None
    return presenters.chat_state(session)


@router.get("/export")
def export_chat(session: SessionState = Depends(current_session)):
    """The conversation as a plain-text transcript."""
    messages = presenters.chat_state(session).messages
    if not messages:
        raise HTTPException(status_code=404, detail="There is nothing to export yet.")
    transcript = "\n\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )
    return Response(
        content=transcript, media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition":
                 'attachment; filename="hydroassist_chat.txt"'},
    )
