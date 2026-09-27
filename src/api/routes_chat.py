"""The consultation.

The conversation belongs to the session and to nobody else. Each route returns
the whole chat state — messages, context, the suggestion card, the retrieved
sources — because that is what the panel redraws after any of them.
"""

import json
import queue
import threading

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

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


@router.post("/messages/stream")
def stream_message(payload: schemas.ChatRequest, response: Response,
                   session: SessionState = Depends(current_session)):
    """The same turn as POST /messages, with the answer sent as it is written.

    An answer takes several seconds, and the first sentence of an interpretation
    usually says whether the fit can be trusted. There is no reason to withhold it
    until the last sentence exists.

    Server-sent events, three kinds:

        {"type": "token", "text": "…"}    a piece of the answer
        {"type": "state", ...}            the whole chat state, once, at the end
        {"type": "error", "detail": "…"}  the turn failed; replace the partial text

    The turn runs in a worker thread and pushes tokens into a queue that this
    response drains — the alternative, generating inside the request, cannot yield
    anything until the model has finished, which is the problem being solved.

    What is *stored* is identical to the non-streaming route, so a reload shows
    the same conversation either way, and the browser can fall back to
    POST /messages whenever streaming is unavailable.
    """
    decision = get_limiter().check(session.id)
    if not decision.allowed:
        raise HTTPException(status_code=429, detail=decision.detail,
                            headers={"Retry-After": str(decision.retry_after_s)})

    tokens: "queue.Queue" = queue.Queue(maxsize=2048)
    DONE = object()
    failure: dict = {}

    def run_turn():
        try:
            session.chat.send_message(payload.message, token_sink=tokens.put)
        except Exception as exc:                            # noqa: BLE001
            # The model is a free-tier service; out of quota or offline is a
            # normal condition the reader should see explained.
            failure["detail"] = f"The consultant is unavailable: {exc}"
        finally:
            tokens.put(DONE)

    def events():
        worker = threading.Thread(target=run_turn, daemon=True)
        worker.start()
        while True:
            piece = tokens.get()
            if piece is DONE:
                break
            yield _event({"type": "token", "text": piece})
        worker.join(timeout=5)

        if failure:
            yield _event({"type": "error", "detail": failure["detail"]})
            return
        # One authoritative state at the end: the messages, this answer's own
        # sources, the suggestion card and any offer the guide attached.
        state = presenters.chat_state(session)
        yield _event({"type": "state",
                      "state": json.loads(state.model_dump_json())})

    stream = StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Proxies that buffer would defeat the whole point of this route.
            "X-Accel-Buffering": "no",
        },
    )

    # `current_session` sets the session cookie on the injected response, and
    # FastAPI merges those headers only when a route returns a model. A route that
    # returns its own Response must carry the cookie across itself — without this,
    # a first visit that starts by streaming would be handed a session it never
    # keeps, and every message would create another one.
    for name, value in response.raw_headers:
        if name.decode("latin-1").lower() == "set-cookie":
            stream.raw_headers.append((name, value))
    return stream


def _event(payload: dict) -> str:
    """One server-sent event. The blank line is what ends it."""
    return f"data: {json.dumps(payload)}\n\n"
