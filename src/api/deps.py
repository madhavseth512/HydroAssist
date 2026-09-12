"""Wiring shared by the routes: the session store, the cookie, the LLM client.

The store and the model client are built once per process and injected, so a
test can supply a stub chat instead of opening the vector store and a
language-model connection for every request.
"""

import os
from functools import lru_cache
from typing import Optional

from fastapi import Cookie, Response

from src.api.sessions import COOKIE_NAME, SessionState, SessionStore

# Streamlit's 50 MB upload cap, carried over: a bound on what a public URL can
# be made to parse.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_store: Optional[SessionStore] = None


def build_chat():
    """A conversation, one per session. Never shared — see sessions.py."""
    from src.core.config import load_config
    from src.core.graph import HydroAssistChat

    return HydroAssistChat(load_config())


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore(chat_factory=build_chat)
    return _store


def set_store(store: Optional[SessionStore]) -> None:
    """Replace the process-wide store. For tests and for app construction."""
    global _store
    _store = store


@lru_cache(maxsize=1)
def inspection_llm():
    """The Gemini client for CSVInspector's layer-3 column-mapping fallback.

    None when it cannot be built (no API key, offline); the inspector then
    degrades to regex and statistical detection rather than failing the upload.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        from src.core.config import load_config

        cfg = load_config()
        return ChatGoogleGenerativeAI(
            model=cfg.llm.model,
            temperature=0.0,          # deterministic column mapping
            max_tokens=1024,
            google_api_key=cfg.llm.api_key,
        )
    except Exception:
        return None


def _cookie_is_secure() -> bool:
    """Set HYDROASSIST_COOKIE_SECURE=1 behind HTTPS, which is how it is served
    in production. Off by default so local http development works."""
    return os.getenv("HYDROASSIST_COOKIE_SECURE", "").lower() in ("1", "true", "yes")


def current_session(
    response: Response,
    hydroassist_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> SessionState:
    """The caller's workspace, created on first contact.

    The cookie is HttpOnly so page scripts cannot read it, and SameSite=Lax so
    another site cannot drive this API with it.
    """
    store = get_store()
    session = store.get(hydroassist_session)
    if session is None:
        session = store.create()
    response.set_cookie(
        COOKIE_NAME, session.id,
        httponly=True, samesite="lax", secure=_cookie_is_secure(),
        max_age=int(store._idle_timeout_s),
    )
    return session
