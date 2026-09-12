"""Per-visitor state, and the boundary around it.

The Streamlit script got one `st.session_state` per browser tab for free. An
HTTP API has to say explicitly what belongs to a visitor: the conversation, the
open project, the last result. Everything else — the calculators, the registry,
the embedding model, the vector store — is shared and stateless.

A shared conversation has already been a real bug here once: the chat object was
cached process-wide, so every visitor read and wrote one chat (fixed in 0c776c6).
Nothing in this module is global, and the chat is built per session.

Sessions live in memory. The free-tier host runs a single worker; projects are
files the user downloads; and losing an unsaved project when the server restarts
is the same failure the Streamlit app had. `max_sessions` and `idle_timeout_s`
bound the memory a public URL can be made to consume — the oldest idle session
is evicted first.
"""

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.project import workspace as ws
from src.project.model import Project, PumpingTest

COOKIE_NAME = "hydroassist_session"

# Two hours of inactivity, and a ceiling low enough that a public URL cannot
# exhaust a 12 GB VM: each session holds a conversation and a project, not data.
IDLE_TIMEOUT_S = 2 * 60 * 60
MAX_SESSIONS = 200


@dataclass
class SessionState:
    """One visitor's workspace."""

    id: str
    chat: object                      # HydroAssistChat
    project: Project
    active_test_id: str
    # The last completed analysis, and the setup it ran with (services.run_record).
    result: object = None             # CalculationResult
    record: dict = field(default_factory=dict)
    # The last prepared dataset (services.data_intake.PreparedData).
    prepared: object = None
    saved_result: object = None       # the result already written to the project
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()

    # ── The active test ───────────────────────────────────────────────────────

    def active_test(self) -> PumpingTest:
        """The test the workspace is showing, recovering if the stored id has
        gone — which happens the moment a different project is opened."""
        try:
            return self.project.get_test(self.active_test_id)
        except KeyError:
            if not self.project.tests:
                self.project.add_test()
            self.active_test_id = self.project.tests[0].id
            return self.project.tests[0]

    def load_test(self, test: PumpingTest) -> PumpingTest:
        """Point the workspace at `test`: its readings, and a clean slate for
        results, so a stale fit is never shown against another test's data."""
        self.active_test_id = test.id
        self.result = None
        self.record = {}
        self.prepared = None
        self.saved_result = None
        self.chat.clear_calculation()
        self.chat.clear_staged_series()
        return test

    def open_project(self, project: Project) -> Project:
        """Replace the open project and show its first test."""
        if not project.tests:
            project.add_test()
        self.project = project
        self.load_test(project.tests[0])
        return project


class SessionStore:
    """Sessions by id, with eviction. Safe to share between threads.

    `chat_factory` is injected rather than imported so tests can supply a stub
    instead of building a language-model client and opening the vector store.
    """

    def __init__(self, chat_factory: Callable[[], object],
                 idle_timeout_s: float = IDLE_TIMEOUT_S,
                 max_sessions: int = MAX_SESSIONS):
        self._chat_factory = chat_factory
        self._idle_timeout_s = idle_timeout_s
        self._max_sessions = max_sessions
        self._sessions: dict = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def create(self) -> SessionState:
        project = ws.new_project()
        session = SessionState(
            id=secrets.token_urlsafe(24),
            chat=self._chat_factory(),
            project=project,
            active_test_id=project.tests[0].id,
        )
        with self._lock:
            self._evict_locked()
            self._sessions[session.id] = session
        return session

    def get(self, session_id: Optional[str]) -> Optional[SessionState]:
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_seen > self._idle_timeout_s:
                del self._sessions[session_id]
                return None
            session.touch()
            return session

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        return self.get(session_id) or self.create()

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _evict_locked(self) -> None:
        """Drop idle sessions, then the least recently seen if still at the cap.

        Called while holding the lock, from `create` — the only point at which
        the store can grow.
        """
        cutoff = time.time() - self._idle_timeout_s
        for session_id in [i for i, s in self._sessions.items()
                           if s.last_seen < cutoff]:
            del self._sessions[session_id]

        while len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
            del self._sessions[oldest.id]
