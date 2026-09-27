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

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.project import workspace as ws
from src.project.model import Project, PumpingTest

logger = logging.getLogger(__name__)

COOKIE_NAME = "hydroassist_session"

# Two hours of inactivity, and a ceiling low enough that a public URL cannot
# exhaust a 12 GB VM: each session holds a conversation and a project, not data.
IDLE_TIMEOUT_S = 2 * 60 * 60
MAX_SESSIONS = 200


class UnavailableChat:
    """The conversation bridge when the real one could not be built.

    Building the consultant needs a model API key and the vector store. Without
    them — no key configured, the free-tier quota exhausted at start-up, the
    store missing — the calculators still work, and they must: fitting a
    pumping test does not need a language model. So the session gets this in
    place of the chat. The workspace can publish into it as usual, and asking it
    a question explains why it cannot answer.
    """

    def __init__(self, reason: str):
        self.reason = reason
        self.state = {
            "messages": [], "user_intent": "unknown", "aquifer_context": "unknown",
            "selected_method": "", "discussed_method": "",
            "retrieved_docs": [], "metadata_filters": {},
            "method_suggestion": {}, "suggestion_confirmed": False,
            "calculation_input": {}, "calculation_result": {},
            "analysis_history": [], "staged_series": {},
        }

    def send_message(self, message: str, token_sink=None) -> str:
        raise RuntimeError(self.reason)

    def reset(self) -> None:
        self.__init__(self.reason)

    def get_context(self) -> dict:
        return {
            "intent": "unknown", "aquifer": "unknown",
            "method": (self.state.get("discussed_method")
                       or self.state.get("selected_method", "")),
            "suggestion_confirmed": self.state.get("suggestion_confirmed", False),
            "method_suggestion": {},
            "has_result": bool(self.state.get("calculation_result")),
            "message_count": 0,
            "retrieved_docs_count": 0,
        }

    def confirm_method(self, method: str) -> None:
        self.state["selected_method"] = method
        self.state["suggestion_confirmed"] = True

    def record_calculation(self, result: dict, calc_input: dict) -> None:
        # The history is kept here too, so a session without a working consultant
        # still has one when the model comes back.
        self.state["calculation_result"] = result or {}
        self.state["calculation_input"] = calc_input or {}
        history = list(self.state.get("analysis_history") or [])
        history.append({"method": (result or {}).get("method", "unknown"),
                        "success": bool((result or {}).get("success")),
                        "T_day": (result or {}).get("T_day"),
                        "S": (result or {}).get("S"),
                        "r_squared": (result or {}).get("r_squared"),
                        "rmse": (result or {}).get("rmse"),
                        "n_points": (calc_input or {}).get("n_points")})
        self.state["analysis_history"] = history[-12:]

    def stage_series(self, time_s, drawdown_m, Q=None) -> None:
        self.state["staged_series"] = {"time_s": time_s, "drawdown_m": drawdown_m,
                                       "Q": Q}

    def clear_staged_series(self) -> None:
        self.state["staged_series"] = {}

    def clear_calculation(self) -> None:
        self.state["calculation_result"] = {}
        self.state["calculation_input"] = {}
        self.state["analysis_history"] = []


def _chat_record(record: dict) -> dict:
    """The slice of a run record the conversation is told about.

    The same projection as `services.analysis.chat_record`, imported lazily so
    session handling does not depend on the calculator layer.
    """
    from src.services.analysis import chat_record

    return chat_record(record or {})


def _build_chat(factory: Callable[[], object]) -> object:
    """The real conversation, or an UnavailableChat that says why not."""
    try:
        return factory()
    except Exception as exc:          # noqa: BLE001 - any failure degrades the same way
        logger.warning("Consultant unavailable for this session: %s", exc)
        return UnavailableChat(
            "The consultant is not available on this server "
            f"({exc}). The calculators still work."
        )


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
    # The last fit for each test, so opening a test again shows its own result.
    # There was one slot for the whole session and switching test emptied it, so
    # a project with two tests showed an analysis for neither.
    results: dict = field(default_factory=dict)   # test id -> (result, record)
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
        """Point the workspace at `test`: its readings, and its own last result.

        Never another test's: showing a fit from Test 1 against Test 2's readings
        would be a silent lie about which numbers produced it. That is why this
        used to clear the result outright — correct, and it meant an analysis
        vanished the moment a second test was opened, which is how two tests came
        to show an analysis for neither.
        """
        self.active_test_id = test.id
        self.prepared = None

        remembered = self.results.get(test.id) or (None, {})
        self.result, self.record = remembered
        self.record = self.record or {}
        # A result recovered this way was not necessarily saved; the save route
        # compares identity, and only an identical object counts as saved.
        self.saved_result = None

        self.chat.clear_calculation()
        self.chat.clear_staged_series()
        # The conversation follows the test. Both parts matter: its own runs, so
        # "compare the two" means the two on this test's readings, and its own
        # result, so the consultant is not told there is no analysis while the
        # workspace is showing one.
        if self.result is not None:
            self.chat.record_calculation(self.result.to_state_dict(),
                                         _chat_record(self.record))
        self._publish_runs(test)
        return test

    def remember_result(self, test_id: str, result, record: dict) -> None:
        """Keep this test's latest fit, for when the user comes back to it."""
        self.results[test_id] = (result, dict(record or {}))

    def forget_results(self, test_id: str) -> None:
        """Drop a test's remembered fit — its readings have been replaced."""
        self.results.pop(test_id, None)

    def _publish_runs(self, test: PumpingTest) -> None:
        """Give the conversation this test's run history."""
        history = [
            {"method": run.method_label, "success": run.success,
             "T_day": run.T_day, "S": run.S, "r_squared": run.r_squared,
             "rmse": run.rmse, "n_points": run.n_points}
            for run in test.current_runs()
        ]
        try:
            self.chat.state["analysis_history"] = history
        except Exception:                                   # noqa: BLE001
            # An unavailable consultant still has a state dict; anything else that
            # cannot take one must not stop the test from opening.
            pass

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
            chat=_build_chat(self._chat_factory),
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
