"""Fixtures for the API tests.

The conversation is stubbed. Every route that matters here — upload, fit, save,
compare, report — only needs the chat to accept what the workspace publishes
into it, and building the real one means a language-model client and the vector
store on every test. `TestStubChat` implements the same bridge, and
`test_api_chat.py` asserts it still matches the real class, so the stub cannot
drift into testing a contract that no longer exists.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import set_store
from src.api.sessions import SessionStore

THEIS_SAMPLE = "tests/theis_cooper_jacob_sample.csv"
RECOVERY_SAMPLE = "tests/theis_recovery_sample.csv"
PC_SAMPLE = "tests/papadopulos_cooper_sample.csv"
TRACER_NATURAL_SAMPLE = "tests/tracer_natural_sample.csv"
TRACER_CONVERGENT_SAMPLE = "tests/tracer_convergent_sample.csv"


class _Message:
    """Stands in for a LangChain message: the presenter reads the class name."""

    def __init__(self, content):
        self.content = content


class HumanMessage(_Message):
    pass


class AIMessage(_Message):
    pass


class TestStubChat:
    """The conversation bridge, without a language model."""

    __test__ = False        # a fixture, not a test class

    fail_with = None        # set to an exception to make send_message fail

    def __init__(self):
        self.state = {
            "messages": [], "user_intent": "unknown", "aquifer_context": "unknown",
            "selected_method": "", "retrieved_docs": [], "metadata_filters": {},
            "method_suggestion": {}, "suggestion_confirmed": False,
            "calculation_input": {}, "calculation_result": {},
            "analysis_history": [], "staged_series": {},
        }

    def send_message(self, message: str, token_sink=None) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self.state["messages"].append(HumanMessage(message))
        reply = f"Stub reply to: {message}"
        # A word at a time, as the real one streams, so a streaming route can be
        # tested without a model.
        if token_sink is not None:
            for word in reply.split(" "):
                token_sink(word + " ")
        self.state["messages"].append(AIMessage(reply))
        self.state["user_intent"] = "consultation"
        return reply

    def reset(self) -> None:
        self.__init__()

    def get_context(self) -> dict:
        return {
            "intent": self.state.get("user_intent", "unknown"),
            "aquifer": self.state.get("aquifer_context", "unknown"),
            "method": self.state.get("selected_method", ""),
            "suggestion_confirmed": self.state.get("suggestion_confirmed", False),
            "method_suggestion": self.state.get("method_suggestion", {}),
            "has_result": bool(self.state.get("calculation_result")),
            "message_count": len(self.state.get("messages", [])),
            "retrieved_docs_count": len(self.state.get("retrieved_docs", [])),
        }

    def confirm_method(self, method: str) -> None:
        self.state["selected_method"] = method
        self.state["suggestion_confirmed"] = True

    def record_calculation(self, result: dict, calc_input: dict) -> None:
        self.state["calculation_result"] = result or {}
        self.state["calculation_input"] = calc_input or {}

    def stage_series(self, time_s, drawdown_m, Q=None) -> None:
        self.state["staged_series"] = {"time_s": time_s, "drawdown_m": drawdown_m,
                                       "Q": Q}

    def clear_staged_series(self) -> None:
        self.state["staged_series"] = {}

    def clear_calculation(self) -> None:
        self.state["calculation_result"] = {}
        self.state["calculation_input"] = {}


@pytest.fixture
def store():
    return SessionStore(chat_factory=TestStubChat)


@pytest.fixture
def client(store):
    """A browser. TestClient keeps cookies, so requests share one session."""
    app = create_app(store)
    with TestClient(app) as test_client:
        yield test_client
    set_store(None)


@pytest.fixture
def second_client(store):
    """A different visitor against the same server."""
    app = create_app(store)
    with TestClient(app) as test_client:
        yield test_client


def upload(client, path=THEIS_SAMPLE, test_id=None):
    """Upload a sample CSV to a test, defaulting to the active one."""
    if test_id is None:
        test_id = client.get("/api/project").json()["active_test_id"]
    with open(path, "rb") as handle:
        return client.post(f"/api/tests/{test_id}/readings",
                           files={"file": (path.split("/")[-1], handle, "text/csv")})


def setup_payload(**overrides) -> dict:
    payload = {"q_value": 500.0, "q_unit": "m³/day", "r_m": 50.0,
               "r_w_m": 0.15, "r_c_m": 0.10, "test_type": "constant_rate",
               "t_p_hours": None, "recovery_clock": "since_stop"}
    payload.update(overrides)
    return payload


def run(client, method="theis_1935", **setup_overrides):
    return client.post("/api/analyses/run",
                       json={"method": method, "setup": setup_payload(**setup_overrides)})
