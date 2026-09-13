"""The workspace must survive the consultant being unavailable.

Found by running the real server: `/api/session` returned 500 because the
language-model client could not be built, which took the calculators down with
it. Fitting a pumping test does not need a language model, and on a free-tier
host the model's quota can run out at any time, so a session is created
regardless and the chat explains itself when asked.
"""

import inspect

import pytest
from fastapi.testclient import TestClient

from conftest import TestStubChat, run, upload

from src.api.app import create_app
from src.api.deps import set_store
from src.api.sessions import SessionStore, UnavailableChat


def _no_key():
    raise ValueError("API key not found in environment variable: GEMINI_API_KEY")


@pytest.fixture
def degraded():
    app = create_app(SessionStore(chat_factory=_no_key))
    with TestClient(app) as client:
        yield client
    set_store(None)


class TestWithoutTheConsultant:

    def test_a_session_is_still_created(self, degraded):
        response = degraded.get("/api/session")

        assert response.status_code == 200
        assert response.json()["project"]["tests"]

    def test_the_calculators_still_work(self, degraded):
        upload(degraded)
        body = run(degraded).json()

        assert body["success"] is True
        assert body["T_day"] == pytest.approx(215.29, rel=1e-3)

    def test_the_report_and_comparison_still_work(self, degraded):
        upload(degraded)
        run(degraded)

        assert degraded.post("/api/analyses/save").status_code == 200
        assert degraded.get("/api/report.pdf").content.startswith(b"%PDF-")

    def test_asking_a_question_explains_why_there_is_no_answer(self, degraded):
        response = degraded.post("/api/chat/messages", json={"message": "hello"})

        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "not available on this server" in detail
        assert "calculators still work" in detail

    def test_the_method_can_still_be_confirmed(self, degraded):
        body = degraded.post("/api/chat/confirm",
                             json={"method": "Theis (1935)"}).json()
        assert body["context"]["suggestion_confirmed"] is True


class TestTheFallbackKeepsTheBridge:

    @pytest.mark.parametrize("name", [
        "send_message", "reset", "get_context", "confirm_method",
        "record_calculation", "stage_series", "clear_staged_series",
        "clear_calculation",
    ])
    def test_it_has_the_same_shape_as_the_real_chat(self, name):
        real = inspect.signature(getattr(TestStubChat, name))
        fallback = inspect.signature(getattr(UnavailableChat, name))
        assert list(real.parameters) == list(fallback.parameters)

    def test_a_working_chat_is_used_when_it_can_be_built(self):
        store = SessionStore(chat_factory=TestStubChat)
        assert isinstance(store.create().chat, TestStubChat)
