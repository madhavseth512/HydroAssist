"""The consultation over HTTP.

The conversation is the one piece of per-visitor state that is expensive to
build, so these tests drive a stub — and the first test here is the one that
keeps the stub honest.
"""

import inspect
import json

import pytest

from conftest import THEIS_SAMPLE, TestStubChat, run, upload


class TestTheStubMatchesTheRealChat:

    @pytest.mark.parametrize("name", [
        "send_message", "reset", "get_context", "confirm_method",
        "record_calculation", "stage_series", "clear_staged_series",
        "clear_calculation",
    ])
    def test_the_bridge_has_the_same_shape(self, name):
        """If HydroAssistChat's bridge changes, the stub must change with it —
        otherwise these tests certify a contract that no longer exists."""
        from src.core.graph import HydroAssistChat

        real = inspect.signature(getattr(HydroAssistChat, name))
        stub = inspect.signature(getattr(TestStubChat, name))

        assert list(real.parameters) == list(stub.parameters)


class TestConversation:

    def test_a_new_session_starts_empty(self, client):
        body = client.get("/api/chat").json()

        assert body["messages"] == []
        assert body["context"]["message_count"] == 0
        assert body["suggestion"] is None

    def test_a_message_is_answered(self, client):
        body = client.post("/api/chat/messages",
                           json={"message": "When should I use Cooper-Jacob?"}).json()

        assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
        assert "Cooper-Jacob" in body["messages"][0]["content"]
        assert body["context"]["message_count"] == 2

    def test_an_empty_message_is_rejected(self, client):
        assert client.post("/api/chat/messages", json={"message": ""}).status_code == 422

    def test_an_unavailable_model_is_explained(self, client, store):
        """Free-tier quota runs out; that is a condition to show, not a crash."""
        session_id = client.get("/api/session").json()["session_id"]
        store.get(session_id).chat.fail_with = RuntimeError("quota exhausted")

        response = client.post("/api/chat/messages", json={"message": "hello"})

        assert response.status_code == 502
        assert "quota exhausted" in response.json()["detail"]

    def test_the_transcript_can_be_exported(self, client):
        assert client.get("/api/chat/export").status_code == 404

        client.post("/api/chat/messages", json={"message": "hello"})
        response = client.get("/api/chat/export")

        assert response.status_code == 200
        assert "User: hello" in response.text
        assert "hydroassist_chat.txt" in response.headers["content-disposition"]


class TestSuggestionCard:

    def _suggest(self, client, store, **overrides):
        session_id = client.get("/api/session").json()["session_id"]
        suggestion = {"method": "Papadopulos-Cooper (1967)",
                      "reason": "Large-diameter well; wellbore storage matters.",
                      "confidence": 0.85, "supported": True}
        suggestion.update(overrides)
        store.get(session_id).chat.state["method_suggestion"] = suggestion
        return suggestion

    def test_a_suggestion_is_offered_with_its_reason(self, client, store):
        self._suggest(client, store)
        body = client.get("/api/chat").json()

        assert body["suggestion"]["method"] == "Papadopulos-Cooper (1967)"
        assert body["suggestion"]["confidence"] == 0.85
        assert "wellbore storage" in body["suggestion"]["reason"]

    def test_an_unsupported_aquifer_is_marked_as_such(self, client, store):
        self._suggest(client, store, method="", supported=False,
                      reason="No calculator for unconfined aquifers yet.")
        body = client.get("/api/chat").json()

        assert body["suggestion"]["supported"] is False
        assert body["suggestion"]["method"] == ""

    def test_confirming_settles_the_method_and_retires_the_card(self, client, store):
        self._suggest(client, store)

        body = client.post("/api/chat/confirm",
                           json={"method": "Cooper-Jacob (1946)"}).json()

        assert body["suggestion"] is None
        assert body["context"]["suggestion_confirmed"] is True
        assert body["context"]["method"] == "Cooper-Jacob (1946)"

    def test_the_confirmed_method_drives_the_form_default(self, client, store):
        client.post("/api/chat/confirm", json={"method": "Cooper-Jacob (1946)"})

        body = client.get("/api/methods", params={"test_type": "constant_rate"}).json()
        offered = [m["label"] for m in body["offered"]]

        assert offered[body["default_index"]] == "Cooper-Jacob (1946)"


class TestResultsReachTheConversation:

    def test_a_completed_analysis_is_published(self, client):
        upload(client)
        run(client)

        context = client.get("/api/chat").json()["context"]

        assert context["has_result"] is True
        assert context["method"] == "Theis (1935)"
        assert context["suggestion_confirmed"] is True

    def test_new_readings_retire_the_previous_result(self, client):
        """Otherwise the consultant goes on discussing numbers from the old
        dataset."""
        upload(client)
        run(client)
        upload(client)

        assert client.get("/api/chat").json()["context"]["has_result"] is False
        assert client.get("/api/analyses/current").status_code == 404

    def test_resetting_clears_the_conversation_and_the_result(self, client):
        upload(client)
        run(client)
        client.post("/api/chat/messages", json={"message": "hello"})

        body = client.post("/api/chat/reset").json()

        assert body["messages"] == []
        assert body["context"]["has_result"] is False
        assert client.get("/api/analyses/current").status_code == 404

    def test_the_project_survives_a_reset(self, client):
        upload(client)
        client.patch("/api/project/info", json={"title": "Kept"})

        client.post("/api/chat/reset")

        project = client.get("/api/project").json()
        assert project["info"]["title"] == "Kept"
        assert project["tests"][0]["n_readings"] == 50


class TestStreamingAnAnswer:
    """The answer sent as it is written.

    An interpretation takes several seconds and its first sentence usually says
    whether the fit can be trusted, so it is sent in pieces. What matters in these
    tests is that streaming changes *when* the reader sees the answer and nothing
    about what is stored: a reload must show the same conversation either way.
    """

    def _events(self, client, message="When should I use Cooper-Jacob?"):
        with client.stream("POST", "/api/chat/messages/stream",
                           json={"message": message}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))
            return events

    def test_the_answer_arrives_in_pieces(self, client):
        events = self._events(client)
        tokens = [event for event in events if event["type"] == "token"]

        assert len(tokens) > 1, "a single piece is not streaming"
        assert "".join(token["text"] for token in tokens).strip() \
            == "Stub reply to: When should I use Cooper-Jacob?"

    def test_the_whole_state_comes_last(self, client):
        events = self._events(client)

        assert events[-1]["type"] == "state"
        state = events[-1]["state"]
        assert [m["role"] for m in state["messages"]] == ["user", "assistant"]
        assert state["context"]["message_count"] == 2

    def test_what_is_stored_is_what_the_plain_route_stores(self, client,
                                                           second_client):
        """The guarantee that lets the browser fall back to POST /messages."""
        streamed = self._events(client)[-1]["state"]
        plain = second_client.post(
            "/api/chat/messages",
            json={"message": "When should I use Cooper-Jacob?"}).json()

        assert [m["content"] for m in streamed["messages"]] \
            == [m["content"] for m in plain["messages"]]

    def test_the_conversation_holds_the_answer_afterwards(self, client):
        self._events(client)

        body = client.get("/api/chat").json()
        assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
        assert "Stub reply" in body["messages"][1]["content"]

    def test_a_failure_is_reported_as_an_event(self, client, monkeypatch):
        """Tokens already sent cannot be recalled, so the browser is told to
        replace what it has rather than leave half an answer looking finished."""
        monkeypatch.setattr(TestStubChat, "fail_with",
                            RuntimeError("the daily allowance is gone"))

        events = self._events(client)

        assert events[-1]["type"] == "error"
        assert "daily allowance" in events[-1]["detail"]

    def test_it_is_rate_limited_like_the_plain_route(self, client, store):
        from src.api.deps import set_limiter
        from src.api.ratelimit import ChatRateLimiter

        set_limiter(ChatRateLimiter(per_session=0, per_day=0))
        try:
            response = client.post("/api/chat/messages/stream",
                                   json={"message": "hello"})
            assert response.status_code == 429
        finally:
            set_limiter(None)

    def test_an_empty_message_is_refused_by_the_schema(self, client):
        response = client.post("/api/chat/messages/stream", json={"message": ""})

        assert response.status_code == 422

    def test_a_first_visit_that_starts_here_keeps_its_session(self, client, store):
        """FastAPI merges the session cookie into a response it serialises, not
        into one a route returns itself. Without carrying it across, a visitor
        whose first request was a streamed message would be given a session they
        never kept — and every message after it another one."""
        before = len(store._sessions)

        self._events(client)
        self._events(client, "and Theis?")

        assert len(store._sessions) == before + 1
        assert len(client.get("/api/chat").json()["messages"]) == 4
