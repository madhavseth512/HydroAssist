"""The consultant's rate limit: generous to one visitor, a ceiling for everyone.

The model runs on a free tier with a daily quota. Without a limit one visitor
could exhaust it for all the others; with one, the service says plainly why it
has paused, and the calculators are never affected.
"""

from datetime import datetime, timezone

import pytest

from conftest import run, upload

from src.api.deps import set_limiter
from src.api.ratelimit import ChatRateLimiter


class Clock:
    """Time that moves only when the test says so."""

    def __init__(self, start: float):
        self.now = start

    def __call__(self) -> float:
        return self.now


NOON = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def clock():
    return Clock(NOON)


@pytest.fixture
def limiter(clock):
    limiter = ChatRateLimiter(per_session=3, window_s=600, per_day=5, clock=clock)
    set_limiter(limiter)
    yield limiter
    set_limiter(None)


def _ask(client):
    return client.post("/api/chat/messages", json={"message": "hello"})


class TestThePerSessionLimit:

    def test_questions_within_the_limit_are_answered(self, client, limiter):
        assert [_ask(client).status_code for _ in range(3)] == [200, 200, 200]

    def test_the_next_question_is_refused_with_a_reason(self, client, limiter):
        for _ in range(3):
            _ask(client)

        response = _ask(client)

        assert response.status_code == 429
        assert "3 questions in the last 10 minutes" in response.json()["detail"]
        assert int(response.headers["retry-after"]) > 0

    def test_another_visitor_is_not_held_back(self, client, second_client, limiter):
        for _ in range(3):
            _ask(client)

        assert _ask(second_client).status_code == 200

    def test_the_window_slides(self, client, limiter, clock):
        for _ in range(3):
            _ask(client)
        clock.now += 601

        assert _ask(client).status_code == 200

    def test_the_calculators_are_never_throttled(self, client, limiter):
        for _ in range(3):
            _ask(client)
        upload(client)

        assert run(client).json()["success"] is True


class TestTheDailyCeiling:

    def test_it_is_shared_by_every_session(self, client, second_client, limiter):
        for _ in range(3):
            assert _ask(client).status_code == 200
        for _ in range(2):
            assert _ask(second_client).status_code == 200

        response = _ask(second_client)

        assert response.status_code == 429
        assert "free tier allows today" in response.json()["detail"]
        assert limiter.used_today == 5

    def test_it_resets_at_midnight_utc(self, client, second_client, limiter, clock):
        for _ in range(3):
            _ask(client)
        for _ in range(2):
            _ask(second_client)
        assert _ask(second_client).status_code == 429

        clock.now += 12 * 3600 + 1        # past midnight UTC

        assert _ask(second_client).status_code == 200

    def test_the_retry_hint_points_at_midnight(self, limiter, clock):
        for index in range(5):
            limiter.check(f"session-{index}")

        decision = limiter.check("another")

        assert not decision.allowed
        assert decision.retry_after_s == pytest.approx(12 * 3600, abs=2)


class TestMemoryStaysBounded:

    def test_idle_sessions_are_forgotten(self, clock):
        limiter = ChatRateLimiter(per_session=5, window_s=60, per_day=10_000, clock=clock)
        for index in range(1001):
            limiter.check(f"visitor-{index}")
        clock.now += 61

        limiter.check("late-arrival")

        assert len(limiter._recent) <= 2
