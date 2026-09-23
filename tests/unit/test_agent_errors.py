"""What the consultant says when the model call fails.

Every failure used to produce "I encountered an error generating a response.
Please try rephrasing your question." That is false in the most common case:
when the daily quota is gone the question was fine, rephrasing cannot help, and
the next attempt fails identically. An evaluation run of 17 questions exhausted
a free key at question nine, so this is the ordinary path, not an edge case.
"""

import pytest

from src.agents import errors

QUOTA_ERROR = (
    "Error calling model 'gemini-2.5-flash' (RESOURCE_EXHAUSTED): 429 "
    "RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota... Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20, model: gemini-2.5-flash. Please retry in 29.84927339s.'}}"
)


class TestRecognisingWhatWentWrong:

    def test_a_quota_failure_is_named_as_one(self):
        assert errors.is_quota_error(Exception(QUOTA_ERROR)) is True

    @pytest.mark.parametrize("text", [
        "RESOURCE_EXHAUSTED", "429 Too Many Requests",
        "You exceeded your current quota", "rate limit reached",
    ])
    def test_its_several_forms(self, text):
        assert errors.is_quota_error(Exception(text)) is True

    def test_an_unrelated_failure_is_not(self):
        assert errors.is_quota_error(Exception("connection reset by peer")) is False

    def test_the_wait_google_asks_for_is_extracted(self):
        assert errors.retry_delay_seconds(Exception(QUOTA_ERROR)) == pytest.approx(29.85, abs=0.01)

    def test_and_is_absent_when_not_offered(self):
        assert errors.retry_delay_seconds(Exception("boom")) is None


class TestWhatTheUserIsTold:

    def test_a_quota_failure_says_so_and_does_not_blame_the_question(self):
        message = errors.explain_model_failure(Exception(QUOTA_ERROR))

        assert "daily allowance" in message
        assert "rephras" not in message.lower()

    def test_it_says_the_calculators_still_work(self):
        """The part that matters to someone mid-analysis: this failure costs
        them the consultant, not their fit, their plots or their report."""
        for error in (QUOTA_ERROR, "DEADLINE_EXCEEDED", "API key not valid", "boom"):
            assert "calculators are unaffected" in errors.explain_model_failure(
                Exception(error))

    @pytest.mark.parametrize("error,expected", [
        (QUOTA_ERROR, "daily allowance"),
        ("504 DEADLINE_EXCEEDED after 60s", "took too long"),
        ("API key not valid. Please pass a valid API key.", "API key"),
        ("something else entirely", "could not answer"),
    ])
    def test_each_cause_gets_its_own_explanation(self, error, expected):
        assert expected in errors.explain_model_failure(Exception(error))

    def test_the_raw_error_never_reaches_the_browser(self):
        """It carries the endpoint, the project's quota metrics and sometimes
        key identifiers. Those belong in the log."""
        message = errors.explain_model_failure(Exception(QUOTA_ERROR))

        for leak in ("generativelanguage.googleapis.com", "RESOURCE_EXHAUSTED",
                     "429", "gemini-2.5-flash"):
            assert leak not in message

    def test_an_authentication_failure_says_it_needs_fixing_elsewhere(self):
        """Nothing the visitor does in the browser can fix a bad key, so the
        message must not invite them to try."""
        message = errors.explain_model_failure(Exception("PERMISSION_DENIED"))

        assert "key" in message.lower()
        assert "nothing you can do in the browser" in message.lower()
