"""Turning a model failure into something true for the person reading it.

The consultant used to answer every failure with "I encountered an error
generating a response. Please try rephrasing your question." That is actively
misleading when the cause is the daily quota: the question was fine, rephrasing
it cannot help, and the next attempt will fail the same way.

The free tier is small enough that this is the common case, not the rare one:
gemini-2.5-flash allows 20 requests a day on a free key, which an evaluation
run of 17 questions exhausts by question nine.
"""

from typing import Optional

# What Google returns when a quota is gone, in the forms it appears in.
_QUOTA_SIGNS = ("RESOURCE_EXHAUSTED", "429", "exceeded your current quota",
                "rate limit", "quota")
_TIMEOUT_SIGNS = ("DEADLINE_EXCEEDED", "timeout", "timed out", "504")
_AUTH_SIGNS = ("API key not valid", "PERMISSION_DENIED", "UNAUTHENTICATED", "401",
               "403")

QUOTA_MESSAGE = (
    "The consultant has used up its daily allowance with Google. The free tier "
    "permits a limited number of questions a day, and they have been spent.\n\n"
    "**The calculators are unaffected** — fitting, plots, the comparison table "
    "and PDF reports all work normally, because none of them uses the language "
    "model.\n\n"
    "The allowance resets daily."
)

TIMEOUT_MESSAGE = (
    "The consultant took too long to answer and the request was abandoned. "
    "This is usually temporary; asking again often works. The calculators are "
    "unaffected."
)

AUTH_MESSAGE = (
    "The consultant cannot reach the language model: its API key is missing, "
    "expired or rejected. This needs the key to be corrected where the "
    "application is configured; nothing you can do in the browser will fix it. "
    "The calculators are unaffected."
)

GENERIC_MESSAGE = (
    "The consultant could not answer that one. If it happens again, the "
    "application log has the detail. The calculators are unaffected."
)


def is_quota_error(error) -> bool:
    text = str(error)
    return any(sign.lower() in text.lower() for sign in _QUOTA_SIGNS)


def explain_model_failure(error) -> str:
    """What to show the user when the model call failed.

    Never includes the raw error: it carries the endpoint, the project's quota
    metrics and sometimes the key's identifiers, none of which belong in a
    browser. The detail stays in the log.
    """
    text = str(error)

    if is_quota_error(error):
        return QUOTA_MESSAGE
    if any(sign.lower() in text.lower() for sign in _TIMEOUT_SIGNS):
        return TIMEOUT_MESSAGE
    if any(sign.lower() in text.lower() for sign in _AUTH_SIGNS):
        return AUTH_MESSAGE
    return GENERIC_MESSAGE


def retry_delay_seconds(error) -> Optional[float]:
    """The wait Google asks for, when it says. Logged, not shown."""
    import re

    match = re.search(r"[Rr]etry in (\d+(?:\.\d+)?)s", str(error))
    return float(match.group(1)) if match else None
