"""Keeping the consultant inside the free tier.

The chat model runs on Gemini's free tier — roughly 1,500 requests a day, and
one question can take two or three model calls. With no limit, a single visitor
or a script on a public URL could spend the whole day's quota in minutes, and
nobody would get an answer until it reset. Two limits prevent that:

  per session   a burst limit, so one visitor cannot monopolise the model
  per day       a ceiling shared by every session, set below the provider's
                quota so the service says clearly why it has stopped instead of
                surfacing the provider's error

Only the consultant is limited. The calculators cost nothing per request and are
never throttled.

The limits can be tuned per deployment through environment variables.
"""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

PER_SESSION = int(os.getenv("HYDROASSIST_CHAT_PER_SESSION", "20"))
WINDOW_S = float(os.getenv("HYDROASSIST_CHAT_WINDOW_S", str(10 * 60)))
# Well under ~1,500 requests/day, since a question may cost several model calls.
PER_DAY = int(os.getenv("HYDROASSIST_CHAT_PER_DAY", "500"))

# Past this many tracked sessions, drop the ones with nothing inside the window,
# so the limiter's memory is bounded however many visitors come and go.
_SWEEP_ABOVE = 1000


@dataclass(frozen=True)
class Decision:
    allowed: bool
    detail: str = ""
    retry_after_s: int = 0


class ChatRateLimiter:
    """A sliding window per session and a counter per UTC day. Thread-safe.

    `clock` is injectable so tests can move time instead of waiting for it.
    """

    def __init__(self, per_session: int = PER_SESSION, window_s: float = WINDOW_S,
                 per_day: int = PER_DAY, clock: Callable[[], float] = time.time):
        self._per_session = per_session
        self._window_s = window_s
        self._per_day = per_day
        self._clock = clock
        self._recent: dict = {}          # session id -> deque of timestamps
        self._day = None
        self._today = 0
        self._lock = threading.Lock()

    def check(self, session_id: str) -> Decision:
        """Record a question if it is allowed, and say why if it is not."""
        now = self._clock()
        day = datetime.fromtimestamp(now, timezone.utc).date()

        with self._lock:
            if day != self._day:
                self._day, self._today = day, 0

            if self._today >= self._per_day:
                return Decision(
                    allowed=False,
                    detail=("The consultant has answered as many questions as the "
                            "free tier allows today. It resets at midnight UTC; "
                            "the calculators are unaffected."),
                    retry_after_s=_seconds_to_utc_midnight(now),
                )

            recent = self._recent.setdefault(session_id, deque())
            while recent and now - recent[0] >= self._window_s:
                recent.popleft()

            if len(recent) >= self._per_session:
                retry = int(self._window_s - (now - recent[0])) + 1
                minutes = max(1, round(self._window_s / 60))
                return Decision(
                    allowed=False,
                    detail=(f"You have asked {self._per_session} questions in the "
                            f"last {minutes} minutes. Please wait about "
                            f"{max(1, round(retry / 60))} minute(s) — this keeps the "
                            "free-tier consultant available to everyone."),
                    retry_after_s=retry,
                )

            recent.append(now)
            self._today += 1
            if len(self._recent) > _SWEEP_ABOVE:
                self._sweep_locked(now)
            return Decision(allowed=True)

    def _sweep_locked(self, now: float) -> None:
        stale = [key for key, times in self._recent.items()
                 if not times or now - times[-1] >= self._window_s]
        for key in stale:
            del self._recent[key]

    @property
    def used_today(self) -> int:
        with self._lock:
            return self._today


def _seconds_to_utc_midnight(now: float) -> int:
    current = datetime.fromtimestamp(now, timezone.utc)
    midnight = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                     microsecond=0)
    return max(1, int((midnight - current).total_seconds()))
