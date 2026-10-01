"""Generic resilience primitives: backoff, retry, circuit breaker.

Used by the LLM gateway (model calls) and by tools/executors (carrier + payment APIs).
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


class TransientError(Exception):
    """A dependency failed in a way that is expected to succeed on retry."""


def backoff_delay(attempt: int, base_s: float, max_s: float, rng: random.Random | None = None) -> float:
    """Exponential backoff with 'equal jitter': half deterministic, half random.

    attempt is 1-based (the delay *after* attempt N failed).
    """
    exp = min(max_s, base_s * (2 ** (attempt - 1)))
    r = (rng or random).random()
    return exp / 2 + r * exp / 2


def retry_call(
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    attempts: int = 3,
    base_delay_s: float = 0.25,
    max_delay_s: float = 4.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call fn, retrying listed exceptions with jittered exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as e:
            if attempt == attempts:
                raise
            if on_retry:
                on_retry(attempt, e)
            sleep(backoff_delay(attempt, base_delay_s, max_delay_s))
    raise AssertionError("unreachable")


@dataclass
class CircuitBreaker:
    """Classic three-state breaker (closed -> open -> half-open), thread-safe.

    While open, callers skip the dependency entirely (fail fast -> fall back) instead of paying a
    timeout per request during an outage. After `cooldown_s` one probe call is allowed through.
    """

    name: str
    failure_threshold: int = 3
    cooldown_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = 0
    _opened_at: float | None = None
    _probe_in_flight: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_unlocked()

    def _state_unlocked(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self.clock() - self._opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        with self._lock:
            state = self._state_unlocked()
            if state == "closed":
                return True
            if state == "half_open" and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._probe_in_flight or self._failures >= self.failure_threshold:
                self._opened_at = self.clock()
            self._probe_in_flight = False
