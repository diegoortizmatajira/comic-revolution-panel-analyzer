"""Cooperative time limit for CPU-bound work running in a worker thread.

A thread cannot be killed or cancelled from outside: when a client gives up (or the request
simply takes too long) the work keeps burning CPU until it finishes by itself, and a few such
requests slow every later one down. The slow code therefore calls ``check_deadline()`` at its
loop boundaries and the caller bounds the work with ``time_limit()``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from app.core.errors import AnalysisTimeoutError

if TYPE_CHECKING:
    from collections.abc import Iterator

_deadline: ContextVar[float | None] = ContextVar("analysis_deadline", default=None)


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Make ``check_deadline()`` raise ``AnalysisTimeoutError`` once ``seconds`` have passed."""
    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def check_deadline() -> None:
    """Raise ``AnalysisTimeoutError`` if the surrounding ``time_limit`` ran out (else no-op)."""
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() > deadline:
        msg = "Analysis took too long and was stopped."
        raise AnalysisTimeoutError(msg)
