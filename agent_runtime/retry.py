"""Retries: what a logical call is allowed to try, how long to wait, and how to wait.

Milestone 4A draws one line that the rest of the runtime leans on::

    a logical call   execution.call("fetch_data", ...)   -- one stable call_id
    an attempt       one concrete invocation of that tool -- attempt = 1, 2, 3

The call id never changes between attempts. Each attempt is journalled
separately, and the answer the state reports is the last attempt's.

Three small pieces live here, and deliberately nothing more:

* :class:`RetryPolicy` -- immutable, five numbers, two questions: should this
  failure be tried again (:meth:`~RetryPolicy.should_retry`), and how long to
  wait before the next attempt (:meth:`~RetryPolicy.delay`). No configuration
  language, no predicates, no jitter.
* :func:`classify_error` -- the only thing that makes a failure retryable is the
  tool saying so, by raising
  :class:`~agent_runtime.exceptions.RetryableToolError`. An unexpected exception
  is *not* retried by default, because "probably transient" is a guess, and this
  milestone refuses to guess on purpose.
* :class:`Sleeper` -- the single seam through which the runtime waits. Production
  passes :class:`RealSleeper`; tests pass :class:`RecordingSleeper` and assert on
  the delays it was asked for. Nothing here patches ``time.sleep``, so a test
  that verifies a backoff schedule does not spend it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable

from .exceptions import (
    PermanentToolError,
    RetryConfigurationError,
    RetryableToolError,
)

__all__ = [
    "ErrorKind",
    "RetryPolicy",
    "RetryDecision",
    "Sleeper",
    "RealSleeper",
    "RecordingSleeper",
    "classify_error",
    "NO_RETRY",
]


class ErrorKind(StrEnum):
    """How a failed attempt is classified, and therefore whether it may be retried."""

    #: The tool raised :class:`~agent_runtime.exceptions.RetryableToolError`.
    RETRYABLE = "RETRYABLE"
    #: The tool raised :class:`~agent_runtime.exceptions.PermanentToolError`.
    PERMANENT = "PERMANENT"
    #: Anything else. Retried only when a policy explicitly opts in.
    UNEXPECTED = "UNEXPECTED"


def classify_error(error: BaseException | str | None) -> ErrorKind:
    """Classify a failure for the purpose of retrying it.

    Deliberately narrow: only the two error types a tool can raise are
    classified. Everything else -- a bug in the tool, a ``TypeError`` -- is
    :attr:`ErrorKind.UNEXPECTED`, which :meth:`RetryPolicy.should_retry` refuses
    by default. Guessing that an unknown exception "looks transient" is exactly
    the sort of inference this milestone is not making.
    """
    if isinstance(error, RetryableToolError):
        return ErrorKind.RETRYABLE
    if isinstance(error, PermanentToolError):
        return ErrorKind.PERMANENT
    return ErrorKind.UNEXPECTED


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """A retry the runtime is about to journal and then carry out.

    This is both the decision and its value: it is what goes into
    ``ToolRetryScheduled``, and what a replay reads back out of that event
    instead of deciding anything again.
    """

    #: The attempt about to be run -- the "attempt=2" of ``ToolRetryScheduled``.
    attempt: int
    #: The attempt that just failed and caused this one.
    failed_attempt: int
    #: Seconds to wait before running :attr:`attempt`.
    delay: float
    #: Why the retry was allowed -- the :class:`ErrorKind` that permitted it.
    reason: str = str(ErrorKind.RETRYABLE)
    #: The failure that triggered it, already JSON-safe.
    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "failed_attempt": self.failed_attempt,
            "delay": self.delay,
            "reason": str(self.reason),
            "error": dict(self.error or {}) if self.error else None,
        }


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """An immutable statement of how often, and how patiently, to try again.

    ::

        RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=2.0, max_delay=10.0)

    ``max_attempts`` counts *attempts*, not retries, and defaults to ``1`` --
    that is, no retry at all. A tool call is therefore never retried unless it
    was configured to be, which is what keeps every pre-Milestone-4A behaviour
    unchanged.

    Attributes:
        max_attempts: Total attempts a logical call may make, including the first.
        initial_delay: Seconds to wait after the first failed attempt.
        multiplier: Factor the delay grows by after each further failure.
        max_delay: Ceiling for the wait, however large the backoff would grow.
        retry_on_unknown: Whether an exception that is neither
            :class:`~agent_runtime.exceptions.RetryableToolError` nor
            :class:`~agent_runtime.exceptions.PermanentToolError` may be retried.
            ``False`` by default: an unexpected exception is a bug until proven
            otherwise, and retrying it only hides the bug.
    """

    max_attempts: int = 1
    initial_delay: float = 0.1
    multiplier: float = 2.0
    max_delay: float = 10.0
    retry_on_unknown: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise RetryConfigurationError(
                "RetryPolicy.max_attempts must be >= 1 (a call always gets one "
                f"attempt), got {self.max_attempts}"
            )
        if self.initial_delay < 0:
            raise RetryConfigurationError(
                f"RetryPolicy.initial_delay must be >= 0, got {self.initial_delay}"
            )
        if self.multiplier < 1:
            raise RetryConfigurationError(
                "RetryPolicy.multiplier must be >= 1 (a backoff must not shrink), "
                f"got {self.multiplier}"
            )
        if self.max_delay < 0:
            raise RetryConfigurationError(
                f"RetryPolicy.max_delay must be >= 0, got {self.max_delay}"
            )

    # -- configuration -------------------------------------------------------

    @classmethod
    def none(cls) -> "RetryPolicy":
        """The policy that never retries -- what a call gets unless configured."""
        return NO_RETRY

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe form that goes into ``ToolRequested``."""
        return {
            "max_attempts": self.max_attempts,
            "initial_delay": self.initial_delay,
            "multiplier": self.multiplier,
            "max_delay": self.max_delay,
            "retry_on_unknown": self.retry_on_unknown,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "RetryPolicy" | None) -> "RetryPolicy":
        """Rebuild a policy from :meth:`to_dict` output; ``None`` means no retry."""
        if data is None:
            return NO_RETRY
        if isinstance(data, RetryPolicy):
            return data
        try:
            return cls(
                max_attempts=int(data["max_attempts"]),
                initial_delay=float(data["initial_delay"]),
                multiplier=float(data["multiplier"]),
                max_delay=float(data["max_delay"]),
                retry_on_unknown=bool(data.get("retry_on_unknown", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RetryConfigurationError(
                f"Cannot read a RetryPolicy from {data!r}: {exc}"
            ) from exc

    # -- decisions -----------------------------------------------------------

    def classify(self, error: BaseException | str | None) -> ErrorKind:
        """How this policy treats ``error``."""
        return classify_error(error)

    def should_retry(self, error: BaseException | str | None, attempt_number: int) -> bool:
        """Whether attempt ``attempt_number`` failed in a way worth repeating.

        Args:
            error: The exception the attempt raised.
            attempt_number: The 1-based number of the attempt that just failed.

        Returns:
            ``True`` only if the policy has attempts left *and* the failure is
            retryable under it.
        """
        self._check_attempt_number(attempt_number)
        if attempt_number >= self.max_attempts:
            return False
        kind = self.classify(error)
        if kind is ErrorKind.PERMANENT:
            return False
        if kind is ErrorKind.RETRYABLE:
            return True
        return self.retry_on_unknown

    def delay(self, attempt_number: int) -> float:
        """Seconds to wait after attempt ``attempt_number`` failed.

        Deterministic exponential backoff with no jitter::

            delay(n) = min(initial_delay * multiplier ** (n - 1), max_delay)

        So with the milestone's example policy ``delay(1) == 0.1``,
        ``delay(2) == 0.2`` and ``delay(3) == 0.4``. Identical inputs give
        identical waits, which is what lets a test assert the schedule and a
        replay reproduce it exactly.
        """
        self._check_attempt_number(attempt_number)
        backoff = self.initial_delay * (self.multiplier ** (attempt_number - 1))
        return min(backoff, self.max_delay)

    def exhausted(self, attempt_number: int) -> bool:
        """True when ``attempt_number`` was the last attempt this policy allows."""
        self._check_attempt_number(attempt_number)
        return attempt_number >= self.max_attempts

    @staticmethod
    def _check_attempt_number(attempt_number: int) -> None:
        if attempt_number < 1:
            raise RetryConfigurationError(
                f"Attempt numbers are 1-based, got {attempt_number}"
            )

    def __str__(self) -> str:
        if self.max_attempts == 1 and not self.retry_on_unknown:
            return "no retries"
        return (
            f"max_attempts={self.max_attempts}, initial_delay={self.initial_delay}, "
            f"multiplier={self.multiplier}, max_delay={self.max_delay}"
            + (", retry_on_unknown" if self.retry_on_unknown else "")
        )


#: The shared "do not retry" policy. A singleton so the common case allocates
#: nothing and identity comparison is a valid fast path.
NO_RETRY: RetryPolicy = RetryPolicy(max_attempts=1)

# ---------------------------------------------------------------------------
# Waiting
# ---------------------------------------------------------------------------


@runtime_checkable
class Sleeper(Protocol):
    """The one seam through which the runtime waits.

    Not an abstraction for its own sake: a backoff that only exists inside
    ``time.sleep`` cannot be tested without waiting, so waiting is a parameter
    and production and tests pass different implementations.
    """

    def sleep(self, delay: float) -> None:  # pragma: no cover - protocol
        """Wait for ``delay`` seconds."""
        ...


class RealSleeper:
    """Production sleeper: actually waits, unless there is nothing to wait for."""

    __slots__ = ()

    def sleep(self, delay: float) -> None:
        if delay > 0:
            time.sleep(delay)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "RealSleeper()"


class RecordingSleeper:
    """A sleeper that records what it was asked to wait and returns immediately.

    This is what makes the backoff schedule testable: a test can assert that the
    schedule was ``[0.1, 0.2]`` in microseconds of wall clock, and a replay can
    reproduce recorded waits without spending them.
    """

    __slots__ = ("_delays",)

    def __init__(self) -> None:
        self._delays: list[float] = []

    @property
    def delays(self) -> tuple[float, ...]:
        """Every requested delay, in order."""
        return tuple(self._delays)

    @property
    def total_delay(self) -> float:
        """Sum of every requested delay -- the time a real sleeper would spend."""
        return sum(self._delays)

    def sleep(self, delay: float) -> None:
        self._delays.append(float(delay))

    def clear(self) -> None:
        """Forget the recorded delays."""
        self._delays.clear()

    def __len__(self) -> int:
        return len(self._delays)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"RecordingSleeper(delays={self.delays!r})"

    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "failed_attempt": self.failed_attempt,
            "delay": self.delay,
            "reason": str(self.reason),
            "error": dict(self.error or {}) if self.error else None,
        }
