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
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from .exceptions import (
    PermanentToolError,
    RetryConfigurationError,
    RetryableToolError,
    ToolCancelledError,
    ToolTimedOutError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .cancellation import CancellationToken

__all__ = [
    "ErrorKind",
    "RetryPolicy",
    "RetryDecision",
    "Sleeper",
    "RealSleeper",
    "RecordingSleeper",
    "classify_error",
    "wait_interrupted",
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
    #: The attempt ran out of time (Milestone 4C). Retried only when the policy
    #: sets ``retry_on_timeout`` -- a deadline is a *statement about time*, and
    #: whether a second attempt would have more of it is the application's call.
    TIMEOUT = "TIMEOUT"
    #: The attempt was cancelled (Milestone 4C). Never retried: cancellation is
    #: a decision, not a fault, and repeating it would repeat the decision.
    CANCELLED = "CANCELLED"


def classify_error(error: BaseException | str | None) -> ErrorKind:
    """Classify a failure for the purpose of retrying it.

    Deliberately narrow: only the error types a tool can raise are classified.
    Everything else -- a bug in the tool, a ``TypeError`` -- is
    :attr:`ErrorKind.UNEXPECTED`, which :meth:`RetryPolicy.should_retry` refuses
    by default. Guessing that an unknown exception "looks transient" is exactly
    the sort of inference this milestone is not making.
    """
    if isinstance(error, ToolCancelledError):
        return ErrorKind.CANCELLED
    if isinstance(error, ToolTimedOutError):
        return ErrorKind.TIMEOUT
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
        retry_on_timeout: Whether an attempt that ran out of time may be retried
            (Milestone 4C). ``False`` by default, and deliberately: a deadline is
            a statement about time, and whether another attempt would get more of
            it is the application's judgement, not the runtime's. A cancellation
            is never retried whatever this says.
    """

    max_attempts: int = 1
    initial_delay: float = 0.1
    multiplier: float = 2.0
    max_delay: float = 10.0
    retry_on_unknown: bool = False
    retry_on_timeout: bool = False

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
            "retry_on_timeout": self.retry_on_timeout,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "RetryPolicy" | None) -> "RetryPolicy":
        """Rebuild a policy from :meth:`to_dict` output; ``None`` means no retry.

        ``retry_on_timeout`` defaults to ``False`` for a journal written before
        Milestone 4C, which is the same answer the default gives -- so an old
        ``ToolRequested`` reads back as exactly the policy it described.
        """
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
                retry_on_timeout=bool(data.get("retry_on_timeout", False)),
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
        if kind is ErrorKind.CANCELLED:
            # Milestone 4C: a cancellation is a decision, not a fault. Retrying
            # it would re-issue the decision nobody asked for a second time.
            return False
        if kind is ErrorKind.TIMEOUT:
            # Milestone 4C: opt-in, because "it ran out of time" does not imply
            # "it would have finished given more time" -- that is a judgement
            # about the tool, and it belongs to whoever configured the policy.
            return self.retry_on_timeout
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
        if self.max_attempts == 1 and not self.retry_on_unknown and not self.retry_on_timeout:
            return "no retries"
        return (
            f"max_attempts={self.max_attempts}, initial_delay={self.initial_delay}, "
            f"multiplier={self.multiplier}, max_delay={self.max_delay}"
            + (", retry_on_unknown" if self.retry_on_unknown else "")
            + (", retry_on_timeout" if self.retry_on_timeout else "")
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

    def interruptible_sleep(
        self, delay: float, token: "CancellationToken | None" = None
    ) -> bool:  # pragma: no cover - protocol
        """Wait for ``delay`` seconds, returning ``True`` if cancelled first.

        Milestone 4C. Optional: :func:`wait_interrupted` falls back to
        :meth:`sleep` for a sleeper that does not implement it, so a 4A-era
        sleeper keeps working.
        """
        self.sleep(delay)
        return bool(token is not None and token.is_cancelled())


class RealSleeper:
    """Production sleeper: actually waits, unless there is nothing to wait for.

    :meth:`interruptible_sleep` is the Milestone 4C addition. A backoff that
    cannot be interrupted is a backoff that keeps a cancelled execution alive
    for the rest of its delay, so the wait is made on the cancellation token's
    :class:`threading.Event` -- which returns the instant another thread calls
    ``cancel()`` -- rather than inside ``time.sleep``.
    """

    __slots__ = ()

    def sleep(self, delay: float) -> None:
        if delay > 0:
            time.sleep(delay)

    def interruptible_sleep(
        self, delay: float, token: "CancellationToken | None" = None
    ) -> bool:
        """Wait ``delay`` seconds, or until ``token`` is cancelled.

        Returns:
            ``True`` if the wait was cut short by cancellation, ``False`` if it
            ran to completion.

        Without a token this is exactly :meth:`sleep`, and it calls it -- so a
        sleeper written for Milestone 4A, or a subclass that overrides ``sleep``
        to inspect the call, still sees every wait go through it.
        """
        if delay <= 0:
            return bool(token is not None and token.is_cancelled())
        if token is None:
            self.sleep(delay)
            return False
        return token.wait(delay)

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

    def interruptible_sleep(
        self, delay: float, token: "CancellationToken | None" = None
    ) -> bool:
        """Record the delay and return immediately; report any cancellation.

        A test never spends a backoff, so the "interruption" here is simply
        whether the token was *already* cancelled when the wait was requested --
        which is exactly the state a test that cancels mid-retry wants to
        exercise, without a thread having to race a clock.

        It goes through :meth:`sleep` so a subclass that overrides ``sleep`` to
        inspect a wait (as several of this repo's tests do) still sees it.
        """
        self.sleep(delay)
        return bool(token is not None and token.is_cancelled())

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"RecordingSleeper(delays={self.delays!r})"


def wait_interrupted(
    sleeper: Sleeper, delay: float, token: "CancellationToken | None" = None
) -> bool:
    """Wait ``delay`` through ``sleeper``, cut short by ``token``.

    Milestone 4C: the retry backoff goes through here, so a cancellation during
    a ten-second wait returns immediately instead of sleeping it out.

    A sleeper that predates :meth:`Sleeper.interruptible_sleep` still works --
    the wait is then made with :meth:`Sleeper.sleep` and only checked
    afterwards -- so third-party sleepers written for Milestone 4A keep working
    unchanged, at the cost of not being interrupted mid-wait.
    """
    interruptible = getattr(sleeper, "interruptible_sleep", None)
    if interruptible is not None:
        return bool(interruptible(delay, token))
    if delay > 0:
        sleeper.sleep(delay)
    return bool(token is not None and token.is_cancelled())

    error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "failed_attempt": self.failed_attempt,
            "delay": self.delay,
            "reason": str(self.reason),
            "error": dict(self.error or {}) if self.error else None,
        }
