"""Timeouts: what a deadline means, and how each execution mode can enforce one.

A timeout is only a timeout if something actually stops. The three ways a tool
can run here have three different answers, and this module is where the runtime
decides which one applies rather than pretending all three are the same.

ASYNC
    ``async def`` tools run under :func:`asyncio.wait_for`, which cancels the
    coroutine at the deadline and delivers ``CancelledError`` inside it. That is
    real structured cancellation: the task is unwound at its own ``await``
    point. A tool that swallows ``CancelledError`` and keeps going has chosen
    to ignore the deadline, and the runtime waits for it rather than reporting a
    stop that did not happen.

COOPERATIVE
    A ``def`` tool that declared ``cancel_token`` runs on a worker thread. At
    the deadline the runtime flips the token and then *waits* for the tool to
    return. If it returns, the stop was real and is reported as one. If it does
    not return within the grace period, the runtime does **not** claim the tool
    stopped -- Python cannot terminate a thread, and saying otherwise would be
    the exact kind of lie this milestone refuses. The attempt is journalled
    ``ToolTimedOut`` with ``enforced=False``, and for a call carrying an
    idempotency key that is the ambiguous case: the side effect may well have
    happened, so the key stays ``PENDING`` and the execution reports
    ``RECOVERY_REQUIRED``.

UNSUPPORTED
    A ``def`` tool with no token cannot be stopped at all. Rather than run it
    and hope, the runtime refuses the configuration up front
    (:class:`~agent_runtime.exceptions.UnsupportedTimeoutError`), naming the
    tool and what it would have to do. This is the milestone's instruction not
    to implement fake timeouts: measuring elapsed time around a call nothing
    can interrupt is not a timeout, it is a lie with a stopwatch on it.

The mode is a property of the *tool*, resolved once and recorded with the
call, so a resumed retry or a replay knows which rule applied without
re-deriving it from a function that may since have changed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping

from .cancellation import CANCEL_TOKEN_PARAMETER, CancellationToken
from .exceptions import TimeoutConfigurationError, UnsupportedTimeoutError

__all__ = [
    "TimeoutMode",
    "ResolvedTimeout",
    "check_timeout",
    "declares_cancel_token",
    "is_async_callable",
    "resolve_timeout_mode",
]


class TimeoutMode(StrEnum):
    """How a deadline is enforced for a given tool."""

    #: A coroutine cancelled with ``asyncio.wait_for``.
    ASYNC = "ASYNC"
    #: A thread asked to stop through its cancellation token, then awaited.
    COOPERATIVE = "COOPERATIVE"
    #: A thread that cannot be interrupted: a timeout is refused, not faked.
    UNSUPPORTED = "UNSUPPORTED"

    @property
    def can_enforce(self) -> bool:
        """Whether the runtime can genuinely stop a tool in this mode."""
        return self is not TimeoutMode.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class ResolvedTimeout:
    """The deadline a call runs under, and the mode that will enforce it.

    Journalled on ``ToolRequested`` so a crash between the request and the
    attempt -- or a replay of the whole thing -- can read back the same rule
    instead of guessing from a tool that may no longer be the same function.
    """

    #: Seconds, or ``None`` for "no deadline on this call".
    seconds: float | None = None
    mode: TimeoutMode = TimeoutMode.UNSUPPORTED

    @property
    def active(self) -> bool:
        """Whether this call actually has a deadline to enforce."""
        return self.seconds is not None

    def to_dict(self) -> dict[str, Any]:
        return {"timeout": self.seconds, "timeout_mode": str(self.mode)}

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | "ResolvedTimeout" | None
    ) -> "ResolvedTimeout":
        """Rebuild a resolved timeout from :meth:`to_dict` output.

        A journal written before Milestone 4C carries neither field, which is
        exactly "no timeout" -- so old histories keep reading back cleanly.
        """
        if data is None:
            return cls()
        if isinstance(data, ResolvedTimeout):
            return data
        seconds = data.get("timeout")
        mode = data.get("timeout_mode") or TimeoutMode.UNSUPPORTED
        try:
            resolved_mode = TimeoutMode(mode)
        except ValueError:
            # A journal naming a mode this build does not know about is not
            # corrupt -- it is simply from a version that had one. An unknown
            # mode is read back as "no deadline to enforce", which is the
            # conservative direction: a replay reproduces the recorded events
            # without acting on a rule it does not understand.
            resolved_mode = TimeoutMode.UNSUPPORTED
        return cls(
            seconds=float(seconds) if seconds is not None else None,
            mode=resolved_mode,
        )

    def __str__(self) -> str:
        if not self.active:
            return "no timeout"
        return f"{self.seconds}s ({self.mode})"


def check_timeout(value: Any, *, where: str = "timeout") -> float | None:
    """Validate a timeout value, returning it as a positive float or ``None``.

    Rejecting this where it is written -- ``@tool(timeout=...)`` or
    ``execution.call(..., timeout=...)`` -- means a nonsensical deadline is a
    ``TimeoutConfigurationError`` on the line that wrote it, rather than a
    surprise discovered during a call that mattered.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimeoutConfigurationError(
            f"{where} must be a number of seconds or None, got {type(value).__name__}"
        )
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        raise TimeoutConfigurationError(
            f"{where} must be a finite number, got {value!r}"
        )
    if value <= 0:
        raise TimeoutConfigurationError(
            f"{where} must be greater than 0 (0 means 'no timeout'; pass None), "
            f"got {value!r}"
        )
    return float(value)


def is_async_callable(func: Callable[..., Any]) -> bool:
    """Whether ``func`` is a coroutine function, following partials and wrappers."""
    target = func
    while hasattr(target, "func") and not inspect.isfunction(target):  # functools.partial
        target = target.func  # type: ignore[assignment]
    return inspect.iscoroutinefunction(target) or inspect.iscoroutinefunction(
        getattr(target, "__call__", None)
    )


def declares_cancel_token(func: Callable[..., Any]) -> bool:
    """Whether a tool opted into cooperative cancellation.

    Opt-in by *declaring the parameter*: a tool that does not ask for a token is
    never handed one, which is what keeps every pre-4C tool callable with
    exactly the arguments it was written for.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return False
    for name, parameter in signature.parameters.items():
        if name == CANCEL_TOKEN_PARAMETER:
            return True
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            # ``**kwargs`` is not an opt-in: a tool that swallows arbitrary
            # keywords has not asked for anything, and quietly receiving a
            # token would make its recorded arguments depend on the runtime.
            return False
    return False


def resolve_timeout_mode(func: Callable[..., Any]) -> TimeoutMode:
    """Which timeout mechanism, if any, can stop this function.

    Decided from the function itself rather than from what the caller hopes:
    a coroutine is cancelled, a token-aware function is asked, and anything
    else gets an honest refusal.
    """
    if is_async_callable(func):
        return TimeoutMode.ASYNC
    if declares_cancel_token(func):
        return TimeoutMode.COOPERATIVE
    return TimeoutMode.UNSUPPORTED


def resolve_timeout(
    func: Callable[..., Any], seconds: float | None, *, tool_name: str
) -> ResolvedTimeout:
    """Pair a validated deadline with the mode that can enforce it.

    A deadline on a tool that cannot be stopped is refused here -- before
    ``ToolRequested``, before the claim, before anything is journalled -- so a
    configuration the runtime cannot honour never gets a call id.
    """
    mode = resolve_timeout_mode(func)
    if seconds is not None and mode is TimeoutMode.UNSUPPORTED:
        raise UnsupportedTimeoutError(
            f"Cannot enforce a {seconds}s timeout on tool {tool_name!r}: it is a "
            "synchronous function with no cancellation token, and Python cannot "
            "terminate a running thread. Write it as 'async def' so a deadline "
            "can cancel it, or declare a 'cancel_token' parameter so the runtime "
            "can ask it to stop. The runtime will not run it and merely measure "
            "how long it took -- that would report a stop that never happened.",
            tool_name=tool_name,
            timeout=seconds,
            mode=mode,
        )
    return ResolvedTimeout(seconds=seconds, mode=mode)

    UNSUPPORTED = "UNSUPPORTED"

    @property
    def can_enforce(self) -> bool:
        """Whether the runtime can genuinely stop a tool in this mode."""
        return self is not TimeoutMode.UNSUPPORTED
