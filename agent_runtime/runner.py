"""The tool runner: the one place that decides *how* a tool call is carried out.

There are exactly two modes, and the difference between them is the whole point
of Milestone 3::

    NORMAL   ToolRequested -> ToolStarted -> execute tool -> ToolCompleted
    REPLAY   ToolRequested -> look up the recorded call -> recorded result

:class:`ToolRunner` runs the registered function. :class:`ReplayToolRunner`
never touches it: given the same request it hands back the outcome the journal
recorded. Because replay dispatches through this same object, the guarantee is
structural -- a replay has no code path that reaches a tool function, so a
``create_github_issue`` cannot run twice no matter what the caller does.

Milestone 4A adds three seams here -- :meth:`ToolRunner.begin_call`,
:meth:`ToolRunner.can_attempt` and :meth:`ToolRunner.retry_decision` -- for the
same reason. NORMAL answers them from the live tool plus the call's policy; the
REPLAY runner answers them from the recorded attempt history, so a replayed
retry is the retry that was journalled rather than one the replay re-decides.

Milestone 4C adds the deadline and the token, and the three execution modes that
result (:meth:`ToolRunner.run` dispatches on them):

    async def        asyncio.wait_for cancels the coroutine -- a real stop
    def + token      the token is flipped, then the tool is *waited for*
    def, no token    refused: Python cannot terminate a running thread

A REPLAY runner has no function to call, so it has nothing to stop:
:meth:`ReplayToolRunner.run` accepts the deadline and the token and discards
both, which is why replaying a five-second timeout costs no time at all.

Scope note: replay substitutes *recorded* tool results. It does not sandbox or
intercept anything else -- ``time``, ``random``, the network and environment
variables are external inputs an execution must treat as tools if they influence
its outcome.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .cancellation import CancellationToken
from .events import describe_error
from .exceptions import (
    TimeoutEnforcementError,
    ToolCancelledError,
    ToolInvocationError,
    ToolTimedOutError,
)
from .retry import RetryDecision, RetryPolicy
from .state import ToolCallStatus
from .timeout import ResolvedTimeout, TimeoutMode
from .tools import ToolRegistry, make_jsonable

__all__ = [
    "ToolRunnerMode",
    "ToolRequest",
    "ToolOutcome",
    "ToolRunner",
]

#: How finely a cooperative tool's stop is polled. Small enough that a
#: cancellation is acted on almost immediately, large enough not to spin.
COOPERATIVE_POLL_INTERVAL = 0.01


async def _with_deadline(coro: Any, seconds: float, token: CancellationToken | None) -> Any:
    """Await ``coro`` under a deadline and a cancellation token.

    The two are unified into one wait: :func:`asyncio.wait_for` supplies the
    deadline, and the token's ``on_cancel`` hook supplies an external cancel by
    cancelling the same task. Whichever fires first, the task is cancelled for
    real and its ``CancelledError`` is allowed to propagate -- that is the whole
    mechanism, and there is no timing wrapper anywhere around it.
    """
    task = asyncio.ensure_future(coro)
    loop = asyncio.get_running_loop()

    def _cancel_from_token() -> None:
        # Called from whichever thread called ``cancel()``; hop to the loop
        # thread, which is the only one allowed to touch the task.
        loop.call_soon_threadsafe(task.cancel)

    if token is not None:
        token.on_cancel(_cancel_from_token)

    try:
        if seconds and seconds > 0:
            return await asyncio.wait_for(asyncio.shield(task), seconds)
        return await task
    except asyncio.TimeoutError:
        # ``wait_for`` already cancelled the inner task; give it a turn to
        # unwind so no coroutine is left running past the deadline.
        try:
            await asyncio.wait_for(asyncio.shield(task), COOPERATIVE_POLL_INTERVAL)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        raise



class ToolRunnerMode(StrEnum):
    """How a tool call is carried out."""

    #: Call the registered function and journal what happened.
    NORMAL = "NORMAL"
    #: Return the outcome the journal already recorded. No function is called.
    REPLAY = "REPLAY"


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """One request to run a tool: a name plus normalized, JSON-safe arguments."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What a tool call produced.

    A successful call carries ``result``; a failed one carries the JSON-safe
    ``error`` payload that the journal will store, so the two halves of a call --
    the outcome and its record -- cannot disagree.

    Milestone 4C keeps the *stops* apart from the failures: a timed-out attempt
    is :attr:`ToolCallStatus.TIMED_OUT` and a cancelled one is
    :attr:`ToolCallStatus.CANCELLED`, neither of which collapses into
    :attr:`ToolCallStatus.FAILED`. A caller can therefore tell "it raised",
    "it ran out of time" and "it was stopped on purpose" without parsing a
    message.
    """

    status: ToolCallStatus
    result: Any = None
    error: Mapping[str, Any] | None = None
    #: The live exception, kept only so the caller can chain it (``raise ... from``).
    #: Never journalled, and ignored for equality.
    cause: BaseException | None = field(default=None, compare=False, repr=False)
    #: Milestone 4C: the deadline this attempt ran under, and whether the
    #: runtime can prove the tool actually stopped at it. ``None``/``True`` for
    #: every outcome that is not a timeout.
    timeout: float | None = None
    timeout_mode: str | None = None
    timeout_enforced: bool = True

    @property
    def succeeded(self) -> bool:
        return self.status is ToolCallStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Whether the call produced no result, for any reason.

        A timeout and a cancellation count: all three are stops, and the
        runtime surfaces all three to the caller rather than returning
        ``None`` as if the tool had produced nothing.
        """
        return self.status in (
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
            ToolCallStatus.TIMED_OUT,
        )

    @property
    def timed_out(self) -> bool:
        return self.status is ToolCallStatus.TIMED_OUT

    @property
    def cancelled(self) -> bool:
        return self.status is ToolCallStatus.CANCELLED

    @property
    def ambiguous(self) -> bool:
        """Whether the outcome of this attempt's side effect is unknown.

        Milestone 4C. True for a timeout whose stop could not be enforced: the
        tool was asked to stop and did not, so whatever it was doing may still
        be happening. A caller with an idempotency key must treat this as
        "unknown", not as "failed".
        """
        return self.timed_out and not self.timeout_enforced

    @classmethod
    def completed(cls, result: Any) -> "ToolOutcome":
        return cls(status=ToolCallStatus.COMPLETED, result=result)

    @classmethod
    def failed_with(cls, error: BaseException | str) -> "ToolOutcome":
        return cls(
            status=ToolCallStatus.FAILED,
            error=describe_error(error),
            cause=error if isinstance(error, BaseException) else None,
        )

    @classmethod
    def timed_out_with(
        cls,
        error: BaseException | str,
        *,
        seconds: float,
        mode: str,
        enforced: bool = True,
    ) -> "ToolOutcome":
        """A deadline expired. Kept distinct from a failure on purpose.

        ``enforced=False`` is the honest report of a tool that was asked to
        stop and did not: the runtime did not terminate it, and anything it was
        doing may still happen (:attr:`ambiguous`).
        """
        return cls(
            status=ToolCallStatus.TIMED_OUT,
            error=describe_error(error),
            cause=error if isinstance(error, BaseException) else None,
            timeout=seconds,
            timeout_mode=mode,
            timeout_enforced=enforced,
        )

    @classmethod
    def cancelled_with(
        cls, error: BaseException | str, *, reason: Any = None
    ) -> "ToolOutcome":
        """A cancellation stopped the attempt. Never eligible for a retry."""
        return cls(
            status=ToolCallStatus.CANCELLED,
            error=describe_error(error),
            cause=error if isinstance(error, BaseException) else None,
        )


class ToolRunner:
    """Runs tools for real. This is the only runner that may call a function.

    Milestone 4C gives :meth:`run` a deadline and a cancellation token, and the
    two are enforced by *actually stopping the tool* rather than by timing it::

        ASYNC         asyncio.wait_for cancels the coroutine at the deadline
        COOPERATIVE   the token is flipped at the deadline and the thread is
                      then waited on; if it does not stop, ``enforced=False``
        UNSUPPORTED   refused before the call (see :mod:`agent_runtime.timeout`)

    A replay substitutes :class:`~agent_runtime.replay.ReplayToolRunner`, which
    overrides all of this: it has no function to call and therefore no timeout
    to enforce, so a replayed timeout costs nothing and waits for nothing.
    """

    mode = ToolRunnerMode.NORMAL

    def __init__(self, registry: ToolRegistry, *, cooperative_grace: float = 1.0) -> None:
        self.registry = registry
        #: How long, past a cooperative deadline, the runtime waits for a tool
        #: to notice it was asked to stop before reporting that it did not.
        #: A parameter, not a constant, so a test can make the unenforceable
        #: case fast and production can be generous.
        self.cooperative_grace = cooperative_grace

    def run(
        self,
        request: ToolRequest,
        *,
        timeout: ResolvedTimeout | None = None,
        token: CancellationToken | None = None,
    ) -> ToolOutcome:
        """Invoke the registered tool and return what it produced.

        One call, one attempt. A tool that raises is an outcome, not an
        exception escaping this method: the caller journals ``ToolFailed`` from
        it, decides about a retry, and only then re-raises, so both the failure
        and any decision about it are durable before anything observes them.

        A deadline has the same shape -- an outcome, not an escape -- and is
        reported as :attr:`ToolCallStatus.TIMED_OUT`, so the journal records a
        timeout as a timeout rather than as an anonymous failure.
        """
        target = self.registry.get(request.tool)
        resolved = timeout or ResolvedTimeout()
        if resolved.active and resolved.mode is TimeoutMode.ASYNC:
            return self._run_async(target, request, resolved, token)
        if resolved.active and resolved.mode is TimeoutMode.COOPERATIVE:
            return self._run_cooperative(target, request, resolved, token)
        return self._run_plain(target, request, token)

    # -- the three execution modes -------------------------------------------

    def _run_plain(
        self, target: Any, request: ToolRequest, token: CancellationToken | None
    ) -> ToolOutcome:
        """No deadline: invoke directly, on this thread.

        A token-aware tool still gets its token, so ``execution.cancel()`` can
        reach a long tool that was given a generous timeout -- or none at all.
        """
        bound = target.bind_with_token((), request.arguments, token)
        try:
            return ToolOutcome.completed(make_jsonable(target.invoke(bound)))
        except ToolCancelledError as cancelled:
            return ToolOutcome.cancelled_with(cancelled, reason=cancelled.reason)
        except Exception as exc:  # noqa: BLE001 - the failure itself is the outcome
            return ToolOutcome.failed_with(exc)

    def _run_async(
        self,
        target: Any,
        request: ToolRequest,
        resolved: ResolvedTimeout,
        token: CancellationToken | None,
    ) -> ToolOutcome:
        """Run a coroutine tool under a real deadline.

        :func:`asyncio.wait_for` is the mechanism, and it is a *real* one: at
        the deadline it cancels the task, which delivers ``CancelledError`` at
        the coroutine's own ``await`` point and unwinds it from there.

        Two details this method is careful about:

        * ``asyncio.TimeoutError`` is an alias of the builtin ``TimeoutError``,
          so it is caught ahead of the generic handler and reported as a
          timeout rather than as a tool failure;
        * a coroutine that *catches* ``CancelledError`` and keeps going has
          chosen to ignore the deadline. It is waited for, and whatever it
          eventually returns is its honest result -- the runtime does not report
          a cancellation that did not happen.
        """
        seconds = float(resolved.seconds or 0.0)

        async def _main() -> Any:
            bound = target.bind_with_token((), request.arguments, token)
            return await target.invoke(bound)

        if token is not None and token.is_cancelled():
            # Already cancelled: the coroutine is never started at all.
            return ToolOutcome.cancelled_with(
                ToolCancelledError(
                    f"Cancelled before starting: {token.reason}",
                    reason=token.reason,
                    tool_name=request.tool,
                )
            )
        try:
            return ToolOutcome.completed(
                make_jsonable(asyncio.run(_with_deadline(_main(), seconds, token)))
            )
        except asyncio.TimeoutError:
            return ToolOutcome.timed_out_with(
                ToolTimedOutError(
                    f"Tool {request.tool!r} exceeded its {seconds}s deadline; the "
                    "coroutine was cancelled at its await point",
                    tool_name=request.tool,
                    timeout=seconds,
                    mode=str(TimeoutMode.ASYNC),
                    enforced=True,
                ),
                seconds=seconds,
                mode=str(TimeoutMode.ASYNC),
            )
        except ToolCancelledError as cancelled:
            return ToolOutcome.cancelled_with(cancelled, reason=cancelled.reason)
        except asyncio.CancelledError:
            return ToolOutcome.cancelled_with(
                ToolCancelledError(
                    f"Tool {request.tool!r} was cancelled", tool_name=request.tool
                )
            )
        except Exception as exc:  # noqa: BLE001 - the failure itself is the outcome
            return ToolOutcome.failed_with(exc)

    def _run_cooperative(
        self,
        target: Any,
        request: ToolRequest,
        resolved: ResolvedTimeout,
        token: CancellationToken | None,
    ) -> ToolOutcome:
        """Run a token-aware tool on a thread, and *wait* for it to stop.

        The honest part is what happens when it does not. Python cannot
        terminate a thread, so this method never pretends it did:

        * the deadline flips the token, and the runtime then waits up to
          :attr:`cooperative_grace` for the tool to return. If it returns, the
          stop was real and is reported as ``enforced=True``;
        * if it does not return, the runtime says so -- ``enforced=False`` --
          instead of reporting a stop that never happened. A caller holding an
          idempotency key reads that as *unknown*, so the key stays ``PENDING``
          and the execution reports ``RECOVERY_REQUIRED``.
        A tool that *returns a partial result* when it notices the stop is still
        a timeout, and that is the subtle case worth stating: reporting it as a
        success would hand a truncated answer back as if it were the real one,
        and would hide the deadline from the retry policy entirely. The stop is
        the fact; what the tool returned on its way out is detail.
        """
        seconds = float(resolved.seconds or 0.0)
        holder: dict[str, Any] = {}

        def _target() -> None:
            try:
                bound = target.bind_with_token((), request.arguments, token)
                holder["result"] = make_jsonable(target.invoke(bound))
            except BaseException as exc:  # noqa: BLE001 - recorded, then re-judged
                holder["error"] = exc

        thread = threading.Thread(
            target=_target, name=f"tool:{request.tool}", daemon=True
        )
        thread.start()

        stopped_because = self._await_stop(thread, seconds, token)
        alive = thread.is_alive()

        if stopped_because == "cancelled":
            # The application asked for this, so it is a cancellation whatever
            # the tool managed to return on its way out.
            return ToolOutcome.cancelled_with(
                ToolCancelledError(
                    f"Tool {request.tool!r} was cancelled: {token.reason}",
                    reason=token.reason,
                    tool_name=request.tool,
                )
            )

        if stopped_because == "deadline":
            if alive:
                return ToolOutcome.timed_out_with(
                    TimeoutEnforcementError(
                        f"Tool {request.tool!r} did not stop within {seconds}s plus a "
                        f"{self.cooperative_grace}s grace period after its "
                        "cancellation token was flipped. Python cannot terminate a "
                        "running thread, so the runtime cannot claim it stopped; any "
                        "side effect it was performing may still happen.",
                        tool_name=request.tool,
                        timeout=seconds,
                        grace=self.cooperative_grace,
                    ),
                    seconds=seconds,
                    mode=str(TimeoutMode.COOPERATIVE),
                    enforced=False,
                )
            return ToolOutcome.timed_out_with(
                ToolTimedOutError(
                    f"Tool {request.tool!r} reached its {seconds}s deadline and "
                    "stopped when asked",
                    tool_name=request.tool,
                    timeout=seconds,
                    mode=str(TimeoutMode.COOPERATIVE),
                    enforced=True,
                ),
                seconds=seconds,
                mode=str(TimeoutMode.COOPERATIVE),
            )

        error = holder.get("error")
        if error is not None:
            if isinstance(error, ToolCancelledError):
                return ToolOutcome.cancelled_with(error, reason=error.reason)
            return ToolOutcome.failed_with(error)
        return ToolOutcome.completed(holder.get("result"))

    def _await_stop(
        self, thread: threading.Thread, seconds: float, token: CancellationToken | None
    ) -> str:
        """Wait for a tool, and report what ended it.

        Returns ``"completed"`` only when the tool finished *on its own*,
        ``"deadline"`` when this method flipped the token because the deadline
        expired, or ``"cancelled"`` when somebody else had already cancelled.

        The distinction the loop below is careful about: a tool that notices the
        stop and **returns a partial result** is still ``"deadline"`` or
        ``"cancelled"``, not ``"completed"``. It finished, but it finished
        *because it was stopped*, and reporting that as a success would hand a
        truncated answer back as if it were the real one -- and would hide the
        deadline from the retry policy entirely.
        """
        if token is None:
            # A cooperative tool always has a token, but the runner refuses to
            # assume it: an unenforceable deadline is honest, and a missing
            # token is a bug rather than something to paper over.
            token = CancellationToken()
            token.cancel("no cancellation token was supplied")
            thread.join(self.cooperative_grace)
            return "cancelled"

        remaining = seconds
        while thread.is_alive():
            if token.is_cancelled():
                break  # somebody else already asked it to stop
            if remaining <= 0:
                token.cancel(f"timeout after {seconds}s")
                break
            # A short slice keeps the loop responsive to an external cancel()
            # without busy-waiting, and bounds how long a late cancel lingers.
            token.wait(min(COOPERATIVE_POLL_INTERVAL, remaining))
            remaining -= COOPERATIVE_POLL_INTERVAL

        # Grace period: the tool has been asked, now give it a bounded moment to
        # act on that. This is where "enforced" is decided -- not before.
        thread.join(self.cooperative_grace)

        if token.is_cancelled():
            # It was stopped, whether or not it managed to return on the way
            # out. Whether the stop actually took effect is a separate question,
            # answered by whether the thread is still running.
            return "cancelled" if self._cancelled_externally(token) else "deadline"
        return "completed"

    @staticmethod
    def _cancelled_externally(token: CancellationToken) -> bool:
        """Whether the token was cancelled by the application, not by a deadline.

        The deadline is the one message the runtime writes itself, so anything
        else came from ``execution.cancel()``. Comparing the two keeps the two
        stops apart in the journal without threading a second flag through.
        """
        return not str(token.reason or "").startswith("timeout after ")

    # -- retry seams ----------------------------------------------------------

    def begin_call(self, call_id: str, request: ToolRequest) -> None:
        """A logical call is starting its first attempt.

        The NORMAL runner has nothing to bind -- the tool is simply there when
        it is run. The hook exists so the REPLAY runner can attach the recorded
        call to the execution, instead of counting calls positionally and
        hoping the two orders agree.
        """

    def can_attempt(self, call_id: str, attempt: int) -> bool:
        """Whether attempt ``attempt`` may be run.

        Always ``True`` for a real call: attempts are limited by the policy, not
        by the journal. A replay asks instead of assuming, because a history
        that ends after ``ToolRetryScheduled`` has nothing recorded for the
        attempt that was scheduled, and inventing one is what replay must never
        do.
        """
        return True

    def retry_decision(
        self,
        request: ToolRequest,
        outcome: ToolOutcome,
        *,
        call_id: str,
        attempt: int,
        policy: RetryPolicy,
    ) -> RetryDecision | None:
        """Whether the attempt that just stopped should be followed by another.

        This is the only place a retry is decided in a real run, and it is
        decided from the live exception plus the policy -- never from a guess
        about how transient the failure looks.

        Milestone 4C keeps the two stops in the same place as the failure and
        lets the *policy* answer for them, so one rule covers all three:

        * a cancellation is never retried, whatever the policy says. It is a
          decision someone made, and repeating it would repeat their decision;
        * a timeout is retried only when the policy opted in with
          ``retry_on_timeout=True``, because "it ran out of time" does not by
          itself mean "it would have finished given more".

        Returns:
            A :class:`~agent_runtime.retry.RetryDecision` to carry out, or
            ``None`` to let the outcome stand.
        """
        if outcome.cancelled:
            return None
        error = outcome.cause
        if not policy.should_retry(error, attempt):
            return None
        return RetryDecision(
            attempt=attempt + 1,
            failed_attempt=attempt,
            delay=policy.delay(attempt),
            reason=str(policy.classify(error)),
            error=dict(outcome.error or {}),
        )

    def invocation_error(
        self, request: ToolRequest, outcome: ToolOutcome
    ) -> ToolInvocationError:
        """Build the error to raise after a stopped call has been journalled.

        Each stop raises its own type, so the caller can catch
        :class:`ToolCancelledError` and :class:`~agent_runtime.exceptions.ToolTimedOutError`
        rather than reading a message to find out which happened.
        """
        error = outcome.error or {}
        message = error.get("message", "unknown error")
        if outcome.cancelled:
            return ToolCancelledError(
                f"Tool {request.tool!r} was cancelled: {message}",
                reason=message,
                tool_name=request.tool,
                traceback_text=error.get("traceback"),
            )
        if outcome.timed_out:
            return ToolTimedOutError(
                f"Tool {request.tool!r} timed out: {message}",
                tool_name=request.tool,
                timeout=outcome.timeout,
                mode=outcome.timeout_mode,
                enforced=outcome.timeout_enforced,
                traceback_text=error.get("traceback"),
            )
        return ToolInvocationError(
            f"Tool {request.tool!r} failed: {message}",
            tool_name=request.tool,
            error_type=error.get("type") or "ToolInvocationError",
            traceback_text=error.get("traceback"),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{type(self).__name__}(mode={self.mode})"
