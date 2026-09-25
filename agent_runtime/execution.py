"""The execution object: the user-facing handle on a journalled run.

Everything an :class:`Execution` reports -- its status, its goal, its tool calls,
what a crash left unfinished -- is derived from the journal by folding its events
through the reducers in :mod:`agent_runtime.state`. In-memory bookkeeping is
never a second source of truth, and a checkpoint is nothing more than one of
those derived states captured at a sequence.

Milestone 4A adds the attempt loop to that story. :meth:`Execution.call` makes
*one logical call* -- one ``call_id`` -- and the loop inside it runs the
attempts::

    ToolRequested
    ToolStarted     attempt=1
    ToolFailed      attempt=1
    ToolRetryScheduled attempt=2
    ToolStarted     attempt=2
    ToolCompleted   attempt=2

The decision to make another attempt is journalled before the wait and before
the attempt, so a process that dies during the backoff leaves the scheduled
retry in the journal rather than only in the memory of a process that was about
to sleep.

Milestone 4B adds the idempotency key to that story, because the attempt loop
alone cannot survive the window between the side effect and its record::

    claim key -> run tool -> [process dies] -> recovery

:meth:`Execution.call` takes an ``idempotency_key``, claims it before the tool
runs, and stores the outcome afterwards. A duplicate of a ``COMPLETED`` key is
answered from the stored result without running anything; a key left ``PENDING``
by a crash is *not* answered at all -- it raises
:class:`~agent_runtime.exceptions.IdempotencyRecoveryRequiredError`, because the
runtime cannot know whether the external effect happened and will not guess.
:meth:`Execution.resolve_idempotency` is the explicit way out.

Milestone 4C adds the two stops a call can end on that are not failures, and
keeps them apart from failures at every layer -- event, status, exception, retry
decision::

    ToolStarted
        ↓
    deadline reached / execution.cancel()
        ↓
    ToolTimedOut  |  ToolCancelled
        ↓
    the same retry policy answers, differently

A timeout is enforced by actually stopping the tool -- an ``asyncio`` deadline
for a coroutine, a cancellation token for a ``def`` tool that declared one --
and is *refused* for a ``def`` tool that did not, because Python cannot
terminate a thread and a wrapper that only measured elapsed time would be
reporting a stop that never happened. A cancellation is never retried, because
cancelling is a decision rather than a fault. Neither collapses into
``ToolFailed``, because an application reading a recovered journal has to be
able to tell them apart without parsing a message.
"""

from __future__ import annotations

from typing import Any, Literal

from .cancellation import CancellationToken
from .checkpoints import Checkpoint, CheckpointStore
from .events import Event, EventType, describe_error, new_id
from .exceptions import (
    AgentRuntimeError,
    IdempotencyResolutionError,
    InvalidRecoveryActionError,
    InvalidStateTransitionError,
    ToolCancelledError,
    ToolTimedOutError,
    UnknownIdempotencyKeyError,
    UnknownToolCallError,
)
from .idempotency import (
    IdempotencyAction,
    IdempotencyGuard,
    IdempotencyRecord,
    IdempotencyStore,
    ReplayIdempotencyGuard,
    StoreIdempotencyGuard,
    unresolved_records,
)
from .journal import EventJournal
from .recovery import RecoveryInfo, recover_execution
from .retry import (
    NO_RETRY,
    RealSleeper,
    RetryDecision,
    RetryPolicy,
    Sleeper,
    wait_interrupted,
)
from .runner import ToolRequest, ToolRunner
from .state import (
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    PendingRetry,
    ToolCall,
    ToolCallStatus,
    detect_pending_retries,
    reconstruct_state,
)
from .timeout import ResolvedTimeout, TimeoutMode, check_timeout, resolve_timeout
from .tools import ToolRegistry, check_retry_policy, make_jsonable

__all__ = ["Execution", "RecoveryAction"]

#: How an application may settle a tool call a crash left open. Milestone 2
#: records decisions the application already made; it never retries on its own.
RecoveryAction = Literal["mark_completed", "mark_failed", "cancel"]

_ACTION_EVENTS: dict[str, EventType] = {
    "mark_completed": EventType.TOOL_COMPLETED,
    "mark_failed": EventType.TOOL_FAILED,
    "cancel": EventType.TOOL_CANCELLED,
}

_ACTION_RESULTS: dict[str, Any] = {
    "mark_completed": None,
    "mark_failed": {"message": "marked failed during recovery"},
    "cancel": {"message": "cancelled during recovery"},
}


class AmbiguousTimeoutError(AgentRuntimeError):
    """A keyed call timed out and the runtime cannot prove its tool stopped.

    The combination Milestone 4C exists to refuse: a side effect that may have
    happened, wrapped in a deadline that expired. Retrying it could perform the
    effect twice, and settling it would be a guess, so the runtime does neither
    -- it raises this, leaves the key ``PENDING``, and lets the application
    check the outside world and decide with
    :meth:`~agent_runtime.execution.Execution.resolve_idempotency`.

    Attributes:
        idempotency_key: The key whose effect is now unknown.
        tool_name: The tool that outlived its deadline.
    """

    def __init__(
        self, summary: str, *, idempotency_key: str, tool_name: str | None = None
    ) -> None:
        super().__init__(summary)
        self.idempotency_key = idempotency_key
        self.tool_name = tool_name

    def __str__(self) -> str:
        return "\n".join(
            [
                "AmbiguousTimeoutError",
                "",
                f"Reason: {self.args[0]}",
                f"Key: {self.idempotency_key}",
                "",
                "The runtime cannot tell whether the external side effect happened,",
                "so it did not retry the call. Decide explicitly:",
                "",
                "    execution.resolve_idempotency(key, 'mark_completed', result=...)",
                "    execution.resolve_idempotency(key, 'mark_failed', error=...)",
                "    execution.resolve_idempotency(key, 'retry')",
            ]
        )


def _default_guard(journal: EventJournal) -> IdempotencyGuard:
    """The guard an execution gets when the caller does not supply one.

    A journalled run keeps its claims in the same SQLite file as its events, so
    the default is a guard over that database. A journal with no store behind it
    -- the in-memory one a replay builds -- gets the store-less replay guard
    instead, because a replay must not write to a real database.
    """
    store = getattr(journal, "store", None)
    if store is None:
        return ReplayIdempotencyGuard()
    return StoreIdempotencyGuard(IdempotencyStore(store))


class Execution:
    """A single execution, backed by its immutable event history."""

    def __init__(
        self,
        journal: EventJournal,
        registry: ToolRegistry,
        execution_id: str,
        *,
        checkpoints: CheckpointStore | None = None,
        recovered_state: ExecutionState | None = None,
        auto_checkpoint: bool = False,
        runner: ToolRunner | None = None,
        sleeper: Sleeper | None = None,
        idempotency: IdempotencyGuard | None = None,
    ) -> None:
        self._journal = journal
        self._registry = registry
        self._id = execution_id
        self._checkpoints = checkpoints
        self._state = recovered_state
        self._auto_checkpoint = auto_checkpoint
        #: Carries out each ``call``. Defaults to the NORMAL runner; replay
        #: passes a REPLAY runner that serves recorded results instead.
        self._runner = runner if runner is not None else ToolRunner(registry)
        #: Decides whether a keyed ``call`` may run its tool (Milestone 4B).
        #: Defaults to a guard over the journal's own SQLite store; a replay --
        #: whose journal is in memory -- substitutes the store-less
        #: :class:`~agent_runtime.idempotency.ReplayIdempotencyGuard`, so it can
        #: never claim, insert or overwrite anything.
        self._idempotency = idempotency if idempotency is not None else _default_guard(journal)
        #: How retry backoff waits. Injectable so a test can assert the
        #: schedule without spending it -- nothing here patches ``time.sleep``.
        self._sleeper = sleeper if sleeper is not None else RealSleeper()
        #: Milestone 4C: the cooperative stop signal for this execution's tools.
        #: One per execution, shared by every call it makes, because
        #: ``cancel()`` cancels the *execution*: a tool that is mid-flight
        #: cannot be told to stop while its siblings carry on.
        self._cancel_token = CancellationToken()

    @property
    def cancel_token(self) -> CancellationToken:
        """The token handed to tools of this execution that declared one.

        Read-only on purpose. Cancelling is :meth:`cancel`'s job, because
        ``cancel()`` is what makes the decision *durable* -- flipping the flag
        alone would stop a tool without leaving a record that it was stopped.
        """
        return self._cancel_token

    @property
    def is_cancel_requested(self) -> bool:
        """Whether this execution's cancellation has been requested."""
        return self._cancel_token.is_cancelled()

    # -- identity / derived state --------------------------------------------

    @property
    def sleeper(self) -> Sleeper:
        """The sleeper this execution waits through, between retry attempts."""
        return self._sleeper

    @property
    def id(self) -> str:
        """The execution id (``exec_...``)."""
        return self._id

    @property
    def events(self) -> list[Event]:
        """The full, sequence-ordered event history read back from SQLite."""
        return self._journal.get_events(self._id)

    @property
    def state(self) -> ExecutionState:
        """The current state of this execution.

        The value is loaded once and kept until something is journalled. Caching
        it is what lets :meth:`checkpoint` persist the state and the sequence
        together: the snapshot has to be of the latest persisted event, so the
        events behind it must not move while the write is in flight. The cache is
        dropped by every write on this object, so it never reports anything the
        journal has not already committed.
        """
        if self._state is None:
            self._state = reconstruct_state(self.events)
        return self._state

    @property
    def last_event_sequence(self) -> int:
        """The sequence of the latest persisted event (``0`` when there is none)."""
        return self._journal.get_last_sequence(self._id)

    @property
    def status(self) -> ExecutionStatus:
        """The execution's status, including anything still unresolved about a key.

        ``RECOVERY_REQUIRED`` is reported for either kind of ambiguity
        (Milestone 2's open tool call, Milestone 4B's ``PENDING`` key), because
        both mean the same thing to a caller: the runtime cannot tell you what
        happened, and will not run the side effect again on its own.

        The stored :class:`~agent_runtime.state.ExecutionState` is untouched --
        a checkpoint still holds the journal-derived status, so recovery re-derives
        this the same way instead of trusting two sources for one field.
        """
        state = self.state
        if state.status is not ExecutionStatus.RUNNING:
            return state.status
        if self.needs_idempotency_resolution:
            return ExecutionStatus.RECOVERY_REQUIRED
        return state.status

    @property
    def goal(self) -> str | None:
        return self.state.goal

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.state.tool_calls

    @property
    def incomplete_tools(self) -> tuple[IncompleteTool, ...]:
        """Tool calls that were started (or requested) but never resolved.

        Non-empty exactly when :attr:`status` is ``RECOVERY_REQUIRED``: the
        journal cannot say what became of them, so the runtime neither invents
        an outcome nor runs them again.
        """
        return self.state.incomplete_tools

    @property
    def needs_recovery(self) -> bool:
        return self.status is ExecutionStatus.RECOVERY_REQUIRED

    # -- idempotency (Milestone 4B) --------------------------------------------

    @property
    def idempotency(self) -> IdempotencyGuard:
        """The guard deciding whether a keyed call may run its tool."""
        return self._idempotency

    @property
    def pending_idempotency(self) -> tuple[IdempotencyRecord, ...]:
        """Every key this execution still holds a claim on.

        A claim with no recorded outcome. Each one is an external side effect
        whose result this database does not hold -- either still running, or
        interrupted by a crash.
        """
        return self._idempotency.pending_records(self._id)

    @property
    def unresolved_idempotency(self) -> tuple[IdempotencyRecord, ...]:
        """The pending keys that need an explicit decision.

        The subset of :attr:`pending_idempotency` with no authorized retry
        outstanding: for those, :meth:`call` raises
        :class:`~agent_runtime.exceptions.IdempotencyRecoveryRequiredError`
        rather than executing, and :attr:`status` reports ``RECOVERY_REQUIRED``.
        """
        return unresolved_records(self.pending_idempotency)

    @property
    def needs_idempotency_resolution(self) -> bool:
        """True when at least one key of this execution is unresolved."""
        return bool(self.unresolved_idempotency)

    def idempotency_record(self, key: str) -> IdempotencyRecord | None:
        """The store's record for ``key``, or ``None`` if nothing claimed it."""
        return self._idempotency.get(key)

    def resolve_idempotency(
        self,
        key: str,
        action: IdempotencyAction,
        *,
        result: Any = None,
        error: Any = None,
        note: str | None = None,
    ) -> IdempotencyRecord:
        """Settle one key explicitly, and store that decision durably.

        This is the *only* way out of a ``PENDING`` or ``FAILED`` key, and it is
        deliberately not something the runtime does by itself::

            execution.resolve_idempotency("email-123", action="retry")
            execution.resolve_idempotency("email-123", action="mark_completed", result=known)
            execution.resolve_idempotency("email-123", action="mark_failed", error="declined")

        ``retry``
            authorizes exactly one more execution of the key. The record stays
            ``PENDING`` -- nobody has said whether the effect happened -- and the
            authorization is consumed by the next :meth:`call` that uses the key.
        ``mark_completed``
            records an externally known successful result. Every later duplicate
            call is answered with it, without running the tool.
        ``mark_failed``
            records a known failure. The key is resolved in the sense that its
            outcome is known, but it still will not run again without an explicit
            ``retry``.

        Args:
            key: The idempotency key to settle.
            action: ``retry``, ``mark_completed`` or ``mark_failed``.
            result: The known result, for ``mark_completed``. Take it from the
                journal when the journal has it -- ``execution.tool_calls``
                already holds what the crashed attempt recorded.
            error: The known failure, for ``mark_failed``.
            note: An optional free-text note recorded with the decision.

        Returns:
            The updated :class:`~agent_runtime.idempotency.IdempotencyRecord`.

        Raises:
            UnknownIdempotencyKeyError: no such key was ever claimed.
            IdempotencyResolutionError: the action is unknown, the key is already
                ``COMPLETED``, or the key belongs to a different execution.
        """
        record = self._idempotency.get(key)
        if record is None:
            raise UnknownIdempotencyKeyError(
                f"Execution {self._id!r} has no idempotency record for key {key!r}",
                idempotency_key=str(key),
            )
        if record.execution_id != self._id:
            raise IdempotencyResolutionError(
                f"Idempotency key {key!r} is held by execution "
                f"{record.execution_id!r}, not {self._id!r}; resolve it there",
                idempotency_key=str(key),
                record=record,
            )
        settled = self._idempotency.resolve(
            key, action, result=result, error=error, note=note
        )
        self._maybe_auto_checkpoint()
        return settled

    @property
    def pending_retries(self) -> tuple[PendingRetry, ...]:
        """Retries the journal scheduled but had not started yet.

        Unlike :attr:`incomplete_tools`, these are decisions rather than
        ambiguities: each one names the attempt that comes next, so a resumed
        execution can carry it on with
        :meth:`continue_pending_retry` instead of asking the application to
        decide what happened to work whose outcome is unknown.
        """
        return detect_pending_retries(self.state)

    def reconstruct_state(self) -> ExecutionState:
        """Explicit alias for :attr:`state` (mirrors ``reconstruct_state(events)``)."""
        return self.state

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable snapshot of the reconstructed state."""
        state = self.state
        latest = self.latest_checkpoint
        return {
            "execution_id": state.execution_id,
            "status": str(state.status),
            "goal": state.goal,
            "last_sequence": state.last_sequence,
            "result": state.result,
            "error": state.error,
            "latest_checkpoint": None if latest is None else latest.sequence,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "tool": call.tool,
                    "arguments": dict(call.arguments),
                    "status": str(call.status),
                    "result": call.result,
                    "error": call.error,
                    "attempt": call.attempt,
                    "attempts": [item.to_dict() for item in call.attempts],
                    "idempotency_key": call.idempotency_key,
                    "timeout": call.timeout,
                    "timeout_mode": call.timeout_mode,
                    "pending_retry": (
                        None if call.pending_retry is None else call.pending_retry.to_dict()
                    ),
                }
                for call in state.tool_calls
            ],
            "incomplete_tools": [item.to_dict() for item in state.incomplete_tools],
            "pending_retries": [item.to_dict() for item in detect_pending_retries(state)],
            "pending_idempotency": [record.to_dict() for record in self.pending_idempotency],
        }

    # -- lifecycle -----------------------------------------------------------

    def call(
        self,
        name: str,
        *args: Any,
        retry_policy: RetryPolicy | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run a registered tool as one logical call, journalling every attempt.

        Emits ``ToolRequested`` once -- the logical call, with its stable
        ``call_id``, retry policy and deadline -- and then one start/outcome
        pair per attempt::

            ToolRequested
            ToolStarted   attempt=1 -> ToolCompleted attempt=1

        or, when the failure is retryable and attempts remain::

            ToolStarted          attempt=1 -> ToolFailed attempt=1
            ToolRetryScheduled   attempt=2      (journalled before the wait)
            ToolStarted          attempt=2 -> ToolCompleted attempt=2

        The policy is resolved call-level first, then tool-level, then "no
        retries" -- so ``execution.call("fetch_data", retry_policy=...)`` always
        wins over the one its tool declares. Only
        :class:`~agent_runtime.exceptions.RetryableToolError` is retried by
        default; a permanent or an unexpected failure is recorded once and
        raised as :class:`ToolInvocationError` after it is durably recorded.

        Timeouts (Milestone 4C)
        ------------------------

        ``timeout=`` sets a deadline for this call, in seconds, overriding the
        tool's ``@tool(timeout=...)`` default::

            execution.call("slow_tool", timeout=5.0)

        The deadline is journalled on ``ToolRequested`` and enforced by actually
        stopping the tool, never by measuring how long it took:

        ========================== ====================================
        ``async def`` tool         :func:`asyncio.wait_for` cancels it
        ``def`` with ``cancel_token``  the token is flipped, then awaited
        ``def`` without one        **refused** -- see below
        ========================== ====================================

        A plain synchronous tool is *rejected* rather than faked. Python cannot
        terminate a running thread, so a wrapper that timed the call and then
        said "timed out" would be claiming a stop that never happened; that
        raises :class:`~agent_runtime.exceptions.UnsupportedTimeoutError` before
        anything is journalled.

        A timeout is its own outcome, journalled ``ToolTimedOut`` rather than
        folded into ``ToolFailed``, and it is retried only when the policy says
        so (``RetryPolicy(retry_on_timeout=True)``). It is also *never* retried
        for a call carrying an idempotency key whose stop could not be enforced:
        the side effect may have happened, so the key stays ``PENDING`` and the
        execution reports ``RECOVERY_REQUIRED`` instead of running it again.

        Cancellation (Milestone 4C)
        ----------------------------

        ``execution.cancel()`` requests cancellation of the whole execution.
        A tool that declared ``cancel_token`` sees ``is_cancelled()`` /
        ``raise_if_cancelled()``; the attempt is journalled ``ToolCancelled``
        and **never** retried, because cancelling is a decision rather than a
        fault.

        The attempt itself is handed to a
        :class:`~agent_runtime.runner.ToolRunner`. In ``NORMAL`` mode that runs
        the function; replay supplies a ``REPLAY`` runner over a recorded
        history instead (see :mod:`agent_runtime.replay`), so both paths journal
        the same way and neither can diverge from the other.

        Idempotency (Milestone 4B)
        --------------------------

        ``idempotency_key`` names the side effect this call is allowed to
        perform, and is what stops a recovery from performing it twice::

            execution.call("send_email", to=..., idempotency_key="welcome-123")

        The key is claimed *before* the tool runs and its outcome stored *after*,
        and the key belongs to the **logical call**, not to an attempt: all three
        attempts above share one claim, because they are one intended side
        effect.

        What the runtime will and will not do with a key it finds:

        * no record       -> claim it (``PENDING``), run the tool, store the outcome;
        * ``COMPLETED``   -> return the stored result and **do not run the tool**;
        * ``PENDING``     -> raise :class:`IdempotencyRecoveryRequiredError`;
        * ``FAILED``      -> raise :class:`IdempotencyKeyFailedError`;
        * retry authorized -> consume the authorization and run the tool.

        A duplicate is still journalled, flagged ``deduplicated``: it is a real
        outcome of *this* execution and replaying the history has to reproduce
        it. See :meth:`resolve_idempotency` for the only way out of the two
        refusals.

        Raises:
            IdempotencyRecoveryRequiredError: the key is ``PENDING`` from a
                previous, unresolved attempt. The tool did not run.
            IdempotencyKeyFailedError: the key is recorded as ``FAILED``.
            ToolInvocationError: the tool itself failed for good.
            ToolCancelledError: the attempt was cancelled.
            ToolTimedOutError: the attempt ran out of time.
            UnsupportedTimeoutError: the deadline cannot be enforced for this
                tool. Raised before anything is journalled.
        """
        self._require_running("call a tool")

        # Resolve + validate before journalling: an unknown tool or a bad
        # argument list is a programming error, and nothing was attempted.
        # The deadline is resolved here too, so an unenforceable one is refused
        # before a call id exists rather than half way through an attempt.
        request = self._prepare_request(name, args, kwargs)
        policy = self._resolve_retry_policy(name, retry_policy)
        deadline = self._resolve_timeout(request.tool, timeout)
        call_id = self._new_call_id()
        self._runner.begin_call(call_id, request)

        if idempotency_key is None:
            self._emit_requested(request, call_id, policy, deadline=deadline)
            return self._attempt_until_settled(
                request, call_id, policy, deadline, attempt=1
            )

        # The claim is committed before the tool runs, and deliberately outside
        # any transaction that could cover it: the external call is not
        # undoable, so the intent has to be durable first. A crash after this
        # point leaves PENDING -- the honest answer, and the one recovery
        # refuses to guess about.
        decision = self._idempotency.begin_call(
            idempotency_key,
            execution_id=self._id,
            call_id=call_id,
            tool=request.tool,
            arguments=request.arguments,
        )
        if decision.deduplicated:
            return self._record_deduplicated(
                request, call_id, policy, idempotency_key, decision.result
            )

        self._emit_requested(
            request, call_id, policy, deadline=deadline, idempotency_key=idempotency_key
        )
        try:
            result = self._attempt_until_settled(
                request,
                call_id,
                policy,
                deadline,
                attempt=1,
                # Threaded through so an *unenforced* timeout on a keyed call
                # knows there is a side effect at risk, and refuses to retry.
                idempotency_key=idempotency_key,
            )
        except AmbiguousTimeoutError:
            # The tool outlived its deadline and the runtime cannot prove it
            # stopped, so the external effect may well have happened. The key is
            # deliberately left PENDING: recording FAILED would let the next
            # process believe the effect did not occur, which is precisely the
            # guess this runtime refuses to make.
            raise
        except ToolTimedOutError as exc:
            # A timeout the runtime *can* prove stopped is different. A coroutine
            # cancelled at its await really did stop, so the side effect did not
            # complete and FAILED is a known outcome rather than an ambiguity --
            # leaving the key PENDING here would invent a recovery problem that
            # nobody has, and block a legitimate retry behind it.
            if exc.enforced:
                self._idempotency.record_outcome(
                    idempotency_key, error=describe_error(exc)
                )
            raise
        except ToolCancelledError:
            # Same reasoning as an unenforced timeout: a cancelled attempt was
            # interrupted, not completed, and nobody has said the effect did not
            # happen.
            raise
        except Exception as exc:
            # A recorded failure is a known outcome, so the key can be settled as
            # FAILED. Only Exception, never BaseException: a KeyboardInterrupt or a
            # SIGKILL means "unknown", and unknown has to stay PENDING.
            self._idempotency.record_outcome(idempotency_key, error=describe_error(exc))
            raise
        self._idempotency.record_outcome(idempotency_key, result=result)
        return result

    def _emit_requested(
        self,
        request: ToolRequest,
        call_id: str,
        policy: RetryPolicy,
        *,
        deadline: ResolvedTimeout | None = None,
        idempotency_key: str | None = None,
        deduplicated: bool = False,
    ) -> None:
        """Journal ``ToolRequested``: one logical call, and its fixed identity.

        The idempotency key and the deadline both go in here and nowhere else,
        so the whole attempt history -- and every replay of it -- hangs off one
        claim and one deadline, no matter how many attempts get made.
        """
        payload: dict[str, Any] = {
            "call_id": call_id,
            "tool": request.tool,
            "arguments": dict(request.arguments),
            # Recorded so that the policy a call ran under -- and the
            # backoff schedule derived from it -- can be read back long
            # after the process that used it is gone.
            "retry_policy": policy.to_dict(),
        }
        if deadline is not None and deadline.active:
            # The deadline, and the mechanism that will enforce it. Both are
            # durable here so a resumed retry or a replay reuses the same rule
            # instead of re-deriving it from a tool that may have changed.
            #
            # Only written when there *is* a deadline: a call with no timeout
            # then produces exactly the ``ToolRequested`` Milestone 4A wrote, so
            # old and new journals stay directly comparable and an old one
            # replays byte for byte.
            payload["timeout"] = deadline.seconds
            payload["timeout_mode"] = str(deadline.mode)
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        if deduplicated:
            payload["deduplicated"] = True
        self._append(EventType.TOOL_REQUESTED, payload)

    def _record_deduplicated(
        self,
        request: ToolRequest,
        call_id: str,
        policy: RetryPolicy,
        idempotency_key: str,
        result: Any,
    ) -> Any:
        """Journal a call the idempotency store answered instead of running.

        No ``ToolStarted`` is written, because nothing started: the tool was not
        invoked, and saying otherwise would put a side effect in the journal that
        never happened. The call is settled with the result the first claim
        recorded, so the history of *this* execution stays complete -- and
        replayable.
        """
        self._emit_requested(
            request, call_id, policy, idempotency_key=idempotency_key, deduplicated=True
        )
        self._append(
            EventType.TOOL_COMPLETED,
            {
                "call_id": call_id,
                "tool": request.tool,
                "attempt": 1,
                "result": result,
                "idempotency_key": idempotency_key,
                "deduplicated": True,
            },
        )
        return result

    def continue_pending_retry(self, call_id: str) -> Any:
        """Carry on the attempt that a crash interrupted.

        A process can die after ``ToolRetryScheduled`` was committed and before
        the attempt it scheduled began. Nothing about that is ambiguous -- the
        journal says exactly which attempt comes next -- so a resumed execution
        can simply make it, using the policy the call was journalled with. The
        already-scheduled backoff is *not* waited for again: it elapsed while
        the process was gone.

        Args:
            call_id: The logical call whose scheduled attempt to run.

        Returns:
            The attempt's result.

        Raises:
            UnknownToolCallError: this execution has no such call.
            InvalidStateTransitionError: the call has no scheduled retry, or the
                execution has already finished.
            ToolInvocationError: this attempt failed for good.
            ToolTimedOutError: this attempt ran out of time.
            ToolCancelledError: the execution was cancelled before it started.
        """
        self._require_running(f"continue the retry of tool call {call_id!r}")
        call = next((c for c in self.tool_calls if c.call_id == call_id), None)
        if call is None:
            raise UnknownToolCallError(
                f"Execution {self._id!r} has no tool call {call_id!r} to retry"
            )
        if call.pending_retry is None:
            raise InvalidStateTransitionError(
                f"Cannot continue tool call {call_id!r}: the journal has no retry "
                f"scheduled for it (it is {call.status})"
            )
        request = ToolRequest(tool=call.tool, arguments=dict(call.arguments))
        self._runner.begin_call(call_id, request)
        return self._attempt_until_settled(
            request,
            call_id,
            self._policy_of_call(call_id),
            # The deadline is read back from the call's own ``ToolRequested``
            # rather than re-resolved: the attempt has to run under the same
            # rule the crashed process was using, or the recovered history
            # would describe a different call than the one that happened.
            self._timeout_of_call(call_id),
            attempt=call.pending_retry.attempt,
            idempotency_key=call.idempotency_key,
        )

    def _timeout_of_call(self, call_id: str) -> ResolvedTimeout:
        """The deadline a past call was journalled with, read from its request.

        The deadline has to come from durable data for the same reason the
        retry policy does: the object that resolved it is long gone, and
        re-deriving it from today's tool definition could silently enforce a
        different rule than the one the journal describes.
        """
        for event in reversed(self.events):
            if (
                event.event_type is EventType.TOOL_REQUESTED
                and event.payload.get("call_id") == call_id
            ):
                return ResolvedTimeout.from_dict(
                    {
                        "timeout": event.payload.get("timeout"),
                        "timeout_mode": event.payload.get("timeout_mode"),
                    }
                )
        return ResolvedTimeout()

    def _resolve_retry_policy(
        self, name: str, call_policy: RetryPolicy | None
    ) -> RetryPolicy:
        """The policy this call runs under: call-level, then tool-level, then none.

        The call-level policy wins outright -- it is the more specific statement
        about this one invocation. When neither says anything, the answer is
        :attr:`RetryPolicy.none`, so a tool call is never retried by accident.
        """
        if call_policy is not None:
            return check_retry_policy(call_policy, tool_name=name) or NO_RETRY
        if self._registry is not None:
            declared = self._registry.get(name).retry_policy
            if declared is not None:
                return declared
        return NO_RETRY

    def _resolve_timeout(
        self, name: str, call_timeout: float | None
    ) -> ResolvedTimeout:
        """The deadline this call runs under: call-level, then tool-level, then none.

        The call-level value wins outright, for the same reason a call-level
        retry policy does: it is the more specific statement about this one
        invocation. A deadline the runtime cannot actually enforce is refused
        here, before ``ToolRequested`` -- a call that cannot honour its own
        deadline should not get a call id at all.
        """
        if self._registry is None:
            return ResolvedTimeout(seconds=check_timeout(call_timeout), mode=TimeoutMode.UNSUPPORTED)
        target = self._registry.get(name)
        if call_timeout is not None:
            seconds = check_timeout(call_timeout, where=f"timeout for call to {name!r}")
        else:
            seconds = target.timeout
        return resolve_timeout(target.func, seconds, tool_name=target.name)

    def _policy_of_call(self, call_id: str) -> RetryPolicy:
        """The policy a past call was journalled with, read back from its request.

        Used to continue a retry after a crash: the policy has to come from
        durable data, because the object that resolved it is long gone.
        """
        for event in reversed(self.events):
            if (
                event.event_type is EventType.TOOL_REQUESTED
                and event.payload.get("call_id") == call_id
            ):
                return RetryPolicy.from_dict(event.payload.get("retry_policy"))
        return NO_RETRY

    def _attempt_until_settled(
        self,
        request: ToolRequest,
        call_id: str,
        policy: RetryPolicy,
        deadline: ResolvedTimeout,
        *,
        attempt: int,
        idempotency_key: str | None = None,
    ) -> Any:
        """Run the attempts of one logical call until it settles; return its result.

        Each attempt is journalled before it happens and its outcome after, and
        a retry is journalled before the wait -- so the journal always describes
        exactly as much as the process actually did, no matter when it died.

        Three stops, three events, three rules (Milestone 4C):

        * a **failure** may be retried, per the policy;
        * a **timeout** may be retried only when the policy opted in, and never
          when the call carries an idempotency key whose stop could not be
          enforced -- there the side effect may have happened, so the key is
          left alone and the call ends;
        * a **cancellation** is never retried, and it also interrupts the wait
          before the next attempt rather than sleeping it out.
        """
        while True:
            if self._cancel_token.is_cancelled():
                # Cancellation arrived between attempts (during a backoff, or
                # while a sibling call was stopping). The call settles as
                # cancelled without ever being attempted -- a retry here would
                # re-issue a decision the application already made.
                self._finish_cancelled(
                    call_id,
                    request.tool,
                    attempt,
                    reason=self._cancel_token.reason,
                )
                raise self._cancelled_error(
                    call_id, request.tool, self._cancel_token.reason, attempt
                )

            if not self._runner.can_attempt(call_id, attempt):
                # Only a replay gets here: the recorded history ends between the
                # scheduled retry and its attempt, so there is nothing to serve
                # and nothing to invent. The call is left exactly as durable as
                # the journal already made it.
                return None

            self._append(
                EventType.TOOL_STARTED,
                {"call_id": call_id, "tool": request.tool, "attempt": attempt},
            )
            outcome = self._runner.run(
                request, timeout=deadline, token=self._cancel_token
            )

            if outcome.succeeded:
                # A tool that returned *because* it was stopped must not have its
                # result recorded. ``cancel()`` from another thread can land
                # between the tool finishing and this line, and a
                # ``ToolCompleted`` written afterwards would silently undo a
                # durable ``ToolCancelled`` -- the call would look successful
                # while the execution says CANCELLED. The token is the authority
                # on whether this attempt was stopped, not the return value.
                if self._cancel_token.is_cancelled():
                    self._finish_cancelled(call_id, request.tool, attempt)
                    raise self._cancelled_error(
                        call_id, request.tool, self._cancel_token.reason, attempt
                    )
                self._append(
                    EventType.TOOL_COMPLETED,
                    {
                        "call_id": call_id,
                        "tool": request.tool,
                        "attempt": attempt,
                        "result": outcome.result,
                    },
                )
                return outcome.result

            if outcome.cancelled:
                # Journalled as ToolCancelled -- never as a failure. Cancellation
                # is a decision, and the journal has to say so plainly enough
                # that a recovery leaves it alone instead of retrying it.
                self._finish_cancelled(
                    call_id,
                    request.tool,
                    attempt,
                    reason=(outcome.error or {}).get("message"),
                )
                error = self._runner.invocation_error(request, outcome)
                error.call_id = call_id
                error.attempts = attempt
                if outcome.cause is not None:
                    raise error from outcome.cause
                raise error

            if outcome.timed_out:
                return self._after_timeout(
                    request,
                    call_id,
                    policy,
                    deadline,
                    attempt,
                    outcome,
                    idempotency_key,
                )

            self._append(
                EventType.TOOL_FAILED,
                {
                    "call_id": call_id,
                    "tool": request.tool,
                    "attempt": attempt,
                    "error": dict(outcome.error or {}),
                },
            )

            decision = self._runner.retry_decision(
                request, outcome, call_id=call_id, attempt=attempt, policy=policy
            )
            if decision is None:
                error = self._runner.invocation_error(request, outcome)
                error.call_id = call_id
                error.attempts = attempt
                if outcome.cause is not None:
                    raise error from outcome.cause
                raise error

            if not self._schedule_retry(call_id, request.tool, decision):
                # Cancelled during the backoff (§9). The wait was interrupted
                # and the decision is durable, so the call settles as cancelled
                # rather than continuing to a retry nobody wants any more.
                return self._after_cancellation_during_backoff(
                    call_id, request.tool, decision
                )
            attempt = decision.attempt

    def _after_timeout(
        self,
        request: ToolRequest,
        call_id: str,
        policy: RetryPolicy,
        deadline: ResolvedTimeout,
        attempt: int,
        outcome: Any,
        idempotency_key: str | None,
    ) -> Any:
        """Journal ``ToolTimedOut`` and decide whether another attempt follows.

        The idempotency check is why this is a method and not a few lines in the
        loop. A timeout whose stop was *not* enforced means the external effect
        may have happened while the runtime was trying to stop a thread.
        Retrying would perform the side effect a second time, so the runtime
        stops here and leaves the key ``PENDING``: the ambiguity is reported,
        not resolved.
        """
        self._append(
            EventType.TOOL_TIMED_OUT,
            {
                "call_id": call_id,
                "tool": request.tool,
                "attempt": attempt,
                "error": dict(outcome.error or {}),
                # The three facts that make a timeout readable after a restart.
                "timeout": outcome.timeout,
                "timeout_mode": outcome.timeout_mode,
                "enforced": outcome.timeout_enforced,
            },
        )

        if outcome.ambiguous and idempotency_key is not None:
            inner = self._runner.invocation_error(request, outcome)
            inner.call_id = call_id
            inner.attempts = attempt
            raise AmbiguousTimeoutError(
                f"Tool {request.tool!r} timed out and the runtime could not stop it, "
                f"so the side effect of idempotency key {idempotency_key!r} may already "
                "have happened. The call was not retried. Check what actually "
                "happened and settle the key with resolve_idempotency(...).",
                idempotency_key=idempotency_key,
                tool_name=request.tool,
            ) from inner

        decision = self._runner.retry_decision(
            request, outcome, call_id=call_id, attempt=attempt, policy=policy
        )
        if decision is None:
            error = self._runner.invocation_error(request, outcome)
            error.call_id = call_id
            error.attempts = attempt
            if outcome.cause is not None:
                raise error from outcome.cause
            raise error

        if not self._schedule_retry(call_id, request.tool, decision):
            return self._after_cancellation_during_backoff(
                call_id, request.tool, decision
            )
        return self._attempt_until_settled(
            request,
            call_id,
            policy,
            deadline,
            attempt=decision.attempt,
            idempotency_key=idempotency_key,
        )

    def _finish_cancelled(
        self, call_id: str, tool: str, attempt: int, *, reason: Any = None
    ) -> None:
        """Settle a call as cancelled, exactly once.

        Idempotent per call on purpose. ``cancel()`` runs on the cancelling
        thread and journals ``ToolCancelled`` for whichever call is open, while
        the call itself is still running and will reach its own cancellation a
        moment later. Two events for one stop would make the history describe
        something that did not happen -- and would let a replay serve an attempt
        that the journal already settled.
        """
        if self._is_settled_as_cancelled(call_id):
            return
        self._settle_cancelled(call_id, tool, attempt=attempt, reason=reason)

    def _is_settled_as_cancelled(self, call_id: str) -> bool:
        """Whether this call already has its durable ``ToolCancelled``."""
        for event in reversed(self.events):
            if (
                event.event_type is EventType.TOOL_CANCELLED
                and event.payload.get("call_id") == call_id
            ):
                return True
        return False

    def _settle_cancelled(
        self, call_id: str, tool: str, *, attempt: int, reason: Any = None
    ) -> None:
        """Journal ``ToolCancelled`` for an attempt that was stopped on purpose.

        The event is a first-class fact rather than a ``ToolFailed`` with a
        message: it is what tells recovery that this call must not be resumed,
        and what tells a replay not to invent a result for it.
        """
        note = make_jsonable(reason)
        self._append(
            EventType.TOOL_CANCELLED,
            {
                "call_id": call_id,
                "tool": tool,
                "attempt": attempt,
                "reason": note,
                "error": {
                    "type": "ToolCancelledError",
                    "message": note or "cancelled",
                    "traceback": None,
                },
            },
        )

    def _cancelled_error(
        self, call_id: str, tool: str, reason: Any, attempt: int = 0
    ) -> ToolCancelledError:
        """The cancellation error a caller sees, with the call it belongs to.

        ``attempts`` is the number the logical call had made, so a caller
        catching this has the same information a caller catching
        :class:`ToolInvocationError` gets from a failure.
        """
        error = ToolCancelledError(
            f"Tool {tool!r} was not attempted: execution {self._id!r} was cancelled",
            reason=reason,
            tool_name=tool,
            call_id=call_id,
            attempts=attempt,
        )
        error.call_id = call_id
        error.attempts = attempt
        return error

    def _after_cancellation_during_backoff(
        self, call_id: str, tool: str, decision: RetryDecision
    ) -> Any:
        """Settle a call whose scheduled retry was cancelled before it ran.

        The retry was already journalled, so the history says a retry *was*
        scheduled; cancellation then says it will not happen. Both are durable
        and the reducers read the second as the final word, so the call ends
        ``CANCELLED`` with no attempt recorded -- exactly what happened. The
        consumed pending slot is gone with it, so nothing can resume it.
        """
        # ``failed_attempt``, not ``attempt``: the attempt the cancelled decision
        # named never started, so the journal has no ``ToolStarted`` for it, and
        # recording it would put a phantom attempt in the folded state.
        self._finish_cancelled(
            call_id,
            tool,
            decision.failed_attempt or 1,
            reason=self._cancel_token.reason,
        )
        # ``attempts`` likewise means how many were *made*.
        raise self._cancelled_error(
            call_id, tool, self._cancel_token.reason, decision.failed_attempt or 1
        )

    def _schedule_retry(
        self, call_id: str, tool: str, decision: RetryDecision
    ) -> bool:
        """Journal the retry decision, then wait for it; ``False`` if cancelled.

        The order is the point: ``ToolRetryScheduled`` is committed *before* the
        sleep, so a process that dies while waiting leaves the decision
        recoverable instead of losing it with the sleep.

        Milestone 4C: the wait itself goes through the sleeper's
        ``interruptible_sleep``, so a cancellation that arrives during a
        ten-second backoff ends it immediately rather than being noticed ten
        seconds later. The ``False`` return is what the caller uses to stop
        instead of starting the attempt the decision named.
        """
        self._append(
            EventType.TOOL_RETRY_SCHEDULED,
            {
                "call_id": call_id,
                "tool": tool,
                "attempt": decision.attempt,
                "failed_attempt": decision.failed_attempt,
                "delay": decision.delay,
                "reason": decision.reason,
                "error": dict(decision.error or {}),
            },
        )
        return not wait_interrupted(
            self._sleeper, decision.delay, self._cancel_token
        )

    def complete(self, result: Any = None) -> None:
        """Mark the execution COMPLETED and journal ``ExecutionCompleted``.

        Refused while an idempotency key is unresolved (Milestone 4B): a side
        effect of unknown outcome is not something to close out quietly. Settle
        the keys with :meth:`resolve_idempotency` -- or give up on the execution
        with :meth:`fail` / :meth:`mark_cancelled`, which are allowed precisely
        because they are decisions.
        """
        self._require_running("complete")
        unresolved = self.unresolved_idempotency
        if unresolved:
            keys = ", ".join(record.idempotency_key for record in unresolved)
            raise InvalidStateTransitionError(
                f"Cannot complete: execution {self._id!r} has unresolved idempotency "
                f"key(s) {keys}; settle them with resolve_idempotency(...) first"
            )
        self._append(EventType.EXECUTION_COMPLETED, {"result": make_jsonable(result)})
        self._maybe_auto_checkpoint()

    def fail(self, error: BaseException | str | None = None) -> None:
        """Mark the execution FAILED and journal ``ExecutionFailed``.

        This is also a way out of ``RECOVERY_REQUIRED``: an execution whose tool
        calls a crash left open can be given up on, and that decision is
        journalled like any other.
        """
        self._require_active("fail")
        self._append(EventType.EXECUTION_FAILED, {"error": describe_error(error)})
        self._maybe_auto_checkpoint()

    def mark_cancelled(self, reason: Any = None) -> None:
        """Mark the execution CANCELLED and journal ``ExecutionCancelled``."""
        self._require_active("cancel")
        self._append(EventType.EXECUTION_CANCELLED, {"result": make_jsonable(reason)})
        self._maybe_auto_checkpoint()

    def cancel(self, reason: Any = None) -> None:
        """Request cancellation of this execution, durably and at once.

        Milestone 4C's cancellation entry point, and the one place the decision
        becomes a fact::

            execution.cancel("user pressed stop")

        What happens, in order:

        1. the execution's :attr:`cancel_token` is flipped, so a tool that is
           blocked right now is interrupted -- and so is a retry backoff, which
           is waited on that token and therefore stops immediately rather than
           sleeping out its remaining seconds;
        2. ``ToolCancelled`` is journalled for any call whose attempt is
           currently open, if there is one;
        3. ``ExecutionCancelled`` is journalled, so the decision survives the
           process.

        Because step 3 is durable, a restart reports ``CANCELLED`` and does
        nothing else: a cancelled execution is not resumed, and its cancelled
        call is not retried. See :attr:`RecoveryState` for how that is
        classified.

        Safe to call from another thread -- that is the usual case, since the
        thread being cancelled is the one inside the tool. Safe to call twice:
        the second call finds the token already cancelled and journals nothing
        new, so a history never grows two ``ExecutionCancelled`` events for one
        decision.

        Args:
            reason: Why. Recorded on ``ToolCancelled`` and ``ExecutionCancelled``
                so a reader can tell a deliberate stop from a timeout.

        Raises:
            InvalidStateTransitionError: the execution already finished as
                something other than ``CANCELLED``.
        """
        if self.state.status is ExecutionStatus.CANCELLED:
            # Already cancelled, durably. A second ``cancel()`` must not append
            # a second cancellation -- the decision is the same decision.
            self._cancel_token.cancel(reason)
            return
        self._require_active("cancel")
        already = self._cancel_token.cancel(reason)
        if not already:
            # Cancelled by an in-flight call's own deadline or an earlier
            # ``cancel()`` on this object; the journal has been written.
            return
        open_call = self._open_call()
        if open_call is not None:
            self._finish_cancelled(
                open_call.call_id,
                open_call.tool,
                max(open_call.attempt, 1),
                reason=reason,
            )
        self._append(EventType.EXECUTION_CANCELLED, {"result": make_jsonable(reason)})
        self._maybe_auto_checkpoint()

    def _open_call(self) -> ToolCall | None:
        """The call whose attempt is currently open, if any.

        "Open" means started and not yet settled -- the one a cancellation has
        to stop. A call that already completed, timed out or failed is not open,
        and neither is one that is merely waiting on a retry, because that
        attempt has not started.
        """
        return next(
            (call for call in self.state.tool_calls if call.status.is_incomplete and call.status is ToolCallStatus.STARTED),
            None,
        )

    # -- checkpoints ---------------------------------------------------------

    @property
    def checkpoints(self) -> CheckpointStore:
        """The checkpoint store this execution reads and writes."""
        if self._checkpoints is None:
            self._checkpoints = CheckpointStore(self._journal.store)
        return self._checkpoints

    @property
    def latest_checkpoint(self) -> Checkpoint | None:
        """The newest stored checkpoint, or ``None`` when the execution has none."""
        return self.checkpoints.get_latest(self._id)

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoints.has_checkpoints(self._id)

    def checkpoint(self) -> Checkpoint:
        """Persist the current state as a recoverable snapshot.

        The snapshot describes this execution immediately after its latest
        persisted event, and that sequence is written together with the state in
        a single transaction -- so a checkpoint can never claim a sequence the
        history has not reached, nor hold a state that does not match the events
        behind it. A mismatch is refused instead of stored.

        Returns the stored :class:`~agent_runtime.checkpoints.Checkpoint`.
        """
        state = self.state
        if self.last_event_sequence != state.last_sequence:
            # Another writer moved the journal on since the state was loaded;
            # rebuild so the snapshot and the sequence describe the same prefix.
            state = reconstruct_state(self.events)
        return self.checkpoints.create(self._id, state)

    # -- recovery ------------------------------------------------------------

    def recovery_info(self) -> RecoveryInfo:
        """Describe what recovery makes of this execution's history.

        Works on a live execution and on a resumed one, and changes nothing::

            print(execution.recovery_info())

        The state it reports is the one :attr:`state` holds: the latest
        checkpoint plus the events durably persisted after it, plus the
        idempotency claims still in flight (Milestone 4B).
        """
        return recover_execution(
            self._journal, self.checkpoints, self._id, idempotency=self._idempotency
        )

    def resolve_recovery(
        self,
        call_id: str,
        action: RecoveryAction,
        *,
        result: Any = None,
        error: Any = None,
    ) -> Checkpoint | None:
        """Settle one tool call a crash left open, and journal that decision.

        Milestone 2 retries nothing by itself -- the caller has already decided.
        What this does is make that decision durable, so the next resume does not
        raise the same question again:

        ================= ===================================================
        ``mark_completed`` the work did finish; journalled as
                          ``ToolCompleted`` carrying ``result``
        ``mark_failed``   the call is given up on; journalled as
                          ``ToolFailed`` carrying ``error``
        ``cancel``        the call was abandoned; journalled as
                          ``ToolCancelled``
        ================= ===================================================

        Settling the last open call moves the execution out of
        ``RECOVERY_REQUIRED`` and back to ``RUNNING``, so it can carry on.
        Re-running an unfinished call is deliberately not offered here: a
        :class:`~agent_runtime.retry.RetryPolicy` retries *recorded failures*,
        never work whose outcome the journal does not know, so resuming this one
        still requires a decision from the application.

        Returns a fresh checkpoint of the resolved state, or ``None`` when
        automatic checkpointing is off.
        """
        self._require_active(f"resolve tool call {call_id!r}")
        try:
            event_type = _ACTION_EVENTS[action]
        except KeyError:
            raise InvalidRecoveryActionError(
                f"Unknown recovery action {action!r} for call {call_id!r}. Supported "
                f"actions: {', '.join(sorted(_ACTION_EVENTS))}"
            ) from None

        call = next((c for c in self.state.tool_calls if c.call_id == call_id), None)
        if call is None:
            raise UnknownToolCallError(
                f"Execution {self._id!r} has no tool call {call_id!r} to resolve"
            )
        if not call.status.is_incomplete:
            hint = (
                "; it has a retry scheduled instead, so carry it on with "
                "continue_pending_retry(...)"
                if call.pending_retry is not None
                else ""
            )
            raise InvalidRecoveryActionError(
                f"Tool call {call_id!r} of execution {self._id!r} is already "
                f"{call.status} and needs no recovery{hint}"
            )

        payload: dict[str, Any] = {
            "call_id": call_id,
            "tool": call.tool,
            "resolution": action,
            # The outcome belongs to the attempt that was left open, so the
            # attempt history keeps making sense after a resolution.
            "attempt": max(call.attempt, 1),
        }
        if event_type is EventType.TOOL_COMPLETED:
            payload["result"] = make_jsonable(result)
        else:
            payload["error"] = make_jsonable(error) or _ACTION_RESULTS[action]
        self._append(event_type, payload)
        return self._maybe_auto_checkpoint()

    # -- internals -----------------------------------------------------------

    def _prepare_request(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ToolRequest:
        """Resolve a call into a request, validating it before anything is journalled.

        An unknown tool or a bad argument list is a programming error, so it
        raises here -- before ``ToolRequested`` -- and nothing was attempted.

        A seam: replay overrides this to normalize against the *recorded*
        arguments instead of the registered signature, so a replay does not
        require the tool function to be importable at all.
        """
        target = self._registry.get(name)
        bound = target.bind(args, kwargs)
        return ToolRequest(tool=target.name, arguments=target.arguments_for(bound))

    def _new_call_id(self) -> str:
        """Identity for the next tool call.

        A seam: replay overrides this so the call it replays keeps the identity
        the journal already gave it, which is what makes a replayed state
        directly comparable with the original.
        """
        return new_id("call")

    def _append(self, event_type: EventType, payload: dict[str, Any]) -> Event:
        """Journal one event and drop the cached state it invalidated."""
        event = self._journal.append_event(self._id, event_type, payload)
        self._state = None
        return event

    def _maybe_auto_checkpoint(self) -> Checkpoint | None:
        """Snapshot the state after a lifecycle write when the runtime asked for it.

        Opt-in (``Runtime(..., auto_checkpoint=True)``), and deliberately at
        boundaries only -- ``complete``, ``fail``, ``mark_cancelled`` and
        recovery resolutions -- not after every tool call. A snapshot taken while
        a call is half-finished is mostly a snapshot of an ambiguity, and paying
        an extra write per call for it is rarely worth it.

        Either way this is only an optimisation: the state a snapshot holds is
        re-derived from the events on the next resume.
        """
        if not self._auto_checkpoint:
            return None
        return self.checkpoint()

    def _require_running(self, action: str) -> None:
        """The *journal's* verdict on whether this execution may do new work.

        Deliberately :attr:`state`-derived rather than :attr:`status`-derived. An
        unresolved idempotency key must not stop an application from asking for
        that key again -- asking is how the runtime gets to answer with
        :class:`~agent_runtime.exceptions.IdempotencyRecoveryRequiredError` and
        its instructions, instead of a bare "this execution is
        ``RECOVERY_REQUIRED``". An open tool call still blocks new work, exactly
        as in Milestone 2.
        """
        self._require_status(
            action, ExecutionStatus.RUNNING, status=self.state.status
        )

    def _require_active(self, action: str) -> None:
        """Allow anything that is not already finished, including recovery."""
        self._require_status(action, None)

    def _require_status(
        self,
        action: str,
        required: ExecutionStatus | None,
        *,
        status: ExecutionStatus | None = None,
    ) -> None:
        current = self.status if status is None else status
        if required is not None and current is not required:
            raise InvalidStateTransitionError(
                f"Cannot {action}: execution {self._id!r} is {current}"
            )
        if required is None and current.is_terminal:
            raise InvalidStateTransitionError(
                f"Cannot {action}: execution {self._id!r} already finished as {current}"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Execution(id={self._id!r}, status={self.status})"

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        lines = [f"Execution {self._id} [{self.status}] goal={self.goal!r}"]
        lines.extend(f"  {call}" for call in self.tool_calls)
        for item in self.incomplete_tools:
            lines.append(f"  ! {item} [{item.status}] stuck at sequence {item.sequence}")
        for retry in self.pending_retries:
            lines.append(f"  ~ {retry} (recorded at sequence {retry.sequence})")
        for record in self.pending_idempotency:
            note = "retry authorized" if record.retry_authorized else "UNRESOLVED"
            lines.append(f"  * key {record.idempotency_key} [{note}] {record}")
        return "\n".join(lines)
