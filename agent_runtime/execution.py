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
"""

from __future__ import annotations

from typing import Any, Literal

from .checkpoints import Checkpoint, CheckpointStore
from .events import Event, EventType, describe_error, new_id
from .exceptions import (
    InvalidRecoveryActionError,
    InvalidStateTransitionError,
    UnknownToolCallError,
)
from .journal import EventJournal
from .recovery import RecoveryInfo, recover_execution
from .retry import (
    NO_RETRY,
    RealSleeper,
    RetryDecision,
    RetryPolicy,
    Sleeper,
)
from .runner import ToolRequest, ToolRunner
from .state import (
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    PendingRetry,
    ToolCall,
    detect_pending_retries,
    reconstruct_state,
)
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
        #: How retry backoff waits. Injectable so a test can assert the
        #: schedule without spending it -- nothing here patches ``time.sleep``.
        self._sleeper = sleeper if sleeper is not None else RealSleeper()

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
        return self.state.status

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
                    "pending_retry": (
                        None if call.pending_retry is None else call.pending_retry.to_dict()
                    ),
                }
                for call in state.tool_calls
            ],
            "incomplete_tools": [item.to_dict() for item in state.incomplete_tools],
            "pending_retries": [item.to_dict() for item in detect_pending_retries(state)],
        }

    # -- lifecycle -----------------------------------------------------------

    def call(
        self,
        name: str,
        *args: Any,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run a registered tool as one logical call, journalling every attempt.

        Emits ``ToolRequested`` once -- the logical call, with its stable
        ``call_id`` and the resolved retry policy -- and then one start/outcome
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

        The attempt itself is handed to a
        :class:`~agent_runtime.runner.ToolRunner`. In ``NORMAL`` mode that runs
        the function; replay supplies a ``REPLAY`` runner over a recorded
        history instead (see :mod:`agent_runtime.replay`), so both paths journal
        the same way and neither can diverge from the other.
        """
        self._require_running("call a tool")

        # Resolve + validate before journalling: an unknown tool or a bad
        # argument list is a programming error, and nothing was attempted.
        request = self._prepare_request(name, args, kwargs)
        policy = self._resolve_retry_policy(name, retry_policy)

        call_id = self._new_call_id()
        self._runner.begin_call(call_id, request)
        self._append(
            EventType.TOOL_REQUESTED,
            {
                "call_id": call_id,
                "tool": request.tool,
                "arguments": request.arguments,
                # Recorded so that the policy a call ran under -- and the
                # backoff schedule derived from it -- can be read back long
                # after the process that used it is gone.
                "retry_policy": policy.to_dict(),
            },
        )
        return self._attempt_until_settled(request, call_id, policy, attempt=1)

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
            attempt=call.pending_retry.attempt,
        )

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
        *,
        attempt: int,
    ) -> Any:
        """Run the attempts of one logical call until it settles; return its result.

        Each attempt is journalled before it happens and its outcome after, and
        a retry is journalled before the wait -- so the journal always describes
        exactly as much as the process actually did, no matter when it died.
        """
        while True:
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
            outcome = self._runner.run(request)

            if outcome.succeeded:
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

            self._schedule_retry(call_id, request.tool, decision)
            attempt = decision.attempt

    def _schedule_retry(
        self, call_id: str, tool: str, decision: RetryDecision
    ) -> None:
        """Journal the retry decision, then wait for it.

        The order is the point: ``ToolRetryScheduled`` is committed *before* the
        sleep, so a process that dies while waiting leaves the decision
        recoverable instead of losing it with the sleep.
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
        self._sleeper.sleep(decision.delay)

    def complete(self, result: Any = None) -> None:
        """Mark the execution COMPLETED and journal ``ExecutionCompleted``."""
        self._require_running("complete")
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
        checkpoint plus the events durably persisted after it.
        """
        return recover_execution(self._journal, self.checkpoints, self._id)

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
        self._require_status(action, ExecutionStatus.RUNNING)

    def _require_active(self, action: str) -> None:
        """Allow anything that is not already finished, including recovery."""
        self._require_status(action, None)

    def _require_status(self, action: str, required: ExecutionStatus | None) -> None:
        status = self.status
        if required is not None and status is not required:
            raise InvalidStateTransitionError(
                f"Cannot {action}: execution {self._id!r} is {status}"
            )
        if required is None and status.is_terminal:
            raise InvalidStateTransitionError(
                f"Cannot {action}: execution {self._id!r} already finished as {status}"
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
        return "\n".join(lines)
