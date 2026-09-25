"""Execution state, reconstructed purely from the event journal.

Nothing in here keeps state between calls: :func:`reconstruct_state` folds the
ordered event stream through :func:`apply_event` and returns the result. The
runtime treats the journal as the source of truth and derives every read from
it -- which is also what makes a checkpoint safe, because a checkpoint is just
this same value captured at one sequence.

Milestone 2 adds the two things recovery needs in order to be honest:

* the sequence each tool call was last seen at, so an unfinished call can be
  reported with the event it stopped at (:func:`detect_incomplete_tools`);
* the statuses a reconstructed history can be *in* rather than only one it
  decided on -- ``CANCELLED`` for a deliberate stop, and
  ``RECOVERY_REQUIRED`` when the journal ends with work still open
  (:func:`resolve_status`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Iterable, Mapping

from .events import Event, EventType
from .exceptions import StateReconstructionError

__all__ = [
    "ExecutionStatus",
    "ToolCallStatus",
    "ToolAttempt",
    "PendingRetry",
    "ToolCall",
    "IncompleteTool",
    "ExecutionState",
    "RecoveryState",
    "TERMINAL_STATUSES",
    "INCOMPLETE_TOOL_STATUSES",
    "initial_state",
    "apply_event",
    "reconstruct_state",
    "detect_incomplete_tools",
    "detect_pending_retries",
    "resolve_status",
    "classify_recovery",
    "finalize_state",
    "replace_state",
]


class ExecutionStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    #: The journal ends with tool work that has no recorded outcome. The
    #: ambiguity is surfaced instead of guessed at -- nothing is retried.
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

    @property
    def is_terminal(self) -> bool:
        """True when the history recorded a final outcome for the execution."""
        return self in TERMINAL_STATUSES


class ToolCallStatus(StrEnum):
    REQUESTED = "REQUESTED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    #: An attempt failed, and the journal recorded that another one is
    #: scheduled. Not incomplete: nothing is ambiguous about it -- the events
    #: say exactly what happens next -- so it does not make an execution
    #: ``RECOVERY_REQUIRED``. It is transient: the next ``ToolStarted``
    #: consumes it.
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"
    #: An attempt's deadline expired (Milestone 4C). Deliberately *not*
    #: ``FAILED``: a timeout says the call ran out of time, and an application
    #: reading a recovered state has to be able to tell that from "the tool
    #: raised" without inspecting a message. It is a settled status -- the
    #: journal recorded what happened -- so it is not an ambiguity either; what
    #: the attempt's ``error.enforced`` flag says is whether the side effect
    #: itself is known to have stopped.
    TIMED_OUT = "TIMED_OUT"

    @property
    def is_incomplete(self) -> bool:
        """True while the journal still says nothing about the outcome."""
        return self in INCOMPLETE_TOOL_STATUSES

    @property
    def is_stop(self) -> bool:
        """True for the three ways a call ends without a result.

        Milestone 4C: failure, timeout and cancellation are deliberately three
        statuses rather than one. This predicate exists for the code that wants
        "the call ended badly" without caring which -- never for a decision
        that depends on the difference.
        """
        return self in (
            ToolCallStatus.FAILED,
            ToolCallStatus.TIMED_OUT,
            ToolCallStatus.CANCELLED,
        )


#: Statuses that mean the execution has finished, one way or another.
TERMINAL_STATUSES: frozenset[ExecutionStatus] = frozenset(
    {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    }
)

#: Tool call statuses that leave the call open for recovery to resolve.
INCOMPLETE_TOOL_STATUSES: frozenset[ToolCallStatus] = frozenset(
    {ToolCallStatus.REQUESTED, ToolCallStatus.STARTED}
)


@dataclass(frozen=True, slots=True)
class ToolAttempt:
    """One concrete execution attempt of a logical tool call.

    Milestone 4A separates the two things Milestone 1 called "a tool call":

    * the **logical call** -- one :class:`ToolCall`, one stable ``call_id``;
    * an **attempt** -- one invocation, numbered from 1.

    A call that fails and is retried therefore owns several of these while
    still being a single call: the attempt history is what the journal records,
    and this is the folded view of it.
    """

    attempt: int
    status: ToolCallStatus = ToolCallStatus.STARTED
    started_sequence: int = 0
    completed_sequence: int = 0
    started_at: str | None = None
    result: Any = None
    error: Mapping[str, Any] | None = None
    #: The retry this attempt's failure scheduled, if it scheduled one.
    #:
    #: Kept on the *attempt* rather than only on the call, because the call's
    #: pending slot is consumed as soon as the next attempt starts. The
    #: decision itself must survive that: a replay reads it back from here to
    #: reproduce the recorded retry rather than making its own.
    scheduled_retry: PendingRetry | None = None

    @property
    def settled(self) -> bool:
        """True when this attempt's outcome was recorded (not left open)."""
        return not self.status.is_incomplete

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "status": str(self.status),
            "started_sequence": self.started_sequence,
            "completed_sequence": self.completed_sequence,
            "started_at": self.started_at,
            "result": self.result,
            "error": self.error,
            "scheduled_retry": (
                None if self.scheduled_retry is None else self.scheduled_retry.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolAttempt":
        scheduled = data.get("scheduled_retry")
        return cls(
            attempt=int(data.get("attempt") or 1),
            status=ToolCallStatus(data.get("status") or ToolCallStatus.STARTED),
            started_sequence=int(data.get("started_sequence") or 0),
            completed_sequence=int(data.get("completed_sequence") or 0),
            started_at=data.get("started_at"),
            result=data.get("result"),
            error=data.get("error"),
            scheduled_retry=None if scheduled is None else PendingRetry.from_dict(scheduled),
        )

    def __str__(self) -> str:
        if self.status is ToolCallStatus.COMPLETED:
            return f"attempt {self.attempt} -> {self.result!r}"
        if self.status is ToolCallStatus.FAILED:
            message = (self.error or {}).get("message", "failed")
            return f"attempt {self.attempt} !! {message}"
        return f"attempt {self.attempt} [{self.status}]"


@dataclass(frozen=True, slots=True)
class PendingRetry:
    """A retry the journal recorded but had not carried out yet.

    Produced by ``ToolRetryScheduled`` and consumed by the next
    ``ToolStarted``. It exists as its own value because a process can die in
    exactly that window -- after the decision was made durable, before the
    attempt began -- and a resumed execution has to be able to say "a retry was
    scheduled for attempt 2" rather than pretending the call simply failed.
    """

    #: The attempt that was scheduled (the ``attempt=2`` of the event).
    attempt: int
    #: The attempt that failed and caused it.
    failed_attempt: int = 0
    #: Seconds the retry was to wait before starting.
    delay: float = 0.0
    #: Why the retry was allowed -- the ``ErrorKind`` that permitted it.
    reason: str = ""
    #: The failure that triggered it.
    error: Mapping[str, Any] | None = None
    #: Where in the journal the decision was recorded.
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "failed_attempt": self.failed_attempt,
            "delay": self.delay,
            "reason": self.reason,
            "error": self.error,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingRetry":
        return cls(
            attempt=int(data.get("attempt") or 1),
            failed_attempt=int(data.get("failed_attempt") or 0),
            delay=float(data.get("delay") or 0.0),
            reason=str(data.get("reason") or ""),
            error=data.get("error"),
            sequence=int(data.get("sequence") or 0),
        )

    def __str__(self) -> str:
        return (
            f"attempt {self.failed_attempt or '?'} failed; retry {self.attempt} "
            f"scheduled after {self.delay}s"
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call, as reconstructed from its events.

    A logical call, however many attempts it took: ``call_id`` is stable across
    retries, ``attempt`` is the final attempt number, and :attr:`attempts` is
    the folded history of how it got there. The journal remains the source of
    truth for that history -- this is the same view the reducers produce from
    it, in the same way a checkpoint is.

    The ``*_sequence`` fields record *where in the journal* each phase of the
    call was last seen. They are what lets recovery name the event an
    unfinished call stopped at, and they make a stored state verifiable
    against its events.
    """

    call_id: str
    tool: str
    arguments: Mapping[str, Any]
    status: ToolCallStatus = ToolCallStatus.REQUESTED
    result: Any = None
    error: Mapping[str, Any] | None = None
    requested_sequence: int = 0
    started_sequence: int = 0
    completed_sequence: int = 0
    started_at: str | None = None
    #: The final attempt number: ``0`` until the first attempt starts, then the
    #: attempt this call ended on. A call that succeeded on its third try
    #: reports ``attempt=3`` and ``status=COMPLETED``, not ``FAILED``.
    attempt: int = 0
    #: Every attempt this logical call made, in order.
    attempts: tuple[ToolAttempt, ...] = ()
    #: A retry the journal recorded but had not started yet, if any.
    pending_retry: PendingRetry | None = None
    #: The idempotency key this call was made under, if any (Milestone 4B).
    #:
    #: Journalled with ``ToolRequested`` and never changed afterwards, so a
    #: deduplicated call -- one the idempotency store answered without running
    #: the tool -- is visible in the reconstructed state exactly like any other
    #: call, and a replay reproduces the same field.
    idempotency_key: str | None = None
    #: The deadline this call was journalled with, in seconds (Milestone 4C),
    #: and the mode that enforces it. Recorded on ``ToolRequested`` so the
    #: value survives a crash, and round-trips through a checkpoint with the
    #: rest of the state. ``None`` for a call with no deadline.
    timeout: float | None = None
    timeout_mode: str | None = None

    @property
    def attempt_count(self) -> int:
        """How many attempts this call made."""
        return len(self.attempts)

    @property
    def timed_out(self) -> bool:
        """Whether this call's final attempt ran out of time."""
        return self.status is ToolCallStatus.TIMED_OUT

    @property
    def timeout_enforced(self) -> bool | None:
        """Whether a timed-out call's tool provably stopped, or ``None``.

        ``False`` is the answer that matters after a restart: the runtime asked
        a thread to stop and it did not, so whatever external effect it was
        performing may still have happened -- and that is what makes such a
        call with an idempotency key a ``RECOVERY_REQUIRED`` rather than a
        failure.
        """
        if self.status is not ToolCallStatus.TIMED_OUT:
            return None
        return bool((self.error or {}).get("enforced", True))

    def attempt_record(self, number: int) -> ToolAttempt | None:
        """The recorded attempt ``number``, or ``None`` if it never happened."""
        return next((a for a in self.attempts if a.attempt == number), None)

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        suffix = f" (attempt {self.attempt})" if self.attempt > 1 else ""
        if self.status is ToolCallStatus.COMPLETED:
            return f"{self.tool}({args}) -> {self.result!r}{suffix}"
        if self.status is ToolCallStatus.FAILED:
            message = (self.error or {}).get("message", "failed")
            return f"{self.tool}({args}) !! {message}{suffix}"
        if self.status is ToolCallStatus.RETRYING:
            return f"{self.tool}({args}) {self.pending_retry}"
        if self.status is ToolCallStatus.CANCELLED:
            return f"{self.tool}({args}) -- cancelled"
        if self.status is ToolCallStatus.TIMED_OUT:
            note = "" if self.timeout_enforced else " (stop NOT enforced)"
            return f"{self.tool}({args}) -- timed out after {self.timeout}s{note}"
        return f"{self.tool}({args}) [{self.status}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "status": str(self.status),
            "result": self.result,
            "error": self.error,
            "requested_sequence": self.requested_sequence,
            "started_sequence": self.started_sequence,
            "completed_sequence": self.completed_sequence,
            "started_at": self.started_at,
            "attempt": self.attempt,
            "attempts": [item.to_dict() for item in self.attempts],
            "pending_retry": None if self.pending_retry is None else self.pending_retry.to_dict(),
            "idempotency_key": self.idempotency_key,
            "timeout": self.timeout,
            "timeout_mode": self.timeout_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCall":
        pending = data.get("pending_retry")
        return cls(
            call_id=data["call_id"],
            tool=data["tool"],
            arguments=data.get("arguments") or {},
            status=ToolCallStatus(data.get("status") or ToolCallStatus.REQUESTED),
            result=data.get("result"),
            error=data.get("error"),
            requested_sequence=int(data.get("requested_sequence") or 0),
            started_sequence=int(data.get("started_sequence") or 0),
            completed_sequence=int(data.get("completed_sequence") or 0),
            started_at=data.get("started_at"),
            attempt=int(data.get("attempt") or 0),
            attempts=tuple(
                ToolAttempt.from_dict(item) for item in data.get("attempts") or ()
            ),
            pending_retry=None if pending is None else PendingRetry.from_dict(pending),
            idempotency_key=data.get("idempotency_key"),
            timeout=data.get("timeout"),
            timeout_mode=data.get("timeout_mode"),
        )


@dataclass(frozen=True, slots=True)
class IncompleteTool:
    """A tool call the journal never recorded an outcome for.

    Produced by :func:`detect_incomplete_tools` after recovery::

        unfinished_tool:
            name: run_tests
            sequence: 17
            status: STARTED
    """

    call_id: str
    tool: str
    status: ToolCallStatus
    sequence: int
    arguments: Mapping[str, Any]
    requested_sequence: int = 0
    started_sequence: int = 0
    started_at: str | None = None

    @classmethod
    def from_call(cls, call: ToolCall) -> "IncompleteTool":
        """Describe an open tool call, pointing at the event recovery found it at."""
        return cls(
            call_id=call.call_id,
            tool=call.tool,
            status=call.status,
            # A started call is reported at the sequence it started at; a call
            # that never got that far is reported at its request.
            sequence=(
                call.started_sequence
                if call.status is ToolCallStatus.STARTED
                else call.requested_sequence
            ),
            arguments=dict(call.arguments),
            requested_sequence=call.requested_sequence,
            started_sequence=call.started_sequence,
            started_at=call.started_at,
        )

    @property
    def was_started(self) -> bool:
        """True when the tool actually began running before the process died."""
        return self.status is ToolCallStatus.STARTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "status": str(self.status),
            "sequence": self.sequence,
            "arguments": dict(self.arguments),
            "requested_sequence": self.requested_sequence,
            "started_sequence": self.started_sequence,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IncompleteTool":
        return cls(
            call_id=data["call_id"],
            tool=data["tool"],
            status=ToolCallStatus(data.get("status") or ToolCallStatus.REQUESTED),
            sequence=int(data.get("sequence") or 0),
            arguments=data.get("arguments") or {},
            requested_sequence=int(data.get("requested_sequence") or 0),
            started_sequence=int(data.get("started_sequence") or 0),
            started_at=data.get("started_at"),
        )

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.tool}({args})"


@dataclass(frozen=True, slots=True)
class ExecutionState:
    """The full state of an execution at a point in its event history.

    The whole object is JSON-serializable (:meth:`to_dict`) and round-trips
    exactly (:meth:`from_dict`). That is what lets it be stored as a checkpoint
    and compared against a fresh reconstruction of the same events.
    """

    execution_id: str = ""
    status: ExecutionStatus = ExecutionStatus.RUNNING
    goal: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    last_sequence: int = 0
    result: Any = None
    error: Mapping[str, Any] | None = None
    incomplete_tools: tuple[IncompleteTool, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": str(self.status),
            "goal": self.goal,
            "last_sequence": self.last_sequence,
            "result": self.result,
            "error": self.error,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "incomplete_tools": [item.to_dict() for item in self.incomplete_tools],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionState":
        """Rebuild a state from :meth:`to_dict` output.

        Malformed input raises ``KeyError``/``ValueError``/``TypeError``; the
        checkpoint layer turns those into
        :class:`~agent_runtime.exceptions.CorruptCheckpointError`.
        """
        return cls(
            execution_id=data["execution_id"],
            status=ExecutionStatus(data["status"]),
            goal=data.get("goal"),
            last_sequence=int(data.get("last_sequence") or 0),
            result=data.get("result"),
            error=data.get("error"),
            tool_calls=tuple(
                ToolCall.from_dict(item) for item in data.get("tool_calls") or ()
            ),
            incomplete_tools=tuple(
                IncompleteTool.from_dict(item) for item in data.get("incomplete_tools") or ()
            ),
        )


def initial_state(execution_id: str = "") -> ExecutionState:
    """The state of an execution that has produced no events yet."""
    return ExecutionState(execution_id=execution_id, status=ExecutionStatus.RUNNING)


def replace_state(
    state: ExecutionState, /, **changes: Any
) -> ExecutionState:
    """Copy ``state`` with ``changes`` applied.

    A thin, re-exported :func:`dataclasses.replace` so callers (tests, checkpoint
    validation) can adjust a single field without importing ``dataclasses``.
    """
    return replace(state, **changes)



# -- reducers ---------------------------------------------------------------


def apply_event(state: ExecutionState, event: Event) -> ExecutionState:
    """Return a new state with ``event`` applied.

    Pure function: the input state is never mutated.
    """
    handler = _REDUCERS.get(event.event_type)
    if handler is None:  # pragma: no cover - EventType is a closed set
        raise StateReconstructionError(
            f"Cannot apply unknown event type {event.event_type!r} "
            f"(sequence {event.sequence} of {event.execution_id!r})"
        )
    if state.execution_id and event.execution_id != state.execution_id:
        raise StateReconstructionError(
            f"Event belongs to execution {event.execution_id!r} but state is "
            f"for {state.execution_id!r}"
        )
    return replace(handler(state, event), last_sequence=event.sequence)


def _on_execution_started(state: ExecutionState, event: Event) -> ExecutionState:
    return replace(
        state,
        execution_id=event.execution_id,
        status=ExecutionStatus.RUNNING,
        goal=event.payload.get("goal"),
    )


def _on_execution_completed(state: ExecutionState, event: Event) -> ExecutionState:
    return replace(
        state,
        execution_id=event.execution_id,
        status=ExecutionStatus.COMPLETED,
        result=event.payload.get("result"),
    )


def _on_execution_failed(state: ExecutionState, event: Event) -> ExecutionState:
    return replace(
        state,
        execution_id=event.execution_id,
        status=ExecutionStatus.FAILED,
        error=event.payload.get("error"),
    )


def _on_execution_cancelled(state: ExecutionState, event: Event) -> ExecutionState:
    return replace(
        state,
        execution_id=event.execution_id,
        status=ExecutionStatus.CANCELLED,
        result=event.payload.get("result"),
    )


def _on_tool_requested(state: ExecutionState, event: Event) -> ExecutionState:
    call = ToolCall(
        call_id=event.payload["call_id"],
        tool=event.payload["tool"],
        arguments=event.payload.get("arguments") or {},
        status=ToolCallStatus.REQUESTED,
        requested_sequence=event.sequence,
        # Milestone 4B: the key is fixed here, for the whole logical call and
        # therefore for every one of its attempts. Journals written before it
        # carry no key, and the field simply stays None.
        idempotency_key=event.payload.get("idempotency_key"),
        # Milestone 4C: the same for the deadline -- fixed once, for the call,
        # so a resumed retry reuses it instead of re-deciding.
        timeout=event.payload.get("timeout"),
        timeout_mode=event.payload.get("timeout_mode"),
    )
    return replace(state, execution_id=event.execution_id, tool_calls=state.tool_calls + (call,))


def _on_tool_started(state: ExecutionState, event: Event) -> ExecutionState:
    attempt = _attempt_number(event)
    call = _call_of(state, event)
    return _update_call(
        state,
        event,
        status=ToolCallStatus.STARTED,
        attempt=attempt,
        attempts=_record_attempt(
            call.attempts,
            ToolAttempt(
                attempt=attempt,
                status=ToolCallStatus.STARTED,
                started_sequence=event.sequence,
                started_at=event.timestamp,
            ),
        ),
        # A scheduled retry is being carried out now, so it is no longer pending.
        pending_retry=None,
        started_sequence=event.sequence,
        started_at=event.timestamp,
    )


def _on_tool_completed(state: ExecutionState, event: Event) -> ExecutionState:
    return _settle_call(
        state, event, ToolCallStatus.COMPLETED, result=event.payload.get("result")
    )


def _on_tool_failed(state: ExecutionState, event: Event) -> ExecutionState:
    return _settle_call(
        state, event, ToolCallStatus.FAILED, error=event.payload.get("error") or {}
    )


def _on_tool_retry_scheduled(state: ExecutionState, event: Event) -> ExecutionState:
    """Record the decision to make another attempt, before any of it happens.

    The call is left ``RETRYING`` with a :class:`PendingRetry`: an unfinished
    but *unambiguous* state, since the journal now says exactly which attempt
    comes next. The same decision is also recorded on the attempt that made it,
    so it survives the next ``ToolStarted`` consuming the pending slot -- which
    is what lets a replay reproduce the retry instead of re-deciding it.
    """
    attempt = _attempt_number(event)
    failed_attempt = event.payload.get("failed_attempt")
    pending = PendingRetry(
        attempt=attempt,
        failed_attempt=int(failed_attempt) if failed_attempt is not None else max(attempt - 1, 0),
        delay=float(event.payload.get("delay") or 0.0),
        reason=str(event.payload.get("reason") or ""),
        error=event.payload.get("error") or {},
        sequence=event.sequence,
    )
    call = _call_of(state, event)
    # Default to the attempt this call is currently on, which is the one that
    # just failed; an event that names its own failed attempt wins.
    failed = pending.failed_attempt or call.attempt or attempt - 1
    failed_record = call.attempt_record(failed)
    updated = _update_call(
        state, event, status=ToolCallStatus.RETRYING, pending_retry=pending
    )
    if failed_record is None:
        # A journal that scheduled a retry for an attempt it never recorded as
        # failed. The pending decision still stands -- it is what happens next --
        # but there is no attempt to attach it to.
        return updated
    return _update_call(
        updated,
        event,
        attempts=_record_attempt(
            call.attempts, replace(failed_record, scheduled_retry=pending)
        ),
    )


def _on_tool_cancelled(state: ExecutionState, event: Event) -> ExecutionState:
    """Settle a call as cancelled, and drop any retry it had scheduled.

    Folding like any other settled attempt -- so a cancelled attempt appears in
    the attempt history, which is what lets a replay reproduce it rather than
    treating the call as unfinished -- and then clearing ``pending_retry``.

    That clearing is the part that matters after a restart: a call cancelled
    *during* a backoff has a ``ToolRetryScheduled`` behind it, and leaving that
    slot populated would let a resume "continue" an attempt the application had
    already decided against. §10, expressed in the reducer.
    """
    settled = _settle_call(
        state,
        event,
        ToolCallStatus.CANCELLED,
        error=event.payload.get("error") or {},
    )
    return _update_call(settled, event, pending_retry=None)


def _on_tool_timed_out(state: ExecutionState, event: Event) -> ExecutionState:
    """Record an attempt that ran out of time (Milestone 4C).

    Folds like any other settled attempt, but keeps the three things that make
    a timeout readable afterwards: the deadline, the mechanism that was going
    to enforce it, and whether the runtime can prove the tool actually stopped.
    That last flag is what makes an unenforceable cooperative timeout an
    *ambiguity* about the side effect, without turning the call itself into an
    unfinished one.
    """
    error = dict(event.payload.get("error") or {})
    if "timeout" in event.payload:
        error.setdefault("timeout", event.payload["timeout"])
    if "timeout_mode" in event.payload:
        error.setdefault("timeout_mode", event.payload["timeout_mode"])
    if "enforced" in event.payload:
        error.setdefault("enforced", bool(event.payload["enforced"]))
    return _settle_call(state, event, ToolCallStatus.TIMED_OUT, error=error)


# -- attempt bookkeeping ---------------------------------------------------


def _attempt_number(event: Event) -> int:
    """The attempt an event belongs to.

    Events written before Milestone 4A carry no ``attempt``, and a single
    attempt is exactly what they describe, so the default is 1. That is what
    keeps every older journal readable.
    """
    try:
        return max(int(event.payload.get("attempt") or 1), 1)
    except (TypeError, ValueError) as exc:
        raise StateReconstructionError(
            f"Event at sequence {event.sequence} of {event.execution_id!r} has a "
            f"non-numeric attempt {event.payload.get('attempt')!r}"
        ) from exc


def _record_attempt(
    existing: tuple[ToolAttempt, ...], entry: ToolAttempt
) -> tuple[ToolAttempt, ...]:
    """Insert ``entry`` into an attempt history, or merge it into the same attempt.

    One attempt is journalled by two events -- a start and an outcome -- so the
    second folds into the first rather than becoming a second attempt.
    """
    for index, current in enumerate(existing):
        if current.attempt == entry.attempt:
            merged = replace(
                entry,
                started_sequence=entry.started_sequence or current.started_sequence,
                started_at=entry.started_at or current.started_at,
                scheduled_retry=entry.scheduled_retry or current.scheduled_retry,
            )
            return existing[:index] + (merged,) + existing[index + 1 :]
    return existing + (entry,)


def _settle_call(
    state: ExecutionState,
    event: Event,
    status: ToolCallStatus,
    *,
    result: Any = None,
    error: Mapping[str, Any] | None = None,
) -> ExecutionState:
    """Record an attempt's outcome on both the attempt history and the call."""
    attempt = _attempt_number(event)
    call = _call_of(state, event)
    started_here = attempt == call.attempt
    return _update_call(
        state,
        event,
        status=status,
        attempt=attempt,
        result=result,
        error=error,
        completed_sequence=event.sequence,
        attempts=_record_attempt(
            call.attempts,
            ToolAttempt(
                attempt=attempt,
                status=status,
                # Carried over from the call when this attempt is the one that
                # was started; an outcome with no start event of its own (a
                # recovery resolution) keeps whatever start the call recorded.
                started_sequence=call.started_sequence if started_here else 0,
                started_at=call.started_at if started_here else None,
                completed_sequence=event.sequence,
                result=result,
                error=error,
            ),
        ),
    )


def _call_of(state: ExecutionState, event: Event) -> ToolCall:
    """The tool call an event refers to, or an explanation that there is none."""
    call_id = event.payload.get("call_id")
    for call in state.tool_calls:
        if call.call_id == call_id:
            return call
    raise StateReconstructionError(
        f"Event at sequence {event.sequence} references unknown tool call {call_id!r} "
        f"(execution {event.execution_id!r})"
    )


def _update_call(state: ExecutionState, event: Event, **changes: Any) -> ExecutionState:
    call = _call_of(state, event)  # raises if the event names no known call
    calls = list(state.tool_calls)
    calls[state.tool_calls.index(call)] = replace(call, **changes)
    return replace(state, tool_calls=tuple(calls))


_REDUCERS: dict[EventType, Any] = {
    EventType.EXECUTION_STARTED: _on_execution_started,
    EventType.EXECUTION_COMPLETED: _on_execution_completed,
    EventType.EXECUTION_FAILED: _on_execution_failed,
    EventType.EXECUTION_CANCELLED: _on_execution_cancelled,
    EventType.TOOL_REQUESTED: _on_tool_requested,
    EventType.TOOL_STARTED: _on_tool_started,
    EventType.TOOL_COMPLETED: _on_tool_completed,
    EventType.TOOL_FAILED: _on_tool_failed,
    EventType.TOOL_RETRY_SCHEDULED: _on_tool_retry_scheduled,
    EventType.TOOL_CANCELLED: _on_tool_cancelled,
    EventType.TOOL_TIMED_OUT: _on_tool_timed_out,
}


def detect_pending_retries(state: ExecutionState) -> tuple[PendingRetry, ...]:
    """Retries the journal scheduled but never started.

    Unlike :func:`detect_incomplete_tools`, these are *not* ambiguities: each
    one says exactly which attempt comes next, so a resumed execution can carry
    it on instead of asking the application to decide.
    """
    return tuple(
        call.pending_retry for call in state.tool_calls if call.pending_retry is not None
    )


def detect_incomplete_tools(state: ExecutionState) -> tuple[IncompleteTool, ...]:
    """The tool calls the journal left open.

    A call is open while the events say it was requested or started and never
    say what became of it -- exactly the shape a crash leaves behind::

        ToolRequested -> ToolStarted -> (process dies)
    """
    return tuple(
        IncompleteTool.from_call(call)
        for call in state.tool_calls
        if call.status.is_incomplete
    )


def resolve_status(state: ExecutionState) -> ExecutionStatus:
    """The status a history leaves an execution in.

    A ``RUNNING`` execution with open tool calls becomes ``RECOVERY_REQUIRED``:
    the journal cannot say whether the tool finished, so the runtime refuses to
    guess and lets the application decide.

    Any other status is returned untouched. ``COMPLETED``, ``FAILED`` and
    ``CANCELLED`` were deliberate decisions that were journalled, and they
    remain the answer even if some tool call in the same history never settled.
    """
    if state.status is ExecutionStatus.RUNNING and detect_incomplete_tools(state):
        return ExecutionStatus.RECOVERY_REQUIRED
    return state.status


class RecoveryState(StrEnum):
    """What a restarted process should do about an execution (Milestone 4C).

    :attr:`ExecutionStatus` says what the journal *is*; this says what that
    means for whoever is holding the baton. The distinction matters because
    ``RUNNING`` covers three very different situations, and Milestone 4C makes
    all three reachable:

    * a retry was scheduled and not yet run -- unambiguous, carry it on;
    * an external effect may have happened -- ambiguous, ask a human;
    * nothing is outstanding -- ordinary, carry on.

    Guessing between them is exactly what this runtime has refused to do since
    Milestone 2, so the classification is derived from events and never stored.
    """

    #: The history recorded a successful end.
    COMPLETED = "COMPLETED"
    #: The history recorded a deliberate failure.
    FAILED = "FAILED"
    #: The history recorded a deliberate cancellation. Stays cancelled: a
    #: cancelled execution does not resume, and nothing retries it.
    CANCELLED = "CANCELLED"
    #: A retry was scheduled and has not started. Not ambiguous -- the journal
    #: says exactly which attempt comes next -- so the new process can carry it
    #: on with ``continue_pending_retry``.
    RETRYABLE = "RETRYABLE"
    #: Something's outcome the journal does not record: an open tool call, or an
    #: idempotency key whose external effect may have happened without the
    #: runtime learning it. Needs an explicit decision; nothing is re-run.
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    #: Ordinary in-flight work with nothing outstanding. Not a special state --
    #: it is listed so the classification is total rather than partial.
    RUNNING = "RUNNING"


def classify_recovery(
    state: ExecutionState, *, unresolved_keys: bool = False
) -> RecoveryState:
    """Classify what a restart should do with ``state``.

    Args:
        state: The recovered state.
        unresolved_keys: Whether the idempotency store holds a key for this
            execution with no recorded outcome. Passed in rather than read
            here because those rows live in SQLite, not in the journal, and this
            module knows nothing about storage.

    The order is the point. A recorded terminal decision is returned as it
    stands -- a cancelled execution stays ``CANCELLED`` even if some call in the
    same history never settled, because the cancellation was deliberate and
    overriding it would be a guess. Only then does ambiguity win over a pending
    retry, because an ambiguity must be answered before anything else proceeds.
    """
    if state.status is ExecutionStatus.COMPLETED:
        return RecoveryState.COMPLETED
    if state.status is ExecutionStatus.FAILED:
        return RecoveryState.FAILED
    if state.status is ExecutionStatus.CANCELLED:
        return RecoveryState.CANCELLED
    if state.status is ExecutionStatus.RECOVERY_REQUIRED or unresolved_keys:
        return RecoveryState.RECOVERY_REQUIRED
    if detect_pending_retries(state):
        return RecoveryState.RETRYABLE
    return RecoveryState.RUNNING


def finalize_state(state: ExecutionState) -> ExecutionState:
    """Add everything a state derives from its events rather than from events.

    Fills in :attr:`ExecutionState.incomplete_tools` and settles the status, so
    every caller sees the same view. It is idempotent, which is what makes it
    safe to run over a checkpointed state that then has more events applied.
    """
    pending = detect_incomplete_tools(state)
    return replace(state, incomplete_tools=pending, status=resolve_status(state))


def reconstruct_state(events: Iterable[Event]) -> ExecutionState:
    """Fold an event stream into the state it describes.

    ``events`` must be ordered by sequence (as returned by
    :meth:`EventJournal.get_events`). The result is finalized, so it already
    carries the incomplete tool calls and the recovery status the stream implies.
    """
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
    return finalize_state(state)
