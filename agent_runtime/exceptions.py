"""Exception hierarchy for the agent execution runtime.

All errors raised by this package derive from :class:`AgentRuntimeError`, so
callers can catch a single base class.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

__all__ = [
    "AgentRuntimeError",
    "StorageError",
    "JournalError",
    "SequenceError",
    "DuplicateSequenceError",
    "EventNotFoundError",
    "CheckpointError",
    "InconsistentCheckpointError",
    "CorruptCheckpointError",
    "ExecutionError",
    "ExecutionExistsError",
    "ExecutionNotFoundError",
    "InvalidStateTransitionError",
    "ToolError",
    "ToolNotFoundError",
    "ToolAlreadyRegisteredError",
    "ToolCallError",
    "ToolArgumentError",
    "ToolInvocationError",
    "RetryableToolError",
    "PermanentToolError",
    "RetryConfigurationError",
    "UnknownToolCallError",
    "InvalidRecoveryActionError",
    "StateReconstructionError",
    "ReplayError",
    "ReplayMismatchError",
    "IdempotencyError",
    "IdempotencyKeyConflictError",
    "IdempotencyRecoveryRequiredError",
    "IdempotencyKeyFailedError",
    "IdempotencyResolutionError",
    "UnknownIdempotencyKeyError",
]


class AgentRuntimeError(Exception):
    """Base class for every error raised by the runtime."""


class StorageError(AgentRuntimeError):
    """Raised when the underlying SQLite store cannot fulfil an operation."""


class JournalError(AgentRuntimeError):
    """Base class for event journal errors."""


class SequenceError(JournalError):
    """An event was appended with a sequence that does not continue the history."""


class DuplicateSequenceError(SequenceError):
    """An event tried to reuse a sequence number already used by an execution."""


class EventNotFoundError(JournalError):
    """A requested event does not exist in the journal."""


class CheckpointError(AgentRuntimeError):
    """Base class for checkpoint failures."""


class InconsistentCheckpointError(CheckpointError):
    """A checkpoint would not describe the state the journal actually holds.

    Raised instead of writing, for example, when the sequence a caller wants to
    store is ahead of or behind the execution's latest event, or when the event
    at that sequence does not exist at all. Either way the snapshot would be a
    state the history does not support, so nothing is persisted.
    """


class CorruptCheckpointError(CheckpointError):
    """A stored checkpoint cannot be read back as an execution state.

    Recovery refuses to guess here: an unreadable checkpoint is surfaced, not
    silently skipped, because quietly falling back to a full replay would hide
    the fact that stored data is broken.
    """


class ExecutionError(AgentRuntimeError):
    """Base class for execution lifecycle errors."""


class ExecutionExistsError(ExecutionError):
    """An execution with the requested id already has events in the journal."""


class ExecutionNotFoundError(ExecutionError):
    """No execution history exists for the requested execution id."""


class InvalidStateTransitionError(ExecutionError):
    """The requested lifecycle transition is not allowed from the current status."""


class ToolError(AgentRuntimeError):
    """Base class for tool registry errors."""


class ToolNotFoundError(ToolError):
    """No tool is registered under the requested name."""


class ToolAlreadyRegisteredError(ToolError):
    """A tool with the same name is already registered."""


class ToolCallError(AgentRuntimeError):
    """Base class for failures raised while invoking a tool."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        call_id: str | None = None,
        error_type: str | None = None,
        traceback_text: str | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.call_id = call_id
        self.error_type = error_type or type(self).__name__
        self.traceback_text = traceback_text
        #: How many attempts the logical call made before failing (Milestone 4A).
        #: ``None`` for failures that are not about a tool call at all.
        self.attempts = attempts


class ToolArgumentError(ToolCallError):
    """The arguments supplied to a tool do not match its signature."""


class ToolInvocationError(ToolCallError):
    """The tool function itself raised an exception.

    The original exception is always chained with ``raise ... from error`` and
    its traceback is stored on the instance.
    """


class RetryableToolError(ToolError):
    """A tool failure that a retry may fix -- a timeout, a rate limit, a 503.

    Raising this is how a tool says "this attempt failed, and running it again
    is worth doing". It is the *only* thing that makes an attempt eligible for
    an automatic retry; nothing is inferred from the exception's message, its
    type or how transient it looks.
    """


class PermanentToolError(ToolError):
    """A tool failure that repeating the attempt cannot fix.

    A validation failure, a missing record, a rejected request. Retrying it
    would only burn time, so the runtime never does, whatever the policy says.
    """


class RetryConfigurationError(AgentRuntimeError):
    """A retry policy or a retry-related argument is not usable.

    Raised while a policy is being defined rather than while a tool is running:
    an impossible ``max_attempts`` or a backoff that is not a policy is a
    programming error, and finding it out at call time would be too late.
    """


class UnknownToolCallError(AgentRuntimeError):
    """No tool call with the requested id exists in this execution's history."""


class InvalidRecoveryActionError(AgentRuntimeError):
    """An unsupported recovery action was requested.

    Milestone 2 records decisions the application has already made; it does not
    retry anything on its own.
    """


class StateReconstructionError(AgentRuntimeError):
    """An event could not be applied while rebuilding execution state."""


class ReplayError(AgentRuntimeError):
    """Base class for deterministic replay failures.

    Milestone 3 replays a *finished* history against itself. Anything that stops
    that -- a divergence between the recorded call and the replayed one, or a
    start point the history cannot support -- is raised rather than smoothed over.
    """


class ReplayMismatchError(ReplayError):
    """A replay diverged from the recorded history.

    Raised the moment replay and the journal disagree: a different tool, different
    arguments, a call with no recorded counterpart (or vice versa), or a final
    state that is not the one the original execution reached.

    The exception carries the machine-readable fields (:attr:`kind`,
    :attr:`sequence`, :attr:`expected`, :attr:`received`) *and* renders the
    human-readable report::

        ReplayMismatchError

        Execution: exec_123
        Sequence: 7

        Expected:
            tool: search_code
            args: {"query": "authentication"}

        Received:
            tool: search_code
            args: {"query": "database"}

    Attributes:
        kind: Stable, matchable label for the divergence (``tool``,
            ``arguments``, ``sequence``, ``missing_recorded_call``,
            ``unexpected_tool_call``, ``unresolved_tool_call``, ``state``,
            ``lifecycle``).
        execution_id: The execution being replayed.
        sequence: The event sequence the divergence was detected at, when known.
        expected: What the journal recorded.
        received: What the replay produced.
        original_state / replayed_state: Present for a ``state`` divergence.
    """

    def __init__(
        self,
        summary: str,
        *,
        kind: str,
        execution_id: str,
        sequence: int | None = None,
        expected: Any = None,
        received: Any = None,
        original_state: Any = None,
        replayed_state: Any = None,
    ) -> None:
        super().__init__(summary)
        self.kind = kind
        self.execution_id = execution_id
        self.sequence = sequence
        self.expected = expected
        self.received = received
        self.original_state = original_state
        self.replayed_state = replayed_state

    def __str__(self) -> str:
        lines = [
            "ReplayMismatchError",
            "",
            f"Reason: {self.args[0]}",
            f"Execution: {self.execution_id}",
        ]
        if self.sequence is not None:
            lines.append(f"Sequence: {self.sequence}")

        expected = _render(self.expected)
        received = _render(self.received)
        if expected or received:
            lines.append("")
            lines.append("Expected:")
            lines.extend(f"    {line}" for line in expected)
            lines.append("")
            lines.append("Received:")
            lines.extend(f"    {line}" for line in received)

        if self.original_state is not None or self.replayed_state is not None:
            lines.append("")
            lines.append("Original state:")
            lines.extend(f"    {line}" for line in _render_state(self.original_state))
            lines.append("")
            lines.append("Replayed state:")
            lines.extend(f"    {line}" for line in _render_state(self.replayed_state))
        return "\n".join(lines)


def _render(value: Any) -> list[str]:
    """Format an expected/received payload as indented ``label: json`` lines."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [f"{key}: {_json(value[key])}" for key in value]
    return [_json(value)]


def _render_state(state: Any) -> list[str]:
    """Summarize an :class:`~agent_runtime.state.ExecutionState` for a report."""
    if state is None:
        return []
    return [
        f"status = {getattr(state, 'status', '?')}",
        f"tool_calls = {len(getattr(state, 'tool_calls', ()))}",
        f"last_sequence = {getattr(state, 'last_sequence', '?')}",
    ]


def _json(value: Any) -> str:
    """Render a payload as JSON, falling back to ``repr`` if it is not JSON-safe."""
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - payloads are JSON-safe
        return repr(value)


# ---------------------------------------------------------------------------
# Idempotency (Milestone 4B)
# ---------------------------------------------------------------------------


class IdempotencyError(AgentRuntimeError):
    """Base class for everything that stops a keyed side effect from running twice.

    Each subclass says something different to the application, because the three
    cases need three different decisions:

    * :class:`IdempotencyKeyConflictError` -- the key is already claimed; the
      call is deduplicated against the recorded outcome;
    * :class:`IdempotencyRecoveryRequiredError` -- the key is ``PENDING`` and the
      runtime does not know whether the external effect happened;
    * :class:`IdempotencyKeyFailedError` -- the key is ``FAILED``; the recorded
      failure stands until the application says otherwise.

    None of them is "the database is unhappy". All of them are the runtime
    refusing to guess, which is the entire point of the milestone.
    """


class IdempotencyKeyConflictError(IdempotencyError):
    """A second claim of an idempotency key that already exists.

    Raised by :meth:`~agent_runtime.idempotency.IdempotencyStore.claim` inside
    its transaction, and by SQLite's ``PRIMARY KEY`` as the backstop. It carries
    the existing :attr:`record` when the claim was refused because a row was
    there to read.
    """

    def __init__(
        self,
        summary: str,
        *,
        idempotency_key: str,
        record: Any | None = None,
    ) -> None:
        super().__init__(summary)
        self.idempotency_key = idempotency_key
        self.record = record


class IdempotencyRecoveryRequiredError(IdempotencyError):
    """A keyed call arrived while its key was still ``PENDING``.

    The exact shape a crash leaves::

        claim key
            -> the external side effect happens
            -> the process dies before the outcome is stored

    On restart the runtime cannot know whether the effect happened -- SQLite
    committed the claim, the world did the side effect, and nothing recorded the
    answer. So it refuses to run the tool and hands the decision back through
    :meth:`~agent_runtime.execution.Execution.resolve_idempotency`.

    Attributes:
        idempotency_key: The key that is unresolved.
        record: The ``PENDING`` :class:`~agent_runtime.idempotency.IdempotencyRecord`.
    """

    def __init__(
        self, summary: str, *, idempotency_key: str, record: Any | None = None
    ) -> None:
        super().__init__(summary)
        self.idempotency_key = idempotency_key
        self.record = record

    def __str__(self) -> str:
        return "\n".join(
            [
                "IdempotencyRecoveryRequiredError",
                "",
                f"Reason: {self.args[0]}",
                f"Key: {self.idempotency_key}",
                "",
                "The runtime cannot tell whether the external side effect happened,",
                "so it did not run the tool. Decide explicitly:",
                "",
                "    execution.resolve_idempotency(key, 'mark_completed', result=...)",
                "    execution.resolve_idempotency(key, 'mark_failed', error=...)",
                "    execution.resolve_idempotency(key, 'retry')",
            ]
        )


class IdempotencyKeyFailedError(IdempotencyError):
    """A keyed call arrived while its key was recorded as ``FAILED``.

    The failure is known, but a failed attempt is not proof that the external
    side effect did not partially happen, so the runtime still does not run the
    tool again on its own.
    """

    def __init__(
        self, summary: str, *, idempotency_key: str, record: Any | None = None
    ) -> None:
        super().__init__(summary)
        self.idempotency_key = idempotency_key
        self.record = record


class IdempotencyResolutionError(IdempotencyError):
    """An explicit idempotency resolution that the store refuses to apply.

    For example resolving a key that is already ``COMPLETED``: a recorded
    outcome stands, and overwriting it is exactly the kind of silent guess this
    milestone avoids.
    """

    def __init__(
        self,
        summary: str,
        *,
        idempotency_key: str | None = None,
        record: Any | None = None,
    ) -> None:
        super().__init__(summary)
        self.idempotency_key = idempotency_key
        self.record = record


class UnknownIdempotencyKeyError(IdempotencyResolutionError):
    """A resolution was asked for a key the idempotency store has never seen."""
