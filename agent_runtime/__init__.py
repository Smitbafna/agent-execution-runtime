"""Agent Execution Runtime -- Milestone 4C: timeouts, cancellation, reliability.

The invariant this milestone adds:

    A timeout stops the tool or the runtime says it could not; a cancellation is
    a decision, never a retry; and an ambiguous side effect is reported rather
    than resolved by a guess.

Quick start::

    from agent_runtime import Runtime

    with Runtime("agent.db") as runtime:
        execution = runtime.start(goal="Process the batch")
        execution.call("slow_tool", timeout=5.0)      # a real deadline
        execution.cancel("user pressed stop")        # a durable decision

A deadline is enforced by actually stopping the tool -- ``asyncio.wait_for`` for
a coroutine, a cancellation token for a ``def`` tool that declared one -- and
*refused* for a ``def`` tool that did not, because Python cannot terminate a
thread and a wrapper that merely measured elapsed time would be reporting a stop
that never happened::

    UnsupportedTimeoutError: Cannot enforce a 1.0s timeout on tool 'unstoppable':
    it is a synchronous function with no cancellation token, and Python cannot
    terminate a running thread. Write it as 'async def' so a deadline can cancel
    it, or declare a 'cancel_token' parameter so the runtime can ask it to stop.

Cancellation is durable and safe to call from another thread, and a cancelled
execution is never resumed and never retried::

    ToolStarted -> ToolCancelled -> ExecutionCancelled

A timeout joins the reliability model without collapsing into a failure, and
without bypassing the idempotency guarantee. ``ToolTimedOut`` records whether
the runtime could *prove* the tool stopped, and that one flag is what decides
whether a keyed call is a known ``FAILED`` outcome or an ambiguity::

    tool starts -> a timeout the runtime could not enforce
        -> the side effect may have happened
        -> the key stays PENDING, the execution reports RECOVERY_REQUIRED
        -> an explicit resolve_idempotency, never a retry and never a guess

Milestone 4B's invariant still holds, and is what makes retries of a keyed call
safe::

    A logical tool call keeps one stable call_id while it may make several
    numbered attempts; every attempt is journalled, and the state reports the
    final one.

Milestone 2's invariant still holds too, and is what makes all of this
recoverable:

    A recovered execution represents a valid state derived from a consistent
    checkpoint plus the events that were durably persisted after it.

And the limit is still stated plainly, because no amount of bookkeeping removes
it:

    This runtime does not provide exactly-once execution of external side
    effects. SQLite cannot commit atomically with an HTTP request, so a key left
    PENDING after a crash may correspond to a request that did happen. That is
    why the answer to a PENDING key is an explicit decision -- retry,
    mark_completed or mark_failed -- and never an assumption.
"""

from __future__ import annotations

from .cancellation import CANCEL_TOKEN_PARAMETER, CancellationToken
from .events import Event, EventType, describe_error
from .exceptions import (
    AgentRuntimeError,
    DuplicateSequenceError,
    ExecutionError,
    ExecutionExistsError,
    ExecutionNotFoundError,
    InvalidStateTransitionError,
    JournalError,
    SequenceError,
    StateReconstructionError,
    StorageError,
    ToolAlreadyRegisteredError,
    ToolArgumentError,
    ToolCallError,
    ToolCancelledError,
    ToolError,
    ToolInvocationError,
    ToolNotFoundError,
)
from .execution import AmbiguousTimeoutError, Execution
from .journal import EventJournal
from .exceptions import (
    AgentRuntimeError,
    CheckpointError,
    CorruptCheckpointError,
    DuplicateSequenceError,
    ExecutionError,
    ExecutionExistsError,
    ExecutionNotFoundError,
    IdempotencyError,
    IdempotencyKeyConflictError,
    IdempotencyKeyFailedError,
    IdempotencyRecoveryRequiredError,
    IdempotencyResolutionError,
    InconsistentCheckpointError,
    InvalidRecoveryActionError,
    InvalidStateTransitionError,
    JournalError,
    PermanentToolError,
    ReplayError,
    ReplayMismatchError,
    RetryConfigurationError,
    RetryableToolError,
    SequenceError,
    StateReconstructionError,
    StorageError,
    ToolAlreadyRegisteredError,
    ToolArgumentError,
    ToolCallError,
    ToolCancelledError,
    ToolError,
    ToolInvocationError,
    ToolNotFoundError,
    ToolTimedOutError,
    ToolTimeoutError,
    TimeoutConfigurationError,
    TimeoutEnforcementError,
    TimeoutError,
    UnsupportedTimeoutError,
    UnknownIdempotencyKeyError,
    UnknownToolCallError,
)
from .checkpoints import Checkpoint, CheckpointStore
from .idempotency import (
    IdempotencyAction,
    IdempotencyDecision,
    IdempotencyGuard,
    IdempotencyRecord,
    IdempotencyStatus,
    IdempotencyStore,
    ReplayIdempotencyGuard,
    StoreIdempotencyGuard,
    unresolved_records,
)
from .recovery import RecoveryInfo, recover_execution
from .replay import (
    RecordedAttempt,
    RecordedToolCall,
    ReplayEngine,
    ReplayExecution,
    ReplayResult,
    ReplayStep,
    ReplayToolRunner,
    replay_execution,
)
from .retry import (
    NO_RETRY,
    ErrorKind,
    RealSleeper,
    RecordingSleeper,
    RetryDecision,
    RetryPolicy,
    Sleeper,
    classify_error,
    wait_interrupted,
)
from .runner import ToolOutcome, ToolRequest, ToolRunner, ToolRunnerMode
from .runtime import Runtime
from .state import (
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    PendingRetry,
    RecoveryState,
    ToolAttempt,
    ToolCall,
    ToolCallStatus,
    apply_event,
    classify_recovery,
    detect_incomplete_tools,
    detect_pending_retries,
    finalize_state,
    initial_state,
    reconstruct_state,
    replace_state,
    resolve_status,
)
from .storage import SQLiteStore
from .timeout import (
    ResolvedTimeout,
    TimeoutMode,
    check_timeout,
    declares_cancel_token,
    is_async_callable,
    resolve_timeout,
    resolve_timeout_mode,
)
from .tools import Tool, ToolRegistry, tool

__version__ = "0.6.0"

__all__ = [
    "__version__",
    # runtime
    "Runtime",
    "Execution",
    "AmbiguousTimeoutError",
    # journal / storage
    "EventJournal",
    "SQLiteStore",
    "Event",
    "EventType",
    # checkpoints / recovery
    "Checkpoint",
    "CheckpointStore",
    "RecoveryInfo",
    "recover_execution",
    # replay
    "RecordedToolCall",
    "RecordedAttempt",
    "ReplayResult",
    "ReplayStep",
    "ReplayToolRunner",
    "ReplayExecution",
    "ReplayEngine",
    "replay_execution",
    "ToolRunnerMode",
    "ToolRunner",
    "ToolRequest",
    "ToolOutcome",
    # retries
    "RetryPolicy",
    "RetryDecision",
    "ErrorKind",
    "classify_error",
    "NO_RETRY",
    "Sleeper",
    "RealSleeper",
    "RecordingSleeper",
    "wait_interrupted",
    # timeouts and cancellation
    "CancellationToken",
    "CANCEL_TOKEN_PARAMETER",
    "TimeoutMode",
    "ResolvedTimeout",
    "check_timeout",
    "declares_cancel_token",
    "is_async_callable",
    "resolve_timeout",
    "resolve_timeout_mode",
    # idempotency
    "IdempotencyStore",
    "IdempotencyRecord",
    "IdempotencyStatus",
    "IdempotencyAction",
    "IdempotencyDecision",
    "IdempotencyGuard",
    "StoreIdempotencyGuard",
    "ReplayIdempotencyGuard",
    "unresolved_records",
    # state
    "ExecutionState",
    "ExecutionStatus",
    "ToolCall",
    "ToolCallStatus",
    "ToolAttempt",
    "PendingRetry",
    "IncompleteTool",
    "RecoveryState",
    "initial_state",
    "apply_event",
    "reconstruct_state",
    "detect_incomplete_tools",
    "detect_pending_retries",
    "resolve_status",
    "classify_recovery",
    "finalize_state",
    "replace_state",
    # tools
    "Tool",
    "ToolRegistry",
    "tool",
    # errors
    "AgentRuntimeError",
    "StorageError",
    "JournalError",
    "SequenceError",
    "DuplicateSequenceError",
    "CheckpointError",
    "InconsistentCheckpointError",
    "CorruptCheckpointError",
    "ExecutionError",
    "ExecutionExistsError",
    "ExecutionNotFoundError",
    "InvalidStateTransitionError",
    "InvalidRecoveryActionError",
    "UnknownToolCallError",
    "IdempotencyError",
    "IdempotencyKeyConflictError",
    "IdempotencyRecoveryRequiredError",
    "IdempotencyKeyFailedError",
    "IdempotencyResolutionError",
    "UnknownIdempotencyKeyError",
    "ToolError",
    "ReplayError",
    "ReplayMismatchError",
    "ToolNotFoundError",
    "ToolAlreadyRegisteredError",
    "ToolCallError",
    "ToolArgumentError",
    "ToolInvocationError",
    "ToolCancelledError",
    "ToolTimedOutError",
    "ToolTimeoutError",
    "TimeoutError",
    "TimeoutConfigurationError",
    "TimeoutEnforcementError",
    "UnsupportedTimeoutError",
    "AmbiguousTimeoutError",
    "RetryableToolError",
    "PermanentToolError",
    "RetryConfigurationError",
    "StateReconstructionError",
    "describe_error",
]
