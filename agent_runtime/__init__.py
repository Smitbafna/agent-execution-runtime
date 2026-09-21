"""Agent Execution Runtime -- Milestone 4B: idempotency, crash-safe side effects.

The invariant this milestone adds:

    A side-effecting tool never runs twice for one idempotency key unless the
    application explicitly asks for it, and every key whose outcome the runtime
    does not know is reported rather than guessed at.

Quick start::

    from agent_runtime import Runtime

    with Runtime("agent.db") as runtime:
        execution = runtime.start(goal="Welcome the new user")
        execution.call(
            "send_email",
            to="user@example.com",
            body="welcome!",
            idempotency_key="welcome-user-123",   # <-- the guard
        )

If that process dies between the claim and the outcome, the key survives as
``PENDING`` and the next process refuses to send the email again::

    execution = runtime.resume(execution_id)
    print(execution.status)                 # RECOVERY_REQUIRED
    print(execution.unresolved_idempotency) # the keys it cannot answer for

    execution.resolve_idempotency("welcome-user-123", action="mark_completed",
                                  result={"message_id": "abc"})  # I checked: it sent
    execution.resolve_idempotency("welcome-user-123", action="retry")          # or run once more

Milestone 4A's invariant still holds, and is what makes retries of a keyed call
safe::

    A logical tool call keeps one stable call_id while it may make several
    numbered attempts; every attempt is journalled, and the state reports the
    final one.

Milestone 2's invariant still holds too, and is what makes all of this
recoverable:

    A recovered execution represents a valid state derived from a consistent
    checkpoint plus the events that were durably persisted after it.

And the limit is stated plainly, because no amount of bookkeeping removes it:

    This runtime does not provide exactly-once execution of external side
    effects. SQLite cannot commit atomically with an HTTP request, so a key left
    PENDING after a crash may correspond to a request that did happen. That is
    why the answer to a PENDING key is an explicit decision -- retry,
    mark_completed or mark_failed -- and never an assumption.
"""

from __future__ import annotations

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
    ToolError,
    ToolInvocationError,
    ToolNotFoundError,
)
from .execution import Execution
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
    ToolError,
    ToolInvocationError,
    ToolNotFoundError,
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
)
from .runner import ToolOutcome, ToolRequest, ToolRunner, ToolRunnerMode
from .runtime import Runtime
from .state import (
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    PendingRetry,
    ToolAttempt,
    ToolCall,
    ToolCallStatus,
    apply_event,
    detect_incomplete_tools,
    detect_pending_retries,
    finalize_state,
    initial_state,
    reconstruct_state,
    replace_state,
    resolve_status,
)
from .storage import SQLiteStore
from .tools import Tool, ToolRegistry, tool

__version__ = "0.5.0"

__all__ = [
    "__version__",
    # runtime
    "Runtime",
    "Execution",
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
    "initial_state",
    "apply_event",
    "reconstruct_state",
    "detect_incomplete_tools",
    "detect_pending_retries",
    "resolve_status",
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
    "RetryableToolError",
    "PermanentToolError",
    "RetryConfigurationError",
    "StateReconstructionError",
    "describe_error",
]
