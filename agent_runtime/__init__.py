"""Agent Execution Runtime -- Milestone 4A: retries and attempt semantics.

The invariant this milestone adds:

    A logical tool call keeps one stable call_id while it may make several
    numbered attempts; every attempt is journalled, and the state reports the
    final one.

Quick start::

    from agent_runtime import PermanentToolError, RetryPolicy, RetryableToolError, Runtime

    @runtime.tool(retry_policy=RetryPolicy(max_attempts=3))
    def fetch_data(url: str) -> str:
        if not reachable(url):
            raise RetryableToolError("upstream is down")   # eligible for retry
        if bad_request(url):
            raise PermanentToolError("400 from upstream")  # never retried
        return download(url)

    execution = runtime.start(goal="Fetch the data")
    execution.call("fetch_data", url="https://example.com/data")

Milestone 2's invariant still holds, and is what makes the above recoverable:

    A recovered execution represents a valid state derived from a consistent
    checkpoint plus the events that were durably persisted after it.

So after a crash::

    with Runtime("agent.db") as runtime:
        execution = runtime.resume(execution_id)       # checkpoint + events after it
        print(execution.pending_retries)              # a retry was scheduled, not run
        print(execution.recovery_info())              # and what, if anything, is unresolved
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
    UnknownToolCallError,
)
from .checkpoints import Checkpoint, CheckpointStore
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

__version__ = "0.4.0"

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
