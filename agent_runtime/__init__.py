"""Agent Execution Runtime -- Milestone 2: checkpoints and crash recovery.

The invariant this package is built around:

    A recovered execution represents a valid state derived from a consistent
    checkpoint plus the events that were durably persisted after it.

Quick start::

    from agent_runtime import Runtime

    with Runtime("agent.db") as runtime:
        execution = runtime.start(goal="Perform some calculations")
        execution.call("add", a=2, b=3)
        execution.checkpoint()          # a snapshot of the state so far
        execution.complete()

    # ... and after a crash, in a brand new process:
    with Runtime("agent.db") as runtime:
        execution = runtime.resume(execution_id)   # checkpoint + events after it
        print(execution.recovery_info())           # what, if anything, is unresolved
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
    ReplayError,
    ReplayMismatchError,
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
    RecordedToolCall,
    ReplayEngine,
    ReplayExecution,
    ReplayResult,
    ReplayStep,
    ReplayToolRunner,
    replay_execution,
)
from .runner import ToolOutcome, ToolRequest, ToolRunner, ToolRunnerMode
from .runtime import Runtime
from .state import (
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    ToolCall,
    ToolCallStatus,
    apply_event,
    detect_incomplete_tools,
    finalize_state,
    initial_state,
    reconstruct_state,
    replace_state,
    resolve_status,
)
from .storage import SQLiteStore
from .tools import Tool, ToolRegistry, tool

__version__ = "0.3.0"

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
    # state
    "ExecutionState",
    "ExecutionStatus",
    "ToolCall",
    "ToolCallStatus",
    "IncompleteTool",
    "initial_state",
    "apply_event",
    "reconstruct_state",
    "detect_incomplete_tools",
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
    "StateReconstructionError",
    "describe_error",
]
