"""Agent Execution Runtime -- Milestone 1: the core execution journal.

The invariant this package is built around:

    The event journal is the source of truth, and the current execution state
    can always be reconstructed from it.

Quick start::

    from agent_runtime import Runtime

    with Runtime("agent.db") as runtime:
        execution = runtime.start(goal="Perform some calculations")
        execution.call("add", a=2, b=3)
        execution.complete()

        recovered = runtime.resume(execution.id)
        assert recovered.state == execution.state
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
from .runtime import Runtime
from .state import (
    ExecutionState,
    ExecutionStatus,
    ToolCall,
    ToolCallStatus,
    apply_event,
    initial_state,
    reconstruct_state,
)
from .storage import SQLiteStore
from .tools import Tool, ToolRegistry, tool

__version__ = "0.1.0"

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
    # state
    "ExecutionState",
    "ExecutionStatus",
    "ToolCall",
    "ToolCallStatus",
    "initial_state",
    "apply_event",
    "reconstruct_state",
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
    "StateReconstructionError",
    "describe_error",
]
