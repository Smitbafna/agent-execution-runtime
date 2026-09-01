"""Exception hierarchy for the agent execution runtime.

All errors raised by this package derive from :class:`AgentRuntimeError`, so
callers can catch a single base class.
"""

from __future__ import annotations

__all__ = [
    "AgentRuntimeError",
    "StorageError",
    "JournalError",
    "SequenceError",
    "DuplicateSequenceError",
    "EventNotFoundError",
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
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.call_id = call_id
        self.error_type = error_type or type(self).__name__
        self.traceback_text = traceback_text


class ToolArgumentError(ToolCallError):
    """The arguments supplied to a tool do not match its signature."""


class ToolInvocationError(ToolCallError):
    """The tool function itself raised an exception.

    The original exception is always chained with ``raise ... from error`` and
    its traceback is stored on the instance.
    """


class StateReconstructionError(AgentRuntimeError):
    """An event could not be applied while rebuilding execution state."""
