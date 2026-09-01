"""The execution object: the user-facing handle on a journalled run.

Everything an :class:`Execution` reports -- its status, its goal, its tool calls
-- is derived from the journal by folding its events through the reducers in
:mod:`agent_runtime.state`. In-memory bookkeeping is never a second source of
truth.
"""

from __future__ import annotations

from typing import Any

from .events import Event, EventType, describe_error, new_id
from .exceptions import InvalidStateTransitionError, ToolInvocationError
from .journal import EventJournal
from .state import (
    ExecutionState,
    ExecutionStatus,
    ToolCall,
    reconstruct_state,
)
from .tools import ToolRegistry, make_jsonable

__all__ = ["Execution"]


class Execution:
    """A single execution, backed by its immutable event history."""

    def __init__(self, journal: EventJournal, registry: ToolRegistry, execution_id: str) -> None:
        self._journal = journal
        self._registry = registry
        self._id = execution_id

    # -- identity / derived state --------------------------------------------

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
        """Current state, reconstructed from the journal on every access."""
        return reconstruct_state(self.events)

    @property
    def status(self) -> ExecutionStatus:
        return self.state.status

    @property
    def goal(self) -> str | None:
        return self.state.goal

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.state.tool_calls

    def reconstruct_state(self) -> ExecutionState:
        """Explicit alias for :attr:`state` (mirrors ``reconstruct_state(events)``)."""
        return self.state

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable snapshot of the reconstructed state."""
        state = self.state
        return {
            "execution_id": state.execution_id,
            "status": str(state.status),
            "goal": state.goal,
            "last_sequence": state.last_sequence,
            "result": state.result,
            "error": state.error,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "tool": call.tool,
                    "arguments": dict(call.arguments),
                    "status": str(call.status),
                    "result": call.result,
                    "error": call.error,
                }
                for call in state.tool_calls
            ],
        }

    # -- lifecycle -----------------------------------------------------------

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Run a registered tool and journal the whole attempt.

        Emits ``ToolRequested`` -> ``ToolStarted`` -> ``ToolCompleted`` on
        success, or ``ToolRequested`` -> ``ToolStarted`` -> ``ToolFailed`` when
        the tool raises, in which case :class:`ToolInvocationError` is raised
        after the failure is durably recorded. Milestone 1 performs no retries.
        """
        self._require_running("call a tool")

        # Resolve + validate before journalling: an unknown tool or a bad
        # argument list is a programming error, and nothing was attempted.
        target = self._registry.get(name)
        bound = target.bind(args, kwargs)
        arguments = target.arguments_for(bound)

        call_id = new_id("call")
        self._append(
            EventType.TOOL_REQUESTED,
            {"call_id": call_id, "tool": target.name, "arguments": arguments},
        )
        self._append(EventType.TOOL_STARTED, {"call_id": call_id, "tool": target.name})

        try:
            result = make_jsonable(target.invoke(bound))
        except Exception as exc:
            error = describe_error(exc)
            self._append(
                EventType.TOOL_FAILED,
                {"call_id": call_id, "tool": target.name, "error": error},
            )
            raise ToolInvocationError(
                f"Tool {target.name!r} failed: {exc}",
                tool_name=target.name,
                call_id=call_id,
                error_type=error["type"],
                traceback_text=error.get("traceback"),
            ) from exc

        self._append(
            EventType.TOOL_COMPLETED,
            {"call_id": call_id, "tool": target.name, "result": result},
        )
        return result

    def complete(self, result: Any = None) -> None:
        """Mark the execution COMPLETED and journal ``ExecutionCompleted``."""
        self._require_running("complete")
        self._append(EventType.EXECUTION_COMPLETED, {"result": make_jsonable(result)})

    def fail(self, error: BaseException | str | None = None) -> None:
        """Mark the execution FAILED and journal ``ExecutionFailed``."""
        self._require_running("fail")
        self._append(EventType.EXECUTION_FAILED, {"error": describe_error(error)})

    # -- internals -----------------------------------------------------------

    def _append(self, event_type: EventType, payload: dict[str, Any]) -> Event:
        return self._journal.append_event(self._id, event_type, payload)

    def _require_running(self, action: str) -> None:
        status = self.status
        if status is not ExecutionStatus.RUNNING:
            raise InvalidStateTransitionError(
                f"Cannot {action}: execution {self._id!r} is {status}"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Execution(id={self._id!r}, status={self.status})"

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        lines = [f"Execution {self._id} [{self.status}] goal={self.goal!r}"]
        lines.extend(f"  {call}" for call in self.tool_calls)
        return "\n".join(lines)
