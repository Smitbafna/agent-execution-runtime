"""Execution state, reconstructed purely from the event journal.

Nothing in here keeps state between calls: :func:`reconstruct_state` folds the
ordered event stream through :func:`apply_event` and returns the result. The
runtime treats the journal as the source of truth and derives every read from
it.
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
    "ToolCall",
    "ExecutionState",
    "initial_state",
    "apply_event",
    "reconstruct_state",
]


class ExecutionStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ToolCallStatus(StrEnum):
    REQUESTED = "REQUESTED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call, as reconstructed from its events."""

    call_id: str
    tool: str
    arguments: Mapping[str, Any]
    status: ToolCallStatus = ToolCallStatus.REQUESTED
    result: Any = None
    error: Mapping[str, Any] | None = None

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        if self.status is ToolCallStatus.COMPLETED:
            return f"{self.tool}({args}) -> {self.result!r}"
        if self.status is ToolCallStatus.FAILED:
            message = (self.error or {}).get("message", "failed")
            return f"{self.tool}({args}) !! {message}"
        return f"{self.tool}({args}) [{self.status}]"


@dataclass(frozen=True, slots=True)
class ExecutionState:
    """The full state of an execution at a point in its event history."""

    execution_id: str = ""
    status: ExecutionStatus = ExecutionStatus.RUNNING
    goal: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    last_sequence: int = 0
    result: Any = None
    error: Mapping[str, Any] | None = None


def initial_state(execution_id: str = "") -> ExecutionState:
    """The state of an execution that has produced no events yet."""
    return ExecutionState(execution_id=execution_id, status=ExecutionStatus.RUNNING)



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


def _on_tool_requested(state: ExecutionState, event: Event) -> ExecutionState:
    call = ToolCall(
        call_id=event.payload["call_id"],
        tool=event.payload["tool"],
        arguments=event.payload.get("arguments") or {},
        status=ToolCallStatus.REQUESTED,
    )
    return replace(state, execution_id=event.execution_id, tool_calls=state.tool_calls + (call,))


def _on_tool_started(state: ExecutionState, event: Event) -> ExecutionState:
    return _update_call(state, event, status=ToolCallStatus.STARTED)


def _on_tool_completed(state: ExecutionState, event: Event) -> ExecutionState:
    return _update_call(
        state, event, status=ToolCallStatus.COMPLETED, result=event.payload.get("result")
    )


def _on_tool_failed(state: ExecutionState, event: Event) -> ExecutionState:
    return _update_call(
        state, event, status=ToolCallStatus.FAILED, error=event.payload.get("error") or {}
    )


def _update_call(state: ExecutionState, event: Event, **changes: Any) -> ExecutionState:
    call_id = event.payload.get("call_id")
    for index, call in enumerate(state.tool_calls):
        if call.call_id == call_id:
            updated = replace(call, **changes)
            calls = state.tool_calls[:index] + (updated,) + state.tool_calls[index + 1 :]
            return replace(state, tool_calls=calls)
    raise StateReconstructionError(
        f"Event at sequence {event.sequence} references unknown tool call {call_id!r} "
        f"(execution {event.execution_id!r})"
    )


_REDUCERS: dict[EventType, Any] = {
    EventType.EXECUTION_STARTED: _on_execution_started,
    EventType.EXECUTION_COMPLETED: _on_execution_completed,
    EventType.EXECUTION_FAILED: _on_execution_failed,
    EventType.TOOL_REQUESTED: _on_tool_requested,
    EventType.TOOL_STARTED: _on_tool_started,
    EventType.TOOL_COMPLETED: _on_tool_completed,
    EventType.TOOL_FAILED: _on_tool_failed,
}


def reconstruct_state(events: Iterable[Event]) -> ExecutionState:
    """Fold an event stream into the state it describes.

    ``events`` must be ordered by sequence (as returned by
    :meth:`EventJournal.get_events`).
    """
    state = initial_state()
    for event in events:
        state = apply_event(state, event)
    return state
