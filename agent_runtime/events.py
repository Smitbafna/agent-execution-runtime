"""Typed, immutable event definitions.

Every fact about an execution is recorded as an :class:`Event`. Events are
frozen dataclasses: once constructed (and therefore once persisted by the
journal) they can never be modified.
"""

from __future__ import annotations

import json
import traceback as _traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "EventType",
    "EVENT_TYPES",
    "INCOMPLETE_EVENT_TYPES",
    "TERMINAL_EVENT_TYPES",
    "Event",
    "describe_error",
    "is_incomplete_event",
    "is_terminal_event",
    "new_id",
    "utc_now_iso",
]


class EventType(StrEnum):
    """The closed set of event types."""

    EXECUTION_STARTED = "ExecutionStarted"
    EXECUTION_COMPLETED = "ExecutionCompleted"
    EXECUTION_FAILED = "ExecutionFailed"
    EXECUTION_CANCELLED = "ExecutionCancelled"
    TOOL_REQUESTED = "ToolRequested"
    TOOL_STARTED = "ToolStarted"
    TOOL_COMPLETED = "ToolCompleted"
    TOOL_FAILED = "ToolFailed"
    #: The decision to run one more attempt, recorded *before* the wait and
    #: before the attempt itself. Durable: a process that dies during the
    #: backoff leaves the scheduled retry visible in the journal rather than
    #: only in the memory of the process that was about to sleep.
    TOOL_RETRY_SCHEDULED = "ToolRetryScheduled"
    TOOL_CANCELLED = "ToolCancelled"
    #: The attempt's deadline expired (Milestone 4C). Durable and explicit,
    #: rather than hidden inside a generic failure: a timeout is *not* a
    #: failure, it says the call ran out of time, and an application -- or a
    #: recovery -- has to be able to tell the two apart. The payload carries
    #: ``timeout``, ``mode`` and ``enforced``, so a reader can tell a coroutine
    #: that was genuinely cancelled from a thread that was asked to stop and
    #: did not.
    TOOL_TIMED_OUT = "ToolTimedOut"


EVENT_TYPES: frozenset[EventType] = frozenset(EventType)

#: Tool events that leave a call open: the journal does not say what happened.
INCOMPLETE_EVENT_TYPES: frozenset[EventType] = frozenset(
    {EventType.TOOL_REQUESTED, EventType.TOOL_STARTED}
)

#: Events that close a call or an execution. Reaching one means the history
#: recorded an outcome, so recovery has nothing left to resolve.
TERMINAL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
        EventType.TOOL_CANCELLED,
        EventType.TOOL_TIMED_OUT,
        EventType.EXECUTION_COMPLETED,
        EventType.EXECUTION_FAILED,
        EventType.EXECUTION_CANCELLED,
    }
)

_READ_ONLY: Mapping[str, Any] = MappingProxyType({})


def is_incomplete_event(event_type: EventType | str) -> bool:
    """True for events that start work without recording its outcome."""
    return EventType(event_type) in INCOMPLETE_EVENT_TYPES


def is_terminal_event(event_type: EventType | str) -> bool:
    """True for events that record a final outcome."""
    return EventType(event_type) in TERMINAL_EVENT_TYPES


def new_id(prefix: str) -> str:
    """Return a short, prefixed, unique identifier (``evt_9f2c...``)."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def describe_error(
    error: BaseException | str | None,
    *,
    include_traceback: bool = True,
) -> dict[str, Any]:
    """Render an exception (or message) as a JSON-safe error payload."""
    if error is None:
        return {"type": "ExecutionError", "message": "unspecified failure", "traceback": None}
    if isinstance(error, str):
        return {"type": "ExecutionError", "message": error, "traceback": None}

    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": None,
    }
    if include_traceback and error.__traceback__ is not None:
        payload["traceback"] = "".join(
            _traceback.format_exception(type(error), error, error.__traceback__)
        )
    return payload


@dataclass(frozen=True, slots=True)
class Event:
    """A single immutable entry in an execution's journal.

    Attributes:
        event_id: Globally unique id of this event.
        execution_id: The execution this event belongs to.
        sequence: 1-based, strictly increasing position within the execution.
        event_type: One of :class:`EventType`.
        timestamp: ISO-8601 UTC time the event was created.
        payload: JSON-serializable event detail (held read-only).
    """

    event_id: str
    execution_id: str
    sequence: int
    event_type: EventType
    timestamp: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Freeze the payload so the event is immutable end to end.
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        if not isinstance(self.event_type, EventType):
            object.__setattr__(self, "event_type", EventType(self.event_type))

    @classmethod
    def create(
        cls,
        execution_id: str,
        sequence: int,
        event_type: EventType | str,
        payload: Mapping[str, Any] | None = None,
        *,
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> "Event":
        """Build a new event, generating id/timestamp when not supplied."""
        return cls(
            event_id=event_id or new_id("evt"),
            execution_id=execution_id,
            sequence=int(sequence),
            event_type=EventType(event_type),
            timestamp=timestamp or utc_now_iso(),
            payload=payload if payload is not None else _READ_ONLY,
        )

    @property
    def is_tool_event(self) -> bool:
        return self.event_type in _TOOL_EVENT_TYPES

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dict copy of the event."""
        return {
            "event_id": self.event_id,
            "execution_id": self.execution_id,
            "sequence": self.sequence,
            "event_type": str(self.event_type),
            "timestamp": self.timestamp,
            "payload": json.loads(json.dumps(dict(self.payload))),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Event":
        """Rebuild an :class:`Event` from :meth:`to_dict` output."""
        return cls(
            event_id=data["event_id"],
            execution_id=data["execution_id"],
            sequence=int(data["sequence"]),
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            payload=data.get("payload") or {},
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"Event(sequence={self.sequence}, event_type={self.event_type}, "
            f"execution_id={self.execution_id!r}, payload={dict(self.payload)!r})"
        )


_TOOL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.TOOL_REQUESTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
        EventType.TOOL_RETRY_SCHEDULED,
        EventType.TOOL_CANCELLED,
        EventType.TOOL_TIMED_OUT,
    }
)
