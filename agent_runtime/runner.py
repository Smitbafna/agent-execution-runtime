"""The tool runner: the one place that decides *how* a tool call is carried out.

There are exactly two modes, and the difference between them is the whole point
of Milestone 3::

    NORMAL   ToolRequested -> ToolStarted -> execute tool -> ToolCompleted
    REPLAY   ToolRequested -> look up the recorded call -> recorded result

:class:`ToolRunner` runs the registered function. :class:`ReplayToolRunner`
never touches it: given the same request it hands back the outcome the journal
recorded. Because replay dispatches through this same object, the guarantee is
structural -- a replay has no code path that reaches a tool function, so a
``create_github_issue`` cannot run twice no matter what the caller does.

Scope note: replay substitutes *recorded* tool results. It does not sandbox or
intercept anything else -- ``time``, ``random``, the network and environment
variables are external inputs an execution must treat as tools if they influence
its outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from .events import describe_error
from .exceptions import ToolInvocationError
from .state import ToolCallStatus
from .tools import ToolRegistry, make_jsonable

__all__ = [
    "ToolRunnerMode",
    "ToolRequest",
    "ToolOutcome",
    "ToolRunner",
]


class ToolRunnerMode(StrEnum):
    """How a tool call is carried out."""

    #: Call the registered function and journal what happened.
    NORMAL = "NORMAL"
    #: Return the outcome the journal already recorded. No function is called.
    REPLAY = "REPLAY"


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """One request to run a tool: a name plus normalized, JSON-safe arguments."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What a tool call produced.

    A successful call carries ``result``; a failed one carries the JSON-safe
    ``error`` payload that the journal will store, so the two halves of a call --
    the outcome and its record -- cannot disagree.
    """

    status: ToolCallStatus
    result: Any = None
    error: Mapping[str, Any] | None = None
    #: The live exception, kept only so the caller can chain it (``raise ... from``).
    #: Never journalled, and ignored for equality.
    cause: BaseException | None = field(default=None, compare=False, repr=False)

    @property
    def succeeded(self) -> bool:
        return self.status is ToolCallStatus.COMPLETED

    @property
    def failed(self) -> bool:
        return self.status in (ToolCallStatus.FAILED, ToolCallStatus.CANCELLED)

    @classmethod
    def completed(cls, result: Any) -> "ToolOutcome":
        return cls(status=ToolCallStatus.COMPLETED, result=result)

    @classmethod
    def failed_with(cls, error: BaseException | str) -> "ToolOutcome":
        return cls(
            status=ToolCallStatus.FAILED,
            error=describe_error(error),
            cause=error if isinstance(error, BaseException) else None,
        )


class ToolRunner:
    """Runs tools for real. This is the only runner that may call a function."""

    mode = ToolRunnerMode.NORMAL

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def run(self, request: ToolRequest) -> ToolOutcome:
        """Invoke the registered tool and return what it produced.

        A tool that raises is an outcome, not an exception escaping this method:
        the caller journals ``ToolFailed`` from it and then re-raises, so the
        failure is durable before anything observes it.
        """
        target = self.registry.get(request.tool)
        bound = target.bind((), request.arguments)
        try:
            return ToolOutcome.completed(make_jsonable(target.invoke(bound)))
        except Exception as exc:  # noqa: BLE001 - the failure itself is the outcome
            return ToolOutcome.failed_with(exc)

    def invocation_error(
        self, request: ToolRequest, outcome: ToolOutcome
    ) -> ToolInvocationError:
        """Build the error to raise after a failed call has been journalled."""
        error = outcome.error or {}
        return ToolInvocationError(
            f"Tool {request.tool!r} failed: {error.get('message', 'unknown error')}",
            tool_name=request.tool,
            error_type=error.get("type") or "ToolInvocationError",
            traceback_text=error.get("traceback"),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{type(self).__name__}(mode={self.mode})"
