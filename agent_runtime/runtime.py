"""The public entry point: :class:`Runtime`.

    runtime = Runtime("agent.db")
    execution = runtime.start(goal="Perform some calculations")
    execution.call("add", a=2, b=3)
    execution.complete()
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .events import Event, EventType, new_id
from .execution import Execution
from .exceptions import ExecutionExistsError, ExecutionNotFoundError
from .journal import EventJournal
from .state import ExecutionState, reconstruct_state
from .storage import DbPath, SQLiteStore
from .tools import Tool, ToolRegistry

__all__ = ["Runtime"]


class Runtime:
    """Owns the SQLite store, the event journal and the tool registry."""

    def __init__(
        self,
        db_path: DbPath = "agent.db",
        *,
        tools: Iterable[Tool | Callable[..., Any]] | None = None,
        register_default_tools: bool = True,
    ) -> None:
        self.store = SQLiteStore(db_path)
        self.journal = EventJournal(self.store)
        self.registry = ToolRegistry(tools, register_defaults=register_default_tools)

    # -- tools ---------------------------------------------------------------

    def register_tool(
        self, item: Tool | Callable[..., Any], name: str | None = None
    ) -> Tool:
        """Make a tool callable from executions (``execution.call(name, ...)``)."""
        return self.registry.register(item, name)

    def tool(self, func: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
        """Decorator form of :meth:`register_tool`::

            @runtime.tool
            def greet(name: str): ...

            @runtime.tool(name="shout")
            def greet(name: str): ...
        """
        if func is None:
            def decorator(target: Callable[..., Any]) -> Tool:
                return self.registry.register(target, kwargs.get("name"))

            return decorator
        return self.registry.register(func, kwargs.get("name"))

    # -- executions ----------------------------------------------------------

    def start(self, goal: str, *, execution_id: str | None = None) -> Execution:
        """Begin a new execution and journal ``ExecutionStarted``."""
        execution_id = execution_id or new_id("exec")
        if self.journal.has_execution(execution_id):
            raise ExecutionExistsError(
                f"Execution {execution_id!r} already exists in the journal"
            )
        self.journal.append_event(
            execution_id, EventType.EXECUTION_STARTED, {"goal": goal}
        )
        return Execution(self.journal, self.registry, execution_id)

    def resume(self, execution_id: str) -> Execution:
        """Rebuild an execution from its journal.

        Loads the events from SQLite and reconstructs the state; the returned
        object reads its state from the journal the same way a fresh execution
        does. Unfinished tool calls are *not* automatically continued in
        Milestone 1.
        """
        if not self.journal.has_execution(execution_id):
            raise ExecutionNotFoundError(f"No journal found for execution {execution_id!r}")
        execution = Execution(self.journal, self.registry, execution_id)
        # Touch the state so reconstruction problems surface here, not later.
        execution.reconstruct_state()
        return execution

    def get_events(self, execution_id: str) -> list[Event]:
        """Raw, sequence-ordered event history for an execution."""
        return self.journal.get_events(execution_id)

    def reconstruct_state(self, execution_id: str) -> ExecutionState:
        """Fold an execution's events into its current state."""
        return reconstruct_state(self.journal.get_events(execution_id))

    def list_executions(self) -> list[str]:
        """Every execution id present in the journal."""
        return self.journal.list_execution_ids()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Runtime(db_path={self.store.db_path!r}, tools={self.registry.names()})"
