"""The public entry point: :class:`Runtime`.

    runtime = Runtime("agent.db")
    execution = runtime.start(goal="Perform some calculations")
    execution.call("add", a=2, b=3)
    execution.checkpoint()
    execution.complete()

After a crash, the same database plus :meth:`Runtime.resume` gives an execution
back, starting from its latest checkpoint instead of from the first event.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .checkpoints import Checkpoint, CheckpointStore
from .events import Event, EventType, new_id
from .execution import Execution
from .exceptions import ExecutionExistsError
from .idempotency import (
    IdempotencyAction,
    IdempotencyRecord,
    IdempotencyStatus,
    IdempotencyStore,
    StoreIdempotencyGuard,
    unresolved_records,
)
from .journal import EventJournal
from .replay import ReplayEngine, ReplayResult, ReplayStep
from .recovery import RecoveryInfo, recover_execution
from .retry import RealSleeper, RetryPolicy, Sleeper
from .state import ExecutionState, reconstruct_state
from .storage import DbPath, SQLiteStore
from .tools import Tool, ToolRegistry, tool  # noqa: F401 - ``tool`` is the decorator

__all__ = ["Runtime"]


class Runtime:
    """Owns the SQLite store, the event journal, the checkpoints and the tools."""

    def __init__(
        self,
        db_path: DbPath = "agent.db",
        *,
        tools: Iterable[Tool | Callable[..., Any]] | None = None,
        register_default_tools: bool = True,
        auto_checkpoint: bool = False,
        sleeper: Sleeper | None = None,
    ) -> None:
        """
        Args:
            db_path: SQLite file to run against.
            tools: extra tools to register up front.
            register_default_tools: register the built-in arithmetic tools.
            auto_checkpoint: also snapshot the state at every lifecycle boundary
                (``complete``, ``fail``, ``mark_cancelled``, recovery
                resolutions). Off by default: milestones call
                :meth:`Execution.checkpoint` where they want a snapshot.
            sleeper: how retry backoff waits. Defaults to
                :class:`~agent_runtime.retry.RealSleeper`; pass a
                :class:`~agent_runtime.retry.RecordingSleeper` in a test to
                assert the schedule without spending it.
        """
        self.store = SQLiteStore(db_path)
        self.journal = EventJournal(self.store)
        self.checkpoints = CheckpointStore(self.store)
        #: The durable claim ledger (Milestone 4B): one row per idempotency key,
        #: in the same database as the events, so a claim and the history that
        #: explains it survive a restart together.
        self.idempotency = IdempotencyStore(self.store)
        #: Shared by every execution this runtime hands out.
        self._idempotency_guard = StoreIdempotencyGuard(self.idempotency)
        self.registry = ToolRegistry(tools, register_defaults=register_default_tools)
        self.auto_checkpoint = auto_checkpoint
        #: Shared by every execution this runtime hands out.
        self.sleeper = sleeper if sleeper is not None else RealSleeper()

    # -- tools ---------------------------------------------------------------

    def register_tool(
        self, item: Tool | Callable[..., Any], name: str | None = None
    ) -> Tool:
        """Make a tool callable from executions (``execution.call(name, ...)``)."""
        return self.registry.register(item, name)

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Decorator form of :meth:`register_tool`::

            @runtime.tool
            def greet(name: str): ...

            @runtime.tool(name="shout")
            def greet(name: str): ...

            @runtime.tool(retry_policy=RetryPolicy(max_attempts=3))
            def greet(name: str): ...

            @runtime.tool(timeout=5.0)            # Milestone 4C
            async def slow_tool(): ...

        ``timeout`` is the tool's *default* deadline; a call-level
        ``execution.call("slow_tool", timeout=1.0)`` overrides it. An invalid
        value is reported here, where it was written.
        """
        if func is None:
            def decorator(target: Callable[..., Any]) -> Tool:
                return self.registry.register(
                    tool(target, name=name, retry_policy=retry_policy, timeout=timeout)
                )

            return decorator
        return self.registry.register(
            tool(func, name=name, retry_policy=retry_policy, timeout=timeout)
        )

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
        return self._execution(execution_id)

    def resume(self, execution_id: str) -> Execution:
        """Rebuild an execution from its latest checkpoint and the events after it.

        With a checkpoint the state is read from the snapshot plus only the
        journal tail; without one the whole journal is folded, exactly as in
        Milestone 1. Either way the returned object's state is already loaded, so
        reading it does not replay the history again.

        The resumed execution reports ``RECOVERY_REQUIRED`` when the journal ends
        with a tool call that was started but never resolved. That is a statement
        about ambiguity, not an invitation to retry: nothing is re-run. Use
        :meth:`Execution.recovery_info` to see what is unresolved and
        :meth:`Execution.resolve_recovery` to decide what happens to it.
        """
        info = self.recovery_info(execution_id)
        return self._execution(execution_id, recovered_state=info.state)

    def get_events(self, execution_id: str) -> list[Event]:
        """Raw, sequence-ordered event history for an execution."""
        return self.journal.get_events(execution_id)

    def reconstruct_state(self, execution_id: str) -> ExecutionState:
        """Fold an execution's *entire* journal into its state.

        This is the ground truth a checkpointed recovery is checked against; see
        :meth:`recover_state` for the checkpoint-accelerated version.
        """
        return reconstruct_state(self.journal.get_events(execution_id))

    def recover_state(self, execution_id: str) -> ExecutionState:
        """The current state, from the latest checkpoint plus the events after it."""
        return self.recovery_info(execution_id).state

    def recovery_info(self, execution_id: str) -> RecoveryInfo:
        """What recovery makes of an execution: its state, source and open calls.

        Includes the idempotency keys that execution still holds (Milestone 4B),
        so an unresolved side effect is reported even when the journal itself
        looks clean, and :attr:`~agent_runtime.recovery.RecoveryInfo.recovery_state`
        is the one-line answer to "what should a restart do about this?"
        (Milestone 4C).
        """
        return recover_execution(
            self.journal,
            self.checkpoints,
            execution_id,
            idempotency=self._idempotency_guard,
        )

    def cancel(self, execution_id: str, reason: Any = None) -> Execution:
        """Cancel a running execution from outside it, and journal that fact.

        Milestone 4C's operator entry point -- the same durable decision
        :meth:`Execution.cancel` makes, reachable from a CLI in a different
        process::

            runtime.cancel("exec_123", reason="operator stopped the queue")

        It resumes the execution (so the journal, not this method, decides what
        is cancellable), calls :meth:`Execution.cancel`, and hands the execution
        back so the caller can inspect what it settled.

        A cancelled execution is ``CANCELLED`` afterwards, in this process and
        in every later one. Nothing retries it: a cancellation is a decision,
        and re-deciding it on restart would be the runtime second-guessing the
        application.
        """
        execution = self.resume(execution_id)
        execution.cancel(reason)
        return execution

    def get_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        """Every stored checkpoint of an execution, oldest first."""
        return self.checkpoints.list_for(execution_id)

    def latest_checkpoint(self, execution_id: str) -> Checkpoint | None:
        """The newest stored checkpoint of an execution, if it has one."""
        return self.checkpoints.get_latest(execution_id)

    def list_executions(self) -> list[str]:
        """Every execution id present in the journal."""
        return self.journal.list_execution_ids()

    # -- idempotency ---------------------------------------------------------

    def idempotency_record(self, key: str) -> IdempotencyRecord | None:
        """The stored claim for ``key``, or ``None`` when nothing claimed it.

        This is the read-only view an operator has after a crash: it says what
        the runtime *committed to* doing, never what the outside world did.
        """
        return self.idempotency.get(key)

    def idempotency_records(
        self,
        *,
        execution_id: str | None = None,
        status: IdempotencyStatus | None = None,
    ) -> tuple[IdempotencyRecord, ...]:
        """Every stored claim matching the filters, oldest first."""
        return self.idempotency.list_for(execution_id=execution_id, status=status)

    def pending_idempotency(
        self, execution_id: str | None = None
    ) -> tuple[IdempotencyRecord, ...]:
        """Claims with no recorded outcome -- the ones recovery must not retry."""
        return self.idempotency.pending_for(execution_id)

    def unresolved_idempotency(
        self, execution_id: str | None = None
    ) -> tuple[IdempotencyRecord, ...]:
        """The pending claims that need an explicit decision right now."""
        return unresolved_records(self.pending_idempotency(execution_id))

    def resolve_idempotency(
        self,
        key: str,
        action: IdempotencyAction,
        *,
        result: Any = None,
        error: Any = None,
        note: str | None = None,
    ) -> IdempotencyRecord:
        """Settle a key without going through an :class:`Execution`.

        The same three actions as
        :meth:`Execution.resolve_idempotency`, and the same refusals. The
        difference is the one an operator needs: this does not require the
        execution that holds the claim to be resumed, which is what makes it
        usable from a CLI in a different process. It is still an explicit
        decision -- the runtime never makes one on its own.

        Returns:
            The updated :class:`~agent_runtime.idempotency.IdempotencyRecord`.
        """
        return self._idempotency_guard.resolve(
            key, action, result=result, error=error, note=note
        )

    # -- replay --------------------------------------------------------------

    def replay(
        self,
        execution_id: str,
        *,
        from_sequence: int = 0,
        on_step: Callable[[ReplayStep], None] | None = None,
    ) -> ReplayResult:
        """Replay an execution from its journal, running no tools.

        The replay re-executes the runtime's own logic -- the same
        :meth:`Execution.call` path, the same reducers -- but every tool call is
        answered from the recorded history instead of being executed, so a
        ``create_github_issue`` or a ``send_email`` cannot happen twice.

            result = runtime.replay(execution_id)
            result.matched          # True: the replay agreed with the original
            result.final_state      # == runtime.reconstruct_state(execution_id)

        Args:
            execution_id: The execution to replay.
            from_sequence: Replay only what came after this sequence, starting
                from the state the history has at that point (a stored checkpoint
                if there is one). ``0`` replays the whole execution.

        Returns:
            A :class:`~agent_runtime.replay.ReplayResult` with the replayed state,
            how much was replayed, and whether it matched.

        Raises:
            ExecutionNotFoundError: the execution has no events.
            ReplayMismatchError: the replay diverged from the recorded history,
                or the replayed state differs from the original. The exception
                names the sequence and shows both sides -- a mismatch is never
                reported as a bare ``False``.

        Replay is read-only: the original journal is not modified.
        """
        return self.replay_engine(
            execution_id, from_sequence=from_sequence, on_step=on_step
        ).run()

    def replay_engine(
        self,
        execution_id: str,
        *,
        from_sequence: int = 0,
        on_step: Callable[[ReplayStep], None] | None = None,
    ) -> ReplayEngine:
        """The :class:`~agent_runtime.replay.ReplayEngine` for an execution.

        Use this when you want to drive the replay yourself -- inspect
        ``engine.replay_execution.steps``, replay calls one at a time -- instead
        of taking the single :class:`~agent_runtime.replay.ReplayResult`.

        ``on_step`` is called with each :class:`~agent_runtime.replay.ReplayStep`
        as it is served, which is how the CLI prints progress while the replay runs.
        """
        return ReplayEngine(
            self.journal,
            self.checkpoints,
            execution_id,
            from_sequence=from_sequence,
            on_step=on_step,
        )

    # -- internals -----------------------------------------------------------

    def _execution(
        self, execution_id: str, *, recovered_state: ExecutionState | None = None
    ) -> Execution:
        return Execution(
            self.journal,
            self.registry,
            execution_id,
            checkpoints=self.checkpoints,
            recovered_state=recovered_state,
            auto_checkpoint=self.auto_checkpoint,
            sleeper=self.sleeper,
            idempotency=self._idempotency_guard,
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Runtime(db_path={self.store.db_path!r}, tools={self.registry.names()})"
