"""Deterministic replay: re-run an execution from its journal, run no tools.

The invariant this milestone is built around:

    Given the same recorded execution history, replay reproduces the same
    execution state without repeating any external side effect.

How it works
------------

A replay is not a shortcut that loads the final answer. It re-executes the
*same* runtime logic the original run used -- the same
:class:`~agent_runtime.execution.Execution` object, the same
:meth:`~agent_runtime.execution.Execution.call` path, the same reducers in
:mod:`agent_runtime.state` -- by handing that object two substitutions:

1. a **REPLAY tool runner** that answers every call with the result (or the
   failure) the journal recorded for it, instead of invoking the function;
2. an **in-memory journal**, so the replayed events are built from scratch and
   the original history is never read for anything but the recorded tool
   outcomes, and is never written to at all.

Because the runtime logic is genuinely re-run, the replayed state is *derived*,
not copied: if a call diverges -- wrong tool, wrong arguments, a missing or
extra call -- the substituted runner raises
:class:`~agent_runtime.exceptions.ReplayMismatchError` at the sequence where
replay and the journal disagree, rather than letting the run continue on a lie.

Replay never executes a real tool, and that is structural rather than a
convention: :class:`ReplayToolRunner` is the only thing that can satisfy a call
during replay, and it never reaches a tool function. ``create_file``,
``delete_file``, ``send_email``, ``create_github_issue``, ``database_write`` and
``http_post`` are all equally unreachable.

Scope note: replay substitutes *recorded tool outputs*. It deliberately does not
attempt to sandbox the rest of Python -- ``time``, ``random``, UUID generation,
environment variables, network and LLM responses are external inputs that an
execution must expose as tools if they influence its outcome. That is recorded
in the determinism tests (``tests/test_replay.py``) rather than papered over with
global interception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .checkpoints import CheckpointStore
from .events import Event, EventType, new_id
from .exceptions import (
    ExecutionNotFoundError,
    ReplayError,
    ReplayMismatchError,
    StateReconstructionError,
    ToolInvocationError,
)
from .execution import Execution
from .journal import EventJournal
from .runner import ToolOutcome, ToolRequest, ToolRunner, ToolRunnerMode
from .state import (
    ExecutionState,
    ToolCall,
    ToolCallStatus,
    apply_event,
    detect_incomplete_tools,
    finalize_state,
    initial_state,
    reconstruct_state,
)
from .tools import make_jsonable

__all__ = [
    "RecordedToolCall",
    "ReplayResult",
    "ReplayStep",
    "ReplayToolRunner",
    "ReplayExecution",
    "ReplayEngine",
    "replay_execution",
]


# ---------------------------------------------------------------------------
# The recorded history, distilled to the tool calls a replay must reproduce
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordedToolCall:
    """One tool call as the journal recorded it: what was asked, and what came back.

    A replay walks these in order. ``call_id`` and ``requested_sequence`` pin
    the call to its exact place in the history, which is what lets a mismatch
    name *where* replay diverged rather than just *that* it did.
    """

    call_id: str
    index: int
    tool: str
    arguments: Mapping[str, Any]
    requested_sequence: int
    status: ToolCallStatus
    result: Any = None
    error: Mapping[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ToolCallStatus.COMPLETED

    @property
    def settled(self) -> bool:
        """True when the journal recorded an outcome for this call."""
        return not self.status.is_incomplete

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "index": self.index,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "requested_sequence": self.requested_sequence,
            "status": str(self.status),
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_tool_call(cls, call: ToolCall, index: int) -> "RecordedToolCall":
        return cls(
            call_id=call.call_id,
            index=index,
            tool=call.tool,
            arguments=dict(call.arguments),
            requested_sequence=call.requested_sequence,
            status=call.status,
            result=call.result,
            error=call.error,
        )

    @classmethod
    def recorded_from(cls, state: ExecutionState, *, since: int = 0) -> tuple["RecordedToolCall", ...]:
        """Every tool call at or after ``since``, which a replay must reproduce."""
        return tuple(
            cls.from_tool_call(call, index)
            for index, call in enumerate(
                call for call in state.tool_calls if call.requested_sequence > since
            )
        )


# ---------------------------------------------------------------------------
# The REPLAY tool runner
# ---------------------------------------------------------------------------


class ReplayToolRunner(ToolRunner):
    """A :class:`ToolRunner` that answers from the journal and never runs a tool.

    It holds the execution's :class:`RecordedToolCall` list in order. Each
    :meth:`run` matches the incoming :class:`ToolRequest` against the *next*
    recorded call and returns that call's recorded outcome:

    * a different tool name or a different set of arguments is a
      :class:`~agent_runtime.exceptions.ReplayMismatchError`, raised at the
      recorded sequence;
    * a recorded *failure* is reproduced as a failure, carrying the recorded
      error type and message, so replay fails exactly where the original did;
    * a call with no recorded outcome cannot be reproduced at all -- the journal
      does not say what became of it -- and that is reported rather than
      invented.

    Because this class never touches a tool function, replaying an execution
    cannot repeat its side effects. There is no registry lookup and no
    ``invoke`` anywhere on this path.
    """

    mode = ToolRunnerMode.REPLAY

    def __init__(self, recorded: tuple[RecordedToolCall, ...], execution_id: str) -> None:
        # Deliberately *not* a ToolRegistry: during replay there is nothing to run,
        # and holding one would be the only way to accidentally run something.
        self.registry = None  # type: ignore[assignment]
        self._recorded = recorded
        self._execution_id = execution_id
        self._cursor = 0

    @property
    def cursor(self) -> int:
        """How many recorded calls have been replayed so far."""
        return self._cursor

    @property
    def remaining(self) -> int:
        """Recorded calls that have not been replayed yet."""
        return len(self._recorded) - self._cursor

    def run(self, request: ToolRequest) -> ToolOutcome:
        """Return the recorded outcome for ``request``. No tool is ever invoked."""
        call = self._match(request)
        self._cursor += 1
        if call.succeeded:
            return ToolOutcome.completed(call.result)
        return ToolOutcome(status=call.status, error=call.error)

    def check(self, request: ToolRequest) -> RecordedToolCall:
        """Validate a request against the recorded history without consuming it.

        Lets a replay find out that it is about to diverge *before* it journals
        anything, so a mismatch leaves no half-written tool call behind. The
        cursor only moves once the call actually happens.
        """
        return self._match(request)

    def invocation_error(
        self, request: ToolRequest, outcome: ToolOutcome
    ) -> ToolInvocationError:
        """Rebuild the failure the original run raised, from the recorded error."""
        error = dict(outcome.error or {})
        message = error.get("message", "the recorded tool failed")
        return ToolInvocationError(
            f"Tool {request.tool!r} failed during replay: {message}",
            tool_name=request.tool,
            error_type=error.get("type") or "RecordedToolFailure",
            traceback_text=error.get("traceback"),
        )

    # -- internals -----------------------------------------------------------

    def _match(self, request: ToolRequest) -> RecordedToolCall:
        """Find the recorded call this request corresponds to, or explain why not."""
        if self._cursor >= len(self._recorded):
            raise ReplayMismatchError(
                f"replay made an extra tool call (#{self._cursor + 1}); the recorded "
                f"execution only has {len(self._recorded)}",
                kind="unexpected_tool_call",
                execution_id=self._execution_id,
                expected={"recorded_calls": len(self._recorded)},
                received=request.to_dict(),
            )

        call = self._recorded[self._cursor]
        if call.tool != request.tool:
            raise ReplayMismatchError(
                f"replay called {request.tool!r} where the recorded history has "
                f"{call.tool!r} (call #{call.index + 1})",
                kind="tool_name",
                execution_id=self._execution_id,
                sequence=call.requested_sequence,
                expected={"tool": call.tool, "arguments": dict(call.arguments)},
                received=request.to_dict(),
            )
        if dict(request.arguments) != dict(call.arguments):
            raise ReplayMismatchError(
                f"replay called {call.tool!r} with different arguments than the "
                f"recorded call #{call.index + 1}",
                kind="arguments",
                execution_id=self._execution_id,
                sequence=call.requested_sequence,
                expected={"tool": call.tool, "arguments": dict(call.arguments)},
                received=request.to_dict(),
            )
        if not call.settled:
            raise ReplayMismatchError(
                f"recorded call #{call.index + 1} of {call.tool!r} never settled "
                f"(status {call.status}); replay has no recorded outcome to reproduce",
                kind="missing_recorded_outcome",
                execution_id=self._execution_id,
                sequence=call.requested_sequence,
                expected={"tool": call.tool, "status": str(call.status)},
                received=request.to_dict(),
            )
        return call

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"ReplayToolRunner(execution_id={self._execution_id!r}, "
            f"recorded={len(self._recorded)}, cursor={self._cursor})"
        )


# ---------------------------------------------------------------------------
# The replay trace (in memory; never written to the original journal)
# ---------------------------------------------------------------------------


class _ReplayJournal:
    """A journal-shaped sink that builds events in memory and never touches SQLite.

    It implements just the surface :class:`Execution` reads and writes -- so the
    real :meth:`Execution.call` code runs unchanged -- but every event lands in a
    list instead of the ``events`` table. Sequences start at whatever the replay
    resumes from, so a replay from checkpoint 500 produces sequences 501, 502,
    ... instead of pretending the execution begins at 1.

    Events are *built* by the runtime, not copied: the type and payload are
    whatever the replayed code produced. The only thing carried over from the
    recorded event at the same position is its identity -- ``event_id`` and
    ``timestamp`` -- so that volatile facts like a tool's start time stay the
    ones the original recorded and the two states can be compared field for
    field.
    """

    def __init__(
        self,
        execution_id: str,
        *,
        first_sequence: int = 1,
        mirror: Sequence[Event] = (),
    ) -> None:
        self._execution_id = execution_id
        self._first_sequence = first_sequence
        self._mirror = list(mirror)
        self._events: list[Event] = []
        self._next_sequence = first_sequence

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def append_event(
        self,
        execution_id: str,
        event_type: EventType | str,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        index = len(self._events)
        if index < len(self._mirror):
            original = self._mirror[index]
            event = Event.create(
                execution_id,
                self._next_sequence,
                event_type,
                payload or {},
                event_id=original.event_id,
                timestamp=original.timestamp,
            )
        else:
            # More events than the original had: the replay is inventing history.
            # Sequence continuity is still enforced, but let the engine's
            # divergence checks be what reports it.
            event = Event.create(execution_id, self._next_sequence, event_type, payload or {})
        self._next_sequence += 1
        self._events.append(event)
        return event

    # -- read surface used by Execution --------------------------------------

    def get_events(self, execution_id: str) -> list[Event]:
        return [e for e in self._events if e.execution_id == execution_id]

    def get_events_from(self, execution_id: str, after_sequence: int) -> list[Event]:
        return [
            e
            for e in self._events
            if e.execution_id == execution_id and e.sequence > after_sequence
        ]

    def get_last_sequence(self, execution_id: str) -> int:
        return max(
            (e.sequence for e in self._events if e.execution_id == execution_id),
            default=self._first_sequence - 1,
        )

    def get_last_event(self, execution_id: str) -> Event | None:
        events = self.get_events(execution_id)
        return events[-1] if events else None

    def count_events(self, execution_id: str) -> int:
        return len(self.get_events(execution_id))

    def has_execution(self, execution_id: str) -> bool:
        return self.count_events(execution_id) > 0
# ---------------------------------------------------------------------------
# The replay trace (in memory; never written to the original journal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One tool call as the replay served it.

    ``executed`` is always ``False`` for a replay: it records that the result
    came from the journal. It stays an explicit field because "this result was
    replayed, not computed" is exactly the claim that has to remain visible.
    """

    sequence: int
    tool: str
    arguments: Mapping[str, Any]
    status: ToolCallStatus
    result: Any = None
    error: Mapping[str, Any] | None = None
    executed: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status is ToolCallStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "status": str(self.status),
            "result": self.result,
            "error": self.error,
            "executed": self.executed,
        }

    def __str__(self) -> str:
        args = ", ".join(f"{key}={value!r}" for key, value in self.arguments.items())
        if self.succeeded:
            return f"{self.tool}({args}) -> {self.result!r}"
        message = (self.error or {}).get("message", "failed")
        return f"{self.tool}({args}) !! {message}"


# ---------------------------------------------------------------------------
# The replay execution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The replay execution
# ---------------------------------------------------------------------------


class ReplayExecution(Execution):
    """An :class:`Execution` that re-runs recorded calls against the journal.

    Everything is the real thing -- the same :meth:`call`, the same lifecycle
    checks, the same reducers -- with three substitutions, all of them in this
    class rather than scattered through the engine:

    * the journal is in memory, so replaying cannot modify the original history;
    * the tool runner is :class:`ReplayToolRunner`, so no tool function runs;
    * calls are normalized against the *recorded* call, so a replay works even
      when the tool function no longer exists or no longer has that signature.

    :attr:`steps` is the in-memory trace of what the replay served. It describes
    the replay and is never persisted: the original journal is only ever read.
    """

    def __init__(
        self,
        journal: Any,
        execution_id: str,
        *,
        runner: ReplayToolRunner,
        base_state: ExecutionState | None = None,
        recorded: tuple[RecordedToolCall, ...] = (),
        on_step: Callable[[ReplayStep], None] | None = None,
    ) -> None:
        # registry=None: this execution never resolves a tool, so there is
        # nothing for a registry to do. Replay must not depend on the tool being
        # importable in the replaying process. The runner is passed explicitly,
        # so it is the REPLAY one and not the default NORMAL one.
        super().__init__(journal, None, execution_id, runner=runner)  # type: ignore[arg-type]
        self._replay_runner = runner
        self._recorded = recorded
        self._cursor = 0
        self._steps: list[ReplayStep] = []
        self._errors: list[Mapping[str, Any]] = []
        self._on_step = on_step
        self._base_state = base_state
        if base_state is not None:
            # Seed the cache so ``status``/``_require_running`` see the state the
            # replay resumes from rather than an empty one.
            self._state = base_state

    # -- replay surface ------------------------------------------------------

    @property
    def runner(self) -> ReplayToolRunner:
        """The REPLAY runner backing this execution."""
        return self._replay_runner

    @property
    def steps(self) -> tuple[ReplayStep, ...]:
        """The in-memory trace of calls served during this replay."""
        return tuple(self._steps)

    @property
    def errors(self) -> tuple[Mapping[str, Any], ...]:
        """The recorded failures this replay reproduced, in order."""
        return tuple(self._errors)

    @property
    def recorded_calls(self) -> tuple[RecordedToolCall, ...]:
        """The calls this replay must reproduce, in recorded order."""
        return self._recorded

    @property
    def replayed_calls(self) -> int:
        """How many recorded calls have been served so far."""
        return self._cursor

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Replay one recorded call: validate it against the journal, serve it.

        The check happens *before* anything is journalled, exactly as the NORMAL
        path validates arguments before it writes ``ToolRequested``. A mismatch
        therefore leaves the replay exactly as it was -- no half-written call, no
        advanced cursor -- instead of half-applying a call it then rejects.
        """
        self._replay_runner.check(self._prepare_request(name, args, kwargs))
        return super().call(name, *args, **kwargs)

    def replay_recorded_call(self, recorded: RecordedToolCall) -> Any:
        """Replay one recorded call by name and arguments, returning its result.

        A recorded failure is reproduced as a failure -- the same
        :class:`~agent_runtime.exceptions.ToolInvocationError`, carrying the
        recorded error type and message -- without the tool running.

        A call the crash left open (requested or started, never settled) has no
        recorded outcome to serve. It is replayed as far as the journal goes:
        the request and start are re-emitted, no outcome is invented, and the
        replay ends in the same ``RECOVERY_REQUIRED`` the original did. Nothing
        is retried.
        """
        if not recorded.settled:
            self._emit_open_call(recorded)
            self._record_step(recorded, None)
            return None

        try:
            result = self.call(recorded.tool, **dict(recorded.arguments))
        except ToolInvocationError as exc:
            self._record_step(recorded, exc)
            # The original execution survived this failure (it went on to call
            # another tool, or to fail deliberately), so the replay has to as well:
            # the failure is collected, not propagated.
            self._errors.append(
                {
                    "tool": recorded.tool,
                    "sequence": recorded.requested_sequence,
                    "call_id": recorded.call_id,
                    "error": dict(recorded.error or {}),
                }
            )
            return None

        self._record_step(recorded, None, result=result)
        return result

    def replay_all(self) -> int:
        """Replay every recorded call in order; returns how many were served.

        Each call goes through the full :meth:`Execution.call` path, so the
        replayed events are produced by the runtime itself rather than copied
        from the journal.
        """
        for recorded in self._recorded:
            self.replay_recorded_call(recorded)
        return self._cursor

    def record_lifecycle(self, event: Event) -> None:
        """Re-emit a recorded lifecycle event (``ExecutionStarted``, ``...Completed``).

        These carry no tool result to substitute, so replay simply puts the
        recorded event through the in-memory journal and lets the reducers
        interpret it -- the same code path the original run used.
        """
        self._append(event.event_type, dict(event.payload))

    def checkpoint(self) -> Any:
        """Replay never writes: checkpointing would mutate the original store."""
        raise ReplayError(
            "replay is read-only and cannot create checkpoints; the state it "
            "produces is derived, not stored"
        )

    # -- internals -----------------------------------------------------------

    @property
    def state(self) -> ExecutionState:
        """Base state + the events this replay produced, folded by the real reducers.

        ``Execution`` rebuilds from its journal alone; a checkpoint replay starts
        from a base state the journal does not contain, so the base is folded in
        first. That fold is what makes the checkpoint and full replays agree: both
        end up passing the same events through the same reducers.
        """
        if self._state is None:
            events = self.events
            self._state = self._fold(self._base_state, events)
        return self._state

    def _fold(self, base: ExecutionState | None, events: list[Event]) -> ExecutionState:
        """Apply ``events`` to ``base`` with the runtime's own reducers."""
        if base is None or not events:
            return reconstruct_state(events)
        if base.execution_id and events and events[0].execution_id != base.execution_id:
            raise StateReconstructionError(
                f"replay base state is for execution {base.execution_id!r} but the "
                f"replayed events belong to {events[0].execution_id!r}"
            )
        for event in events:
            base = apply_event(base, event)
        return finalize_state(base)

    def _emit_open_call(self, recorded: RecordedToolCall) -> None:
        """Re-emit the request and start of a call the journal never settled."""
        self._new_call_id()  # keep the recorded identity, as a settled call would
        self._append(
            EventType.TOOL_REQUESTED,
            {
                "call_id": recorded.call_id,
                "tool": recorded.tool,
                "arguments": dict(recorded.arguments),
            },
        )
        self._append(
            EventType.TOOL_STARTED, {"call_id": recorded.call_id, "tool": recorded.tool}
        )

    def _prepare_request(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ToolRequest:
        """Normalize the call the way the *original* run did: from the journal.

        Deliberately does not touch the tool registry. Replay compares what was
        asked with what was recorded; it does not re-derive arguments from a
        signature that may have changed since, and it must work for a tool whose
        function is not registered in this process at all.
        """
        if args:
            # Positional arguments cannot be mapped to names without the
            # registered signature; recorded calls are replayed by keyword.
            raise ReplayMismatchError(
                f"replay of {name!r} used positional arguments {args!r}; recorded "
                "tool calls are replayed with keyword arguments",
                kind="arguments",
                execution_id=self.id,
                received={"tool": name, "args": list(args), "kwargs": dict(kwargs)},
            )
        return ToolRequest(tool=name, arguments=make_jsonable(dict(kwargs)))

    def _new_call_id(self) -> str:
        """Keep the identity the journal gave this call.

        Reusing the recorded ``call_id`` is what lets the replayed state be
        compared with the original field for field, rather than merely having the
        same shape.
        """
        if self._cursor < len(self._recorded):
            call_id = self._recorded[self._cursor].call_id
            self._cursor += 1
            return call_id
        return new_id("call")

    def _record_step(
        self,
        recorded: RecordedToolCall,
        exc: ToolInvocationError | None,
        *,
        result: Any = None,
    ) -> None:
        """Append one entry to the in-memory replay trace, and notify any listener."""
        step = ReplayStep(
            sequence=recorded.requested_sequence,
            tool=recorded.tool,
            arguments=dict(recorded.arguments),
            status=recorded.status,
            result=result if exc is None else None,
            error=recorded.error if exc is not None else None,
            # The whole point: nothing was executed, only looked up.
            executed=False,
        )
        self._steps.append(step)
        if self._on_step is not None:
            self._on_step(step)


# ---------------------------------------------------------------------------
# The replay result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """What one replay produced, and whether it agreed with the original.

    ::

        result = runtime.replay(execution_id)
        print(result)
        result.matched            # True
        result.final_state.status # ExecutionStatus.COMPLETED
        result.tools_replayed     # 13

    Attributes:
        execution_id: The execution that was replayed.
        final_state: The state the replay derived, by re-running the runtime.
        original_state: The state the recorded history folds into.
        events_replayed: How many events the replay produced.
        tools_replayed: How many recorded tool calls were served from the journal.
        duration: Wall-clock seconds the replay took.
        matched: True when the replayed state equals the original.
        from_sequence: Where the replay started (``0`` for a full replay).
        steps: The in-memory replay trace.
        journal_unchanged: Events in the original journal before and after; equal
            is the read-only guarantee of §8, asserted by the engine itself.
    """

    execution_id: str
    final_state: ExecutionState
    original_state: ExecutionState
    events_replayed: int
    tools_replayed: int
    duration: float
    matched: bool
    from_sequence: int = 0
    steps: tuple[ReplayStep, ...] = ()
    journal_unchanged: bool = True
    errors: tuple[Mapping[str, Any], ...] = ()

    # The spec's example reads ``replayed.state``; keep both names working.
    @property
    def state(self) -> ExecutionState:
        """Alias for :attr:`final_state`."""
        return self.final_state

    @property
    def status(self) -> str:
        """``MATCHED`` or ``MISMATCHED``."""
        return "MATCHED" if self.matched else "MISMATCHED"

    @property
    def tool_names(self) -> list[str]:
        """The tools served during this replay, in order."""
        return [step.tool for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "matched": self.matched,
            "status": self.status,
            "events_replayed": self.events_replayed,
            "tools_replayed": self.tools_replayed,
            "duration": round(self.duration, 6),
            "from_sequence": self.from_sequence,
            "journal_unchanged": self.journal_unchanged,
            "original_state": self.original_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

    def __str__(self) -> str:
        lines = [
            "ReplayResult",
            "",
            f"Execution: {self.execution_id}",
            f"Status: {self.status}",
            "",
            f"Events replayed: {self.events_replayed}",
            f"Tools replayed: {self.tools_replayed}",
        ]
        if self.from_sequence:
            lines.append(f"Resumed from sequence: {self.from_sequence}")
        if not self.journal_unchanged:
            lines.append("Original journal: MUTATED")
        lines.extend(
            [
                "",
                "Original state:",
                f"    status = {self.original_state.status}",
                f"    tool_calls = {len(self.original_state.tool_calls)}",
                "",
                "Replay state:",
                f"    status = {self.final_state.status}",
                f"    tool_calls = {len(self.final_state.tool_calls)}",
            ]
        )
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

#: Events that say "this execution began" or "it ended". These are replayed by
#: re-emitting them into the in-memory journal, because they carry no tool
#: result to substitute -- the runtime's own reducers decide what they mean.
_LIFECYCLE_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.EXECUTION_STARTED,
        EventType.EXECUTION_COMPLETED,
        EventType.EXECUTION_FAILED,
        EventType.EXECUTION_CANCELLED,
    }
)

#: Tool events a replay must not re-emit itself: :meth:`Execution.call` produces
#: them as it re-runs, so replaying them again would double-count them. Only
#: ``ToolRequested`` is dispatched, and it dispatches the whole call.
_CALL_PHASE_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.TOOL_REQUESTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
        EventType.TOOL_CANCELLED,
    }
)


class ReplayEngine:
    """Replays one execution's recorded history and checks it against itself.

    The engine's whole job is to answer three questions without ever running a
    tool: what did the original do, what does the same code do when fed those
    answers, and do the two agree?

    It works by constructing a real :class:`ReplayExecution` over an in-memory
    journal, replaying the recorded lifecycle events, and driving
    :meth:`ReplayExecution.replay_recorded_call` for every recorded call in
    order. The state that comes out is therefore *produced by the runtime*, not
    copied from the journal -- which is what makes a divergence visible at all.
    """

    def __init__(
        self,
        journal: EventJournal,
        checkpoints: CheckpointStore,
        execution_id: str,
        *,
        from_sequence: int = 0,
        on_step: Callable[[ReplayStep], None] | None = None,
    ) -> None:
        self.journal = journal
        self.checkpoints = checkpoints
        self.execution_id = execution_id
        self.from_sequence = from_sequence
        self.on_step = on_step
        self._execution: ReplayExecution | None = None
        self._original_state: ExecutionState | None = None
        self._recorded: tuple[RecordedToolCall, ...] = ()
        self._mirror: Sequence[Event] = ()
        self._replay_journal: _ReplayJournal | None = None
        self._validate_start()

    # -- public API ----------------------------------------------------------

    def prepare(self) -> ReplayExecution:
        """Build the replay execution without driving it.

        :meth:`run` is "prepare, then replay every recorded call". This is the
        first half on its own, for a caller that wants to drive the replay
        itself -- asking for the recorded calls one at a time, or asking for
        something else and finding out whether it matches.

        The returned execution is a real :class:`~agent_runtime.execution.Execution`
        with a REPLAY runner, so ``.call(...)`` on it behaves exactly like the
        original run's ``.call(...)`` except that every answer comes from the
        journal.
        """
        if self._execution is not None:
            return self._execution

        original_state = reconstruct_state(self.journal.get_events(self.execution_id))
        base_state = self._base_state()
        recorded = RecordedToolCall.recorded_from(original_state, since=self.from_sequence)
        mirror = self.journal.get_events_from(self.execution_id, self.from_sequence)

        self._original_state = original_state
        self._recorded = recorded
        self._mirror = mirror
        self._replay_journal = _ReplayJournal(
            self.execution_id, first_sequence=self.from_sequence + 1, mirror=mirror
        )
        self._execution = ReplayExecution(
            self._replay_journal,
            self.execution_id,
            runner=ReplayToolRunner(recorded, self.execution_id),
            base_state=base_state,
            recorded=recorded,
            on_step=self.on_step,
        )

        # Start the replayed execution before handing it to anyone. Without this
        # the handle would have no goal and no execution id, and could not be
        # driven directly; with it, a caller can call it exactly as the original
        # run did. ``_drive`` skips these, so they are never emitted twice.
        for event in mirror:
            if event.event_type is not EventType.EXECUTION_STARTED:
                break
            self._execution.record_lifecycle(event)
        return self._execution

    def run(self) -> ReplayResult:
        """Replay the execution and return the result.

        Raises:
            ReplayMismatchError: the replay diverged from the recorded history,
                or the replayed state differs from the original state. A mismatch
                is never reported by returning ``matched=False`` alone: the
                caller gets the exception, with the sequence and both sides of
                the divergence.
        """
        started = time.perf_counter()
        events_before = self.journal.count_events(self.execution_id)

        execution = self.prepare()
        self._drive(execution, self._mirror, self._recorded)

        final_state = execution.state
        duration = time.perf_counter() - started
        events_after = self.journal.count_events(self.execution_id)

        result = ReplayResult(
            execution_id=self.execution_id,
            final_state=final_state,
            original_state=self._original_state,
            events_replayed=len(self._replay_journal.events),
            tools_replayed=execution.replayed_calls,
            duration=duration,
            matched=final_state == self._original_state,
            from_sequence=self.from_sequence,
            steps=execution.steps,
            journal_unchanged=events_before == events_after,
            errors=execution.errors,
        )
        self._assert_matched(result)
        return result

    @property
    def replay_execution(self) -> ReplayExecution | None:
        """The replay execution, once :meth:`prepare` or :meth:`run` has built it."""
        return self._execution

    # -- internals -----------------------------------------------------------

    def _drive(
        self,
        execution: ReplayExecution,
        mirror: Sequence[Event],
        recorded: tuple[RecordedToolCall, ...],
    ) -> None:
        """Walk the recorded tail in order, feeding the replay one event at a time.

        Order matters and is not incidental: ``ExecutionCompleted`` must be
        replayed *after* the calls that preceded it, or the replay would be
        marking an execution finished while it still had calls to make. So the
        walk follows the recorded sequence and dispatches each event:

        * ``ExecutionStarted``/``Completed``/``Failed``/``Cancelled`` carry no
          tool result to substitute, so they are re-emitted into the in-memory
          journal and the reducers decide what they mean -- the same decision the
          original run made;
        * a ``ToolRequested`` hands the recorded call to
          :meth:`ReplayExecution.replay_recorded_call`, which validates it and
          serves the recorded outcome;
        * the other phases of a call (``ToolStarted``, ``ToolCompleted``, ...) are
          produced by :meth:`Execution.call` itself as it re-runs, and replaying
          them again here would double-count them.
        """
        by_sequence = {call.requested_sequence: call for call in recorded}
        for event in mirror:
            if event.event_type is EventType.EXECUTION_STARTED:
                continue  # already emitted by prepare(), so the handle is usable
            if event.event_type in _LIFECYCLE_EVENTS:
                execution.record_lifecycle(event)
            elif event.event_type is EventType.TOOL_REQUESTED:
                call = by_sequence.get(event.sequence)
                if call is None:
                    # The prefix fold and the recorded call index disagree about
                    # what happened; report it rather than skipping the call.
                    raise ReplayMismatchError(
                        f"replay found a recorded tool request at sequence "
                        f"{event.sequence} with no recorded call to match",
                        kind="missing_recorded_call",
                        execution_id=self.execution_id,
                        sequence=event.sequence,
                        expected={"tool": event.payload.get("tool")},
                        received={"payload": dict(event.payload)},
                    )
                execution.replay_recorded_call(call)

    def _state_at(self, sequence: int) -> ExecutionState:
        """The state this execution's history describes at ``sequence``.

        A stored checkpoint at exactly that sequence is used when there is one --
        which is the point of §7, replaying 501..700 instead of 1..700 -- and the
        event prefix is folded otherwise. Both routes answer the same question, so
        a replay may start from either and reach the same state.
        """
        if sequence == 0:
            return initial_state(self.execution_id)

        checkpoint = self.checkpoints.get(self.execution_id, sequence)
        if checkpoint is not None:
            return checkpoint.state
        events = [
            e
            for e in self.journal.get_events(self.execution_id)
            if e.sequence <= sequence
        ]
        return reconstruct_state(events)

    def _base_state(self) -> ExecutionState:
        """The state a replay starts from, before any recorded event is re-applied."""
        return self._state_at(self.from_sequence)

    def _validate_start(self) -> None:
        """Reject a start point the history does not support, before replaying."""
        last = self.journal.get_last_sequence(self.execution_id)
        if last == 0:
            raise ExecutionNotFoundError(
                f"No journal found for execution {self.execution_id!r}"
            )
        if self.from_sequence < 0:
            raise ReplayMismatchError(
                f"replay start sequence must not be negative, got {self.from_sequence}",
                kind="sequence",
                execution_id=self.execution_id,
                expected={"from_sequence": ">= 0"},
                received={"from_sequence": self.from_sequence},
            )
        if self.from_sequence > last:
            raise ReplayMismatchError(
                f"replay asked to start at sequence {self.from_sequence} but execution "
                f"{self.execution_id!r} ends at {last}",
                kind="sequence",
                execution_id=self.execution_id,
                expected={"from_sequence": f"0..{last}"},
                received={"from_sequence": self.from_sequence},
            )
        if self.from_sequence == 0:
            # The beginning of the history: nothing is unresolved by definition.
            return

        # The only start points a history cannot be read from are the ones where a
        # tool call is still open. The events after such a point resolve a call
        # the replay never issued -- it would see a ToolCompleted for a
        # ToolRequested that is behind it -- so the state there is ambiguous, and
        # replay refuses rather than guessing. This is Milestone 2's rule applied
        # to a start point: the journal does not say what became of that call, so
        # nothing downstream of it can be replayed.
        open_calls = detect_incomplete_tools(self._state_at(self.from_sequence))
        if open_calls:
            stuck = open_calls[0]
            raise ReplayMismatchError(
                f"replay cannot start at sequence {self.from_sequence}: the call to "
                f"{stuck.tool!r} is still open there, and the events after it "
                "resolve a tool call the replay never made",
                kind="sequence",
                execution_id=self.execution_id,
                sequence=self.from_sequence,
                expected="a sequence where no tool call is still open",
                received={"unresolved_tool": stuck.tool, "call_id": stuck.call_id},
            )

    def _assert_matched(self, result: ReplayResult) -> None:
        """Raise if the replay and the original disagree, or the journal moved."""
        if not result.journal_unchanged:
            raise ReplayMismatchError(
                "replay modified the original event journal; replay must be read-only",
                kind="journal_mutated",
                execution_id=self.execution_id,
                expected={"journal": "unchanged"},
                received={"journal": "modified"},
            )
        if not result.matched:
            raise ReplayMismatchError(
                "the replayed state differs from the original state",
                kind="state",
                execution_id=self.execution_id,
                sequence=result.final_state.last_sequence,
                original_state=result.original_state,
                replayed_state=result.final_state,
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"ReplayEngine(execution_id={self.execution_id!r}, "
            f"from_sequence={self.from_sequence})"
        )


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


def replay_execution(
    journal: EventJournal,
    checkpoints: CheckpointStore,
    execution_id: str,
    *,
    from_sequence: int = 0,
    on_step: Callable[[ReplayStep], None] | None = None,
) -> ReplayResult:
    """Replay ``execution_id`` and return the result.

    The functional form of :meth:`agent_runtime.Runtime.replay`; the runtime
    method is a thin wrapper, so a caller holding a journal directly can replay
    an execution without constructing a :class:`~agent_runtime.runtime.Runtime`.
    """
    return ReplayEngine(
        journal,
        checkpoints,
        execution_id,
        from_sequence=from_sequence,
        on_step=on_step,
    ).run()

