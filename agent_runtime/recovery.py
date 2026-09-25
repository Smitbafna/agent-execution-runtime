"""Rebuilding an execution from its checkpoint plus the events after it.

Recovery is deliberately boring::

    latest checkpoint
            +
    events after the checkpoint
            ↓
      current state

and when there is no checkpoint it folds the whole journal instead. Either way
the result comes from durable data only, and the journal remains the source of
truth: a checkpoint that cannot be read back is reported, never ignored.

Nothing here retries, replays deterministically or branches. The most this
module does is *say* what a crash left unfinished, through :class:`RecoveryInfo`.

Milestone 4B adds a second thing a crash can leave unfinished, which does not
live in the journal at all: an idempotency key that was claimed before the tool
ran and never got an outcome. Recovery reports those too
(:attr:`RecoveryInfo.pending_idempotency`) rather than letting them hide behind
a clean-looking history -- an execution whose last event is ``ToolRequested`` can
still be the one whose payment already went through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .checkpoints import CheckpointStore
from .events import Event
from .exceptions import (
    CorruptCheckpointError,
    ExecutionNotFoundError,
    StateReconstructionError,
)
from .idempotency import IdempotencyGuard, IdempotencyRecord, unresolved_records
from .journal import EventJournal
from .state import (
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    PendingRetry,
    RecoveryState,
    ToolCall,
    ToolCallStatus,
    apply_event,
    classify_recovery,
    detect_incomplete_tools,
    detect_pending_retries,
    finalize_state,
    reconstruct_state,
)

__all__ = [
    "RecoveryInfo",
    "RecoverySource",
    "recover_execution",
    "apply_events_from",
]

#: Where a recovered state came from: a stored snapshot, or the whole journal.
RecoverySource = Literal["checkpoint", "events"]


@dataclass(frozen=True, slots=True)
class RecoveryInfo:
    """What recovery made of an execution's history.

    This is the object an application inspects after a crash::

        Execution: exec_123
        Status: RECOVERY_REQUIRED
        ...
    """

    execution_id: str
    status: ExecutionStatus
    source: RecoverySource
    state: ExecutionState
    last_sequence: int
    checkpoint_sequence: int | None = None
    checkpoint_id: str | None = None
    events_after_checkpoint: int = 0
    incomplete_tools: tuple[IncompleteTool, ...] = ()
    pending_retries: tuple[PendingRetry, ...] = ()
    #: Idempotency keys this execution claimed and never settled (Milestone 4B).
    #: Read from the idempotency store, not from the journal: a key can be left
    #: ``PENDING`` by a crash that left no ambiguous tool call behind at all.
    pending_idempotency: tuple[IdempotencyRecord, ...] = ()

    @property
    def needs_resolution(self) -> bool:
        """True when the history left tool work with no recorded outcome.

        Milestone 2 stopped here on purpose, and Milestone 4A did not move the
        line: a retry policy repeats *recorded failures*, never work whose
        outcome the journal does not know. Whether to retry, resume or mark such
        a call failed still belongs to the application.
        """
        return bool(self.incomplete_tools)

    @property
    def needs_idempotency_resolution(self) -> bool:
        """True when an idempotency key is unresolved (Milestone 4B).

        The other half of "this crash left something unresolved", and the one
        that survives a perfectly clean journal: the side effect may already have
        happened even though every recorded event looks fine.
        """
        return bool(unresolved_records(self.pending_idempotency))

    @property
    def has_pending_retries(self) -> bool:
        """True when the journal scheduled retries that never started.

        The opposite of :attr:`needs_resolution`: nothing here needs a decision,
        each scheduled attempt is simply the next thing to do (see
        :meth:`~agent_runtime.execution.Execution.continue_pending_retry`).
        """
        return bool(self.pending_retries)

    @property
    def recovery_state(self) -> RecoveryState:
        """What a restarted process should do with this execution.

        Milestone 4C's §12 answer, and the one field that says all of it::

            RecoveryInfo(...).recovery_state   # RecoveryState.RECOVERY_REQUIRED

        Derived from the events plus the idempotency store, never stored, and
        deliberately not a guess: a recorded terminal decision wins outright, an
        ambiguity outranks a scheduled retry, and only then does a pending retry
        mean "carry this on". See :func:`~agent_runtime.state.classify_recovery`.
        """
        return classify_recovery(
            self.state, unresolved_keys=self.needs_idempotency_resolution
        )

    @property
    def timed_out_calls(self) -> tuple[ToolCall, ...]:
        """Calls whose last recorded attempt ran out of time.

        Distinct from failed ones on purpose (§11): "it raised" and "it ran out
        of time" call for different responses, and a recovery that could not
        tell them apart would be guessing.
        """
        return tuple(
            call
            for call in self.state.tool_calls
            if call.status is ToolCallStatus.TIMED_OUT
        )

    @property
    def cancelled_calls(self) -> tuple[ToolCall, ...]:
        """Calls whose last recorded attempt was cancelled.

        Reported so a restart can be *seen* to leave them alone: a cancelled
        call is a decision, and nothing here resumes or retries one.
        """
        return tuple(
            call
            for call in self.state.tool_calls
            if call.status is ToolCallStatus.CANCELLED
        )

    @property
    def unenforced_timeouts(self) -> tuple[ToolCall, ...]:
        """Timed-out calls the runtime could **not** prove stopped.

        The important subset of :attr:`timed_out_calls`. For these the external
        side effect may still have happened, so a keyed call among them is a
        ``RECOVERY_REQUIRED`` rather than a failure -- the runtime asked a
        thread to stop and the thread kept running.
        """
        return tuple(call for call in self.timed_out_calls if call.timeout_enforced is False)

    @property
    def events_replayed(self) -> int:
        """How many events recovery had to apply on top of the checkpoint."""
        return self.events_after_checkpoint

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": str(self.status),
            "source": self.source,
            "last_sequence": self.last_sequence,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sequence": self.checkpoint_sequence,
            "events_after_checkpoint": self.events_after_checkpoint,
            "events_replayed": self.events_replayed,
            "needs_resolution": self.needs_resolution,
            "incomplete_tools": [item.to_dict() for item in self.incomplete_tools],
            "pending_retries": [item.to_dict() for item in self.pending_retries],
            "needs_idempotency_resolution": self.needs_idempotency_resolution,
            "pending_idempotency": [
                record.to_dict() for record in self.pending_idempotency
            ],
        }

    def __str__(self) -> str:
        lines = [f"Execution: {self.execution_id}", f"Status: {self.status}"]
        if self.source == "checkpoint" and self.checkpoint_sequence is not None:
            lines.append(
                f"Recovered from: checkpoint {self.checkpoint_id} @ sequence "
                f"{self.checkpoint_sequence}"
            )
            lines.append(f"Events replayed after checkpoint: {self.events_after_checkpoint}")
        else:
            lines.append("Recovered from: full journal replay (no checkpoint)")
        lines.append(f"Last event sequence: {self.last_sequence}")

        if self.incomplete_tools:
            lines.append("")
            lines.append("Incomplete operations:")
            for item in self.incomplete_tools:
                lines.append("")
                lines.append(f"Tool: {item.tool}")
                lines.append(f"Started at sequence: {item.sequence}")
                lines.append(f"Status: {item.status}")
                if item.started_at:
                    lines.append(f"Started at: {item.started_at}")
                lines.append("Arguments:")
                for key, value in item.arguments.items():
                    lines.append(f"    {key}={value!r}")
            lines.append("")
            lines.append(
                "These calls have no recorded outcome, so no retry policy applies to "
                "them: resolve them with execution.resolve_recovery(...), or "
                "fail/cancel the execution."
            )

        if self.pending_retries:
            lines.append("")
            lines.append("Scheduled retries:")
            for retry in self.pending_retries:
                lines.append("")
                lines.append(f"    {retry}")
                lines.append(f"    recorded at sequence: {retry.sequence}")
            lines.append("")
            lines.append(
                "These are decisions, not ambiguities: carry them on with "
                "execution.continue_pending_retry(call_id)."
            )

        if self.pending_idempotency:
            lines.append("")
            lines.append("Unresolved idempotency keys:")
            for record in self.pending_idempotency:
                lines.append("")
                lines.append(f"    {record}")
                if record.is_unresolved:
                    lines.append(f"    claimed at: {record.created_at}")
                    lines.append(f"    tool: {record.tool_name}({record.call_id})")
            lines.append("")
            lines.append(
                "A PENDING key means the runtime committed the intent to run the "
                "side effect and never recorded an outcome -- the process may have "
                "died between the two. It will not run the tool again on its own; "
                "settle each key with execution.resolve_idempotency(key, ...)."
            )
        return "\n".join(lines)


def apply_events_from(
    state: ExecutionState, events: list[Event], *, expected_execution_id: str
) -> ExecutionState:
    """Fold post-checkpoint events onto a stored state.

    The checkpoint is checked against the execution being recovered before
    anything is applied, and every event is then guarded exactly as a full
    replay would guard it. The result is finalized, so the resumed execution
    carries the same incomplete-tool diagnosis a full replay would give.
    """
    if state.execution_id and state.execution_id != expected_execution_id:
        raise StateReconstructionError(
            f"Checkpoint holds state for execution {state.execution_id!r} but recovery "
            f"was for {expected_execution_id!r}"
        )
    for event in events:
        state = apply_event(state, event)
    return finalize_state(state)


def recover_execution(
    journal: EventJournal,
    checkpoints: CheckpointStore,
    execution_id: str,
    *,
    idempotency: IdempotencyGuard | None = None,
) -> RecoveryInfo:
    """Rebuild an execution's current state, starting from its latest checkpoint.

    The checkpoint is only ever a shortcut over events that are still there: the
    events after it are read from the journal and applied on top, so the result
    is always "checkpoint + durable events" and never "checkpoint as-is".

    ``idempotency`` supplies the claims to report alongside (Milestone 4B).
    Leaving it out reports none -- which is what a caller that has no
    idempotency store (a replay) wants. Note what recovery does *not* do with
    them: a ``PENDING`` key is reported, never executed, because "was it
    claimed" is not "did it happen".

    Raises:
        ExecutionNotFoundError: the execution has no events at all.
        CorruptCheckpointError: the latest checkpoint cannot be read back.
    """
    last_sequence = journal.get_last_sequence(execution_id)
    if last_sequence == 0:
        raise ExecutionNotFoundError(f"No journal found for execution {execution_id!r}")

    checkpoint = checkpoints.get_latest(execution_id)
    pending_keys: tuple[IdempotencyRecord, ...] = (
        () if idempotency is None else idempotency.pending_records(execution_id)
    )

    if checkpoint is None:
        state = reconstruct_state(journal.get_events(execution_id))
        return RecoveryInfo(
            execution_id=execution_id,
            status=_status_with_keys(state.status, pending_keys),
            source="events",
            state=state,
            last_sequence=last_sequence,
            events_after_checkpoint=last_sequence,
            incomplete_tools=state.incomplete_tools,
            pending_retries=detect_pending_retries(state),
            pending_idempotency=pending_keys,
        )

    if checkpoint.sequence > last_sequence:
        # The snapshot is ahead of the history, so it describes events that are
        # no longer there. Applying "the events after it" would silently return
        # the snapshot alone, which is not a state this journal supports.
        raise CorruptCheckpointError(
            f"Checkpoint {checkpoint.checkpoint_id!r} of execution {execution_id!r} is "
            f"at sequence {checkpoint.sequence} but the journal ends at {last_sequence}"
        )

    tail = journal.get_events_from(execution_id, checkpoint.sequence)
    state = apply_events_from(
        checkpoint.state, tail, expected_execution_id=execution_id
    )
    return RecoveryInfo(
        execution_id=execution_id,
        status=_status_with_keys(state.status, pending_keys),
        source="checkpoint",
        state=state,
        last_sequence=last_sequence,
        checkpoint_sequence=checkpoint.sequence,
        checkpoint_id=checkpoint.checkpoint_id,
        events_after_checkpoint=len(tail),
        incomplete_tools=state.incomplete_tools,
        pending_retries=detect_pending_retries(state),
        pending_idempotency=pending_keys,
    )


def _status_with_keys(
    status: ExecutionStatus, pending_keys: tuple[IdempotencyRecord, ...]
) -> ExecutionStatus:
    """Report ``RECOVERY_REQUIRED`` when an unresolved key joins an open status.

    The journal-derived status is only upgraded, never downgraded: a deliberate
    ``COMPLETED``/``FAILED``/``CANCELLED`` stays, exactly as
    :func:`~agent_runtime.state.resolve_status` decided. What is added is the
    case the journal cannot see at all -- a side effect that may have happened
    with nothing in the history saying so.
    """
    if status is ExecutionStatus.RUNNING and unresolved_records(pending_keys):
        return ExecutionStatus.RECOVERY_REQUIRED
    return status
