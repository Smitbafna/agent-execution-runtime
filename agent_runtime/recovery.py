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
from .journal import EventJournal
from .state import (
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    apply_event,
    finalize_state,
    reconstruct_state,
)

__all__ = ["RecoveryInfo", "RecoverySource", "recover_execution", "apply_events_from"]

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

    @property
    def needs_resolution(self) -> bool:
        """True when the history left tool work with no recorded outcome.

        Milestone 2 stops here on purpose: the decision of whether to retry,
        resume or mark the call failed belongs to the application.
        """
        return bool(self.incomplete_tools)

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
                "This runtime does not retry automatically: resolve these with "
                "execution.resolve_recovery(...), or fail/cancel the execution."
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
    journal: EventJournal, checkpoints: CheckpointStore, execution_id: str
) -> RecoveryInfo:
    """Rebuild an execution's current state, starting from its latest checkpoint.

    The checkpoint is only ever a shortcut over events that are still there: the
    events after it are read from the journal and applied on top, so the result
    is always "checkpoint + durable events" and never "checkpoint as-is".

    Raises:
        ExecutionNotFoundError: the execution has no events at all.
        CorruptCheckpointError: the latest checkpoint cannot be read back.
    """
    last_sequence = journal.get_last_sequence(execution_id)
    if last_sequence == 0:
        raise ExecutionNotFoundError(f"No journal found for execution {execution_id!r}")

    checkpoint = checkpoints.get_latest(execution_id)

    if checkpoint is None:
        state = reconstruct_state(journal.get_events(execution_id))
        return RecoveryInfo(
            execution_id=execution_id,
            status=state.status,
            source="events",
            state=state,
            last_sequence=last_sequence,
            events_after_checkpoint=last_sequence,
            incomplete_tools=state.incomplete_tools,
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
        status=state.status,
        source="checkpoint",
        state=state,
        last_sequence=last_sequence,
        checkpoint_sequence=checkpoint.sequence,
        checkpoint_id=checkpoint.checkpoint_id,
        events_after_checkpoint=len(tail),
        incomplete_tools=state.incomplete_tools,
    )
