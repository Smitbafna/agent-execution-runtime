"""Durable snapshots of execution state, and the consistency rules for them.

A checkpoint is the reconstructed state of an execution *immediately after* a
specific event sequence. It is a performance device, never a source of truth:
recovery replays the checkpoint plus the events durably persisted after it, and
the journal still decides what happened.

The invariant :meth:`CheckpointStore.create` protects::

    a checkpoint's sequence is always the execution's latest persisted event,
    and the state stored beside it is the state of exactly that prefix

so both halves of the row are written in one transaction, and a snapshot that
would disagree with the journal is rejected instead of stored.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .events import new_id, utc_now_iso
from .exceptions import (
    CorruptCheckpointError,
    InconsistentCheckpointError,
)
from .state import ExecutionState, ExecutionStatus, detect_incomplete_tools
from .storage import SQLiteStore

__all__ = ["Checkpoint", "CheckpointStore", "storable_state"]


def storable_state(state: ExecutionState) -> ExecutionState:
    """The form of ``state`` that is safe to write to the ``checkpoints`` table.

    ``RECOVERY_REQUIRED`` is a *diagnosis recovery makes*, never a decision that
    was journalled, so it is not stored: recovery re-derives it from the events
    it applies, and keeping it in the snapshot as well would mean trusting two
    sources for one field. ``incomplete_tools`` is recomputed from the snapshot's
    own tool calls so the stored row is self-consistent.
    """
    status = (
        ExecutionStatus.RUNNING
        if state.status is ExecutionStatus.RECOVERY_REQUIRED
        else state.status
    )
    return replace(state, status=status, incomplete_tools=detect_incomplete_tools(state))


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One persisted snapshot, tied to the event sequence it describes."""

    checkpoint_id: str
    execution_id: str
    sequence: int
    state: ExecutionState
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            execution_id=data["execution_id"],
            sequence=int(data["sequence"]),
            state=ExecutionState.from_dict(data["state"]),
            created_at=data["created_at"],
        )

    def __str__(self) -> str:
        return (
            f"Checkpoint {self.checkpoint_id} of {self.execution_id} @ "
            f"sequence {self.sequence} (status={self.state.status})"
        )


class CheckpointStore:
    """Reads and writes :class:`Checkpoint` rows.

    Writes go through :meth:`create`, the only supported way to put a checkpoint
    in the database -- it is what makes the sequence and the state agree with the
    journal.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    # -- writing --------------------------------------------------------------

    def create(self, execution_id: str, state: ExecutionState) -> Checkpoint:
        """Persist ``state`` as the snapshot of ``execution_id`` after its last event.

        ``state.last_sequence`` is the sequence the snapshot describes. The
        snapshot and that sequence are written in a single transaction that only
        commits when the journal agrees the sequence exists and is the
        execution's latest event; otherwise nothing is stored and
        :class:`~agent_runtime.exceptions.InconsistentCheckpointError` is raised.

        Storing the same sequence twice returns the checkpoint already on file: a
        snapshot is a pure function of its event prefix, so there is nothing to
        update.
        """
        snapshot = storable_state(state)
        sequence = state.last_sequence
        if sequence < 1:
            raise InconsistentCheckpointError(
                f"Cannot checkpoint execution {execution_id!r}: sequence {sequence} "
                "does not correspond to any event"
            )

        existing = self.get(execution_id, sequence)
        if existing is not None:
            if existing.state == snapshot:
                return existing
            raise InconsistentCheckpointError(
                f"Execution {execution_id!r} already has a different checkpoint at "
                f"sequence {sequence}; the journal and the stored state disagree"
            )

        checkpoint = Checkpoint(
            checkpoint_id=new_id("ckpt"),
            execution_id=execution_id,
            sequence=sequence,
            state=snapshot,
            created_at=utc_now_iso(),
        )
        # Everything the snapshot is checked against is decided above, so the
        # transaction is the one INSERT: it commits the sequence and the state
        # together, or leaves no trace at all.
        with self.store.transaction() as conn:
            cursor = self._insert(conn, checkpoint)
        if cursor.rowcount != 1:
            raise InconsistentCheckpointError(self._mismatch(execution_id, sequence, snapshot))
        return checkpoint

    @staticmethod
    def _insert(conn: sqlite3.Connection, checkpoint: Checkpoint) -> sqlite3.Cursor:
        """Insert a checkpoint, but only if the journal agrees with the snapshot.

        The whole consistency rule is one statement, so it is evaluated and
        written under the same transaction:

        * an event must exist at ``sequence`` (enforced again by the table's
          foreign key), otherwise the sequence refers to nothing;
        * ``sequence`` must be the execution's ``MAX(sequence)``, so a snapshot
          can never claim a sequence the history has not reached, nor one it has
          already moved past;
        * the snapshot's own ``last_sequence`` must equal it, so the state cannot
          describe a different prefix than the row claims.
        """
        state = checkpoint.state
        payload = json.dumps(state.to_dict(), sort_keys=True)
        try:
            return conn.execute(
                """
                INSERT INTO checkpoints
                    (checkpoint_id, execution_id, sequence, state, created_at)
                SELECT ?, ?, ?, ?, ?
                WHERE EXISTS (
                          SELECT 1 FROM events
                          WHERE execution_id = ? AND sequence = ?
                      )
                  AND ? = (SELECT MAX(sequence) FROM events WHERE execution_id = ?)
                  AND ? = ?
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.execution_id,
                    checkpoint.sequence,
                    payload,
                    checkpoint.created_at,
                    checkpoint.execution_id,
                    checkpoint.sequence,
                    checkpoint.sequence,
                    checkpoint.execution_id,
                    state.last_sequence,
                    checkpoint.sequence,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # Reached only for UNIQUE (execution_id, sequence): the pre-check
            # above missed it because another writer committed in between.
            raise InconsistentCheckpointError(
                f"Cannot store a checkpoint for execution {checkpoint.execution_id!r} "
                f"at sequence {checkpoint.sequence}: one is already stored for that "
                "sequence"
            ) from exc

    def _mismatch(self, execution_id: str, sequence: int, snapshot: ExecutionState) -> str:
        """Explain, after a rejected write, which rule the journal broke.

        This happens outside the transaction on purpose: the write has already
        been refused, and reporting *why* must not risk committing part of it.
        """
        row = self.store.query_all(
            "SELECT MAX(sequence) AS last FROM events WHERE execution_id = ?",
            (execution_id,),
        )[0]
        last = 0 if row["last"] is None else int(row["last"])
        if last == 0:
            return (
                f"Cannot checkpoint execution {execution_id!r} at sequence {sequence}: "
                "the execution has no events in the journal"
            )
        if snapshot.last_sequence != sequence:
            return (
                f"Cannot checkpoint execution {execution_id!r}: the state describes "
                f"sequence {snapshot.last_sequence} but the checkpoint claims "
                f"{sequence}"
            )
        if last != sequence:
            return (
                f"Cannot checkpoint execution {execution_id!r} at sequence {sequence}: "
                f"the journal's latest event is at sequence {last}"
            )
        return (
            f"Cannot checkpoint execution {execution_id!r} at sequence {sequence}: "
            "no event exists at that sequence"
        )

    # -- reading --------------------------------------------------------------

    def get(self, execution_id: str, sequence: int) -> Checkpoint | None:
        """The checkpoint stored at an exact sequence, if there is one."""
        rows = self.store.query_all(
            "SELECT * FROM checkpoints WHERE execution_id = ? AND sequence = ?",
            (execution_id, sequence),
        )
        return self._row_to_checkpoint(rows[0]) if rows else None

    def get_latest(self, execution_id: str) -> Checkpoint | None:
        """The newest checkpoint of an execution -- the one recovery starts from."""
        rows = self.store.query_all(
            "SELECT * FROM checkpoints WHERE execution_id = ? ORDER BY sequence DESC LIMIT 1",
            (execution_id,),
        )
        return self._row_to_checkpoint(rows[0]) if rows else None

    def latest_sequence(self, execution_id: str) -> int:
        """The sequence of the newest checkpoint, or ``0`` when there is none."""
        checkpoint = self.get_latest(execution_id)
        return 0 if checkpoint is None else checkpoint.sequence

    def list_for(self, execution_id: str) -> list[Checkpoint]:
        """Every checkpoint of an execution, oldest first."""
        rows = self.store.query_all(
            "SELECT * FROM checkpoints WHERE execution_id = ? ORDER BY sequence ASC",
            (execution_id,),
        )
        return [self._row_to_checkpoint(row) for row in rows]

    def count(self, execution_id: str) -> int:
        row = self.store.query_all(
            "SELECT COUNT(*) AS total FROM checkpoints WHERE execution_id = ?",
            (execution_id,),
        )[0]
        return int(row["total"])

    def has_checkpoints(self, execution_id: str) -> bool:
        return self.count(execution_id) > 0

    def list_execution_ids(self) -> list[str]:
        rows = self.store.query_all(
            "SELECT DISTINCT execution_id FROM checkpoints ORDER BY execution_id"
        )
        return [row["execution_id"] for row in rows]

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
        """Rebuild a checkpoint, refusing to guess when the stored state is unreadable.

        ``CorruptCheckpointError`` propagates on purpose: silently replaying the
        journal instead would hide broken stored data behind a working recovery.
        """
        try:
            state = ExecutionState.from_dict(json.loads(row["state"]))
        except (TypeError, ValueError, KeyError) as exc:
            raise CorruptCheckpointError(
                f"Checkpoint {row['checkpoint_id']!r} of execution "
                f"{row['execution_id']!r} at sequence {row['sequence']} does not hold "
                f"a readable execution state: {exc}"
            ) from exc
        sequence = int(row["sequence"])
        if state.last_sequence != sequence:
            # The row claims one sequence and the state beside it describes
            # another. Writes cannot produce that, so if it is ever read back the
            # stored data was tampered with or damaged -- either way it is not a
            # state the history supports, and recovery must not use it.
            raise CorruptCheckpointError(
                f"Checkpoint {row['checkpoint_id']!r} of execution "
                f"{row['execution_id']!r} claims sequence {sequence} but the state it "
                f"holds describes sequence {state.last_sequence}"
            )
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            execution_id=row["execution_id"],
            sequence=sequence,
            state=state,
            created_at=row["created_at"],
        )
