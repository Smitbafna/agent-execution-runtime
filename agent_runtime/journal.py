"""The append-only event journal.

The journal is the source of truth. It only ever inserts rows: there is no
update or delete API, which is what makes the history immutable.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping

from .events import Event, EventType
from .exceptions import DuplicateSequenceError, EventNotFoundError, SequenceError
from .storage import SQLiteStore

__all__ = ["EventJournal"]


class EventJournal:
    """Persists and queries :class:`~agent_runtime.events.Event` objects."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    # -- writing --------------------------------------------------------------

    def append(self, event: Event) -> Event:
        """Append a fully-formed event.

        The event's sequence must be exactly ``last_sequence + 1`` for its
        execution; anything else is rejected so that histories stay gapless.
        """
        if event.sequence < 1:
            raise SequenceError(
                f"Event sequence must be >= 1, got {event.sequence} "
                f"for execution {event.execution_id!r}"
            )

        with self.store.transaction() as conn:
            expected = self._last_sequence(conn, event.execution_id) + 1
            if event.sequence != expected:
                raise SequenceError(
                    f"Out-of-order sequence for execution {event.execution_id!r}: "
                    f"expected {expected}, got {event.sequence}"
                )
            self._insert(conn, event)
        return event

    def append_event(
        self,
        execution_id: str,
        event_type: EventType | str,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        """Allocate the next sequence and append an event in one transaction."""
        with self.store.transaction() as conn:
            sequence = self._last_sequence(conn, execution_id) + 1
            event = Event.create(execution_id, sequence, event_type, payload or {})
            self._insert(conn, event)
        return event

    # -- reading --------------------------------------------------------------

    def get_events(self, execution_id: str) -> list[Event]:
        """All events for an execution, ordered by sequence."""
        rows = self.store.query_all(
            "SELECT * FROM events WHERE execution_id = ? ORDER BY sequence ASC",
            (execution_id,),
        )
        return [self._row_to_event(row) for row in rows]

    def get_events_from(self, execution_id: str, after_sequence: int) -> list[Event]:
        """Events strictly after ``after_sequence``, ordered by sequence.

        This is what makes a checkpoint worth having: recovery reads the snapshot
        plus only the tail of the journal instead of the whole history.
        """
        rows = self.store.query_all(
            "SELECT * FROM events WHERE execution_id = ? AND sequence > ? ORDER BY sequence ASC",
            (execution_id, after_sequence),
        )
        return [self._row_to_event(row) for row in rows]

    def get_event(self, execution_id: str, sequence: int) -> Event:
        """A single event by execution id and sequence."""
        rows = self.store.query_all(
            "SELECT * FROM events WHERE execution_id = ? AND sequence = ?",
            (execution_id, sequence),
        )
        if not rows:
            raise EventNotFoundError(
                f"No event with sequence {sequence} for execution {execution_id!r}"
            )
        return self._row_to_event(rows[0])

    def get_last_sequence(self, execution_id: str) -> int:
        """Highest sequence for an execution, or ``0`` when it has no events."""
        return self._last_sequence(self.store.connection, execution_id)

    def get_last_event(self, execution_id: str) -> Event | None:
        """The most recent event of an execution, or ``None`` when it has none.

        The sequence a checkpoint must name: a snapshot is only ever valid for
        the execution's latest persisted event.
        """
        rows = self.store.query_all(
            "SELECT * FROM events WHERE execution_id = ? ORDER BY sequence DESC LIMIT 1",
            (execution_id,),
        )
        return self._row_to_event(rows[0]) if rows else None

    def count_events(self, execution_id: str) -> int:
        row = self.store.query_all(
            "SELECT COUNT(*) AS total FROM events WHERE execution_id = ?",
            (execution_id,),
        )[0]
        return int(row["total"])

    def has_execution(self, execution_id: str) -> bool:
        return self.count_events(execution_id) > 0

    def list_execution_ids(self) -> list[str]:
        """Every execution id known to the journal, sorted."""
        rows = self.store.query_all("SELECT DISTINCT execution_id FROM events ORDER BY execution_id")
        return [row["execution_id"] for row in rows]

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _last_sequence(conn: sqlite3.Connection, execution_id: str) -> int:
        row = conn.execute(
            "SELECT MAX(sequence) AS last FROM events WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None or row["last"] is None:
            return 0
        return int(row["last"])

    @staticmethod
    def _insert(conn: sqlite3.Connection, event: Event) -> None:
        try:
            conn.execute(
                """
                INSERT INTO events (event_id, execution_id, sequence, event_type, payload, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.execution_id,
                    event.sequence,
                    str(event.event_type),
                    json.dumps(dict(event.payload)),
                    event.timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # The UNIQUE(execution_id, sequence) / PRIMARY KEY(event_id) constraints.
            raise DuplicateSequenceError(
                f"Cannot append event {event.event_id!r}: duplicate id or sequence "
                f"({event.execution_id!r}, {event.sequence})"
            ) from exc

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            event_id=row["event_id"],
            execution_id=row["execution_id"],
            sequence=int(row["sequence"]),
            event_type=EventType(row["event_type"]),
            timestamp=row["timestamp"],
            payload=json.loads(row["payload"]),
        )
