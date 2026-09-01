"""SQLite persistence layer.

This module owns the connection and the schema; it knows nothing about events.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .exceptions import StorageError

__all__ = ["SQLiteStore", "SCHEMA_STATEMENTS", "DbPath"]

DbPath = str | Path

MEMORY = ":memory:"

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id     TEXT    NOT NULL PRIMARY KEY,
        execution_id TEXT    NOT NULL,
        sequence     INTEGER NOT NULL,
        event_type   TEXT    NOT NULL,
        payload      TEXT    NOT NULL,
        timestamp    TEXT    NOT NULL,
        UNIQUE (execution_id, sequence)
    )
    """,
)


class SQLiteStore:
    """A thin, durable wrapper around a single SQLite connection.

    The connection is opened with ``isolation_level=None`` (autocommit) so that
    transactions are explicit, and with ``synchronous=FULL`` so that a committed
    event really is on disk.
    """

    def __init__(self, db_path: DbPath = "agent.db") -> None:
        self.db_path = str(db_path)
        try:
            self._conn = sqlite3.connect(
                self.db_path,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:  # pragma: no cover - depends on filesystem
            raise StorageError(f"Could not open database {self.db_path!r}: {exc}") from exc

        self._conn.row_factory = sqlite3.Row
        if self.db_path != MEMORY:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._closed = False
        self.create_schema()

    # -- schema ---------------------------------------------------------------

    def create_schema(self) -> None:
        """Create the tables/indexes if they do not exist yet (idempotent)."""
        for statement in SCHEMA_STATEMENTS:
            self._conn.execute(statement)

    # -- access ---------------------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        if self._closed:
            raise StorageError("Store is closed")
        return self._conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, params).fetchall())

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside an ``IMMEDIATE`` transaction.

        ``IMMEDIATE`` takes the write lock straight away, which keeps concurrent
        writers from interleaving and losing sequence numbers.
        """
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
