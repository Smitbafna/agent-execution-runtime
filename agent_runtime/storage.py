"""SQLite persistence layer.

This module owns the connection and the schema; it knows nothing about events.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .exceptions import StorageError

__all__ = ["SQLiteStore", "SCHEMA_STATEMENTS", "DbPath", "MEMORY"]

DbPath = str | Path

MEMORY = ":memory:"

SCHEMA_STATEMENTS: tuple[str, ...] = (
    # The append-only journal. UNIQUE (execution_id, sequence) is what keeps
    # histories gapless; FOREIGN KEY support is enabled on every connection.
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
    # A checkpoint is the state of an execution immediately after `sequence`.
    #
    #   * the composite FOREIGN KEY pins the sequence to a row that really is in
    #     the journal, so a checkpoint cannot reference an event that never
    #     happened;
    #   * UNIQUE (execution_id, sequence) keeps one snapshot per sequence, so a
    #     committed checkpoint can never be half-overwritten by another;
    #   * state is stored as JSON -- the checkpoint is a cache of the journal,
    #     never an independent source of truth.
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id TEXT    NOT NULL PRIMARY KEY,
        execution_id  TEXT    NOT NULL,
        sequence      INTEGER NOT NULL,
        state         TEXT    NOT NULL,
        created_at    TEXT    NOT NULL,
        UNIQUE (execution_id, sequence),
        FOREIGN KEY (execution_id, sequence) REFERENCES events (execution_id, sequence)
    )
    """,
    # Recovery asks for "the newest checkpoint of this execution" on every
    # resume; this index keeps that a single index seek.
    """
    CREATE INDEX IF NOT EXISTS idx_checkpoints_execution
        ON checkpoints (execution_id, sequence DESC)
    """,
    # The idempotency ledger (Milestone 4B): one row per idempotency key.
    #
    #   * PRIMARY KEY is the whole guarantee of uniqueness: two local executions
    #     racing for one key cannot both insert, so one of them sees the other's
    #     claim instead (see IdempotencyStore.claim);
    #   * the row is *updated* in place -- unlike events, which are append-only --
    #     because a record has a lifecycle: PENDING -> COMPLETED / FAILED. It is a
    #     claim ledger, not a history;
    #   * retry_authorized is how an explicit "run it again" survives without
    #     pretending an outcome is known: the record stays PENDING, and exactly
    #     one further claim is allowed to consume the authorization.
    """
    CREATE TABLE IF NOT EXISTS idempotency_records (
        idempotency_key  TEXT    NOT NULL PRIMARY KEY,
        execution_id     TEXT    NOT NULL,
        call_id          TEXT    NOT NULL,
        tool_name        TEXT    NOT NULL,
        arguments        TEXT    NOT NULL,
        status           TEXT    NOT NULL,
        result           TEXT,
        error            TEXT,
        created_at       TEXT    NOT NULL,
        updated_at       TEXT    NOT NULL,
        attempts         INTEGER NOT NULL DEFAULT 1,
        retry_authorized INTEGER NOT NULL DEFAULT 0,
        resolution       TEXT,
        resolution_note  TEXT,
        resolved_at      TEXT
    )
    """,
    # Recovery asks "which keys does this execution still have in flight?" on
    # every resume, and the CLI asks for every unresolved key in the database.
    """
    CREATE INDEX IF NOT EXISTS idx_idempotency_execution
        ON idempotency_records (execution_id, status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_idempotency_status
        ON idempotency_records (status)
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
        # Two local writers -- two processes, or two Runtime objects over one file
        # -- serialize on SQLite's write lock. Without a busy timeout the loser
        # would get "database is locked" instead of waiting its turn, which is how
        # a second claim of an idempotency key would surface as a storage error
        # rather than as "this key is already claimed".
        self._conn.execute("PRAGMA busy_timeout=5000")
        #: Milestone 4C: ``execution.cancel()`` may be called from a different
        #: thread than the one running the tool, so two threads can reach the
        #: journal through this one connection. SQLite's own file lock does not
        #: help there -- the problem is a second ``BEGIN`` on a connection that
        #: already has one open -- so transactions are serialized here. An
        #: ``RLock`` because a nested transaction on the same thread must be
        #: able to join the one already running.
        self._transaction_lock = threading.RLock()
        #: How deep *this* thread is, so nesting joins rather than restarts.
        self._local = threading.local()
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
        """Run a block of statements inside an ``IMMEDIATE`` transaction.

        ``IMMEDIATE`` takes the write lock straight away, which keeps concurrent
        writers from interleaving and losing sequence numbers.

        Milestone 4C adds the lock. ``execution.cancel()`` is documented as
        callable from another thread while a tool is running, which means two
        threads can reach ``journal.append_event`` at the same moment -- on one
        shared connection, where the second ``BEGIN IMMEDIATE`` would otherwise
        fail with "cannot start a transaction within a transaction". The
        ``RLock`` makes the writers serialize, and the per-thread depth makes a
        nested transaction *join* the one already open rather than opening a
        second, so a caller that nests still works.

        Everything in the block commits together or not at all, which makes a
        transaction a poor fit for two kinds of work:

        * anything that has to be *split* across commits (a checkpoint has to be
          one commit -- see :meth:`agent_runtime.checkpoints.CheckpointStore.create`);
        * "check, then act" control flow. A ``finally`` block that swallowed an
          error would commit the first statement anyway, so read what you need
          to decide before opening one, or express the check in SQL as part of
          the statement that writes.
        """
        conn = self.connection
        if self._depth_of_current_thread() > 0:
            # Already inside a transaction on *this* thread: the outer one owns
            # the commit, so this block is just a (checked) part of it.
            with self._transaction_lock:
                self._local.depth += 1
            try:
                yield conn
            finally:
                with self._transaction_lock:
                    self._local.depth -= 1
            return

        with self._transaction_lock:
            conn.execute("BEGIN IMMEDIATE")
            self._local.depth = 1
            try:
                yield conn
            except BaseException:
                self._local.depth = 0
                conn.execute("ROLLBACK")
                raise
            self._local.depth = 0
            conn.execute("COMMIT")

    def _depth_of_current_thread(self) -> int:
        """How many transactions the calling thread already has open."""
        return int(getattr(self._local, "depth", 0))

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
