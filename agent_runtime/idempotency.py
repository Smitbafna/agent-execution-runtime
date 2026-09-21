"""Idempotency: a durable, explicit answer to "did this side effect happen?".

Milestone 4B exists because of one crash window that no amount of journalling
closes::

    execution.call("send_email", idempotency_key="email-123")
        |
        +-- claim the key          (committed to SQLite)
        +-- the email is sent      (outside SQLite, cannot be rolled back)
        +-- store the outcome      <-- the process dies HERE
        |
        +-- recovery: "did the email go out?"

If the answer were assumed to be "no", every recovery would send the email
again. If it were assumed to be "yes", a crash *before* the request would lose
it. Neither assumption is knowledge, so this module does not make either one.

What it does instead
--------------------

* A **key** names an intended logical side effect
  (``execution.call("send_email", ..., idempotency_key="welcome-user-123")``),
  and the PRIMARY KEY of ``idempotency_records`` is what makes "at most one
  claim at a time" true across processes, not just across threads.
* A **claim** is committed *before* the tool runs and is ``PENDING`` until an
  outcome is recorded. A ``PENDING`` record means "committed intent, no answer
  yet" -- which after a crash is exactly the ambiguity, and it is surfaced as
  such (:class:`IdempotencyRecoveryRequiredError`) instead of being resolved by
  a guess.
* A **duplicate** of a ``COMPLETED`` key is answered from the stored result.
  The tool is not executed.
* An **explicit resolution** is the only way out of a ``PENDING`` or ``FAILED``
  key: ``retry`` (allow exactly one more attempt), ``mark_completed`` (record an
  externally known result), ``mark_failed`` (record a known failure).

What is deliberately not claimed
--------------------------------

    This runtime does not provide exactly-once execution of external side
    effects, and no implementation of it can.

The claim commits and the side effect happens in different systems, and nothing
can make those two writes atomic::

    BEGIN
    <external API call>      <-- not in the transaction, not reversible
    INSERT outcome
    COMMIT

A crash in the middle of that external call leaves a ``PENDING`` record whether
or not the effect happened. So the guarantee this module *does* offer is the one
that can be kept:

    A side-effecting tool never runs twice for one key unless the application
    explicitly asked for it, and every unresolved key is visible.

Two other boundaries are equally deliberate:

* **retries keep the key.** Milestone 4A's logical call has one ``call_id`` for
  all its attempts, so all attempts share one key and one claim; attempt 3 of
  ``call-123`` is not a new claim of ``payment-456``, it is the second attempt
  of the first.
* **replay is read-only.** :class:`ReplayIdempotencyGuard` has no store behind
  it: a replay cannot claim a key, insert a ``PENDING`` row, overwrite a result
  or run a tool, because there is no code path that reaches the database.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from .events import utc_now_iso
from .exceptions import (
    IdempotencyError,
    IdempotencyKeyConflictError,
    IdempotencyKeyFailedError,
    IdempotencyRecoveryRequiredError,
    IdempotencyResolutionError,
    StorageError,
    UnknownIdempotencyKeyError,
)
from .storage import SQLiteStore
from .tools import make_jsonable

__all__ = [
    "IdempotencyStatus",
    "IdempotencyAction",
    "IdempotencyRecord",
    "IdempotencyStore",
    "IdempotencyDecision",
    "IdempotencyGuard",
    "StoreIdempotencyGuard",
    "ReplayIdempotencyGuard",
    "unresolved_records",
]


class IdempotencyStatus(StrEnum):
    """Where a key is in its lifecycle.

    ::

        claim -> PENDING -> COMPLETED
                        -> FAILED

    ``COMPLETED`` and ``FAILED`` are *recorded outcomes*. ``PENDING`` is not an
    outcome at all: it means the runtime committed the intent to run a side
    effect and has no answer yet -- which, after a crash, is an ambiguity the
    application has to settle.
    """

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def is_resolved(self) -> bool:
        """True when the record holds an outcome at all, as opposed to ``PENDING``.

        "Resolved" here means *settled*, not *successful*: a ``FAILED`` record is
        resolved in that it says the attempt failed, while still not being
        permission to run again -- :attr:`IdempotencyRecord.is_unresolved` is the
        question that matters to :meth:`Execution.call`.
        """
        return self in RESOLVED_IDEMPOTENCY_STATUSES


#: The statuses that record an outcome. Anything else is still in flight.
RESOLVED_IDEMPOTENCY_STATUSES: frozenset[IdempotencyStatus] = frozenset(
    {IdempotencyStatus.COMPLETED, IdempotencyStatus.FAILED}
)

#: What an application may ask the runtime to do about an unresolved key. There
#: is no "assume it worked" and no "assume it didn't": both are decisions a
#: human or an upstream system has to make, with evidence the runtime does not have.
IdempotencyAction = Literal["retry", "mark_completed", "mark_failed"]

IDEMPOTENCY_ACTIONS: tuple[str, ...] = ("retry", "mark_completed", "mark_failed")


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """What the runtime durably knows about one idempotency key.

    A row in ``idempotency_records``, read back as a value. The fields that
    decide behaviour are :attr:`status` -- a recorded outcome, or ``PENDING`` --
    and :attr:`retry_authorized`, which is how an explicit "run it once more" is
    stored *without* pretending an outcome is known.

    ::

        IdempotencyRecord(
            idempotency_key='payment-456',
            status=IdempotencyStatus.PENDING,
            call_id='call_9f2c',
            attempts=2,
            retry_authorized=1,
        )
    """

    idempotency_key: str
    execution_id: str
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    status: IdempotencyStatus = IdempotencyStatus.PENDING
    result: Any = None
    error: Mapping[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""
    #: How many claims this key has had -- one per deliberate execution of the
    #: side effect, so an authorized retry shows up as ``attempts=2``.
    attempts: int = 1
    #: How many further claims the application has explicitly authorized.
    retry_authorized: int = 0
    #: How the key was last settled: ``completed``, ``failed``,
    #: ``mark_completed``, ``mark_failed`` or ``retry``. The last decision, not
    #: a full history -- the event journal is the audit trail for what an
    #: execution did.
    resolution: str | None = None
    resolution_note: str | None = None
    resolved_at: str | None = None

    @property
    def is_pending(self) -> bool:
        """True while the runtime holds a claim with no recorded outcome."""
        return self.status is IdempotencyStatus.PENDING

    @property
    def is_unresolved(self) -> bool:
        """True when the runtime must not execute this key again on its own.

        ``PENDING`` *and* no authorized retry outstanding. An authorized retry
        is a decision the application already made, so it is not an ambiguity --
        but the record stays ``PENDING`` all the same, because nobody has said
        yet whether the external effect happened.
        """
        return self.is_pending and self.retry_authorized <= 0

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable copy of the record."""
        return {
            "idempotency_key": self.idempotency_key,
            "execution_id": self.execution_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "status": str(self.status),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempts": self.attempts,
            "retry_authorized": self.retry_authorized,
            "resolution": self.resolution,
            "resolution_note": self.resolution_note,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IdempotencyRecord":
        return cls(
            idempotency_key=data["idempotency_key"],
            execution_id=data["execution_id"],
            call_id=data["call_id"],
            tool_name=data["tool_name"],
            arguments=data.get("arguments") or {},
            status=IdempotencyStatus(data.get("status") or IdempotencyStatus.PENDING),
            result=data.get("result"),
            error=data.get("error"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            attempts=int(data.get("attempts") or 1),
            retry_authorized=int(data.get("retry_authorized") or 0),
            resolution=data.get("resolution"),
            resolution_note=data.get("resolution_note"),
            resolved_at=data.get("resolved_at"),
        )

    def __str__(self) -> str:
        call = f"{self.tool_name}({self.call_id})"
        if self.status is IdempotencyStatus.COMPLETED:
            return f"{self.idempotency_key} [{self.status}] {call} -> {self.result!r}"
        if self.status is IdempotencyStatus.FAILED:
            message = (self.error or {}).get("message", "failed")
            return f"{self.idempotency_key} [{self.status}] {call} !! {message}"
        extra = (
            f", retry authorized ({self.retry_authorized})"
            if self.retry_authorized
            else ", outcome unknown"
        )
        return (
            f"{self.idempotency_key} [{self.status}] {call} claimed by "
            f"{self.execution_id} (attempt {self.attempts}{extra})"
        )


def unresolved_records(
    records: tuple[IdempotencyRecord, ...] | list[IdempotencyRecord],
) -> tuple[IdempotencyRecord, ...]:
    """The subset of ``records`` that still needs an application decision."""
    return tuple(record for record in records if record.is_unresolved)


class IdempotencyStore:
    """Durable storage for idempotency claims.

    This is a **claim ledger**, not a source of execution truth. It answers one
    question -- *may this key's side effect run, and what happened last time?* --
    and it answers it across process restarts.

    What SQLite guarantees here, and all it guarantees:

    * **uniqueness** -- ``idempotency_key`` is the PRIMARY KEY, so two local
      executions cannot both hold a claim: the second insert fails;
    * **atomicity** -- claiming, resolving and consuming an authorization each
      happen inside one ``IMMEDIATE`` transaction, so a crash leaves a key
      either claimed or not, never half-claimed.

    What SQLite cannot guarantee, and this class never pretends to:

    * **anything about the external world.** The claim commits *before* the tool
      runs. If the process dies between the two, the row says ``PENDING`` while
      the email may or may not have been sent. No transaction here spans the
      database and an HTTP call, so ``PENDING`` is surfaced as an ambiguity
      instead of being resolved by a guess.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    # -- reading --------------------------------------------------------------

    def get(self, key: str) -> IdempotencyRecord | None:
        """The record for ``key``, or ``None`` when nothing has claimed it."""
        rows = self.store.query_all(
            "SELECT * FROM idempotency_records WHERE idempotency_key = ?",
            (_normalize_key(key),),
        )
        return self._row_to_record(rows[0]) if rows else None

    def list_for(
        self,
        *,
        execution_id: str | None = None,
        status: IdempotencyStatus | str | None = None,
    ) -> tuple[IdempotencyRecord, ...]:
        """Records matching the filters, oldest first."""
        clauses: list[str] = []
        params: list[Any] = []
        if execution_id is not None:
            clauses.append("execution_id = ?")
            params.append(execution_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(str(IdempotencyStatus(status)))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.store.query_all(
            f"SELECT * FROM idempotency_records{where} "
            "ORDER BY created_at ASC, idempotency_key ASC",
            tuple(params),
        )
        return tuple(self._row_to_record(row) for row in rows)

    def pending_for(
        self, execution_id: str | None = None
    ) -> tuple[IdempotencyRecord, ...]:
        """Every claim still in flight, optionally narrowed to one execution."""
        return self.list_for(execution_id=execution_id, status=IdempotencyStatus.PENDING)

    def count(self) -> int:
        """How many idempotency records exist."""
        row = self.store.query_all("SELECT COUNT(*) AS total FROM idempotency_records")[0]
        return int(row["total"])

    # -- writing --------------------------------------------------------------

    def claim(
        self,
        key: str,
        *,
        execution_id: str,
        call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> IdempotencyRecord:
        """Claim ``key`` for a side effect that is about to be attempted.

        The one place a claim is allowed to happen, and therefore the one place
        uniqueness matters:

        1. ``BEGIN IMMEDIATE`` takes SQLite's write lock, so this read-then-write
           is serialized against every other claimer -- local threads and other
           processes alike;
        2. no row means the key is free: insert ``PENDING`` and return it;
        3. a row means somebody got here first. If that row is ``COMPLETED`` or
           ``FAILED``, or is ``PENDING`` with no authorized retry, the claim is
           refused and the existing record travels with the exception;
        4. a row *with* an outstanding authorization is re-claimed: the
           authorization is consumed, the attempt counter goes up, and the key
           goes back to ``PENDING``.

        The ``PRIMARY KEY`` is the backstop for step 1's assumption; if two
        writers ever did collide there, the loser gets an
        :class:`~agent_runtime.exceptions.IdempotencyKeyConflictError` instead of
        a second claim.

        Raises:
            IdempotencyKeyConflictError: the key exists and this claim is not
                authorized. Carries the existing record.
        """
        normalized = _normalize_key(key)
        now = utc_now_iso()
        payload = _dump(dict(arguments))
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM idempotency_records WHERE idempotency_key = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                try:
                    conn.execute(
                        """
                        INSERT INTO idempotency_records (
                            idempotency_key, execution_id, call_id, tool_name,
                            arguments, status, result, error, created_at,
                            updated_at, attempts, retry_authorized, resolution,
                            resolution_note, resolved_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, 1, 0, NULL, NULL, NULL)
                        """,
                        (
                            normalized,
                            execution_id,
                            call_id,
                            tool_name,
                            payload,
                            str(IdempotencyStatus.PENDING),
                            now,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError as exc:  # pragma: no cover - backstop
                    raise IdempotencyKeyConflictError(
                        f"Idempotency key {normalized!r} was claimed by another writer "
                        "at the same moment",
                        idempotency_key=normalized,
                    ) from exc
                return self._require(conn, normalized)

            existing = self._row_to_record(row)
            if existing.status is IdempotencyStatus.COMPLETED:
                raise IdempotencyKeyConflictError(
                    f"Idempotency key {normalized!r} already completed with result "
                    f"{existing.result!r}; it will not run again",
                    idempotency_key=normalized,
                    record=existing,
                )
            if existing.retry_authorized <= 0:
                raise IdempotencyKeyConflictError(
                    f"Idempotency key {normalized!r} is {existing.status} and holds no "
                    "authorized retry; resolve it explicitly before running the tool again",
                    idempotency_key=normalized,
                    record=existing,
                )

            # Explicitly authorized: this claim consumes the authorization and
            # moves the key back to PENDING. It never becomes COMPLETED here --
            # nobody has said the external effect happened.
            conn.execute(
                """
                UPDATE idempotency_records
                   SET execution_id = ?,
                       call_id = ?,
                       tool_name = ?,
                       arguments = ?,
                       status = ?,
                       error = NULL,
                       attempts = attempts + 1,
                       retry_authorized = retry_authorized - 1,
                       updated_at = ?
                 WHERE idempotency_key = ? AND retry_authorized > 0
                """,
                (
                    execution_id,
                    call_id,
                    tool_name,
                    payload,
                    str(IdempotencyStatus.PENDING),
                    now,
                    normalized,
                ),
            )
            return self._require(conn, normalized)

    def mark_completed(
        self,
        key: str,
        result: Any,
        *,
        resolution: str = "completed",
        note: str | None = None,
    ) -> IdempotencyRecord:
        """Record that the side effect succeeded, with the result it produced.

        Refuses to overwrite a key that is already ``COMPLETED``: the stored
        result is the answer every future duplicate call is given, and replacing
        it would be a silent guess about which of two side effects actually ran.
        """
        normalized = _normalize_key(key)
        now = utc_now_iso()
        with self.store.transaction() as conn:
            existing = self._require(conn, normalized)
            if existing.status is IdempotencyStatus.COMPLETED:
                raise IdempotencyResolutionError(
                    f"Idempotency key {normalized!r} is already COMPLETED with result "
                    f"{existing.result!r}; a recorded outcome is not overwritten",
                    idempotency_key=normalized,
                    record=existing,
                )
            conn.execute(
                """
                UPDATE idempotency_records
                   SET status = ?, result = ?, error = NULL, retry_authorized = 0,
                       resolution = ?, resolution_note = ?, resolved_at = ?, updated_at = ?
                 WHERE idempotency_key = ?
                """,
                (
                    str(IdempotencyStatus.COMPLETED),
                    _dump(make_jsonable(result)),
                    resolution,
                    note,
                    now,
                    now,
                    normalized,
                ),
            )
            return self._require(conn, normalized)

    def mark_failed(
        self,
        key: str,
        error: Mapping[str, Any] | str | None,
        *,
        resolution: str = "failed",
        note: str | None = None,
    ) -> IdempotencyRecord:
        """Record that the attempt failed, and why.

        ``FAILED`` is a recorded outcome but not permission to run again: a
        failed attempt is not proof that the external effect did not partially
        happen. Calling the key again raises
        :class:`~agent_runtime.exceptions.IdempotencyKeyFailedError` until the
        application resolves it.
        """
        normalized = _normalize_key(key)
        now = utc_now_iso()
        payload = _dump(make_jsonable(_as_error(error)))
        with self.store.transaction() as conn:
            existing = self._require(conn, normalized)
            if existing.status is IdempotencyStatus.COMPLETED:
                raise IdempotencyResolutionError(
                    f"Idempotency key {normalized!r} is already COMPLETED with result "
                    f"{existing.result!r}; a recorded outcome is not overwritten",
                    idempotency_key=normalized,
                    record=existing,
                )
            conn.execute(
                """
                UPDATE idempotency_records
                   SET status = ?, result = NULL, error = ?, retry_authorized = 0,
                       resolution = ?, resolution_note = ?, resolved_at = ?, updated_at = ?
                 WHERE idempotency_key = ?
                """,
                (
                    str(IdempotencyStatus.FAILED),
                    payload,
                    resolution,
                    note,
                    now,
                    now,
                    normalized,
                ),
            )
            return self._require(conn, normalized)

    def authorize_retry(self, key: str, *, note: str | None = None) -> IdempotencyRecord:
        """Authorize exactly one more claim of ``key``.

        The record keeps the status it had -- ``PENDING`` stays ``PENDING``,
        because nobody has said whether the external effect happened -- and
        :attr:`IdempotencyRecord.retry_authorized` goes up by one. The next
        claim of that key consumes it; a second claim without another
        authorization is refused again. "Retry" is therefore one deliberate
        decision, not a standing permission.
        """
        normalized = _normalize_key(key)
        now = utc_now_iso()
        with self.store.transaction() as conn:
            existing = self._require(conn, normalized)
            if existing.status is IdempotencyStatus.COMPLETED:
                raise IdempotencyResolutionError(
                    f"Idempotency key {normalized!r} is COMPLETED with result "
                    f"{existing.result!r}; authorizing a retry would repeat a "
                    "recorded side effect",
                    idempotency_key=normalized,
                    record=existing,
                )
            conn.execute(
                """
                UPDATE idempotency_records
                   SET retry_authorized = retry_authorized + 1,
                       resolution = ?, resolution_note = ?, resolved_at = ?, updated_at = ?
                 WHERE idempotency_key = ?
                """,
                ("retry", note, now, now, normalized),
            )
            return self._require(conn, normalized)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _require(conn: sqlite3.Connection, key: str) -> IdempotencyRecord:
        """Read a key back inside the caller's transaction, or say it is unknown."""
        row = conn.execute(
            "SELECT * FROM idempotency_records WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if row is None:
            raise UnknownIdempotencyKeyError(
                f"No idempotency record for key {key!r}", idempotency_key=key
            )
        return IdempotencyStore._row_to_record(row)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> IdempotencyRecord:
        """Rebuild a record, refusing to guess when a stored column is unreadable."""
        key = row["idempotency_key"]
        try:
            return IdempotencyRecord(
                idempotency_key=key,
                execution_id=row["execution_id"],
                call_id=row["call_id"],
                tool_name=row["tool_name"],
                arguments=_load(row["arguments"], key, "arguments") or {},
                status=IdempotencyStatus(row["status"]),
                result=_load(row["result"], key, "result"),
                error=_load(row["error"], key, "error"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                attempts=int(row["attempts"]),
                retry_authorized=int(row["retry_authorized"]),
                resolution=row["resolution"],
                resolution_note=row["resolution_note"],
                resolved_at=row["resolved_at"],
            )
        except (TypeError, ValueError) as exc:
            raise StorageError(
                f"Idempotency record {key!r} does not hold readable data: {exc}"
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"IdempotencyStore(db_path={self.store.db_path!r})"


# ---------------------------------------------------------------------------
# The guard: the one seam that decides whether a keyed call may run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdempotencyDecision:
    """What to do with a keyed call, once the key has been consulted.

    A decision is only ever returned when the runtime has *cleared* the call:

    * :attr:`deduplicated` -- the key is ``COMPLETED``: return :attr:`result`
      and do not run the tool;
    * otherwise -- the claim succeeded and the tool may run.

    Everything the runtime refuses to guess about raises instead of coming back
    as a decision, so there is no "maybe" branch here to get wrong.
    """

    action: str
    call_id: str
    result: Any = None
    #: The record the decision was made from. ``None`` only for a replay, which
    #: has no store to read.
    record: IdempotencyRecord | None = None

    @property
    def deduplicated(self) -> bool:
        """True when the stored outcome answers the call without executing."""
        return self.action == "deduplicated"

    @property
    def executes(self) -> bool:
        """True when the tool is allowed to run."""
        return self.action == "execute"

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return f"IdempotencyDecision({self.action}, call_id={self.call_id!r})"


@runtime_checkable
class IdempotencyGuard(Protocol):
    """The seam between :class:`~agent_runtime.execution.Execution` and the keys.

    A real run answers from :class:`IdempotencyStore`; a replay answers from
    nothing at all, because a replay must not claim, insert or overwrite
    anything. Having the replay substitute a different guard is what makes that
    structural rather than a convention somebody has to remember.
    """

    def begin_call(
        self,
        key: str,
        *,
        execution_id: str,
        call_id: str,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> IdempotencyDecision:  # pragma: no cover - protocol
        """Decide whether a keyed call may run, claiming the key when it may."""
        ...

    def record_outcome(
        self,
        key: str,
        *,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
    ) -> IdempotencyRecord | None:  # pragma: no cover - protocol
        """Store the final outcome of the logical call that claimed ``key``."""
        ...

    def resolve(
        self,
        key: str,
        action: IdempotencyAction,
        *,
        result: Any = None,
        error: Mapping[str, Any] | str | None = None,
        note: str | None = None,
    ) -> IdempotencyRecord:  # pragma: no cover - protocol
        """Apply an explicit application decision to ``key``."""
        ...

    def get(self, key: str) -> IdempotencyRecord | None:  # pragma: no cover
        """The record for ``key``, or ``None``."""
        ...

    def pending_records(
        self, execution_id: str | None = None
    ) -> tuple[IdempotencyRecord, ...]:  # pragma: no cover - protocol
        """Every claim still in flight, optionally narrowed to one execution."""
        ...


class StoreIdempotencyGuard:
    """The guard a real run uses: :class:`IdempotencyStore` plus the refusals.

    This is where the milestone's policy actually lives:

    * ``COMPLETED``       -> deduplicate: run nothing, hand back the stored result;
    * ``PENDING``         -> :class:`~agent_runtime.exceptions.IdempotencyRecoveryRequiredError`;
    * ``FAILED``          -> :class:`~agent_runtime.exceptions.IdempotencyKeyFailedError`;
    * authorized retry    -> execute, consuming the authorization.
    """

    __slots__ = ("store",)

    def __init__(self, store: IdempotencyStore) -> None:
        self.store = store

    def begin_call(
        self,
        key: str,
        *,
        execution_id: str,
        call_id: str,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> IdempotencyDecision:
        try:
            record = self.store.claim(
                key,
                execution_id=execution_id,
                call_id=call_id,
                tool_name=tool,
                arguments=arguments,
            )
        except IdempotencyKeyConflictError as conflict:
            existing = conflict.record or self.store.get(conflict.idempotency_key)
            if existing is None:  # pragma: no cover - only a lost PRIMARY KEY race
                raise
            if existing.status is IdempotencyStatus.COMPLETED:
                return IdempotencyDecision(
                    action="deduplicated",
                    call_id=call_id,
                    result=existing.result,
                    record=existing,
                )
            if existing.status is IdempotencyStatus.FAILED:
                message = (existing.error or {}).get("message", "the attempt failed")
                raise IdempotencyKeyFailedError(
                    f"Idempotency key {existing.idempotency_key!r} is recorded as "
                    f"FAILED ({message}); it will not run again until it is resolved "
                    "explicitly",
                    idempotency_key=existing.idempotency_key,
                    record=existing,
                ) from None
            raise IdempotencyRecoveryRequiredError(
                f"Idempotency key {existing.idempotency_key!r} is still PENDING: it was "
                f"claimed by {existing.call_id} in execution {existing.execution_id} "
                "and no outcome was ever recorded, so the runtime cannot tell whether "
                "the external side effect happened",
                idempotency_key=existing.idempotency_key,
                record=existing,
            ) from None
        return IdempotencyDecision(
            action="execute", call_id=call_id, result=None, record=record
        )

    def record_outcome(
        self,
        key: str,
        *,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
    ) -> IdempotencyRecord | None:
        """Store the outcome the logical call reached.

        Called once per logical call, after the journal already holds the same
        outcome, so the event history and the claim ledger agree. A crash in
        between leaves the key ``PENDING`` -- conservative, and never a second
        side effect.
        """
        if error is not None:
            return self.store.mark_failed(key, error)
        return self.store.mark_completed(key, result)

    def resolve(
        self,
        key: str,
        action: IdempotencyAction,
        *,
        result: Any = None,
        error: Mapping[str, Any] | str | None = None,
        note: str | None = None,
    ) -> IdempotencyRecord:
        """Apply ``action`` to ``key``: ``retry``, ``mark_completed`` or ``mark_failed``.

        The three actions are the only ways out of an unresolved key, and each
        one is a statement the *application* can back up:

        * ``retry``          -- "run it once more"; the key stays unresolved;
        * ``mark_completed`` -- "I checked, it did happen", plus the real result;
        * ``mark_failed``    -- "I checked, it did not happen", plus the reason.
        """
        if action == "retry":
            return self.store.authorize_retry(key, note=note)
        if action == "mark_completed":
            return self.store.mark_completed(
                key, result, resolution="mark_completed", note=note
            )
        if action == "mark_failed":
            return self.store.mark_failed(
                key, error, resolution="mark_failed", note=note
            )
        raise IdempotencyResolutionError(
            f"Unknown idempotency resolution {action!r}; expected one of "
            f"{', '.join(IDEMPOTENCY_ACTIONS)}",
            idempotency_key=str(key),
        )

    def get(self, key: str) -> IdempotencyRecord | None:
        return self.store.get(key)

    def pending_records(
        self, execution_id: str | None = None
    ) -> tuple[IdempotencyRecord, ...]:
        return self.store.pending_for(execution_id)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"StoreIdempotencyGuard({self.store!r})"


class ReplayIdempotencyGuard:
    """The guard a replay uses: it never writes, and it has no store at all.

    A replay re-runs the runtime's own logic through
    :meth:`~agent_runtime.execution.Execution.call`, so this is what keeps it
    from touching the live ledger. Its answer to a keyed call comes from the
    *recorded history* rather than from a key lookup::

        idempotency key -> the recorded ToolRequested -> the recorded result

    Concretely:

    * :meth:`begin_call` replays the recorded outcome of a call the original
      deduplicated, and clears everything else;
    * :meth:`record_outcome` does nothing, because the recorded outcome is
      already in the journal;
    * :meth:`resolve` refuses, because a replay has nothing to decide;
    * :meth:`pending_records` reports nothing, because a replay claims nothing.

    No database is opened, so "replay does not modify the idempotency store" is
    structural rather than a rule somebody has to keep.

    Args:
        recorded: The calls the journal recorded for the replay, as objects
            exposing ``call_id``, ``deduplicated`` and ``result``. Structural
            typing on purpose: :mod:`agent_runtime.idempotency` must not import
            :mod:`agent_runtime.replay`, which imports this module.
    """

    __slots__ = ("_recorded",)

    def __init__(self, recorded: Sequence[Any] = ()) -> None:
        self._recorded = {item.call_id: item for item in recorded}

    def begin_call(
        self,
        key: str,
        *,
        execution_id: str,
        call_id: str,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> IdempotencyDecision:
        """Serve a recorded deduplicated call; clear everything else.

        The recorded flag is what keeps a replay faithful: the original ran this
        call's tool only once, and a replay that "ran" it again would both
        diverge from the recorded history and put a second side effect in the
        (in-memory) journal.
        """
        recorded = self._recorded.get(call_id)
        if recorded is not None and getattr(recorded, "deduplicated", False):
            return IdempotencyDecision(
                action="deduplicated", call_id=call_id, result=recorded.result
            )
        return IdempotencyDecision(action="execute", call_id=call_id)

    def record_outcome(
        self,
        key: str,
        *,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
    ) -> None:
        """No-op: the recorded outcome is already in the journal."""

    def resolve(
        self,
        key: str,
        action: IdempotencyAction,
        *,
        result: Any = None,
        error: Mapping[str, Any] | str | None = None,
        note: str | None = None,
    ) -> IdempotencyRecord:
        raise IdempotencyResolutionError(
            "A replay never resolves idempotency keys: it reproduces the recorded "
            "history, which is read-only. Resolve the key on a real runtime instead.",
            idempotency_key=str(key),
        )

    def get(self, key: str) -> IdempotencyRecord | None:
        """Always ``None``: a replay has no store, and the journal is its history."""
        return None

    def pending_records(
        self, execution_id: str | None = None
    ) -> tuple[IdempotencyRecord, ...]:
        """Always empty: a replay never claims a key, so it has none pending."""
        return ()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "ReplayIdempotencyGuard()"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _normalize_key(key: str) -> str:
    """The key as it is stored: a non-empty string.

    An empty key would make "no key at all" indistinguishable from "every call
    shares one key", which is the opposite of what the caller asked for.
    """
    if not isinstance(key, str) or not key.strip():
        raise IdempotencyError(
            f"An idempotency key must be a non-empty string, got {key!r}"
        )
    return key


def _dump(value: Any) -> str:
    """JSON-encode a stored column."""
    return json.dumps(make_jsonable(value), sort_keys=True)


def _load(raw: Any, key: str, column: str) -> Any:
    """Decode a stored column, refusing to guess if it is unreadable."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise StorageError(
            f"Idempotency record {key!r} has an unreadable {column} column: {exc}"
        ) from exc


def _as_error(error: Mapping[str, Any] | str | None) -> Mapping[str, Any]:
    """Coerce whatever a caller passed as an error into a JSON-safe payload."""
    if error is None:
        return {"type": "ExecutionError", "message": "unspecified failure"}
    if isinstance(error, str):
        return {"type": "ExecutionError", "message": error}
    return dict(error)
