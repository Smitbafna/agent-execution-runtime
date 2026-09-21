"""Idempotency: one key, at most one side effect, and no guessing.

The invariant under test throughout:

    A side-effecting tool never runs twice for one idempotency key unless the
    application explicitly asks for it, and every key whose outcome the runtime
    does not know is reported rather than guessed at.

``sent_emails`` in most tests here is the fake "external service": a list that
only grows when the tool actually runs. Counting it is how "the tool did not
execute" is proved, rather than asserted.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import (
    ExecutionStatus,
    IdempotencyKeyFailedError,
    IdempotencyRecoveryRequiredError,
    IdempotencyRecord,
    IdempotencyResolutionError,
    IdempotencyStatus,
    PermanentToolError,
    RetryableToolError,
    RetryPolicy,
    StoreIdempotencyGuard,
    ToolCallStatus,
    ToolInvocationError,
    UnknownIdempotencyKeyError,
)
from agent_runtime.exceptions import IdempotencyError, InvalidStateTransitionError


@pytest.fixture
def inbox():
    """The fake mail server: a list that grows only when the tool really runs."""
    return []


@pytest.fixture
def mailer(runtime, restarted, inbox):
    """A side-effecting tool registered on both runtimes over the same file.

    Registering it on ``restarted`` too is what makes "the key survived the
    restart" mean something: the second process *could* run the tool, so the
    only thing stopping the duplicate is the claim it reads from SQLite.
    """

    def send_email(to: str, body: str = "") -> dict:
        inbox.append(to)
        return {"message_id": f"msg-{len(inbox)}", "to": to}

    runtime.register_tool(send_email, name="send_email")
    restarted.register_tool(send_email, name="send_email")
    return send_email


# -- basic: first call executes, duplicate does not ----------------------------


def test_a_first_idempotent_call_runs_the_tool_and_records_the_outcome(
    runtime, execution, mailer, inbox
):
    result = execution.call(
        "send_email", to="user@example.com", body="hi", idempotency_key="email-1"
    )

    assert result == {"message_id": "msg-1", "to": "user@example.com"}
    assert inbox == ["user@example.com"]

    record = execution.idempotency_record("email-1")
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.result == result
    assert record.tool_name == "send_email"
    assert record.arguments == {"to": "user@example.com", "body": "hi"}
    assert record.attempts == 1


def test_a_duplicate_call_returns_the_stored_result(runtime, execution, mailer):
    first = execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    second = execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    assert second == first


def test_a_duplicate_call_does_not_execute_the_tool(runtime, execution, mailer, inbox):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    for _ in range(3):
        execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    assert inbox == ["a@x.com"], "the side effect happened more than once"


def test_a_duplicate_is_journalled_as_a_completed_call_with_no_start(
    runtime, execution, mailer
):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    requests = [
        event
        for event in execution.events
        if event.event_type == "ToolRequested"
    ]
    duplicate = requests[1]
    assert duplicate.payload["deduplicated"] is True
    assert duplicate.payload["idempotency_key"] == "email-1"
    # Nothing started, because nothing ran: a ToolStarted here would put a side
    # effect in the history that never happened.
    assert not any(
        event.event_type == "ToolStarted"
        and event.payload["call_id"] == duplicate.payload["call_id"]
        for event in execution.events
    )
    assert execution.tool_calls[1].status is ToolCallStatus.COMPLETED
    assert execution.tool_calls[1].idempotency_key == "email-1"


def test_the_claim_survives_a_restart(runtime, restarted, execution, mailer, inbox):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    recovered = restarted.resume(execution.id)
    result = recovered.call("send_email", to="a@x.com", idempotency_key="email-1")

    assert result == {"message_id": "msg-1", "to": "a@x.com"}
    assert inbox == ["a@x.com"], "a restarted process re-sent the email"


def test_a_key_is_not_required_for_a_call(runtime, execution, mailer, inbox):
    """Without a key, nothing changes: the tool runs on every call."""
    execution.call("send_email", to="a@x.com")
    execution.call("send_email", to="a@x.com")

    assert len(inbox) == 2
    assert runtime.idempotency.count() == 0


def test_an_empty_key_is_refused(runtime, execution, mailer, inbox):
    with pytest.raises(IdempotencyError):
        execution.call("send_email", to="a@x.com", idempotency_key="   ")

    assert inbox == []


# -- multiple keys -------------------------------------------------------------


def test_different_keys_execute_independently(runtime, execution, mailer, inbox):
    first = execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    second = execution.call("send_email", to="b@x.com", idempotency_key="email-2")

    assert inbox == ["a@x.com", "b@x.com"]
    assert first != second
    assert runtime.idempotency.count() == 2


def test_a_key_is_shared_across_executions(runtime, restarted, execution, mailer, inbox):
    """The key names the side effect, not the execution that happened to make it."""
    execution.call("send_email", to="a@x.com", idempotency_key="welcome-123")

    other = restarted.start(goal="a second execution")
    result = other.call("send_email", to="a@x.com", idempotency_key="welcome-123")

    assert result == {"message_id": "msg-1", "to": "a@x.com"}
    assert inbox == ["a@x.com"]
    assert other.tool_calls[0].idempotency_key == "welcome-123"


# -- failure -------------------------------------------------------------------


@pytest.fixture
def failures():
    """How many more attempts the ``flaky`` tool should refuse."""
    return {"left": 2}


@pytest.fixture
def flaky(runtime, failures):
    """A payment tool that declines while ``failures`` says so, then succeeds."""

    @runtime.tool(retry_policy=RetryPolicy(max_attempts=1))
    def charge(amount: int) -> dict:
        if failures["left"]:
            failures["left"] -= 1
            raise PermanentToolError("card declined")
        return {"charge_id": f"ch-{amount}"}

    return charge


def test_a_failed_idempotent_call_is_recorded_as_failed(
    runtime, execution, flaky, inbox
):
    with pytest.raises(ToolInvocationError):
        execution.call("charge", amount=10, idempotency_key="pay-1")

    record = execution.idempotency_record("pay-1")
    assert record.status is IdempotencyStatus.FAILED
    assert "declined" in record.error["message"]


def test_a_failed_key_is_not_re_executed_by_a_later_call(runtime, execution, flaky):
    with pytest.raises(ToolInvocationError):
        execution.call("charge", amount=10, idempotency_key="pay-1")
    events_before = len(runtime.journal.get_events(execution.id))

    with pytest.raises(IdempotencyKeyFailedError) as excinfo:
        execution.call("charge", amount=10, idempotency_key="pay-1")

    assert "pay-1" in str(excinfo.value)
    # The refusal happens before anything is journalled: nothing was attempted,
    # so nothing is recorded as an attempt.
    assert len(runtime.journal.get_events(execution.id)) == events_before


def test_an_explicit_retry_re_executes_and_can_succeed(
    runtime, execution, flaky, failures
):
    failures["left"] = 1  # only the first attempt declines
    with pytest.raises(ToolInvocationError):
        execution.call("charge", amount=10, idempotency_key="pay-1")

    execution.resolve_idempotency("pay-1", "retry")
    result = execution.call("charge", amount=10, idempotency_key="pay-1")

    record = execution.idempotency_record("pay-1")
    assert result == {"charge_id": "ch-10"}
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.attempts == 2, "the retry reused the key instead of making a new one"


def test_a_retry_is_authorized_once_not_forever(runtime, execution, flaky, failures):
    failures["left"] = 3  # every attempt in this test declines
    with pytest.raises(ToolInvocationError):
        execution.call("charge", amount=10, idempotency_key="pay-1")
    execution.resolve_idempotency("pay-1", "retry")
    with pytest.raises(ToolInvocationError):
        execution.call("charge", amount=10, idempotency_key="pay-1")

    with pytest.raises(IdempotencyKeyFailedError):
        execution.call("charge", amount=10, idempotency_key="pay-1")

    assert failures["left"] == 1, "two attempts ran, and then the runtime stopped"


def test_a_recorded_outcome_is_never_overwritten(runtime, execution, mailer):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    with pytest.raises(IdempotencyResolutionError) as excinfo:
        execution.resolve_idempotency("email-1", "mark_failed", error="it did not send")

    assert "not overwritten" in str(excinfo.value)
    assert execution.idempotency_record("email-1").status is IdempotencyStatus.COMPLETED


# -- retries keep one key across attempts (Milestone 4A + 4B) ------------------


def test_a_retrying_call_keeps_one_key_across_all_attempts(runtime, execution, inbox):
    """Attempt 3 of a logical call is not a second claim of the key."""
    runs = {"count": 0}

    @runtime.tool(retry_policy=RetryPolicy(max_attempts=3, initial_delay=0))
    def fetch(url: str) -> str:
        runs["count"] += 1
        inbox.append(url)
        if runs["count"] < 3:
            raise RetryableToolError("upstream is down")
        return "payload"

    result = execution.call("fetch", url="https://x/data", idempotency_key="fetch-1")

    assert result == "payload"
    record = execution.idempotency_record("fetch-1")
    assert record.attempts == 1, "three attempts of one call, one claim"
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.result == "payload"

    keys = {
        event.payload.get("idempotency_key")
        for event in execution.events
        if event.event_type == "ToolRequested"
    }
    assert keys == {"fetch-1"}
    assert execution.tool_calls[0].attempt == 3
    assert runtime.idempotency.count() == 1


# -- pending / recovery --------------------------------------------------------


def test_a_pending_key_makes_recovery_required(runtime, execution):
    """The central case: claimed, then the process never learned the outcome."""
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )

    # The journal is perfectly clean: no open tool call at all.
    assert execution.incomplete_tools == ()
    assert execution.status is ExecutionStatus.RECOVERY_REQUIRED

    record = execution.unresolved_idempotency[0]
    assert record.idempotency_key == "email-1"
    assert record.status is IdempotencyStatus.PENDING


def test_recovery_reports_a_pending_key(runtime, restarted, execution):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )

    info = restarted.recovery_info(execution.id)

    assert info.status is ExecutionStatus.RECOVERY_REQUIRED
    assert info.needs_idempotency_resolution is True
    assert [record.idempotency_key for record in info.pending_idempotency] == ["email-1"]
    assert "Unresolved idempotency keys" in str(info)


def test_calling_a_pending_key_does_not_run_the_tool(runtime, execution, mailer, inbox):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )

    with pytest.raises(IdempotencyRecoveryRequiredError) as excinfo:
        execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    assert inbox == [], "the runtime ran the tool for an unresolved key"
    assert "email-1" in str(excinfo.value)
    assert "resolve_idempotency" in str(excinfo.value)


def test_reading_state_never_executes_or_resolves_anything(
    runtime, execution, mailer, inbox
):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )

    assert execution.state.status is ExecutionStatus.RUNNING  # the journal is clean
    assert execution.status is ExecutionStatus.RECOVERY_REQUIRED
    assert execution.to_dict()["pending_idempotency"][0]["status"] == "PENDING"
    assert str(execution)
    assert inbox == []
    assert runtime.idempotency_record("email-1").status is IdempotencyStatus.PENDING


def test_an_authorized_key_is_no_longer_unresolved(runtime, execution, mailer, inbox):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )
    assert execution.status is ExecutionStatus.RECOVERY_REQUIRED

    execution.resolve_idempotency("email-1", "retry")

    assert execution.unresolved_idempotency == ()
    assert execution.status is ExecutionStatus.RUNNING
    assert execution.call(
        "send_email", to="a@x.com", idempotency_key="email-1"
    ) == {"message_id": "msg-1", "to": "a@x.com"}
    assert inbox == ["a@x.com"]


# -- explicit resolution -------------------------------------------------------


def test_mark_completed_records_an_externally_known_result(runtime, execution, mailer, inbox):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )

    settled = execution.resolve_idempotency(
        "email-1", "mark_completed", result={"message_id": "checked-by-hand"}
    )

    assert settled.status is IdempotencyStatus.COMPLETED
    assert settled.resolution == "mark_completed"
    assert execution.status is ExecutionStatus.RUNNING

    # The duplicate is answered from what the operator recorded, still without
    # running anything.
    assert execution.call(
        "send_email", to="a@x.com", idempotency_key="email-1"
    ) == {"message_id": "checked-by-hand"}
    assert inbox == []


def test_mark_failed_records_a_known_failure(runtime, execution, mailer, inbox):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )

    settled = execution.resolve_idempotency(
        "email-1", "mark_failed", error="the mail server has no record of it"
    )

    assert settled.status is IdempotencyStatus.FAILED
    assert "no record" in settled.error["message"]
    assert settled.resolution == "mark_failed"
    assert execution.status is ExecutionStatus.RUNNING, "FAILED is an outcome, not an ambiguity"
    with pytest.raises(IdempotencyKeyFailedError):
        execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    assert inbox == []


def test_a_resolution_records_its_note(runtime, execution):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={},
    )

    settled = execution.resolve_idempotency("email-1", "retry", note="asked the provider")

    assert settled.resolution_note == "asked the provider"
    assert settled.retry_authorized == 1
    assert settled.status is IdempotencyStatus.PENDING


def test_resolving_an_unknown_key_is_refused(runtime, execution):
    with pytest.raises(UnknownIdempotencyKeyError) as excinfo:
        execution.resolve_idempotency("never-seen", "retry")

    assert "never-seen" in str(excinfo.value)


def test_an_unknown_action_is_refused(runtime, execution):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={},
    )

    with pytest.raises(IdempotencyResolutionError) as excinfo:
        execution.resolve_idempotency("email-1", "assume_it_worked")

    assert "assume_it_worked" in str(excinfo.value)


def test_a_key_belonging_to_another_execution_is_resolved_there(runtime, restarted, execution):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={},
    )
    other = restarted.start(goal="elsewhere")

    with pytest.raises(IdempotencyResolutionError) as excinfo:
        other.resolve_idempotency("email-1", "retry")
    assert "resolve it there" in str(excinfo.value)

    # The runtime-level API is the operator's: no execution handle required.
    settled = restarted.resolve_idempotency("email-1", "retry")
    assert settled.retry_authorized == 1


def test_an_unresolved_key_stops_the_execution_from_completing(runtime, execution):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={},
    )

    with pytest.raises(InvalidStateTransitionError) as excinfo:
        execution.complete()

    assert "email-1" in str(excinfo.value)
    # Giving up is still allowed: it is a decision, not an oversight.
    execution.fail("the side effect could not be confirmed")
    assert execution.status is ExecutionStatus.FAILED


def test_resolving_the_key_lets_the_execution_complete(runtime, execution, mailer, inbox):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={},
    )
    execution.resolve_idempotency("email-1", "mark_failed", error="never sent")

    execution.complete("done")

    assert execution.status is ExecutionStatus.COMPLETED


# -- checkpoints ---------------------------------------------------------------


def test_a_checkpoint_never_causes_a_completed_operation_to_run_again(
    runtime, restarted, execution, mailer, inbox
):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    execution.checkpoint()
    execution.call("add", a=1, b=2)
    execution.checkpoint()

    recovered = restarted.resume(execution.id)
    assert recovered.call(
        "send_email", to="a@x.com", idempotency_key="email-1"
    ) == {"message_id": "msg-1", "to": "a@x.com"}
    assert inbox == ["a@x.com"]


def test_recovery_from_a_checkpoint_still_reports_a_pending_key(
    runtime, restarted, execution, mailer
):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    execution.checkpoint()
    runtime.idempotency.claim(
        "email-2",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "b@x.com"},
    )

    info = restarted.recovery_info(execution.id)

    assert info.source == "checkpoint"
    assert info.status is ExecutionStatus.RECOVERY_REQUIRED
    assert [record.idempotency_key for record in info.pending_idempotency] == ["email-2"]


def test_a_checkpoint_does_not_carry_idempotency_state(runtime, restarted, execution, mailer):
    """Checkpoints stay a cache of the journal; keys live in their own table."""
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    checkpoint = execution.checkpoint()

    assert "idempotency_key" in json.dumps(checkpoint.state.to_dict())
    stored = restarted.store.query_all(
        "SELECT status, result FROM idempotency_records WHERE idempotency_key = ?",
        ("email-1",),
    )
    assert stored[0]["status"] == "COMPLETED"
    assert json.loads(stored[0]["result"]) == {"message_id": "msg-1", "to": "a@x.com"}


# -- transaction boundaries ----------------------------------------------------


def test_a_failure_storing_the_outcome_leaves_the_key_pending(
    runtime, execution, mailer, inbox, monkeypatch
):
    """The window between the side effect and its record -- conservatively.

    If the outcome cannot be written, the runtime keeps the key ``PENDING``
    rather than pretending the email was never sent.
    """

    def no_room_for_the_outcome(self, key, *, result=None, error=None):
        raise OSError("disk full")

    monkeypatch.setattr(StoreIdempotencyGuard, "record_outcome", no_room_for_the_outcome)

    with pytest.raises(OSError):
        execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    record = runtime.idempotency_record("email-1")
    assert record.status is IdempotencyStatus.PENDING
    assert inbox == ["a@x.com"], "the effect really did happen"
    with pytest.raises(IdempotencyRecoveryRequiredError):
        execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    assert inbox == ["a@x.com"], "and the runtime refuses to send a second one"


def test_a_claim_that_cannot_be_written_stops_the_call(
    runtime, execution, mailer, inbox, monkeypatch
):
    """No durable claim means no side effect: the call never starts."""

    def no_room_for_the_claim(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(StoreIdempotencyGuard, "begin_call", no_room_for_the_claim)

    with pytest.raises(OSError):
        execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    assert inbox == []
    assert runtime.idempotency.count() == 0
    assert [event.event_type for event in execution.events] == ["ExecutionStarted"]


def test_a_resolution_that_cannot_be_written_rolls_back(
    runtime, execution, monkeypatch
):
    runtime.idempotency.claim(
        "email-1",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={},
    )

    def no_space(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr("agent_runtime.idempotency.json.dumps", no_space)
    with pytest.raises(OSError):
        execution.resolve_idempotency("email-1", "mark_completed", result={"ok": True})

    record = runtime.idempotency_record("email-1")
    assert record.status is IdempotencyStatus.PENDING
    assert record.retry_authorized == 0


# -- the store itself ----------------------------------------------------------


def test_the_table_exists_with_the_key_as_primary_key(runtime):
    columns = {
        row["name"]: row for row in runtime.store.query_all("PRAGMA table_info(idempotency_records)")
    }

    assert columns["idempotency_key"]["pk"] == 1
    for required in ("execution_id", "call_id", "tool_name", "arguments", "status", "created_at"):
        assert columns[required]["notnull"] == 1


def test_sqlite_refuses_a_second_row_for_one_key(runtime):
    runtime.idempotency.claim(
        "email-1",
        execution_id="exec_1",
        call_id="call_1",
        tool_name="send_email",
        arguments={},
    )

    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        runtime.store.execute(
            "INSERT INTO idempotency_records (idempotency_key, execution_id, call_id,"
            " tool_name, arguments, status, created_at, updated_at)"
            " VALUES ('email-1', 'exec_2', 'call_2', 'send_email', '{}', 'PENDING', 'x', 'x')"
        )


def test_records_round_trip_through_to_dict(runtime, execution, mailer):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")

    record = runtime.idempotency_record("email-1")

    assert IdempotencyRecord.from_dict(record.to_dict()) == record
    assert json.loads(json.dumps(record.to_dict())) == record.to_dict()
    assert str(record).startswith("email-1 [COMPLETED] send_email")


def test_a_record_is_narrowed_by_execution_and_status(runtime, execution, mailer, inbox):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    runtime.idempotency.claim(
        "email-2",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={},
    )
    other = runtime.start(goal="elsewhere")
    runtime.idempotency.claim(
        "email-3",
        execution_id=other.id,
        call_id="call_other",
        tool_name="send_email",
        arguments={},
    )

    assert [r.idempotency_key for r in runtime.pending_idempotency(execution.id)] == ["email-2"]
    assert [r.idempotency_key for r in runtime.pending_idempotency()] == ["email-2", "email-3"]
    assert [r.idempotency_key for r in runtime.unresolved_idempotency()] == ["email-2", "email-3"]
    assert len(runtime.idempotency_records(status=IdempotencyStatus.COMPLETED)) == 1
    assert len(runtime.idempotency_records(execution_id=execution.id)) == 2


def test_unreadable_stored_data_is_reported_not_guessed(runtime, execution, mailer):
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    runtime.store.execute(
        "UPDATE idempotency_records SET result = ? WHERE idempotency_key = ?",
        ("{not json", "email-1"),
    )

    from agent_runtime import StorageError

    with pytest.raises(StorageError) as excinfo:
        runtime.idempotency_record("email-1")

    assert "unreadable result" in str(excinfo.value)
