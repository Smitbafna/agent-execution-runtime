"""Replay must not touch the idempotency ledger -- and cannot.

§13 asks for four things a replay must not do: claim a key, insert a
``PENDING`` record, execute a real tool, overwrite a result. Three of them are
proved here by comparing the *whole* table before and after; the fourth is
structural: :class:`ReplayIdempotencyGuard` has no store behind it, so there is
no code path from a replay to the database at all.

A replay answers a keyed call from the recorded history instead::

    idempotency key -> the recorded ToolRequested -> the recorded result
"""

from __future__ import annotations

import pytest

from agent_runtime import (
    ExecutionStatus,
    IdempotencyResolutionError,
    IdempotencyStatus,
    ReplayIdempotencyGuard,
    ReplayMismatchError,
    Runtime,
)


@pytest.fixture
def inbox():
    return []


@pytest.fixture
def mailer(runtime, inbox):
    def send_email(to: str, body: str = "") -> dict:
        inbox.append(to)
        return {"message_id": f"msg-{len(inbox)}", "to": to}

    runtime.register_tool(send_email, name="send_email")
    return send_email


def snapshot(runtime: Runtime) -> list[dict]:
    """Every row of the idempotency table, as plain data.

    Compared before and after a replay: "the store was not modified" has to mean
    every column of every row, not just the count.
    """
    return [
        {key: row[key] for key in row.keys()}
        for row in runtime.store.query_all(
            "SELECT * FROM idempotency_records ORDER BY idempotency_key"
        )
    ]


@pytest.fixture
def keyed_execution(runtime, execution, mailer, inbox):
    """An execution with a real keyed call, a duplicate, and a pending key."""
    execution.call("send_email", to="a@x.com", body="hi", idempotency_key="email-1")
    execution.call("send_email", to="a@x.com", body="hi", idempotency_key="email-1")
    runtime.idempotency.claim(
        "email-2",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "b@x.com"},
    )
    return execution


def test_replay_does_not_modify_the_idempotency_store(runtime, keyed_execution, inbox):
    before = snapshot(runtime)
    assert len(before) == 2

    runtime.replay(keyed_execution.id)

    assert snapshot(runtime) == before
    assert runtime.idempotency.count() == 2
    assert runtime.idempotency_record("email-1").status is IdempotencyStatus.COMPLETED
    assert runtime.idempotency_record("email-2").status is IdempotencyStatus.PENDING


def test_replay_does_not_execute_the_side_effect(runtime, keyed_execution, inbox):
    assert inbox == ["a@x.com"]

    result = runtime.replay(keyed_execution.id)

    assert result.matched is True
    assert all(step.executed is False for step in result.steps)
    assert inbox == ["a@x.com"], "replay sent the email again"


def test_replay_reproduces_the_deduplicated_call_from_the_history(
    runtime, keyed_execution
):
    result = runtime.replay(keyed_execution.id)

    assert result.matched is True
    assert result.tools_replayed == 2
    replayed = result.final_state.tool_calls
    assert [call.result for call in replayed] == [
        {"message_id": "msg-1", "to": "a@x.com"},
        {"message_id": "msg-1", "to": "a@x.com"},
    ]
    assert [call.idempotency_key for call in replayed] == ["email-1", "email-1"]
    assert all(call.started_sequence == 0 for call in replayed[:0] + replayed[1:])


def test_replaying_from_a_checkpoint_leaves_the_store_alone(
    runtime, restarted, keyed_execution, inbox
):
    keyed_execution.checkpoint()
    before = snapshot(restarted)

    restarted.replay(keyed_execution.id)

    assert snapshot(restarted) == before
    assert inbox == ["a@x.com"]


def test_replaying_an_unresolved_key_creates_no_new_record(runtime, keyed_execution):
    """A claim with no journal trace of its call is still never touched."""
    before = snapshot(runtime)

    result = runtime.replay(keyed_execution.id)

    assert result.matched is True
    assert snapshot(runtime) == before, "the replay claimed a key of its own"
    assert runtime.idempotency.count() == 2
    assert runtime.idempotency_record("email-2").status is IdempotencyStatus.PENDING


def test_replaying_a_crashed_call_ends_unresolved_and_writes_nothing(
    runtime, execution, mailer
):
    """The crash shape: ``ToolRequested`` + ``ToolStarted``, key claimed, no outcome."""
    from conftest import crash_mid_tool

    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    runtime.idempotency.claim(
        "email-2",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "b@x.com"},
    )
    crash_mid_tool(runtime, "send_email", start=True, execution_id=execution.id)
    before = snapshot(runtime)

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state.status is ExecutionStatus.RECOVERY_REQUIRED
    assert snapshot(runtime) == before


def test_a_deduplicated_call_in_the_middle_replays_correctly(
    runtime, execution, mailer, inbox
):
    """Deduplication in the middle of a history, not only at the end of it."""
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    execution.call("send_email", to="b@x.com", idempotency_key="email-2")
    before = snapshot(runtime)

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.tools_replayed == 3
    assert inbox == ["a@x.com", "b@x.com"]
    assert snapshot(runtime) == before
    # The duplicate started nothing in the original, so it starts nothing here.
    started = [call.started_sequence > 0 for call in result.final_state.tool_calls]
    assert started == [True, False, True]


def test_the_replay_guard_has_no_store_to_write_to():
    guard = ReplayIdempotencyGuard()

    assert guard.get("email-1") is None
    assert guard.pending_records() == ()
    with pytest.raises(IdempotencyResolutionError) as excinfo:
        guard.resolve("email-1", "retry")
    assert "read-only" in str(excinfo.value)


def test_the_replay_guard_refuses_to_record_an_outcome():
    guard = ReplayIdempotencyGuard()

    assert guard.record_outcome("email-1", result={"anything": True}) is None


def test_a_replay_with_a_different_key_is_reported(runtime, keyed_execution):
    replay = runtime.replay_engine(keyed_execution.id).prepare()

    with pytest.raises(ReplayMismatchError) as excinfo:
        replay.call(
            "send_email", to="a@x.com", body="hi", idempotency_key="a-different-key"
        )

    assert "email-1" in str(excinfo.value)
    assert excinfo.value.kind == "arguments"


def test_a_replay_without_the_recorded_key_is_reported(runtime, keyed_execution):
    replay = runtime.replay_engine(keyed_execution.id).prepare()

    with pytest.raises(ReplayMismatchError):
        replay.call(
            "send_email", to="a@x.com", body="hi", idempotency_key=None
        )


def test_replaying_twice_is_stable(runtime, keyed_execution, inbox):
    first = runtime.replay(keyed_execution.id)
    second = runtime.replay(keyed_execution.id)

    assert first.final_state.to_dict() == second.final_state.to_dict()
    assert [step.to_dict() for step in first.steps] == [
        step.to_dict() for step in second.steps
    ]
    assert inbox == ["a@x.com"]
