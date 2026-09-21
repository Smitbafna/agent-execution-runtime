"""Two local executions must not both win the same key.

The guarantee is exactly what SQLite can give, and nothing more:

    the PRIMARY KEY of ``idempotency_records`` plus one ``IMMEDIATE``
    transaction around the read-then-write, so two local writers -- two
    processes, or two Runtime objects over one file -- serialize and exactly one
    of them holds the claim.

There is no distributed lock here, and no claim about two machines. Each test
opens its own connections the way separate processes would.
"""

from __future__ import annotations

import threading

import pytest

from agent_runtime import (
    IdempotencyKeyConflictError,
    IdempotencyRecord,
    IdempotencyStatus,
    Runtime,
)


@pytest.fixture
def side_by_side(db_path):
    """Two :class:`Runtime` objects over one file -- two local writers."""
    with Runtime(db_path) as first, Runtime(db_path) as second:
        yield first, second


def test_the_second_execution_sees_the_first_ones_claim(side_by_side):
    first, second = side_by_side
    first_execution = first.start(goal="first")
    second_execution = second.start(goal="second")

    claimed = first.idempotency.claim(
        "email-1",
        execution_id=first_execution.id,
        call_id="call_a",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )
    assert claimed.status is IdempotencyStatus.PENDING

    with pytest.raises(IdempotencyKeyConflictError) as excinfo:
        second.idempotency.claim(
            "email-1",
            execution_id=second_execution.id,
            call_id="call_b",
            tool_name="send_email",
            arguments={"to": "a@x.com"},
        )

    assert excinfo.value.record.call_id == "call_a"
    assert excinfo.value.idempotency_key == "email-1"


def test_a_duplicate_tool_call_across_two_runtimes_runs_once(side_by_side):
    first, second = side_by_side
    sent: list[str] = []

    def send_email(to: str) -> dict:
        sent.append(to)
        return {"message_id": f"msg-{len(sent)}"}

    first.register_tool(send_email, name="send_email")
    second.register_tool(send_email, name="send_email")

    one = first.start(goal="one")
    two = second.start(goal="two")
    first_result = one.call("send_email", to="a@x.com", idempotency_key="email-1")
    second_result = two.call("send_email", to="a@x.com", idempotency_key="email-1")

    assert first_result == second_result == {"message_id": "msg-1"}
    assert sent == ["a@x.com"]


def test_simultaneous_claims_produce_exactly_one_winner(side_by_side):
    """Both threads aim at the same key at the same instant."""
    first, second = side_by_side
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def claim(runtime: Runtime, label: str) -> None:
        barrier.wait(timeout=10)
        try:
            runtime.idempotency.claim(
                "email-1",
                execution_id=runtime.start(goal=label).id,
                call_id=f"call_{label}",
                tool_name="send_email",
                arguments={"to": "a@x.com"},
            )
        except IdempotencyKeyConflictError as exc:
            outcome = f"{label}: conflict ({exc.record.call_id})"
        else:
            outcome = f"{label}: claimed"
        with lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=claim, args=(first, "a")),
        threading.Thread(target=claim, args=(second, "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(outcomes) == 2, outcomes
    winner = next(outcome for outcome in outcomes if outcome.endswith("claimed"))
    loser = next(outcome for outcome in outcomes if "conflict" in outcome)
    winner_label = winner.split(":")[0]
    assert winner_label != loser.split(":")[0], "both threads claimed it"
    assert f"(call_{winner_label})" in loser, "the loser saw the winner's claim"

    record = first.idempotency_record("email-1")
    assert record.status is IdempotencyStatus.PENDING
    assert record.attempts == 1


def test_a_racing_resolution_cannot_reopen_a_completed_key(side_by_side):
    """Whoever loses the race still sees a recorded outcome it must not overwrite."""
    first, second = side_by_side
    first.idempotency.claim(
        "email-1",
        execution_id=first.start(goal="one").id,
        call_id="call_a",
        tool_name="send_email",
        arguments={},
    )
    first.idempotency.mark_completed("email-1", {"message_id": "msg-1"})

    with pytest.raises(Exception) as excinfo:
        second.idempotency.mark_failed("email-1", "I think it failed")
    assert "not overwritten" in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        second.idempotency.authorize_retry("email-1")
    assert "would repeat a recorded side effect" in str(excinfo.value)

    record = second.idempotency_record("email-1")
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.result == {"message_id": "msg-1"}


def test_an_authorized_retry_is_consumed_by_exactly_one_of_two_writers(side_by_side):
    first, second = side_by_side
    first.idempotency.claim(
        "email-1",
        execution_id=first.start(goal="one").id,
        call_id="call_a",
        tool_name="send_email",
        arguments={},
    )
    first.resolve_idempotency("email-1", "retry")

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def claim(runtime: Runtime, label: str) -> None:
        barrier.wait(timeout=10)
        try:
            runtime.idempotency.claim(
                "email-1",
                execution_id=runtime.start(goal=label).id,
                call_id=f"call_{label}",
                tool_name="send_email",
                arguments={},
            )
        except IdempotencyKeyConflictError:
            outcome = f"{label}: conflict"
        else:
            outcome = f"{label}: claimed"
        with lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=claim, args=(first, "a")),
        threading.Thread(target=claim, args=(second, "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(outcomes) == 2, outcomes
    claimed = [outcome for outcome in outcomes if outcome.endswith("claimed")]
    assert len(claimed) == 1, "one authorization cannot become two claims"

    record = first.idempotency_record("email-1")
    assert record.attempts == 2
    assert record.retry_authorized == 0
    assert record.status is IdempotencyStatus.PENDING


def test_a_record_read_back_from_another_connection_is_identical(side_by_side):
    first, second = side_by_side
    first.idempotency.claim(
        "email-1",
        execution_id=first.start(goal="one").id,
        call_id="call_a",
        tool_name="send_email",
        arguments={"to": "a@x.com"},
    )
    first.idempotency.mark_completed("email-1", {"message_id": "msg-1"})

    read_back = second.idempotency.get("email-1")

    assert isinstance(read_back, IdempotencyRecord)
    assert read_back == first.idempotency.get("email-1")
    assert IdempotencyRecord.from_dict(read_back.to_dict()) == read_back
