"""The CLI half of Milestone 4B: looking at keys, and deciding about them.

``agent-runtime idempotency ...`` is what an operator has when the process that
made the claim is gone. What matters is that it *shows* the ambiguity and
records a decision that was actually made -- never one the tool inferred.
"""

from __future__ import annotations

import io
import json

import pytest

from agent_runtime import IdempotencyStatus, Runtime
from agent_runtime.cli import main


@pytest.fixture
def claimed(runtime, execution):
    """One settled key and one the "process" never finished."""
    runtime.register_tool(lambda to: {"message_id": "msg-1"}, name="send_email")
    execution.call("send_email", to="a@x.com", idempotency_key="email-1")
    runtime.idempotency.claim(
        "email-2",
        execution_id=execution.id,
        call_id="call_crashed",
        tool_name="send_email",
        arguments={"to": "b@x.com"},
    )
    return execution


def run(db_path, *argv) -> tuple[int, str]:
    buffer = io.StringIO()
    code = main(["--db", db_path, *argv], out=buffer)
    return code, buffer.getvalue()


def test_list_shows_the_unresolved_keys_first(db_path, claimed):
    code, output = run(db_path, "idempotency", "list")

    assert code == 0
    assert "email-2" in output
    assert "email-1" not in output, "a settled key is not something to act on"
    assert "PENDING" in output
    assert "--action {retry|mark_completed|mark_failed}" in output


def test_list_all_shows_every_record(db_path, claimed):
    code, output = run(db_path, "idempotency", "list", "--all")

    assert code == 0
    assert "email-1" in output and "COMPLETED" in output
    assert "email-2" in output and "PENDING" in output


def test_list_json_is_machine_readable(db_path, claimed):
    code, output = run(db_path, "idempotency", "list", "--all", "--json")

    records = json.loads(output)
    assert code == 0
    assert {record["idempotency_key"] for record in records} == {"email-1", "email-2"}


def test_list_can_be_narrowed_to_an_execution(db_path, claimed, restarted):
    other = restarted.start(goal="elsewhere")
    runtime = Runtime(db_path)
    runtime.idempotency.claim(
        "email-3",
        execution_id=other.id,
        call_id="call_other",
        tool_name="send_email",
        arguments={},
    )
    runtime.close()

    code, output = run(db_path, "idempotency", "list", "--execution", claimed.id)

    assert code == 0
    assert "email-2" in output
    assert "email-3" not in output


def test_show_prints_one_record(db_path, claimed):
    code, output = run(db_path, "idempotency", "show", "email-2")

    assert code == 0
    assert "idempotency_key: email-2" in output
    assert "status: PENDING" in output
    assert "call_crashed" in output


def test_show_of_an_unknown_key_reports_it(db_path, claimed):
    code, output = run(db_path, "idempotency", "show", "never-claimed")

    assert code == 2
    assert "never-claimed" in output


def test_resolve_records_a_known_failure(db_path, claimed):
    code, output = run(
        db_path,
        "idempotency",
        "resolve",
        "email-2",
        "--action",
        "mark_failed",
        "--error",
        "the provider has no record of it",
        "--note",
        "checked by hand",
    )

    assert code == 0
    assert "mark_failed" in output

    with Runtime(db_path) as runtime:
        record = runtime.idempotency_record("email-2")
    assert record.status is IdempotencyStatus.FAILED
    assert "no record" in record.error["message"]
    assert record.resolution_note == "checked by hand"


def test_resolve_records_a_known_success_with_a_json_result(db_path, claimed):
    code, _ = run(
        db_path,
        "idempotency",
        "resolve",
        "email-2",
        "--action",
        "mark_completed",
        "--result",
        '{"message_id": "checked"}',
    )

    assert code == 0
    with Runtime(db_path) as runtime:
        record = runtime.idempotency_record("email-2")
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.result == {"message_id": "checked"}


def test_resolve_retry_authorizes_one_more_attempt(db_path, claimed):
    code, output = run(db_path, "idempotency", "resolve", "email-2", "--action", "retry")

    assert code == 0
    assert "retry authorized x1" in output

    with Runtime(db_path) as runtime:
        record = runtime.idempotency_record("email-2")
        # Still unresolved: the outcome is unknown, the next call just may run.
        assert record.status is IdempotencyStatus.PENDING
        assert record.retry_authorized == 1
        assert record.resolution == "retry"
        assert runtime.unresolved_idempotency() == ()


def test_resolve_will_not_overwrite_a_settled_key(db_path, claimed, capsys):
    code, _ = run(
        db_path, "idempotency", "resolve", "email-1", "--action", "mark_failed"
    )

    # The refusal goes to stderr, the way every runtime error does from here.
    assert code == 2
    assert "not overwritten" in capsys.readouterr().err


def test_the_execution_listing_counts_unresolved_keys(db_path, claimed):
    code, output = run(db_path, "list")

    assert code == 0
    assert f"{claimed.id}" in output
    assert "1 unresolved key(s)" in output
