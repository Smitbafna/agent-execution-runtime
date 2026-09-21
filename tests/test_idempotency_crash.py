"""The crash this milestone exists for, in a real child process.

    claim key        (committed to SQLite)
    run the tool     (the "external" side effect really happens -- and is
                      appended to a file, because after ``os._exit`` the only
                      trace of it that survives is one the outside world wrote)
    [os._exit]       (no ToolCompleted, no idempotency outcome, no close, no atexit)

Then a *second* process opens the same database and has to answer the only
question that matters: does the runtime send the email again?

The answer asserted here is no -- and not because it knows the mail went out.
It cannot know that. It refuses to run the tool at all until an application
decides, which is the only honest answer available.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from agent_runtime import (
    ExecutionStatus,
    IdempotencyRecoveryRequiredError,
    IdempotencyStatus,
    InvalidStateTransitionError,
    Runtime,
    ToolCallStatus,
)
from conftest import PROJECT_ROOT

#: Run by a child process that dies without unwinding at CRASH_POINT. Both
#: points are *after* the claim is committed and *after* the side effect, which
#: is the whole point: the difference between them is only whether the outcome
#: had started to be written.
CHILD = r'''
import os, sys

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from agent_runtime import Runtime, StoreIdempotencyGuard

point = os.environ["CRASH_POINT"]
db = os.environ["CRASH_DB"]
effects = os.environ["CRASH_EFFECTS"]
execution_id = os.environ["CRASH_EXECUTION_ID"]


def send_effect(to: str) -> dict:
    """The fake mail server. Its write is the only trace that outlives the crash."""
    with open(effects, "a") as handle:
        handle.write(to + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if point == "mid_tool":
        # The mail is sent; the process dies before the runtime hears anything.
        os._exit(70)
    return {"message_id": "msg-1", "to": to}


if point == "before_outcome":
    def die_before_the_outcome(self, key, *, result=None, error=None):
        """Die between the recorded ToolCompleted and the stored idempotency result."""
        os._exit(71)

    StoreIdempotencyGuard.record_outcome = die_before_the_outcome

with Runtime(db) as runtime:
    runtime.register_tool(send_effect, name="send_email")
    execution = runtime.start(goal="welcome", execution_id=execution_id)
    execution.call("send_email", to="crashed@example.com", idempotency_key="email-1")

os._exit(0)
'''


def run_child(db_path: str, effects_path: str, point: str) -> int:
    """Run the crashing child; it never returns normally."""
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CRASH_DB": db_path,
        "CRASH_EFFECTS": effects_path,
        "CRASH_POINT": point,
        "CRASH_EXECUTION_ID": "exec_crashed",
    }
    completed = subprocess.run(
        [sys.executable, "-c", CHILD], env=env, capture_output=True, text=True
    )
    return completed.returncode


def effects_in(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [line for line in handle.read().splitlines() if line]


def sending_tool(effects_path: str):
    """A tool that appends to the *same* file the child wrote.

    Counting executions in one file that both processes write is what makes
    "the tool ran exactly once more" checkable across a process boundary.
    """

    def send_email(to: str) -> dict:
        with open(effects_path, "a") as handle:
            handle.write(to + "\n")
        return {"message_id": "msg-2", "to": to}

    return send_email


@pytest.fixture
def crashed(db_path, tmp_path):
    """A database whose keyed side effect happened, and whose outcome never landed."""
    effects = str(tmp_path / "effects.txt")
    returncode = run_child(db_path, effects, "mid_tool")
    assert returncode == 70, "the child was supposed to die mid-tool"
    return db_path, effects


@pytest.mark.parametrize("point", ["mid_tool", "before_outcome"])
def test_a_crash_after_the_side_effect_leaves_the_key_pending(db_path, tmp_path, point):
    effects = str(tmp_path / "effects.txt")
    assert run_child(db_path, effects, point) in (70, 71)

    with Runtime(db_path) as runtime:
        record = runtime.idempotency_record("email-1")

    assert record.status is IdempotencyStatus.PENDING
    assert record.result is None
    assert effects_in(effects) == ["crashed@example.com"], "the email really was sent"


@pytest.mark.parametrize("point", ["mid_tool", "before_outcome"])
def test_recovery_reports_the_unresolved_key_after_a_crash(db_path, tmp_path, point):
    effects = str(tmp_path / "effects.txt")
    run_child(db_path, effects, point)

    with Runtime(db_path) as runtime:
        execution = runtime.resume("exec_crashed")
        info = runtime.recovery_info("exec_crashed")

        assert execution.status is ExecutionStatus.RECOVERY_REQUIRED
        assert info.needs_idempotency_resolution is True
        assert [record.idempotency_key for record in info.pending_idempotency] == ["email-1"]
        assert execution.unresolved_idempotency[0].call_id == execution.tool_calls[0].call_id
        assert info.to_dict()["pending_idempotency"][0]["status"] == "PENDING"


def test_the_journal_reports_exactly_what_the_child_committed(crashed):
    db_path, _ = crashed
    with Runtime(db_path) as runtime:
        state = runtime.reconstruct_state("exec_crashed")

    # The claim and both events are durable; the outcome is not.
    assert [call.status for call in state.tool_calls] == [ToolCallStatus.STARTED]
    assert state.tool_calls[0].idempotency_key == "email-1"


def settle_the_open_call(execution, *, result=None):
    """Resolve the tool call the crash left open, when there is one.

    Separate on purpose: the journal's open call and the idempotency key are two
    different unresolved things with two different answers, and neither stands in
    for the other. ``before_outcome`` dies later than ``mid_tool`` -- by then the
    journal already holds ``ToolCompleted`` -- so there is nothing to settle,
    and the key is the only thing left unresolved.
    """
    if not execution.incomplete_tools:
        return None
    [stuck] = execution.incomplete_tools
    return execution.resolve_recovery(
        stuck.call_id, "mark_completed", result=result or {"message_id": "msg-1"}
    )


def test_an_open_tool_call_stops_the_duplicate(db_path, tmp_path):
    """The first refusal is Milestone 2's: an unfinished call blocks new work."""
    effects = str(tmp_path / "effects.txt")
    run_child(db_path, effects, "mid_tool")

    with Runtime(db_path) as runtime:
        runtime.register_tool(sending_tool(effects), name="send_email")
        execution = runtime.resume("exec_crashed")

        with pytest.raises(InvalidStateTransitionError) as excinfo:
            execution.call(
                "send_email", to="crashed@example.com", idempotency_key="email-1"
            )
        assert "RECOVERY_REQUIRED" in str(excinfo.value)

    assert effects_in(effects) == ["crashed@example.com"], "the email was sent twice"


@pytest.mark.parametrize("point", ["mid_tool", "before_outcome"])
def test_the_key_stops_the_duplicate_whatever_the_journal_says(
    db_path, tmp_path, point
):
    """Settle whatever the journal left open, and the key is still saying no.

    ``before_outcome`` is the sharper case: the journal is completely clean --
    the call completed -- and the key alone keeps the second email from going
    out, because the outcome was never stored.
    """
    effects = str(tmp_path / "effects.txt")
    run_child(db_path, effects, point)

    with Runtime(db_path) as runtime:
        runtime.register_tool(sending_tool(effects), name="send_email")
        execution = runtime.resume("exec_crashed")
        settle_the_open_call(execution)

        assert execution.state.status is ExecutionStatus.RUNNING
        assert execution.status is ExecutionStatus.RECOVERY_REQUIRED

        with pytest.raises(IdempotencyRecoveryRequiredError) as excinfo:
            execution.call(
                "send_email", to="crashed@example.com", idempotency_key="email-1"
            )

    assert "email-1" in str(excinfo.value)
    assert effects_in(effects) == ["crashed@example.com"], "the email was sent twice"


def test_an_operator_can_record_what_actually_happened(crashed):
    db_path, effects = crashed

    with Runtime(db_path) as runtime:
        runtime.register_tool(sending_tool(effects), name="send_email")
        execution = runtime.resume("exec_crashed")
        settle_the_open_call(execution)
        # "I checked the provider: it really was sent."
        execution.resolve_idempotency(
            "email-1", "mark_completed", result={"message_id": "msg-1"}
        )

        assert execution.status is ExecutionStatus.RUNNING
        # The duplicate is now answered from what was recorded -- still no send.
        assert execution.call(
            "send_email", to="crashed@example.com", idempotency_key="email-1"
        ) == {"message_id": "msg-1"}

    assert effects_in(effects) == ["crashed@example.com"]


def test_an_operator_can_authorize_exactly_one_more_send(crashed):
    db_path, effects = crashed

    with Runtime(db_path) as runtime:
        runtime.register_tool(sending_tool(effects), name="send_email")
        execution = runtime.resume("exec_crashed")
        settle_the_open_call(execution)
        execution.resolve_idempotency("email-1", "retry", note="confirmed not sent")

        result = execution.call(
            "send_email", to="crashed@example.com", idempotency_key="email-1"
        )

        record = runtime.idempotency_record("email-1")
        assert result == {"message_id": "msg-2", "to": "crashed@example.com"}
        assert record.status is IdempotencyStatus.COMPLETED
        assert record.attempts == 2

    assert effects_in(effects) == ["crashed@example.com", "crashed@example.com"]


def test_an_operator_can_record_that_it_never_happened(crashed):
    db_path, effects = crashed

    with Runtime(db_path) as runtime:
        execution = runtime.resume("exec_crashed")
        execution.resolve_idempotency(
            "email-1", "mark_failed", error="the provider has no record of it"
        )

        record = runtime.idempotency_record("email-1")
        assert record.status is IdempotencyStatus.FAILED
        assert record.resolution == "mark_failed"

    assert effects_in(effects) == ["crashed@example.com"]


def test_the_crash_is_recoverable_from_another_process_entirely(db_path, tmp_path):
    """Everything after the crash happens through a runtime this test opens later."""
    effects = str(tmp_path / "effects.txt")
    run_child(db_path, effects, "mid_tool")

    # A separate process resolves the key...
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CRASH_DB": db_path,
    }
    resolver = subprocess.run(
        [sys.executable, "-c", RESOLVER],
        env=env,
        capture_output=True,
        text=True,
    )
    assert resolver.returncode == 0, resolver.stderr

    # ...and this one sees the settled outcome.
    with Runtime(db_path) as runtime:
        record = runtime.idempotency_record("email-1")
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.result == {"message_id": "checked-by-hand"}
    assert effects_in(effects) == ["crashed@example.com"]


RESOLVER = r'''
import os, sys

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from agent_runtime import Runtime

with Runtime(os.environ["CRASH_DB"]) as runtime:
    record = runtime.resolve_idempotency(
        "email-1", "mark_completed", result={"message_id": "checked-by-hand"}
    )
    print(record.status)

os._exit(0)
'''
