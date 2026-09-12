"""Crash-scenario tests.

Two layers:

* in-process, where a failure is injected at a precise point to prove the
  transaction rolls back and leaves no partial checkpoint;
* across a real process boundary, where the child calls :func:`os._exit` -- no
  ``finally``, no ``atexit``, no close -- so what survives is only what SQLite
  committed.

In every case the central invariant is checked: a recovered execution is a state
derived from a consistent checkpoint plus the events durably persisted after it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from agent_runtime import (
    CheckpointStore,
    CorruptCheckpointError,
    ExecutionStatus,
    Runtime,
    ToolCallStatus,
    replace_state,
)
from agent_runtime.exceptions import InconsistentCheckpointError
from conftest import PROJECT_ROOT, crash_mid_tool

#: Run by a child process that dies, without unwinding, at CRASH_POINT.
CHILD = r'''
import contextlib, os, sys

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from agent_runtime import Runtime

point = os.environ["CRASH_POINT"]
execution_id = os.environ["CRASH_EXECUTION_ID"]
store_holder = {}


@contextlib.contextmanager
def never_commits():
    """Open a transaction, let the write happen, then die before COMMIT."""
    conn = store_holder["store"].connection
    conn.execute("BEGIN IMMEDIATE")
    yield conn
    os._exit(70)


with Runtime(os.environ["CRASH_DB"]) as runtime:
    store_holder["store"] = runtime.store
    execution = runtime.resume(execution_id)
    call_id = "call_run_tests"

    if point == "before_tool_requested":
        os._exit(70)

    runtime.journal.append_event(
        execution_id,
        "ToolRequested",
        {"call_id": call_id, "tool": "run_tests", "arguments": {"suite": "unit"}},
    )
    if point == "after_tool_requested":
        os._exit(70)

    runtime.journal.append_event(
        execution_id, "ToolStarted", {"call_id": call_id, "tool": "run_tests"}
    )
    if point == "after_tool_started":
        os._exit(70)

    if point == "before_checkpoint_commit":
        # The INSERT runs; the process dies before the COMMIT that would make it
        # durable. Nothing of the checkpoint may be visible afterwards.
        runtime.store.transaction = never_commits
        execution.checkpoint()
        os._exit(71)  # unreachable: never_commits exits first

    execution.checkpoint()
    if point == "after_checkpoint_commit":
        os._exit(70)

    runtime.journal.append_event(
        execution_id,
        "ToolCompleted",
        {"call_id": call_id, "tool": "run_tests", "result": "passed"},
    )
    if point == "after_tool_completed":
        os._exit(70)

os._exit(0)
'''

#: What must be true after each kind of death: events, checkpoints, status and
#: the incomplete call that recovery has to report.
EXPECTATIONS = {
    "before_tool_requested": (1, 0, ExecutionStatus.RUNNING, ()),
    "after_tool_requested": (2, 0, ExecutionStatus.RECOVERY_REQUIRED, (ToolCallStatus.REQUESTED,)),
    "after_tool_started": (3, 0, ExecutionStatus.RECOVERY_REQUIRED, (ToolCallStatus.STARTED,)),
    "before_checkpoint_commit": (3, 0, ExecutionStatus.RECOVERY_REQUIRED, (ToolCallStatus.STARTED,)),
    "after_checkpoint_commit": (3, 1, ExecutionStatus.RECOVERY_REQUIRED, (ToolCallStatus.STARTED,)),
    "after_tool_completed": (4, 1, ExecutionStatus.RUNNING, ()),
    "clean": (4, 1, ExecutionStatus.RUNNING, ()),
}


def run_child(db_path: str, execution_id: str, point: str, *, expect_crash: bool = True) -> None:
    """Run the crash script in a separate process that really dies."""
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CRASH_DB": db_path,
        "CRASH_EXECUTION_ID": execution_id,
        "CRASH_POINT": point,
    }
    result = subprocess.run(
        [sys.executable, "-c", CHILD], env=env, capture_output=True, text=True
    )
    if expect_crash:
        assert result.returncode != 0, (
            f"the child was supposed to die at {point!r} but exited cleanly\n"
            f"{result.stdout}{result.stderr}"
        )
    else:
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("point", sorted(EXPECTATIONS))
def test_recovery_after_a_real_process_crash(runtime, db_path, point):
    events, checkpoints, status, incomplete = EXPECTATIONS[point]

    execution = runtime.start(goal="Perform calculations")
    run_child(db_path, execution.id, point, expect_crash=point != "clean")

    # Everything below is read by a new process, from what SQLite committed.
    with Runtime(db_path) as recovered_runtime:
        recovered = recovered_runtime.resume(execution.id)
        info = recovered.recovery_info()

        assert recovered_runtime.journal.get_last_sequence(execution.id) == events
        assert recovered_runtime.checkpoints.count(execution.id) == checkpoints
        assert recovered.status is status
        assert tuple(item.status for item in recovered.incomplete_tools) == incomplete
        # The invariant: a consistent checkpoint plus the durable events after it
        # gives exactly what folding the whole journal gives.
        assert recovered.state == recovered_runtime.reconstruct_state(execution.id)
        assert info.state == recovered.state
        assert info.needs_resolution is bool(incomplete)


def test_crash_before_the_checkpoint_commit_leaves_no_trace(runtime, db_path):
    """The INSERT ran, the COMMIT did not: the row must not exist."""
    execution = runtime.start(goal="Perform calculations")
    run_child(db_path, execution.id, "before_checkpoint_commit")

    rows = runtime.store.query_all("SELECT * FROM checkpoints")
    assert rows == []
    # The journal events written before the crash are untouched.
    assert [e.event_type for e in runtime.get_events(execution.id)] == [
        "ExecutionStarted",
        "ToolRequested",
        "ToolStarted",
    ]

    # And the execution can be checkpointed normally afterwards.
    recovered = runtime.resume(execution.id)
    checkpoint = recovered.checkpoint()
    assert checkpoint.sequence == 3
    assert runtime.checkpoints.get_latest(execution.id) == checkpoint


def test_a_committed_checkpoint_survives_a_later_crash(runtime, db_path):
    """The snapshot committed before a crash is what the next recovery uses."""
    execution = runtime.start(goal="Perform calculations")
    run_child(db_path, execution.id, "after_checkpoint_commit")

    # The tool finishes after the crash was noticed, and a third process reads
    # the result: the checkpoint plus this one event.
    with Runtime(db_path) as writer:
        writer.journal.append_event(
            execution.id,
            "ToolCompleted",
            {"call_id": "call_run_tests", "tool": "run_tests", "result": "passed"},
        )

    with Runtime(db_path) as recovered_runtime:
        recovered = recovered_runtime.resume(execution.id)
        assert recovered.status is ExecutionStatus.RUNNING
        assert recovered.state == recovered_runtime.reconstruct_state(execution.id)
        assert recovered.recovery_info().checkpoint_sequence == 3
        assert recovered.recovery_info().events_after_checkpoint == 1


# -- injected failures ------------------------------------------------------


def test_a_failure_inside_the_checkpoint_write_rolls_everything_back(runtime, execution, monkeypatch):
    execution.call("add", a=2, b=3)
    sequence = runtime.journal.get_last_sequence(execution.id)

    def explode(conn, checkpoint):
        raise RuntimeError("power cut")

    monkeypatch.setattr(CheckpointStore, "_insert", staticmethod(explode))

    with pytest.raises(RuntimeError):
        execution.checkpoint()

    assert runtime.checkpoints.count(execution.id) == 0
    # The journal is exactly as it was, and recovery from it still works.
    assert runtime.journal.get_last_sequence(execution.id) == sequence
    assert runtime.resume(execution.id).state == runtime.reconstruct_state(execution.id)


def test_a_failure_while_serializing_the_state_writes_nothing(runtime, execution, monkeypatch):
    execution.call("add", a=2, b=3)

    def no_space(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr("agent_runtime.checkpoints.json.dumps", no_space)

    with pytest.raises(OSError):
        execution.checkpoint()

    assert runtime.checkpoints.count(execution.id) == 0


def test_a_torn_checkpoint_is_reported_rather_than_used(restarted, execution):
    """A row whose sequence and state disagree cannot be treated as valid."""
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()
    execution._journal.store.execute(  # tamper with the stored state
        "UPDATE checkpoints SET state = replace(state, ?, ?) WHERE checkpoint_id = ?",
        ('"last_sequence": 4', '"last_sequence": 5', checkpoint.checkpoint_id),
    )

    with pytest.raises(CorruptCheckpointError) as excinfo:
        restarted.resume(execution.id)

    assert "claims sequence 4" in str(excinfo.value)


def test_a_checkpoint_ahead_of_the_journal_is_reported(restarted, execution):
    execution.call("add", a=2, b=3)
    execution.checkpoint()
    # Simulate losing events, e.g. restoring a database from an older backup.
    # Foreign keys are dropped for the delete only; the runtime still refuses to
    # trust the snapshot that now describes events which are gone.
    store = execution._journal.store
    store.execute("PRAGMA foreign_keys=OFF")
    try:
        store.execute(
            "DELETE FROM events WHERE execution_id = ? AND sequence > ?",
            (execution.id, 1),
        )
    finally:
        store.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(CorruptCheckpointError) as excinfo:
        restarted.resume(execution.id)

    assert "journal ends at" in str(excinfo.value)


def test_an_unfinished_call_never_runs_itself_again(runtime, restarted, execution):
    """Recovery reports the open call; it does not re-run it."""
    calls_before = runtime.journal.count_events(execution.id)
    crash_mid_tool(runtime, "run_tests", execution_id=execution.id)

    recovered = restarted.resume(execution.id)
    recovered.state  # reading the state must not invoke anything

    assert restarted.journal.count_events(execution.id) == calls_before + 2
    assert recovered.incomplete_tools[0].tool == "run_tests"


def test_recovered_state_stays_within_the_allowed_statuses(runtime, restarted, execution):
    execution.call("add", a=2, b=3)
    execution.checkpoint()
    crash_mid_tool(runtime, "run_tests", execution_id=execution.id)

    recovered = restarted.resume(execution.id)

    assert recovered.state.status is recovered.status
    assert recovered.status is ExecutionStatus.RECOVERY_REQUIRED
    assert recovered.state.last_sequence == runtime.journal.get_last_sequence(execution.id)
    assert recovered.to_dict()["incomplete_tools"] == [
        item.to_dict() for item in recovered.incomplete_tools
    ]


def test_checkpoint_of_a_state_that_never_happened_is_refused(runtime, execution):
    """A state claiming a sequence the journal has not reached is not stored."""
    execution.call("add", a=2, b=3)
    fabricated = replace_state(execution.state, last_sequence=execution.state.last_sequence + 1)

    with pytest.raises(InconsistentCheckpointError):
        runtime.checkpoints.create(execution.id, fabricated)

    assert runtime.checkpoints.count(execution.id) == 0
