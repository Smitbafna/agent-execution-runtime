"""Checkpoint tests: persistence, exact sequences, state contents, consistency."""

from __future__ import annotations

import json
import sqlite3

import pytest

from agent_runtime import (
    CorruptCheckpointError,
    ExecutionState,
    ExecutionStatus,
    InconsistentCheckpointError,
    ToolCallStatus,
    ToolInvocationError,
    replace_state,
)
from agent_runtime.checkpoints import Checkpoint, storable_state


# -- persisted --------------------------------------------------------------


def test_checkpoint_is_persisted(runtime, execution):
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()

    rows = runtime.store.query_all(
        "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint.checkpoint_id,)
    )

    assert len(rows) == 1
    assert rows[0]["execution_id"] == execution.id
    assert isinstance(rows[0]["state"], str)
    assert rows[0]["created_at"] == checkpoint.created_at


def test_checkpoint_records_the_exact_sequence(runtime, execution):
    execution.call("add", a=2, b=3)
    last_event = runtime.get_events(execution.id)[-1]

    checkpoint = execution.checkpoint()

    assert checkpoint.sequence == last_event.sequence
    assert checkpoint.sequence == runtime.journal.get_last_sequence(execution.id)
    assert checkpoint.state.last_sequence == checkpoint.sequence


def test_checkpoint_contains_the_reconstructed_state(runtime, execution):
    execution.call("add", a=2, b=3)
    execution.call("multiply", a=5, b=10)

    checkpoint = execution.checkpoint()
    stored = runtime.checkpoints.get(execution.id, checkpoint.sequence)

    assert stored is not None
    assert stored.state == execution.state
    assert stored.state.goal == execution.goal
    assert [call.tool for call in stored.state.tool_calls] == ["add", "multiply"]
    assert [call.result for call in stored.state.tool_calls] == [5, 50]
    assert all(call.status is ToolCallStatus.COMPLETED for call in stored.state.tool_calls)


def test_checkpoint_state_is_stored_as_json(runtime, execution):
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()

    raw = runtime.store.query_all(
        "SELECT state FROM checkpoints WHERE checkpoint_id = ?", (checkpoint.checkpoint_id,)
    )[0]["state"]
    payload = json.loads(raw)

    assert payload["execution_id"] == execution.id
    assert payload["last_sequence"] == checkpoint.sequence
    assert payload["tool_calls"][0]["result"] == 5


def test_checkpoint_state_round_trips_exactly(runtime, execution):
    execution.call("add", a=2, b=3)
    with pytest.raises(ToolInvocationError):
        execution.call("divide", a=1, b=0)  # fails; its error must survive too
    original = execution.checkpoint().state

    reloaded = runtime.checkpoints.get_latest(execution.id).state

    assert reloaded == original
    assert reloaded.tool_calls[1].status is ToolCallStatus.FAILED
    assert reloaded.tool_calls[1].error["type"] == "ZeroDivisionError"
    assert ExecutionState.from_dict(json.loads(json.dumps(reloaded.to_dict()))) == original


def test_checkpoint_survives_a_new_connection(restarted, execution):
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()

    latest = restarted.latest_checkpoint(execution.id)

    assert latest is not None
    assert latest.checkpoint_id == checkpoint.checkpoint_id
    assert latest.state == execution.state


# -- several checkpoints ----------------------------------------------------


def test_multiple_checkpoints_can_exist(runtime, execution):
    execution.call("add", a=2, b=3)
    first = execution.checkpoint()
    execution.call("multiply", a=5, b=10)
    second = execution.checkpoint()
    execution.call("subtract", a=9, b=4)
    third = execution.checkpoint()

    stored = runtime.get_checkpoints(execution.id)

    assert [c.sequence for c in stored] == [first.sequence, second.sequence, third.sequence]
    assert [c.sequence for c in stored] == sorted(c.sequence for c in stored)
    assert len({c.checkpoint_id for c in stored}) == 3
    assert runtime.checkpoints.count(execution.id) == 3


def test_latest_checkpoint_can_be_retrieved(runtime, execution):
    execution.call("add", a=2, b=3)
    execution.checkpoint()
    execution.call("multiply", a=5, b=10)
    newest = execution.checkpoint()

    latest = runtime.latest_checkpoint(execution.id)

    assert latest is not None
    assert latest.checkpoint_id == newest.checkpoint_id
    assert latest.sequence == newest.sequence
    assert len(latest.state.tool_calls) == 2
    assert runtime.checkpoints.latest_sequence(execution.id) == newest.sequence


def test_no_checkpoint_exists_before_one_is_written(runtime, execution):
    execution.call("add", a=2, b=3)

    assert runtime.latest_checkpoint(execution.id) is None
    assert execution.has_checkpoint is False
    assert execution.latest_checkpoint is None
    assert runtime.checkpoints.latest_sequence(execution.id) == 0


def test_checkpoint_at_the_same_sequence_is_not_duplicated(runtime, execution):
    execution.call("add", a=2, b=3)

    first = execution.checkpoint()
    again = execution.checkpoint()

    # A snapshot is a pure function of its event prefix, so storing it again is
    # a no-op that hands back the row already on file.
    assert again.checkpoint_id == first.checkpoint_id
    assert runtime.checkpoints.count(execution.id) == 1


def test_checkpoints_are_isolated_per_execution(runtime):
    first = runtime.start(goal="first")
    first.call("add", a=1, b=1)
    first.checkpoint()
    second = runtime.start(goal="second")
    second.call("add", a=2, b=2)
    second.checkpoint()

    assert runtime.checkpoints.count(first.id) == 1
    assert runtime.checkpoints.count(second.id) == 1
    assert sorted(runtime.checkpoints.list_execution_ids()) == sorted([first.id, second.id])
    assert runtime.latest_checkpoint(first.id).state.goal == "first"


def test_get_checkpoint_by_sequence(runtime, execution):
    execution.call("add", a=2, b=3)
    first = execution.checkpoint()
    execution.call("multiply", a=5, b=10)
    execution.checkpoint()

    assert runtime.checkpoints.get(execution.id, first.sequence) == first
    assert runtime.checkpoints.get(execution.id, 999) is None


# -- consistency ------------------------------------------------------------


def test_checkpoint_rejects_a_sequence_the_journal_has_not_reached(runtime, execution):
    execution.call("add", a=2, b=3)
    ahead = replace_state(execution.state, last_sequence=execution.state.last_sequence + 1)

    with pytest.raises(InconsistentCheckpointError):
        runtime.checkpoints.create(execution.id, ahead)

    assert runtime.checkpoints.count(execution.id) == 0


def test_checkpoint_rejects_a_sequence_the_journal_has_passed(runtime, execution):
    execution.call("add", a=2, b=3)
    behind = replace_state(execution.state, last_sequence=execution.state.last_sequence - 1)

    with pytest.raises(InconsistentCheckpointError) as excinfo:
        runtime.checkpoints.create(execution.id, behind)

    assert "latest event" in str(excinfo.value)
    assert runtime.checkpoints.count(execution.id) == 0


def test_checkpoint_rejects_an_unknown_execution(runtime):
    with pytest.raises(InconsistentCheckpointError):
        runtime.checkpoints.create("exec_never_ran", ExecutionState(execution_id="exec_never_ran"))


def test_checkpoint_rejects_a_sequence_with_no_event(runtime, execution):
    execution.call("add", a=2, b=3)
    # A gap in the journal cannot be produced through the API, so the guard is
    # exercised directly: a sequence no event occupies must not be check-pointable.
    state = replace_state(execution.state, last_sequence=runtime.journal.get_last_sequence(execution.id) + 5)

    with pytest.raises(InconsistentCheckpointError):
        runtime.checkpoints.create(execution.id, state)


def test_foreign_key_rejects_a_checkpoint_for_a_missing_event(runtime, execution):
    execution.call("add", a=2, b=3)
    last = runtime.journal.get_last_sequence(execution.id)

    with pytest.raises(sqlite3.IntegrityError):
        runtime.store.execute(
            "INSERT INTO checkpoints (checkpoint_id, execution_id, sequence, state, created_at)"
            " VALUES ('ckpt_raw', ?, ?, '{}', 'now')",
            (execution.id, last + 1),
        )


def test_unique_sequence_rejects_two_checkpoints_at_one_sequence(runtime, execution):
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()

    with pytest.raises(sqlite3.IntegrityError):
        runtime.store.execute(
            "INSERT INTO checkpoints (checkpoint_id, execution_id, sequence, state, created_at)"
            " VALUES ('ckpt_other', ?, ?, '{}', 'now')",
            (execution.id, checkpoint.sequence),
        )


def test_storable_state_keeps_the_recovery_status_out_of_storage(execution):
    execution.call("add", a=2, b=3)
    needs_recovery = replace_state(execution.state, status=ExecutionStatus.RECOVERY_REQUIRED)

    stored = storable_state(needs_recovery)

    # RECOVERY_REQUIRED is a diagnosis recovery re-derives from the events, so it
    # must not be persisted as though it were part of the state.
    assert stored.status is ExecutionStatus.RUNNING
    completed = replace_state(execution.state, status=ExecutionStatus.COMPLETED)
    assert storable_state(completed).status is ExecutionStatus.COMPLETED


def test_checkpoint_of_an_execution_needing_recovery_keeps_the_open_call(runtime, execution):
    call_id = "call_stuck"
    runtime.journal.append_event(
        execution.id, "ToolRequested", {"call_id": call_id, "tool": "run_tests", "arguments": {}}
    )
    runtime.journal.append_event(
        execution.id, "ToolStarted", {"call_id": call_id, "tool": "run_tests"}
    )

    checkpoint = execution.checkpoint()

    assert str(checkpoint.state.status) == "RUNNING"
    assert [item.call_id for item in checkpoint.state.incomplete_tools] == [call_id]
    assert checkpoint.state.incomplete_tools[0].sequence == execution.state.last_sequence


def test_corrupt_checkpoint_state_is_reported(runtime, execution):
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()
    runtime.store.execute(
        "UPDATE checkpoints SET state = ? WHERE checkpoint_id = ?",
        ('{"execution_id": "x", ', checkpoint.checkpoint_id),
    )

    with pytest.raises(CorruptCheckpointError):
        runtime.latest_checkpoint(execution.id)


def test_stored_checkpoint_is_not_rewritten_by_later_events(runtime, execution):
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()

    with pytest.raises(AttributeError):
        checkpoint.sequence = 99  # type: ignore[misc]

    execution.call("multiply", a=5, b=10)
    assert runtime.checkpoints.get(execution.id, checkpoint.sequence).state == checkpoint.state


def test_checkpoint_helpers_agree_with_the_store(runtime, execution):
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()

    assert Checkpoint.from_dict(checkpoint.to_dict()) == checkpoint
    assert runtime.checkpoints.has_checkpoints(execution.id) is True
    assert str(checkpoint).startswith(f"Checkpoint {checkpoint.checkpoint_id}")
