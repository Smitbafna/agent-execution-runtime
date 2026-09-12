"""Recovery tests: checkpoint-based resume, equivalence, incomplete tool detection."""

from __future__ import annotations

import pytest

from agent_runtime import (
    CorruptCheckpointError,
    ExecutionNotFoundError,
    ExecutionStatus,
    InvalidRecoveryActionError,
    InvalidStateTransitionError,
    ToolCallStatus,
    UnknownToolCallError,
)
from conftest import assert_recovery_required, crash_mid_tool


# -- without a checkpoint ---------------------------------------------------


def test_resume_without_a_checkpoint(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    execution_id = execution.id

    recovered = restarted.resume(execution_id)

    assert recovered.id == execution_id
    assert recovered.goal == "Perform calculations"
    assert [call.result for call in recovered.tool_calls] == [5]
    assert recovered.status is ExecutionStatus.RUNNING


def test_recovery_without_a_checkpoint_replays_every_event(runtime, restarted):
    execution = runtime.start(goal="no checkpoints here")
    for value in range(1, 4):
        execution.call("add", a=value, b=value)

    info = restarted.recovery_info(execution.id)

    assert info.source == "events"
    assert info.checkpoint_sequence is None
    assert info.events_replayed == runtime.journal.get_last_sequence(execution.id)
    assert info.state == runtime.reconstruct_state(execution.id)


def test_resume_reports_an_unknown_execution(runtime):
    with pytest.raises(ExecutionNotFoundError):
        runtime.resume("exec_does_not_exist")


# -- with a checkpoint ------------------------------------------------------


def test_resume_with_a_checkpoint_matches_a_full_reconstruction(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()          # checkpoint at sequence 4
    execution.call("multiply", a=5, b=10)         # events 5, 6, 7 land after it
    execution.call("subtract", a=9, b=4)

    recovered = restarted.resume(execution.id)

    assert recovered.state == runtime.reconstruct_state(execution.id)
    assert [call.tool for call in recovered.tool_calls] == ["add", "multiply", "subtract"]
    assert [call.result for call in recovered.tool_calls] == [5, 50, 5]


def test_recovery_only_replays_the_events_after_the_checkpoint(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()
    execution.call("multiply", a=5, b=10)

    info = restarted.recovery_info(execution.id)

    assert info.source == "checkpoint"
    assert info.checkpoint_sequence == checkpoint.sequence
    assert info.checkpoint_id == checkpoint.checkpoint_id
    assert info.events_after_checkpoint == 3
    assert [e.sequence for e in runtime.journal.get_events_from(execution.id, checkpoint.sequence)] == [
        checkpoint.sequence + 1,
        checkpoint.sequence + 2,
        checkpoint.sequence + 3,
    ]


def test_resume_uses_the_latest_checkpoint(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    stale = execution.checkpoint()
    execution.call("multiply", a=5, b=10)
    fresh = execution.checkpoint()

    info = restarted.recovery_info(execution.id)

    assert info.checkpoint_id == fresh.checkpoint_id != stale.checkpoint_id
    assert info.events_after_checkpoint == 0
    assert len(info.state.tool_calls) == 2


def test_resume_works_when_the_checkpoint_is_the_very_latest_event(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    execution.checkpoint()

    recovered = restarted.resume(execution.id)

    assert recovered.state == runtime.reconstruct_state(execution.id)
    assert recovered.recovery_info().events_after_checkpoint == 0


def test_recovery_is_idempotent(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    execution.checkpoint()
    execution.call("multiply", a=5, b=10)

    first = restarted.recovery_info(execution.id)
    second = restarted.recovery_info(execution.id)

    assert first == second
    assert first.state == runtime.reconstruct_state(execution.id)


def test_corrupt_checkpoint_is_reported_instead_of_ignored(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()
    runtime.store.execute(
        "UPDATE checkpoints SET state = 'not json' WHERE checkpoint_id = ?",
        (checkpoint.checkpoint_id,),
    )

    # Falling back to a full replay would hide that stored data is broken.
    with pytest.raises(CorruptCheckpointError):
        restarted.resume(execution.id)


def test_completed_execution_resumes_as_completed(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    execution.complete(result="done")

    recovered = restarted.resume(execution.id)

    assert recovered.status is ExecutionStatus.COMPLETED
    assert recovered.state.result == "done"
    assert recovered.incomplete_tools == ()


def test_cancelled_execution_resumes_as_cancelled(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)
    execution.mark_cancelled("operator stopped it")

    recovered = restarted.resume(execution.id)

    assert recovered.status is ExecutionStatus.CANCELLED
    assert recovered.state.result == "operator stopped it"
    with pytest.raises(InvalidStateTransitionError):
        recovered.call("add", a=1, b=1)


# -- incomplete tool detection ---------------------------------------------


def test_incomplete_tool_after_tool_started_is_detected(restarted, crashed_after_tool_started):
    recovered = restarted.resume(crashed_after_tool_started)
    call_id = assert_recovery_required(recovered)

    (stuck,) = recovered.incomplete_tools
    assert stuck.call_id == call_id
    assert stuck.tool == "run_tests"
    assert stuck.status is ToolCallStatus.STARTED
    assert stuck.sequence == 3           # the sequence of the ToolStarted event
    assert stuck.arguments == {"suite": "unit"}
    assert stuck.was_started is True
    assert stuck.started_at is not None


def test_incomplete_tool_after_tool_requested_is_detected(restarted, crashed_after_tool_requested):
    recovered = restarted.resume(crashed_after_tool_requested)
    call_id = assert_recovery_required(recovered)

    (stuck,) = recovered.incomplete_tools
    assert stuck.status is ToolCallStatus.REQUESTED
    assert stuck.sequence == 2           # the sequence of the ToolRequested event
    assert stuck.started_sequence == 0
    assert stuck.was_started is False


def test_resolved_tool_leaves_the_execution_running(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)

    recovered = restarted.resume(execution.id)

    assert recovered.status is ExecutionStatus.RUNNING
    assert recovered.incomplete_tools == ()
    assert recovered.needs_recovery is False


def test_finished_tool_call_does_not_need_recovery_even_with_open_ones(runtime, restarted):
    execution = runtime.start(goal="Perform calculations")
    execution.call("add", a=2, b=3)                       # resolved
    call_id = crash_mid_tool(runtime, "run_tests", execution_id=execution.id)

    recovered = restarted.resume(execution.id)

    assert recovered.status is ExecutionStatus.RECOVERY_REQUIRED
    assert [c.call_id for c in recovered.incomplete_tools] == [call_id]
    assert recovered.tool_calls[0].status is ToolCallStatus.COMPLETED


def test_an_execution_may_need_recovery_twice(runtime, restarted):
    """A second crash after the first one was resolved shows up as well."""
    execution = runtime.start(goal="Perform calculations")
    first_id = crash_mid_tool(runtime, "run_tests", execution_id=execution.id)
    resumed = restarted.resume(execution.id)
    resumed.resolve_recovery(first_id, "mark_completed", result="passed")
    second_id = crash_mid_tool(runtime, "lint", execution_id=execution.id)

    again = restarted.resume(execution.id)

    assert [item.call_id for item in again.incomplete_tools] == [second_id]
    assert again.state == runtime.reconstruct_state(execution.id)


def test_incomplete_execution_refuses_to_run_more_tools(runtime, restarted, crashed_after_tool_started):
    recovered = restarted.resume(crashed_after_tool_started)

    with pytest.raises(InvalidStateTransitionError) as excinfo:
        recovered.call("add", a=1, b=1)

    assert "RECOVERY_REQUIRED" in str(excinfo.value)
    # The rejection happened before anything was journalled.
    assert runtime.journal.get_last_sequence(crashed_after_tool_started) == 3


# -- recovery information ---------------------------------------------------


def test_recovery_info_describes_the_situation(runtime, restarted, crashed_after_tool_started):
    info = restarted.recovery_info(crashed_after_tool_started)

    assert info.execution_id == crashed_after_tool_started
    assert info.status is ExecutionStatus.RECOVERY_REQUIRED
    assert info.needs_resolution is True
    assert info.last_sequence == 3
    assert [item.tool for item in info.incomplete_tools] == ["run_tests"]

    text = str(info)
    assert f"Execution: {crashed_after_tool_started}" in text
    assert "Status: RECOVERY_REQUIRED" in text
    assert "Tool: run_tests" in text
    assert "Started at sequence: 3" in text
    assert "suite='unit'" in text

    payload = info.to_dict()
    assert payload["status"] == "RECOVERY_REQUIRED"
    assert payload["incomplete_tools"][0]["sequence"] == 3


def test_recovery_info_of_a_clean_execution_says_so(runtime, restarted, execution):
    execution.call("add", a=2, b=3)
    execution.checkpoint()

    info = restarted.recovery_info(execution.id)

    assert info.needs_resolution is False
    assert info.status is ExecutionStatus.RUNNING
    assert "Incomplete operations" not in str(info)


def test_recovery_info_changes_nothing(runtime, crashed_after_tool_started):
    before = runtime.journal.get_last_sequence(crashed_after_tool_started)
    info = runtime.recovery_info(crashed_after_tool_started)

    assert runtime.journal.get_last_sequence(crashed_after_tool_started) == before
    assert runtime.checkpoints.count(crashed_after_tool_started) == 0
    assert info.state == runtime.reconstruct_state(crashed_after_tool_started)


# -- resolving what a crash left unfinished ---------------------------------
#
# Milestone 2 does not retry. These tests cover the other half of the contract:
# the application decides, the runtime records that decision durably.


def test_resolve_recovery_as_completed_lets_the_execution_continue(restarted, crashed_after_tool_started):
    recovered = restarted.resume(crashed_after_tool_started)
    call_id = assert_recovery_required(recovered)

    recovered.resolve_recovery(call_id, "mark_completed", result="tests passed")

    assert recovered.status is ExecutionStatus.RUNNING
    assert recovered.incomplete_tools == ()
    assert recovered.tool_calls[0].result == "tests passed"
    # The execution carries on from where the journal says it was.
    assert recovered.call("add", a=2, b=3) == 5
    assert recovered.state == restarted.reconstruct_state(crashed_after_tool_started)


def test_resolve_recovery_is_recorded_as_an_event(runtime, restarted, crashed_after_tool_started):
    recovered = restarted.resume(crashed_after_tool_started)
    call_id = assert_recovery_required(recovered)

    recovered.resolve_recovery(call_id, "mark_completed", result="tests passed")

    events = runtime.get_events(crashed_after_tool_started)
    assert events[-1].event_type == "ToolCompleted"
    assert events[-1].payload["resolution"] == "mark_completed"
    assert events[-1].payload["result"] == "tests passed"
    # A fresh process sees the resolved call, not the original problem.
    assert restarted.resume(crashed_after_tool_started).status is ExecutionStatus.RUNNING


def test_resolve_recovery_as_failed_records_the_given_error(runtime, crashed_after_tool_started):
    recovered = runtime.resume(crashed_after_tool_started)
    call_id = assert_recovery_required(recovered)

    recovered.resolve_recovery(call_id, "mark_failed", error={"message": "test run timed out"})

    call = recovered.tool_calls[0]
    assert call.status is ToolCallStatus.FAILED
    assert call.error["message"] == "test run timed out"
    assert runtime.get_events(crashed_after_tool_started)[-1].event_type == "ToolFailed"
    # Nothing is left unresolved, so the execution is running again.
    assert recovered.status is ExecutionStatus.RUNNING


def test_resolve_recovery_can_cancel_a_call(runtime, crashed_after_tool_started):
    recovered = runtime.resume(crashed_after_tool_started)
    call_id = assert_recovery_required(recovered)

    recovered.resolve_recovery(call_id, "cancel")

    assert recovered.tool_calls[0].status is ToolCallStatus.CANCELLED
    assert runtime.get_events(crashed_after_tool_started)[-1].event_type == "ToolCancelled"
    assert recovered.status is ExecutionStatus.RUNNING


def test_recovery_can_be_ended_by_failing_the_execution(restarted, crashed_after_tool_started):
    recovered = restarted.resume(crashed_after_tool_started)
    assert_recovery_required(recovered)

    recovered.fail("gave up on the unfinished call")

    assert recovered.status is ExecutionStatus.FAILED
    # The unfinished call stays visible in the history for debugging, but a
    # deliberate FAILED decision is not "still needs recovery".
    assert restarted.resume(crashed_after_tool_started).status is ExecutionStatus.FAILED


def test_recovery_can_be_ended_by_cancelling_the_execution(restarted, crashed_after_tool_started):
    recovered = restarted.resume(crashed_after_tool_started)
    assert_recovery_required(recovered)

    recovered.mark_cancelled("operator stopped it")

    assert recovered.status is ExecutionStatus.CANCELLED
    assert restarted.resume(crashed_after_tool_started).status is ExecutionStatus.CANCELLED


def test_resolution_is_refused_for_an_unknown_call(runtime, crashed_after_tool_started):
    recovered = runtime.resume(crashed_after_tool_started)

    with pytest.raises(UnknownToolCallError):
        recovered.resolve_recovery("call_nope", "mark_completed")


def test_resolution_is_refused_for_an_unknown_action(runtime, crashed_after_tool_started):
    recovered = runtime.resume(crashed_after_tool_started)
    call_id = assert_recovery_required(recovered)

    with pytest.raises(InvalidRecoveryActionError) as excinfo:
        recovered.resolve_recovery(call_id, "retry")

    assert "retry" in str(excinfo.value)
    assert runtime.journal.get_last_sequence(crashed_after_tool_started) == 3


def test_resolution_is_refused_for_a_call_that_is_already_settled(runtime, execution):
    execution.call("add", a=2, b=3)

    with pytest.raises(InvalidRecoveryActionError) as excinfo:
        execution.resolve_recovery(execution.tool_calls[0].call_id, "mark_completed")

    assert "already COMPLETED" in str(excinfo.value)


def test_resolution_is_refused_once_the_execution_has_finished(runtime, execution):
    execution.call("add", a=2, b=3)
    call_id = execution.tool_calls[0].call_id
    execution.complete()

    with pytest.raises(InvalidStateTransitionError):
        execution.resolve_recovery(call_id, "mark_completed")


def test_resolved_state_matches_a_full_reconstruction_after_checkpointing(runtime, restarted, crashed_after_tool_started):
    recovered = runtime.resume(crashed_after_tool_started)
    call_id = assert_recovery_required(recovered)
    recovered.checkpoint()                        # snapshot the ambiguous state
    recovered.resolve_recovery(call_id, "mark_completed", result="passed")
    recovered.checkpoint()                        # and the resolved one

    again = restarted.resume(crashed_after_tool_started)

    assert again.status is ExecutionStatus.RUNNING
    assert again.state == runtime.reconstruct_state(crashed_after_tool_started)
    assert again.recovery_info().events_after_checkpoint == 0
