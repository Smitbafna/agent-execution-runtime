"""State-model tests: statuses, incomplete tool detection, serialization."""

from __future__ import annotations

import json

import pytest

from agent_runtime import (
    Event,
    EventType,
    ExecutionState,
    ExecutionStatus,
    IncompleteTool,
    StateReconstructionError,
    ToolCallStatus,
    apply_event,
    detect_incomplete_tools,
    finalize_state,
    initial_state,
    reconstruct_state,
    replace_state,
    resolve_status,
)
from agent_runtime.events import is_incomplete_event, is_terminal_event

CALL = {"call_id": "c1", "tool": "run_tests", "arguments": {"suite": "unit"}}


def event(sequence: int, event_type: str, **payload) -> Event:
    return Event.create("exec_1", sequence, event_type, payload)


# -- the five statuses ------------------------------------------------------


def test_the_execution_statuses_are_the_documented_five():
    assert {str(status) for status in ExecutionStatus} == {
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "RECOVERY_REQUIRED",
    }


@pytest.mark.parametrize(
    "event_type, expected",
    [
        ("ExecutionCompleted", ExecutionStatus.COMPLETED),
        ("ExecutionFailed", ExecutionStatus.FAILED),
        ("ExecutionCancelled", ExecutionStatus.CANCELLED),
    ],
)
def test_terminal_execution_events_set_their_status(event_type, expected):
    state = reconstruct_state([event(1, "ExecutionStarted", goal="g"), event(2, event_type)])

    assert state.status is expected
    assert state.status.is_terminal is True


def test_only_deliberate_outcomes_are_terminal():
    assert ExecutionStatus.RECOVERY_REQUIRED.is_terminal is False
    assert ExecutionStatus.RUNNING.is_terminal is False
    assert ExecutionStatus.CANCELLED.is_terminal is True


# -- incomplete tool detection ----------------------------------------------


def test_initial_state_is_running_and_empty():
    state = initial_state("exec_1")

    assert state.execution_id == "exec_1"
    assert state.status is ExecutionStatus.RUNNING
    assert state.last_sequence == 0
    assert state.tool_calls == ()
    assert state.incomplete_tools == ()


def test_a_requested_call_counts_as_incomplete():
    state = reconstruct_state(
        [event(1, "ExecutionStarted", goal="g"), event(2, "ToolRequested", **CALL)]
    )

    assert state.status is ExecutionStatus.RECOVERY_REQUIRED
    (stuck,) = state.incomplete_tools
    assert stuck.tool == "run_tests"
    assert stuck.status is ToolCallStatus.REQUESTED
    assert stuck.sequence == 2


def test_a_started_call_counts_as_incomplete():
    state = reconstruct_state(
        [
            event(1, "ExecutionStarted", goal="g"),
            event(2, "ToolRequested", **CALL),
            event(3, "ToolStarted", call_id="c1", tool="run_tests"),
        ]
    )

    (stuck,) = state.incomplete_tools
    assert stuck.status is ToolCallStatus.STARTED
    assert stuck.sequence == 3
    assert stuck.was_started is True


@pytest.mark.parametrize(
    "closing, expected",
    [
        ("ToolCompleted", ToolCallStatus.COMPLETED),
        ("ToolFailed", ToolCallStatus.FAILED),
        ("ToolCancelled", ToolCallStatus.CANCELLED),
    ],
)
def test_a_closed_call_is_not_incomplete(closing, expected):
    payload = {"call_id": "c1", "tool": "run_tests"}
    if closing != "ToolCompleted":
        payload["error"] = {"message": "nope"}
    state = reconstruct_state(
        [
            event(1, "ExecutionStarted", goal="g"),
            event(2, "ToolRequested", **CALL),
            event(3, "ToolStarted", call_id="c1", tool="run_tests"),
            event(4, closing, **payload),
        ]
    )

    assert state.status is ExecutionStatus.RUNNING
    assert state.incomplete_tools == ()
    assert state.tool_calls[0].status is expected


def test_a_finished_execution_is_not_pulled_into_recovery():
    """A deliberate FAILED stays FAILED even with an open call in the history."""
    state = reconstruct_state(
        [
            event(1, "ExecutionStarted", goal="g"),
            event(2, "ToolRequested", **CALL),
            event(3, "ExecutionFailed", error={"message": "gave up"}),
        ]
    )

    assert state.status is ExecutionStatus.FAILED
    assert len(state.incomplete_tools) == 1
    assert resolve_status(state) is ExecutionStatus.FAILED


def test_detect_incomplete_tools_can_be_used_directly():
    state = apply_event(initial_state("exec_1"), event(1, "ExecutionStarted", goal="g"))
    assert detect_incomplete_tools(state) == ()
    assert state.incomplete_tools == ()  # only finalize_state populates the field

    opened = apply_event(state, event(2, "ToolRequested", **CALL))
    (stuck,) = detect_incomplete_tools(opened)
    assert isinstance(stuck, IncompleteTool)
    assert stuck.arguments == {"suite": "unit"}


def test_finalize_state_is_idempotent():
    state = reconstruct_state(
        [event(1, "ExecutionStarted", goal="g"), event(2, "ToolRequested", **CALL)]
    )

    assert finalize_state(state) == state
    assert finalize_state(finalize_state(state)) == state


# -- sequences --------------------------------------------------------------


def test_every_phase_of_a_call_records_its_sequence():
    state = reconstruct_state(
        [
            event(1, "ExecutionStarted", goal="g"),
            event(2, "ToolRequested", **CALL),
            event(3, "ToolStarted", call_id="c1", tool="run_tests"),
            event(4, "ToolCompleted", call_id="c1", tool="run_tests", result="ok"),
        ]
    )

    call = state.tool_calls[0]
    assert (call.requested_sequence, call.started_sequence, call.completed_sequence) == (2, 3, 4)
    assert call.started_at is not None
    assert state.last_sequence == 4


def test_apply_event_tracks_the_sequence_and_is_pure():
    state = initial_state("exec_1")

    updated = apply_event(state, event(1, "ExecutionStarted", goal="g"))

    assert updated.last_sequence == 1
    assert state.last_sequence == 0
    assert state.goal is None


def test_an_event_from_another_execution_is_refused():
    with pytest.raises(StateReconstructionError):
        apply_event(initial_state("exec_1"), Event.create("exec_2", 1, "ExecutionStarted", {}))


def test_an_event_for_an_unknown_call_is_refused():
    state = apply_event(initial_state("exec_1"), event(1, "ExecutionStarted", goal="g"))

    with pytest.raises(StateReconstructionError):
        apply_event(state, event(2, "ToolCompleted", call_id="ghost", tool="t", result=1))


# -- serialization ----------------------------------------------------------


def test_state_round_trips_through_json():
    state = reconstruct_state(
        [
            event(1, "ExecutionStarted", goal="g"),
            event(2, "ToolRequested", **CALL),
            event(3, "ToolStarted", call_id="c1", tool="run_tests"),
        ]
    )

    restored = ExecutionState.from_dict(json.loads(json.dumps(state.to_dict())))

    assert restored == state
    assert restored.status is ExecutionStatus.RECOVERY_REQUIRED
    assert restored.incomplete_tools[0] == state.incomplete_tools[0]


def test_replace_state_returns_a_copy():
    state = initial_state("exec_1")
    other = replace_state(state, status=ExecutionStatus.COMPLETED)

    assert state.status is ExecutionStatus.RUNNING
    assert other.status is ExecutionStatus.COMPLETED


# -- event classification ---------------------------------------------------


@pytest.mark.parametrize("event_type", ["ToolRequested", "ToolStarted"])
def test_incomplete_events(event_type):
    assert is_incomplete_event(event_type) is True
    assert is_terminal_event(event_type) is False


@pytest.mark.parametrize(
    "event_type",
    [
        "ToolCompleted",
        "ToolFailed",
        "ToolCancelled",
        "ExecutionCompleted",
        "ExecutionFailed",
        "ExecutionCancelled",
    ],
)
def test_terminal_events(event_type):
    assert is_terminal_event(event_type) is True
    assert is_incomplete_event(event_type) is False


def test_event_types_round_trip_from_their_wire_values():
    assert EventType("ExecutionCancelled") is EventType.EXECUTION_CANCELLED
    assert EventType("ToolCancelled") is EventType.TOOL_CANCELLED
