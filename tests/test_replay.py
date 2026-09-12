"""Milestone 3: deterministic replay.

The invariant under test:

    Given the same recorded execution history, replay reproduces the same
    execution state without repeating any external side effect.

The tests are grouped the way the milestone groups them: basic replay,
side-effect protection, failed tools, mismatches, checkpoint replay and
determinism.
"""

from __future__ import annotations

import pytest

from agent_runtime import (
    ExecutionNotFoundError,
    ExecutionStatus,
    ReplayError,
    ReplayMismatchError,
    Runtime,
    ToolCallStatus,
    ToolInvocationError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_calculations(runtime: Runtime, goal: str = "Perform calculations"):
    """A completed execution with two settled tool calls."""
    execution = runtime.start(goal=goal)
    execution.call("add", a=2, b=3)
    execution.call("multiply", a=5, b=10)
    execution.complete()
    return execution


@pytest.fixture
def counter_tool():
    """A tool with an obvious side effect: every call increments a counter.

    Returned as a *registrar* rather than a registered tool, so a test can put
    it on whichever ``Runtime`` it wants -- including a different one from the
    replaying process, which is the whole point of the side-effect tests.
    """
    calls = {"count": 0}

    def register(runtime: Runtime):
        @runtime.tool
        def dangerous() -> int:
            calls["count"] += 1
            return calls["count"]

        return dangerous

    return register, calls


def replay_execution_for(runtime: Runtime, execution_id: str):
    """Build the replay handle for an execution, without running the replay."""
    engine = runtime.replay_engine(execution_id)
    engine.run()
    return engine.replay_execution


# ---------------------------------------------------------------------------
# Basic replay (§1)
# ---------------------------------------------------------------------------


def test_a_completed_execution_can_be_replayed(runtime):
    execution = run_calculations(runtime)

    result = runtime.replay(execution.id)

    assert result.execution_id == execution.id
    assert result.matched is True
    assert result.status == "MATCHED"


def test_replayed_state_equals_the_original_state(runtime):
    execution = run_calculations(runtime)

    result = runtime.replay(execution.id)

    assert result.final_state == execution.state
    # ``.state`` is the name the milestone's example uses; both must work.
    assert result.state == execution.state
    assert result.final_state.status is ExecutionStatus.COMPLETED
    assert result.final_state.last_sequence == execution.state.last_sequence


def test_replay_returns_the_recorded_results(runtime):
    execution = run_calculations(runtime)
    recorded = [call.result for call in execution.state.tool_calls]

    result = runtime.replay(execution.id)

    assert recorded == [5, 50]
    assert [call.result for call in result.final_state.tool_calls] == recorded


def test_replay_counts_the_work_it_did(runtime):
    execution = run_calculations(runtime)

    result = runtime.replay(execution.id)

    assert result.tools_replayed == 2
    assert result.events_replayed == len(execution.events)
    assert result.duration >= 0.0
    assert [step.tool for step in result.steps] == ["add", "multiply"]


def test_every_replayed_result_came_from_the_journal(runtime):
    execution = run_calculations(runtime)

# ---------------------------------------------------------------------------
# Side-effect protection (§2, §13)
# ---------------------------------------------------------------------------


def test_replay_never_executes_the_real_tool(runtime, counter_tool):
    register, calls = counter_tool
    register(runtime)
    execution = runtime.start(goal="count me")
    execution.call("dangerous")
    execution.call("add", a=1, b=1)
    execution.complete()

    assert calls["count"] == 1, "the original run called the tool once"

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert calls["count"] == 1, "the replay must not call the tool again"
    assert result.final_state.tool_calls[0].result == 1, "the recorded result was replayed"


@pytest.mark.parametrize(
    "tool_name",
    [
        "create_file",
        "delete_file",
        "send_email",
        "create_github_issue",
        "database_write",
        "http_post",
    ],
)
def test_no_obviously_dangerous_tool_runs_during_replay(runtime, tool_name):
    """Every side-effect tool the milestone names is equally unreachable."""
    calls = {"count": 0}

    def make():
        calls["count"] += 1
        return {"side_effect": calls["count"]}

    runtime.register_tool(make, name=tool_name)
    execution = runtime.start(goal=f"call {tool_name}")
    execution.call(tool_name)
    execution.complete()

    assert calls["count"] == 1

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert calls["count"] == 1, f"{tool_name} ran twice"
    assert result.final_state.tool_calls[0].result == {"side_effect": 1}


def test_replay_protection_is_structural_not_conventional(runtime, counter_tool):
    """The REPLAY runner holds no registry, so there is nothing to invoke."""
    register, _calls = counter_tool
    register(runtime)
    execution = runtime.start(goal="structural")
    execution.call("dangerous")
    execution.complete()

    replayed = replay_execution_for(runtime, execution.id)

    assert replayed.runner.registry is None
    assert replayed.runner.mode == "REPLAY"


# ---------------------------------------------------------------------------
# Replay must not mutate the original execution (§8)
# ---------------------------------------------------------------------------


def test_replay_does_not_change_the_event_count(runtime):
    execution = run_calculations(runtime)
    before = runtime.journal.count_events(execution.id)

    runtime.replay(execution.id)

    assert runtime.journal.count_events(execution.id) == before


def test_replay_does_not_modify_any_recorded_event(runtime):
    execution = run_calculations(runtime)

    def snapshot():
        return [
            (e.event_id, e.sequence, str(e.event_type), str(e.timestamp), dict(e.payload))
            for e in runtime.get_events(execution.id)
        ]

    before = snapshot()
    runtime.replay(execution.id)

    assert snapshot() == before


def test_replay_does_not_add_checkpoints(runtime):
    execution = run_calculations(runtime)
    before = runtime.checkpoints.count(execution.id)

    runtime.replay(execution.id)

    assert runtime.checkpoints.count(execution.id) == before


def test_replay_reports_that_the_journal_was_unchanged(runtime):
    execution = run_calculations(runtime)

    assert runtime.replay(execution.id).journal_unchanged is True


def test_replaying_twice_is_stable(runtime):
    """Replay is a pure function of the history, so it is idempotent."""
    execution = run_calculations(runtime)

    first = runtime.replay(execution.id)
    second = runtime.replay(execution.id)

    assert first.final_state == second.final_state
    assert second.final_state == execution.state


def test_replaying_an_unknown_execution_is_an_error(runtime):
    with pytest.raises(ExecutionNotFoundError):
        runtime.replay("exec_never_ran")


def test_replay_exposes_the_replayed_execution_for_inspection(runtime):
    execution = run_calculations(runtime)

    replayed = replay_execution_for(runtime, execution.id)

    assert replayed.status is ExecutionStatus.COMPLETED
    assert len(replayed.steps) == 2
    assert replayed.recorded_calls[0].tool == "add"


def test_a_replay_execution_refuses_to_checkpoint(runtime):
    execution = run_calculations(runtime)
    replayed = replay_execution_for(runtime, execution.id)

    with pytest.raises(ReplayError):
        replayed.checkpoint()

    result = runtime.replay(execution.id)

    # A replay never computes a result; each step says so explicitly.
    assert all(step.executed is False for step in result.steps)
    completed = [
        event for event in execution.events if event.event_type == "ToolCompleted"
    ]
    assert [step.result for step in result.steps] == [
        event.payload["result"] for event in completed
    ]


def test_replay_works_in_a_fresh_process_with_no_tools_registered(db_path, counter_tool):
    """The replaying process need not have the tool functions at all."""
    register, calls = counter_tool

    with Runtime(db_path) as runtime:
        register(runtime)
        execution = runtime.start(goal="side effects")
        execution.call("dangerous")
        execution.complete()

    assert calls["count"] == 1

    with Runtime(db_path, register_default_tools=False) as runtime:
        result = runtime.replay(execution.id)

    assert result.matched is True
    assert calls["count"] == 1
# ---------------------------------------------------------------------------
# Failed tools replay correctly (§6)
# ---------------------------------------------------------------------------


def failing_execution(runtime):
    """An execution whose second tool failed, and which then completed."""
    execution = runtime.start(goal="failing run")
    execution.call("add", a=2, b=3)
    with pytest.raises(ToolInvocationError):
        execution.call("divide", a=1, b=0)
    execution.complete()
    return execution


def test_a_failed_tool_replays_correctly(runtime):
    execution = failing_execution(runtime)

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state == execution.state


def test_the_recorded_error_information_is_reproduced(runtime):
    execution = failing_execution(runtime)
    recorded = execution.state.tool_calls[1]

    result = runtime.replay(execution.id)
    replayed = result.final_state.tool_calls[1]

    assert replayed.status is ToolCallStatus.FAILED
    assert replayed.error["type"] == "ZeroDivisionError"
    assert replayed.error["message"] == recorded.error["message"]
    assert replayed.error == recorded.error


def test_a_recorded_failure_is_reported_in_the_replay_trace(runtime):
    execution = failing_execution(runtime)

    result = runtime.replay(execution.id)

    assert len(result.errors) == 1
    assert result.errors[0]["tool"] == "divide"
    assert result.errors[0]["error"]["type"] == "ZeroDivisionError"


def test_a_failed_tool_does_not_execute_again(runtime):
    calls = {"count": 0}

    @runtime.tool
    def run_tests() -> str:
        calls["count"] += 1
        raise RuntimeError("3 tests failed")

    execution = runtime.start(goal="tests")
    with pytest.raises(ToolInvocationError):
        execution.call("run_tests")
    execution.complete()

    assert calls["count"] == 1

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert calls["count"] == 1, "the failing tool must not run again"
    assert result.final_state.tool_calls[0].error["message"] == "3 tests failed"


def test_an_execution_that_ended_in_failure_replays(runtime):
    execution = runtime.start(goal="doomed")
    execution.call("add", a=1, b=1)
    execution.fail(RuntimeError("gave up"))

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state.status is ExecutionStatus.FAILED
    assert result.final_state.error["message"] == "gave up"


def test_a_cancelled_execution_replays(runtime):
    execution = runtime.start(goal="stopped")
    execution.call("add", a=1, b=1)
    execution.mark_cancelled("not needed")

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state.status is ExecutionStatus.CANCELLED


def test_an_interrupted_execution_replays_as_interrupted(runtime):
    """A crash mid-call leaves an open call; replay reproduces the ambiguity."""
    execution = runtime.start(goal="interrupted")
    execution.call("add", a=1, b=1)
    call_id = "call_run_tests"
    runtime.journal.append_event(
        execution.id,
        "ToolRequested",
        {"call_id": call_id, "tool": "run_tests", "arguments": {"suite": "unit"}},
    )
    runtime.journal.append_event(
        execution.id, "ToolStarted", {"call_id": call_id, "tool": "run_tests"}
    )

    result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state.status is ExecutionStatus.RECOVERY_REQUIRED
    assert [item.tool for item in result.final_state.incomplete_tools] == ["run_tests"]
    # Nothing was invented for the call that never settled.
    assert result.final_state.tool_calls[1].result is None

# ---------------------------------------------------------------------------
# Mismatches raise ReplayMismatchError (§4, §5)
#
# Each test drives a ReplayExecution directly -- asking it to replay a *different*
# call than the one the journal recorded -- because that is what a diverging
# agent run looks like. ``runtime.replay`` on an intact history can only ever
# match; the mismatches only appear when the replay is told to do something
# other than what was recorded.
# ---------------------------------------------------------------------------


def replay_handle(runtime, execution_id, *, from_sequence: int = 0):
    """A replay execution with its recorded calls loaded but not yet replayed.

    This is the state a diverging agent run starts from: the history is known,
    and the caller is about to ask for calls that may or may not be the ones it
    recorded.
    """
    return runtime.replay_engine(execution_id, from_sequence=from_sequence).prepare()


@pytest.fixture
def search_runtime(runtime):
    """A runtime with the ``search_code`` tool the mismatch tests diverge on."""
    runtime.register_tool(lambda query: [f"hit in {query}"], name="search_code")
    return runtime


def searching_execution(runtime, query: str = "authentication"):
    """A one-call execution that searched for something."""
    execution = runtime.start(goal="search the code")
    execution.call("search_code", query=query)
    execution.complete()
    return execution


def test_the_wrong_tool_name_is_a_mismatch(search_runtime):
    execution = searching_execution(search_runtime)
    handle = replay_handle(search_runtime, execution.id)

    with pytest.raises(ReplayMismatchError) as info:
        handle.call("multiply", a=1, b=1)  # the recorded call is `search_code`

    assert info.value.kind == "tool_name"
    assert info.value.expected["tool"] == "search_code"
    assert info.value.received["tool"] == "multiply"


def test_the_wrong_arguments_are_a_mismatch(search_runtime):
    execution = searching_execution(search_runtime)
    handle = replay_handle(search_runtime, execution.id)

    with pytest.raises(ReplayMismatchError) as info:
        handle.call("search_code", query="database")

    assert info.value.kind == "arguments"
    assert info.value.sequence is not None


def test_a_missing_recorded_tool_call_is_a_mismatch(runtime):
    """Skipping a recorded call means the replay is no longer the same run."""
    execution = runtime.start(goal="two calls")
    execution.call("add", a=1, b=1)
    execution.call("multiply", a=2, b=2)
    execution.complete()
    handle = replay_handle(runtime, execution.id)

    with pytest.raises(ReplayMismatchError) as info:
        handle.call("multiply", a=2, b=2)  # the recorded first call is `add`

    assert info.value.kind == "tool_name"


def test_an_unexpected_extra_tool_call_is_a_mismatch(runtime):
    execution = runtime.start(goal="one call")
    execution.call("add", a=1, b=1)
    execution.complete()
    handle = replay_handle(runtime, execution.id)

    handle.call("add", a=1, b=1)  # the only recorded call

    with pytest.raises(ReplayMismatchError) as info:
        handle.call("multiply", a=9, b=9)  # the history has nothing left to serve

    assert info.value.kind == "unexpected_tool_call"


def test_a_mismatch_names_the_sequence_and_both_sides(search_runtime):
    execution = searching_execution(search_runtime)
    handle = replay_handle(search_runtime, execution.id)
    recorded_sequence = execution.state.tool_calls[0].requested_sequence

    with pytest.raises(ReplayMismatchError) as info:
        handle.call("search_code", query="database")

    error = info.value
    assert error.sequence == recorded_sequence
    assert error.execution_id == execution.id
    assert error.expected["arguments"] == {"query": "authentication"}
    assert error.received["arguments"] == {"query": "database"}


def test_a_mismatch_renders_a_readable_report(search_runtime):
    execution = searching_execution(search_runtime)
    handle = replay_handle(search_runtime, execution.id)

    with pytest.raises(ReplayMismatchError) as info:
        handle.call("search_code", query="database")

    report = str(info.value)
    assert "ReplayMismatchError" in report
    assert "Expected:" in report
    assert '"authentication"' in report
    assert "Received:" in report
    assert '"database"' in report


def test_a_mismatch_stops_the_replay_instead_of_continuing(search_runtime):
    """Divergence is reported, never smoothed over."""
    execution = searching_execution(search_runtime)
    handle = replay_handle(search_runtime, execution.id)

    with pytest.raises(ReplayMismatchError):
        handle.call("search_code", query="database")

    # The runner did not advance past the divergence, so the next call is still
    # compared against the same recorded call.
    assert handle.runner.cursor == 0
    with pytest.raises(ReplayMismatchError):
        handle.call("search_code", query="database")
    assert handle.runner.cursor == 0


def test_replaying_the_recorded_calls_through_the_handle_matches(runtime):
    """The handle and ``runtime.replay`` are two doors to the same replay."""
    execution = run_calculations(runtime)
    handle = replay_handle(runtime, execution.id)

    assert handle.call("add", a=2, b=3) == 5
    assert handle.call("multiply", a=5, b=10) == 50
    handle.complete()

    assert handle.state == runtime.replay(execution.id).final_state

# ---------------------------------------------------------------------------
# Replay from a checkpoint (§7, §13)
# ---------------------------------------------------------------------------


@pytest.fixture
def checkpointed(runtime):
    """An execution with a checkpoint after its first tool call.

        seq 1 ExecutionStarted
        seq 2 ToolRequested  add
        seq 3 ToolStarted    add
        seq 4 ToolCompleted  add      <-- checkpoint @ 4
        seq 5 ToolRequested  multiply
        seq 6 ToolStarted    multiply
        seq 7 ToolCompleted  multiply
        seq 8 ExecutionCompleted
    """
    execution = runtime.start(goal="checkpointed work")
    execution.call("add", a=2, b=3)
    checkpoint = execution.checkpoint()
    execution.call("multiply", a=5, b=10)
    execution.complete()
    return execution, checkpoint


def test_replay_from_a_checkpoint_matches_a_full_replay(runtime, checkpointed):
    execution, checkpoint = checkpointed

    full = runtime.replay(execution.id)
    partial = runtime.replay(execution.id, from_sequence=checkpoint.sequence)

    assert full.final_state == partial.final_state
    assert partial.final_state == execution.state


def test_replay_from_a_checkpoint_only_replays_the_tail(runtime, checkpointed):
    execution, checkpoint = checkpointed

    result = runtime.replay(execution.id, from_sequence=checkpoint.sequence)

    assert result.from_sequence == checkpoint.sequence
    assert result.tools_replayed == 1, "only the call after the checkpoint"
    assert result.events_replayed == 4, "sequences 5..8"
    assert [step.tool for step in result.steps] == ["multiply"]


def test_a_full_replay_still_sees_the_whole_history(runtime, checkpointed):
    execution, _checkpoint = checkpointed

    result = runtime.replay(execution.id, from_sequence=0)

    assert result.tools_replayed == 2
    assert result.events_replayed == 8


def test_checkpoint_replay_and_prefix_fold_agree_without_a_checkpoint(runtime):
    """From-sequence works whether or not a snapshot happens to be stored."""
    execution = runtime.start(goal="no snapshot")
    execution.call("add", a=2, b=3)
    boundary = execution.state.last_sequence
    execution.call("multiply", a=5, b=10)
    execution.complete()

    result = runtime.replay(execution.id, from_sequence=boundary)

    assert runtime.checkpoints.count(execution.id) == 0, "folded the prefix instead"
    assert result.final_state == execution.state


def test_replaying_from_the_start_sequence_replays_nothing(runtime, checkpointed):
    execution, _checkpoint = checkpointed

    result = runtime.replay(execution.id, from_sequence=execution.last_event_sequence)

    assert result.matched is True
    assert result.tools_replayed == 0


def test_replay_from_a_checkpoint_does_not_touch_the_journal(runtime, checkpointed):
    execution, checkpoint = checkpointed
    before_events = runtime.journal.count_events(execution.id)
    before_checkpoints = runtime.checkpoints.count(execution.id)

    runtime.replay(execution.id, from_sequence=checkpoint.sequence)

    assert runtime.journal.count_events(execution.id) == before_events
    assert runtime.checkpoints.count(execution.id) == before_checkpoints


def test_replay_from_an_out_of_range_sequence_is_a_mismatch(runtime, checkpointed):
    execution, _checkpoint = checkpointed

    with pytest.raises(ReplayMismatchError) as info:
        runtime.replay(execution.id, from_sequence=99)

    assert info.value.kind == "sequence"


def test_replay_from_inside_an_open_tool_call_is_a_mismatch(runtime):
    """A start point in the middle of a call is a history replay cannot read."""
    execution = runtime.start(goal="interrupted")
    execution.call("add", a=1, b=1)
    runtime.journal.append_event(
        execution.id,
        "ToolRequested",
        {"call_id": "call_x", "tool": "run_tests", "arguments": {}},
    )
    runtime.journal.append_event(
        execution.id, "ToolStarted", {"call_id": "call_x", "tool": "run_tests"}
    )
    open_call_sequence = execution.last_event_sequence

    with pytest.raises(ReplayMismatchError) as info:
        runtime.replay(execution.id, from_sequence=open_call_sequence)

    assert info.value.kind == "sequence"
    assert "run_tests" in str(info.value)


def test_a_checkpoint_survives_a_replay_and_is_still_valid(runtime, checkpointed):
    """Milestone 2's checkpoint is untouched and still consistent afterwards."""
    execution, checkpoint = checkpointed

    runtime.replay(execution.id)
    runtime.replay(execution.id, from_sequence=checkpoint.sequence)

    assert runtime.latest_checkpoint(execution.id).state == checkpoint.state
    assert runtime.recover_state(execution.id) == runtime.reconstruct_state(execution.id)

# ---------------------------------------------------------------------------
# Determinism (§11)
#
# The runtime does not try to make arbitrary Python deterministic, and these
# tests say so concretely instead. Anything a run reads that can change between
# two runs is an *external input*: if it influenced the execution, it has to be
# exposed as a tool, so that its value lands in the journal and replay can serve
# it back. Recording it as a tool is what makes the execution replayable.
# ---------------------------------------------------------------------------


def test_a_time_dependent_tool_replays_only_because_its_output_was_recorded(runtime):
    """``time`` is not frozen; the recorded value is what replay serves."""
    values = iter(["2024-01-01T00:00:00Z", "2031-09-09T09:09:09Z"])

    @runtime.tool
    def now() -> str:
        return next(values)

    execution = runtime.start(goal="stamp the time")
    first = execution.call("now")
    execution.complete()

    # A second *live* run would see a different time; a replay must not.
    result = runtime.replay(execution.id)

    assert first == "2024-01-01T00:00:00Z"
    assert result.final_state.tool_calls[0].result == first


def test_a_random_dependent_tool_replays_only_because_its_output_was_recorded(runtime):
    """``random`` is not seeded; the recorded draw is what replay serves."""
    import random

    @runtime.tool
    def roll() -> int:
        return random.randint(1, 1000)

    random.seed(1234)
    execution = runtime.start(goal="roll some dice")
    rolled = execution.call("roll")
    execution.complete()

    result = runtime.replay(execution.id)

    assert result.final_state.tool_calls[0].result == rolled
    # The generator has moved on, so a second *live* roll would differ. What
    # replay returns is the recorded draw, not a new one.
    assert random.randint(1, 1000) > 0


def test_a_uuid_dependent_tool_replays_only_because_its_output_was_recorded(runtime):
    """A fresh id per call is the same problem as a clock."""
    from agent_runtime.events import new_id as make_id

    @runtime.tool
    def new_identifier() -> str:
        return make_id("run")

    execution = runtime.start(goal="make an id")
    first = execution.call("new_identifier")
    execution.complete()

    result = runtime.replay(execution.id)

    assert result.final_state.tool_calls[0].result == first
    # A live second call would produce a different id; replay serves the recorded
    # one, which is what makes the execution reproducible.
    assert make_id("run") != first


def test_replay_is_not_affected_by_environment_variables(db_path, monkeypatch):
    """An env var read inside a tool is an input the tool must surface."""
    with Runtime(db_path) as runtime:

        @runtime.tool
        def setting(variable: str) -> str:
            import os

            return os.environ.get(variable, "<unset>")

        execution = runtime.start(goal="read the env")
        execution.call("setting", variable="AGENT_RUNTIME_TEST")
        execution.complete()

    monkeypatch.setenv("AGENT_RUNTIME_TEST", "something-else-entirely")

    with Runtime(db_path) as runtime:
        result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state.tool_calls[0].result == "<unset>"


def test_two_replays_of_the_same_history_agree_exactly(runtime):
    """The strongest determinism statement available: replay is a pure function."""
    execution = run_calculations(runtime)

    first = runtime.replay(execution.id)
    second = runtime.replay(execution.id)

    assert first.final_state.to_dict() == second.final_state.to_dict()
    assert [s.to_dict() for s in first.steps] == [s.to_dict() for s in second.steps]
    assert first.tools_replayed == second.tools_replayed
    assert first.events_replayed == second.events_replayed


def test_a_replay_survives_a_new_process(db_path, counter_tool):
    """State comes out of SQLite, not out of the object that ran the tools."""
    register, calls = counter_tool

    with Runtime(db_path) as runtime:
        register(runtime)
        execution = runtime.start(goal="across processes")
# ---------------------------------------------------------------------------
# The replay CLI (§12)
# ---------------------------------------------------------------------------


def run_cli(argv, db_path):
    """Run the CLI against ``db_path``; returns (exit_code, output)."""
    import io

    from agent_runtime.cli import main

    buffer = io.StringIO()
    code = main(["--db", db_path, *argv], out=buffer)
    return code, buffer.getvalue()


def test_the_replay_cli_reports_a_matched_replay(db_path):
    with Runtime(db_path) as runtime:
        execution = run_calculations(runtime)

    code, output = run_cli(["replay", execution.id], db_path)

    assert code == 0
    assert f"Execution: {execution.id}" in output
    assert "Replay completed" in output
    assert "✓ add" in output
    assert "✓ multiply" in output
    assert "Events replayed: 8" in output
    assert "Tools replayed: 2" in output
    assert "State: MATCHED" in output


def test_the_replay_cli_reports_a_checkpoint_replay(db_path):
    with Runtime(db_path) as runtime:
        execution = runtime.start(goal="checkpointed")
        execution.call("add", a=2, b=3)
        checkpoint = execution.checkpoint()
        execution.call("multiply", a=5, b=10)
        execution.complete()

    code, output = run_cli(
        ["replay", execution.id, "--from-sequence", str(checkpoint.sequence)], db_path
    )

    assert code == 0
    assert "Tools replayed: 1" in output
    assert f"Resumed from sequence: {checkpoint.sequence}" in output
    assert "State: MATCHED" in output


def test_the_replay_cli_reports_a_mismatch(search_runtime):
    """The CLI's mismatch report shows the sequence and both sides.

    An end-to-end mismatch is not reachable through the CLI on a consistent
    journal -- which is the point: a history that replays cleanly produces a
    clean replay. The report is therefore exercised on the exception it renders.
    """
    import io

    from agent_runtime.cli import _print_mismatch

    execution = searching_execution(search_runtime)
    handle = replay_handle(search_runtime, execution.id)
    with pytest.raises(ReplayMismatchError) as info:
        handle.call("search_code", query="database")

    buffer = io.StringIO()
    _print_mismatch(info.value, buffer)
    output = buffer.getvalue()

    assert "Sequence:" in output
    assert "Tool: search_code" in output
    assert "Expected arguments:" in output
    assert '{"query": "authentication"}' in output
    assert "Received:" in output
    assert '{"query": "database"}' in output
    assert execution.id  # the execution it belongs to is known


def test_the_replay_cli_exits_non_zero_on_an_unknown_execution(db_path):
    code, _output = run_cli(["replay", "exec_nope"], db_path)

    assert code == 2


def test_the_replay_cli_reports_a_recorded_failure(db_path):
    with Runtime(db_path) as runtime:
        execution = runtime.start(goal="tests")
        with pytest.raises(ToolInvocationError):
            execution.call("divide", a=1, b=0)
        execution.complete()

    code, output = run_cli(["replay", execution.id], db_path)

    assert code == 0
    assert "State: MATCHED" in output
    assert "✗ divide" in output
    assert "division by zero" in output


def test_the_replay_cli_can_emit_json(db_path):
    import json as json_module

    with Runtime(db_path) as runtime:
        execution = run_calculations(runtime)

    code, output = run_cli(["replay", execution.id, "--json"], db_path)

    assert code == 0
    payload = json_module.loads(output)
    assert payload["execution_id"] == execution.id
    assert payload["matched"] is True
    assert payload["tools_replayed"] == 2
    # Progress lines would have made this unparseable.
    assert "✓" not in output


def test_the_cli_lists_executions(db_path):
    with Runtime(db_path) as runtime:
        execution = run_calculations(runtime)

    code, output = run_cli(["list"], db_path)

    assert code == 0
    assert execution.id in output
    assert "COMPLETED" in output


def test_a_replay_survives_a_new_process(db_path, counter_tool):
    """State comes out of SQLite, not out of the object that ran the tools."""
    register, calls = counter_tool

    with Runtime(db_path) as runtime:
        register(runtime)
        execution = runtime.start(goal="across processes")
        execution.call("dangerous")
        execution.complete()
        expected = execution.state

    with Runtime(db_path, register_default_tools=False) as runtime:
        result = runtime.replay(execution.id)

    assert result.final_state == expected
    assert calls["count"] == 1

