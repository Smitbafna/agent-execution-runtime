"""Milestone 4A, part 4: replaying a run that retried.

The invariant under test:

    Replay substitutes *recorded* attempts. A run that failed twice and then
    succeeded replays the two failures and the success without executing the
    tool once, and lands on exactly the state the original reached -- including
    the attempt numbers and the backoff schedule.
"""

from __future__ import annotations

import pytest

from agent_runtime import (
    ExecutionStatus,
    RecordingSleeper,
    RetryPolicy,
    RetryableToolError,
    Runtime,
    ToolCallStatus,
    ToolInvocationError,
)

POLICY = RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=2.0, max_delay=10.0)


@pytest.fixture
def sleeper() -> RecordingSleeper:
    return RecordingSleeper()


def flaky_runtime(db_path, counter, *, failures: int, sleeper=None, retry_policy=POLICY):
    """A runtime with a tool that fails ``failures`` times, then succeeds."""
    runtime = Runtime(db_path, sleeper=sleeper or RecordingSleeper())

    @runtime.tool(name="fetch_data", retry_policy=retry_policy)
    def fetch_data(url: str) -> dict:
        counter["calls"] += 1
        if counter["calls"] <= failures:
            raise RetryableToolError(f"attempt {counter['calls']} failed")
        return {"url": url, "attempt": counter["calls"]}

    return runtime


def retried_execution(runtime, failures: int = 2):
    """One logical call that fails ``failures`` times and then succeeds."""
    execution = runtime.start(goal="Fetch data that is flaky")
    result = execution.call("fetch_data", url="https://example.com")
    execution.complete()
    return execution, result


def event_shape(events):
    """The (type, attempt, failed_attempt) triples of an event stream."""
    return [
        (str(event.event_type), event.payload.get("attempt"), event.payload.get("failed_attempt"))
        for event in events
    ]


# ---------------------------------------------------------------------------
# Replaying a retry sequence (§11)
# ---------------------------------------------------------------------------


def test_a_retried_call_replays_correctly(db_path):
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, result = retried_execution(runtime)
        original_state = execution.state

        replayed = runtime.replay(execution.id)

    assert replayed.matched is True
    assert replayed.final_state == original_state
    assert replayed.final_state.status is ExecutionStatus.COMPLETED
    # The recorded result came back, attempts and all.
    assert result == {"url": "https://example.com", "attempt": 3}
    assert replayed.final_state.tool_calls[0].result == result


def test_the_replayed_state_has_the_same_attempts(db_path):
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, _ = retried_execution(runtime)
        original_call = execution.state.tool_calls[0]

        replayed = runtime.replay(execution.id)

    replayed_call = replayed.final_state.tool_calls[0]
    assert replayed_call.call_id == original_call.call_id
    assert replayed_call.attempt == 3
    assert [a.attempt for a in replayed_call.attempts] == [1, 2, 3]
    assert [a.status for a in replayed_call.attempts] == [
        ToolCallStatus.FAILED,
        ToolCallStatus.FAILED,
        ToolCallStatus.COMPLETED,
    ]
    assert replayed_call.attempts[0].error == original_call.attempts[0].error
    assert replayed_call.attempts[0].scheduled_retry.attempt == 2
    assert replayed_call.attempts[1].scheduled_retry.delay == pytest.approx(0.2)


def test_the_replay_reproduces_the_whole_event_sequence(db_path):
    """Same events, same order, same attempt numbers -- produced, not copied."""
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, _ = retried_execution(runtime)
        original = event_shape(execution.events)

        engine = runtime.replay_engine(execution.id)
        result = engine.run()
        replayed_events = event_shape(engine.replay_execution.events)

    assert original == replayed_events
    assert result.events_replayed == len(original)
    assert original == [
        ("ExecutionStarted", None, None),
        ("ToolRequested", None, None),
        ("ToolStarted", 1, None),
        ("ToolFailed", 1, None),
        ("ToolRetryScheduled", 2, 1),
        ("ToolStarted", 2, None),
        ("ToolFailed", 2, None),
        ("ToolRetryScheduled", 3, 2),
        ("ToolStarted", 3, None),
        ("ToolCompleted", 3, None),
        ("ExecutionCompleted", None, None),
    ]


def test_no_tool_runs_during_a_retry_replay(db_path):
    """The counter is the proof: the real function was called three times total."""
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, _ = retried_execution(runtime)

        assert counter["calls"] == 3, "the original run made three attempts"

        replayed = runtime.replay(execution.id)

        assert counter["calls"] == 3, "the replay must not call the tool again"

    assert all(step.executed is False for step in replayed.steps)
    assert replayed.matched is True


def test_a_replay_needs_no_tools_registered_at_all(db_path):
    """A different process, with the tool function not even registered."""
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, _ = retried_execution(runtime)
        execution_id = execution.id

    with Runtime(db_path, register_default_tools=False) as replaying:
        result = replaying.replay(execution_id)

    assert result.matched is True
    assert result.final_state.tool_calls[0].attempt == 3
    assert counter["calls"] == 3


def test_the_replay_counts_the_retries_it_reproduced(db_path):
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, _ = retried_execution(runtime)

        result = runtime.replay(execution.id)

    assert result.retries_replayed == 2
    assert result.delays_replayed == (0.1, 0.2)
    assert result.tools_replayed == 1, "one logical call, three attempts"


def test_a_replay_reproduces_the_backoff_without_spending_it(db_path):
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, _ = retried_execution(runtime)
        execution_id = execution.id

    replay_sleeper = RecordingSleeper()
    with Runtime(db_path, sleeper=replay_sleeper, register_default_tools=False) as replaying:
        result = replaying.replay(execution_id)

    assert result.delays_replayed == (0.1, 0.2)
    assert replay_sleeper.delays == (), "a replay waits through its own sleeper"


def test_the_trace_marks_a_retried_call(db_path):
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution, _ = retried_execution(runtime)

        result = runtime.replay(execution.id)

    (step,) = result.steps
    assert step.attempt == 3
    assert step.attempts == 3
    assert step.retried is True
    assert "attempt 3/3" in str(step)


def test_a_replay_does_not_invent_retries_for_a_call_that_never_retried(db_path):
    """A recorded failure stays a failure, whatever policy this process holds."""
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=1, retry_policy=None) as runtime:
        execution = runtime.start(goal="one attempt, no policy")
        with pytest.raises(ToolInvocationError):
            execution.call("fetch_data", url="https://example.com")
        execution.complete()
        execution_id = execution.id

    # The replaying process is full of retryable errors and willing policies.
    with Runtime(db_path, register_default_tools=False) as replaying:
        result = replaying.replay(execution_id)

    assert result.matched is True
    assert result.retries_replayed == 0
    assert result.delays_replayed == ()
    assert counter["calls"] == 1


def test_a_call_that_exhausted_its_attempts_replays_as_failed(db_path):
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=99) as runtime:
        execution = runtime.start(goal="doomed")
        with pytest.raises(ToolInvocationError):
            execution.call("fetch_data", url="https://example.com")
        execution.fail("gave up")
        execution_id = execution.id

        result = runtime.replay(execution_id)

    assert result.matched is True
    assert result.retries_replayed == 2
    call = result.final_state.tool_calls[0]
    assert call.status is ToolCallStatus.FAILED
    assert call.attempt == 3
    assert len(result.errors) == 1
    assert result.errors[0]["attempts"] == 3
    assert result.errors[0]["error"]["type"] == "RetryableToolError"


def test_a_replay_from_a_checkpoint_agrees_with_a_full_replay(db_path):
    counter = {"calls": 0}
    with flaky_runtime(db_path, counter, failures=2) as runtime:
        execution = runtime.start(goal="Fetch data that is flaky")
        execution.call("fetch_data", url="https://example.com/first")
        checkpoint = execution.checkpoint()
        execution.call("fetch_data", url="https://example.com/second")
        execution.complete()
        execution_id = execution.id

        full = runtime.replay(execution_id)
        partial = runtime.replay(execution_id, from_sequence=checkpoint.sequence)

    assert full.matched is True
    assert full.retries_replayed == 2
    assert partial.matched is True
    assert partial.final_state == full.final_state
    # The two retries this run made happened before the checkpoint, so the
    # partial replay starts after them -- and still agrees on the final state.
    assert partial.retries_replayed == 0


# ---------------------------------------------------------------------------
# Interrupted retries (§11)
# ---------------------------------------------------------------------------


class ProcessDied(BaseException):
    """What a crash looks like to a tool: uncatchable, so nothing is journalled."""


def interrupted_execution(runtime):
    """A call that failed once, retried, and died *during* attempt 2."""
    calls = {"n": 0}

    @runtime.tool(name="fetch_data", retry_policy=POLICY)
    def fetch_data(url: str) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RetryableToolError("upstream unavailable")
        raise ProcessDied("the process died mid-attempt")

    execution = runtime.start(goal="Fetch data across a crash")
    with pytest.raises(ProcessDied):
        execution.call("fetch_data", url="https://example.com")
    return execution


def test_a_crash_during_a_retried_attempt_replays_as_interrupted(db_path):
    """The earlier failure and its retry must survive into the replayed state."""
    with Runtime(db_path, sleeper=RecordingSleeper()) as runtime:
        execution = interrupted_execution(runtime)
        original_state = execution.state

        result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state == original_state
    assert result.final_state.status is ExecutionStatus.RECOVERY_REQUIRED
    call = result.final_state.tool_calls[0]
    assert call.attempt == 2
    assert [a.status for a in call.attempts] == [ToolCallStatus.FAILED, ToolCallStatus.STARTED]
    # The retry between them was reproduced rather than dropped.
    assert result.retries_replayed == 1
    assert call.attempts[0].scheduled_retry.attempt == 2
    # ... and nothing was invented for the attempt that never finished.
    assert call.attempts[1].error is None
    assert result.final_state.tool_calls[0].pending_retry is None


def test_a_crash_during_the_backoff_replays_with_its_pending_retry(db_path):
    """A scheduled retry the process never carried out is replayed as scheduled."""
    with Runtime(db_path, sleeper=RecordingSleeper()) as runtime:
        execution = runtime.start(goal="died during the backoff")
        error = {"type": "RetryableToolError", "message": "upstream down"}

        # The exact shape a crash mid-backoff leaves behind.
        runtime.journal.append_event(
            execution.id,
            "ToolRequested",
            {"call_id": "call_x", "tool": "fetch_data", "arguments": {"url": "u"}},
        )
        runtime.journal.append_event(
            execution.id,
            "ToolStarted",
            {"call_id": "call_x", "tool": "fetch_data", "attempt": 1},
        )
        runtime.journal.append_event(
            execution.id,
            "ToolFailed",
            {"call_id": "call_x", "tool": "fetch_data", "attempt": 1, "error": error},
        )
        runtime.journal.append_event(
            execution.id,
            "ToolRetryScheduled",
            {
                "call_id": "call_x",
                "tool": "fetch_data",
                "attempt": 2,
                "failed_attempt": 1,
                "delay": 0.1,
                "reason": "RETRYABLE",
                "error": error,
            },
        )
        original_state = runtime.reconstruct_state(execution.id)

        result = runtime.replay(execution.id)

    assert result.matched is True
    assert result.final_state == original_state
    assert result.retries_replayed == 1
    assert result.delays_replayed == (0.1,)
    call = result.final_state.tool_calls[0]
    assert call.status is ToolCallStatus.RETRYING
    assert call.pending_retry.attempt == 2
    assert call.pending_retry.delay == pytest.approx(0.1)
