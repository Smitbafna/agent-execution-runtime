"""Milestone 4A, part 2: retrying a tool call.

The invariant under test:

    A logical call keeps one call_id across every attempt; each attempt is
    journalled with its number; and the state reports the final attempt's
    outcome -- so a call that failed twice and then succeeded is COMPLETED on
    attempt 3, not FAILED.

The tests are grouped the way the milestone groups them: the basic retry
outcomes, error classification, the backoff schedule and the sleeper, the
policy's two configuration levels, and the journal/state the attempts produce.
"""

from __future__ import annotations

import json
import time

import pytest

from agent_runtime import (
    ExecutionState,
    PermanentToolError,
    RecordingSleeper,
    RetryConfigurationError,
    RetryPolicy,
    RetryableToolError,
    Runtime,
    ToolCallStatus,
    ToolInvocationError,
    tool,
)

#: The milestone's example policy: three attempts, 0.1 doubling, capped at 10s.
POLICY = RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=2.0, max_delay=10.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sleeper() -> RecordingSleeper:
    """A sleeper that records delays instead of spending them."""
    return RecordingSleeper()


@pytest.fixture
def flaky_runtime(db_path, sleeper):
    """A runtime whose executions wait through ``sleeper``."""
    with Runtime(db_path, sleeper=sleeper) as runtime:
        yield runtime


def register_flaky(
    runtime: Runtime,
    counter: dict,
    *,
    error=PermanentToolError,
    name: str = "fetch_data",
    retry_policy: RetryPolicy | None = None,
):
    """A tool that fails its first ``counter['failures']`` invocations."""

    @tool(retry_policy=retry_policy)
    def fetch_data(url: str = "https://example.com") -> dict:
        counter["calls"] += 1
        if counter["calls"] <= counter["failures"]:
            raise error(f"attempt {counter['calls']} failed")
        return {"url": url, "attempt": counter["calls"]}

    runtime.register_tool(fetch_data, name=name)
    return fetch_data


def event_types(execution) -> list[str]:
    return [str(event.event_type) for event in execution.events]


# ---------------------------------------------------------------------------
# Basic retries (§12)
# ---------------------------------------------------------------------------


def test_a_call_that_succeeds_first_time_is_not_retried(flaky_runtime):
    counter = {"calls": 0, "failures": 0}
    register_flaky(flaky_runtime, counter)
    execution = flaky_runtime.start(goal="fetch")

    result = execution.call("fetch_data")

    assert result == {"url": "https://example.com", "attempt": 1}
    assert counter["calls"] == 1
    assert event_types(execution) == [
        "ExecutionStarted",
        "ToolRequested",
        "ToolStarted",
        "ToolCompleted",
    ]


def test_a_failure_then_a_success_completes_the_call(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    result = execution.call("fetch_data", retry_policy=POLICY)

    assert result == {"url": "https://example.com", "attempt": 2}
    assert counter["calls"] == 2
    assert event_types(execution) == [
        "ExecutionStarted",
        "ToolRequested",
        "ToolStarted",
        "ToolFailed",
        "ToolRetryScheduled",
        "ToolStarted",
        "ToolCompleted",
    ]


def test_several_failures_then_a_success_complete_the_call(flaky_runtime):
    """The milestone's worked example: fail, fail, succeed."""
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    result = execution.call("fetch_data", retry_policy=POLICY)

    assert result["attempt"] == 3
    assert counter["calls"] == 3
    assert event_types(execution) == [
        "ExecutionStarted",
        "ToolRequested",
        "ToolStarted",
        "ToolFailed",
        "ToolRetryScheduled",
        "ToolStarted",
        "ToolFailed",
        "ToolRetryScheduled",
        "ToolStarted",
        "ToolCompleted",
    ]


def test_the_maximum_number_of_attempts_is_respected(flaky_runtime):
    counter = {"calls": 0, "failures": 99}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError) as excinfo:
        execution.call("fetch_data", retry_policy=POLICY)

    assert counter["calls"] == 3, "three attempts, no more"
    assert excinfo.value.attempts == 3
    # Two retries were scheduled, and no third attempt was started after them.
    assert event_types(execution).count("ToolRetryScheduled") == 2
    assert event_types(execution).count("ToolStarted") == 3
    assert event_types(execution)[-1] == "ToolFailed"


def test_the_raised_error_still_chains_the_original_exception(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError) as excinfo:
        execution.call("fetch_data", retry_policy=RetryPolicy(max_attempts=1))

    assert isinstance(excinfo.value.__cause__, RetryableToolError)
    assert excinfo.value.call_id == execution.state.tool_calls[0].call_id


# ---------------------------------------------------------------------------
# Logical call versus attempt (§2)
# ---------------------------------------------------------------------------


def test_every_attempt_of_a_call_shares_one_call_id(flaky_runtime):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)

    call_ids = {
        event.payload["call_id"]
        for event in execution.events
        if "call_id" in event.payload
    }
    assert len(call_ids) == 1
    assert execution.state.tool_calls[0].call_id in call_ids


def test_two_logical_calls_get_two_ids_even_when_they_are_the_same_tool(flaky_runtime):
    counter = {"calls": 0, "failures": 0}
    register_flaky(flaky_runtime, counter)
    execution = flaky_runtime.start(goal="fetch twice")

    execution.call("fetch_data")
    execution.call("fetch_data")

    ids = [call.call_id for call in execution.state.tool_calls]
    assert len(set(ids)) == 2
    assert [call.attempt for call in execution.state.tool_calls] == [1, 1]


def test_a_call_without_a_policy_is_never_retried(flaky_runtime):
    """The pre-Milestone-4A behaviour, unchanged: one attempt, then it raises."""
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError):
        execution.call("fetch_data")

    assert counter["calls"] == 1
    assert "ToolRetryScheduled" not in event_types(execution)


# ---------------------------------------------------------------------------
# Error classification in the loop (§4, §12)
# ---------------------------------------------------------------------------


def test_a_retryable_failure_is_retried(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)

    assert counter["calls"] == 2
    assert execution.state.tool_calls[0].status is ToolCallStatus.COMPLETED


def test_a_permanent_failure_is_not_retried(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=PermanentToolError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError):
        execution.call("fetch_data", retry_policy=POLICY)

    assert counter["calls"] == 1, "a permanent failure is never repeated"
    assert "ToolRetryScheduled" not in event_types(execution)
    assert execution.state.tool_calls[0].status is ToolCallStatus.FAILED


def test_an_unexpected_exception_is_not_retried_by_default(flaky_runtime):
    """A bug is not a transient condition, and the runtime will not pretend."""
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RuntimeError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError):
        execution.call("fetch_data", retry_policy=POLICY)

    assert counter["calls"] == 1
    assert "ToolRetryScheduled" not in event_types(execution)


def test_an_unexpected_exception_is_retried_when_the_policy_opts_in(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RuntimeError)
    execution = flaky_runtime.start(goal="fetch")

    result = execution.call(
        "fetch_data", retry_policy=RetryPolicy(max_attempts=3, retry_on_unknown=True)
    )

    assert counter["calls"] == 2
    assert result["attempt"] == 2
    assert execution.state.tool_calls[0].attempt == 2


def test_the_classification_is_recorded_with_the_retry(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)

    scheduled = next(
        e for e in execution.events if e.event_type == "ToolRetryScheduled"
    )
    assert scheduled.payload["reason"] == "RETRYABLE"
    assert scheduled.payload["error"]["type"] == "RetryableToolError"
# ---------------------------------------------------------------------------
# Backoff and the sleeper in the loop (§7, §8, §12)
# ---------------------------------------------------------------------------


def test_the_sleeper_records_the_backoff_schedule(flaky_runtime, sleeper):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)

    assert sleeper.delays == (0.1, 0.2)
    assert sleeper.total_delay == pytest.approx(0.3)


def test_the_sleeper_is_asked_to_wait_before_each_new_attempt(flaky_runtime, sleeper):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.5))

    (scheduled,) = [e for e in execution.events if e.event_type == "ToolRetryScheduled"]
    assert scheduled.payload["delay"] == 0.5
    assert sleeper.delays == (0.5,)


def test_no_test_ever_spends_the_backoff(db_path, sleeper):
    """A policy whose waits would total minutes completes in milliseconds."""
    counter = {"calls": 0, "failures": 2}
    slow = RetryPolicy(max_attempts=3, initial_delay=600.0, max_delay=600.0)
    with Runtime(db_path, sleeper=sleeper) as runtime:
        register_flaky(runtime, counter, error=RetryableToolError)
        execution = runtime.start(goal="slow but not really")

        started = time.perf_counter()
        execution.call("fetch_data", retry_policy=slow)
        elapsed = time.perf_counter() - started

    assert sleeper.delays == (600.0, 600.0)
    assert elapsed < 2.0, "the schedule was recorded, not waited for"


def test_a_call_that_never_retries_never_waits(flaky_runtime, sleeper):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError):
        execution.call("fetch_data")

    assert sleeper.delays == ()


def test_the_max_delay_caps_the_schedule_the_execution_waits_through(flaky_runtime, sleeper):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    policy = RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=100.0, max_delay=0.15)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=policy)

    assert sleeper.delays == (0.1, 0.15)

# ---------------------------------------------------------------------------
# Configuring the policy (§5)
# ---------------------------------------------------------------------------


def test_a_tool_may_declare_its_own_retry_policy(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError, retry_policy=POLICY)
    execution = flaky_runtime.start(goal="fetch")

    result = execution.call("fetch_data")

    assert counter["calls"] == 2
    assert result["attempt"] == 2


def test_a_call_level_policy_overrides_the_tool_level_one(flaky_runtime):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError, retry_policy=POLICY)
    execution = flaky_runtime.start(goal="fetch")

    result = execution.call("fetch_data", retry_policy=RetryPolicy(max_attempts=5))

    assert counter["calls"] == 3, "the call's five attempts beat the tool's three"
    assert result["attempt"] == 3
    assert execution.state.tool_calls[0].attempt == 3


def test_a_call_level_policy_can_also_refuse_to_retry(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError, retry_policy=POLICY)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError):
        execution.call("fetch_data", retry_policy=RetryPolicy(max_attempts=1))

    assert counter["calls"] == 1


def test_a_tool_without_a_policy_is_not_retried_by_its_callers(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError):
        execution.call("fetch_data")

    assert counter["calls"] == 1


def test_the_resolved_policy_is_recorded_with_the_call(flaky_runtime):
    counter = {"calls": 0, "failures": 0}
    register_flaky(flaky_runtime, counter)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)

    requested = next(e for e in execution.events if e.event_type == "ToolRequested")
    assert requested.payload["retry_policy"]["max_attempts"] == 3
    assert requested.payload["retry_policy"]["initial_delay"] == 0.1


def test_the_runtime_decorator_accepts_a_retry_policy(db_path, sleeper):
    counter = {"calls": 0, "failures": 1}

    with Runtime(db_path, sleeper=sleeper) as runtime:
        @runtime.tool(name="shaky", retry_policy=POLICY)
        def shaky() -> str:
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise RetryableToolError("not yet")
            return "settled"

        execution = runtime.start(goal="decorated")
        assert execution.call("shaky") == "settled"
        assert counter["calls"] == 2


def test_a_non_policy_retry_configuration_is_refused(flaky_runtime):
    with pytest.raises(RetryConfigurationError):
        tool(lambda: None, retry_policy="three times")


def test_a_non_policy_call_argument_is_refused(flaky_runtime):
    counter = {"calls": 0, "failures": 0}
    register_flaky(flaky_runtime, counter)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(RetryConfigurationError):
        execution.call("fetch_data", retry_policy=3)


def test_renaming_a_tool_keeps_its_retry_policy(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(
        flaky_runtime, counter, error=RetryableToolError, name="original", retry_policy=POLICY
    )
    flaky_runtime.registry.register(flaky_runtime.registry.get("original"), name="renamed")
    execution = flaky_runtime.start(goal="renamed")

    assert flaky_runtime.registry.get("renamed").retry_policy == POLICY
    assert execution.call("renamed")["attempt"] == 2

# ---------------------------------------------------------------------------
# The journal of attempts (§6) and the state it reconstructs (§9)
# ---------------------------------------------------------------------------


def test_the_journal_reconstructs_the_whole_retry_sequence(flaky_runtime):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)

    call_id = execution.state.tool_calls[0].call_id
    journal = [
        (str(e.event_type), e.payload.get("attempt"), e.payload.get("failed_attempt"))
        for e in execution.events
        if "call_id" in e.payload
    ]
    assert journal == [
        ("ToolRequested", None, None),
        ("ToolStarted", 1, None),
        ("ToolFailed", 1, None),
        ("ToolRetryScheduled", 2, 1),
        ("ToolStarted", 2, None),
        ("ToolFailed", 2, None),
        ("ToolRetryScheduled", 3, 2),
        ("ToolStarted", 3, None),
        ("ToolCompleted", 3, None),
    ]
    assert {e.payload["call_id"] for e in execution.events if "call_id" in e.payload} == {call_id}


def test_failed_intermediate_attempts_stay_in_the_journal(flaky_runtime):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)

    failures = [e for e in execution.events if e.event_type == "ToolFailed"]
    assert [e.payload["attempt"] for e in failures] == [1, 2]
    assert failures[0].payload["error"]["message"] == "attempt 1 failed"
    assert failures[1].payload["error"]["message"] == "attempt 2 failed"
    # The final outcome does not erase them.
    assert execution.state.tool_calls[0].status is ToolCallStatus.COMPLETED


def test_the_state_reports_the_final_attempt_of_a_successful_call(flaky_runtime):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)
    call = execution.state.tool_calls[0]

    assert call.status is ToolCallStatus.COMPLETED, "not FAILED: the last attempt won"
    assert call.attempt == 3
    assert call.attempt_count == 3
    assert call.result["attempt"] == 3
    assert call.error is None
    assert call.pending_retry is None


def test_the_state_exposes_every_attempt_of_the_call(flaky_runtime):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)
    attempts = execution.state.tool_calls[0].attempts

    assert [a.attempt for a in attempts] == [1, 2, 3]
    assert [a.status for a in attempts] == [
        ToolCallStatus.FAILED,
        ToolCallStatus.FAILED,
        ToolCallStatus.COMPLETED,
    ]
    assert attempts[0].error["message"] == "attempt 1 failed"
    assert attempts[2].result["attempt"] == 3
    assert attempts[2].settled is True


def test_each_attempt_records_the_retry_it_scheduled(flaky_runtime):
    counter = {"calls": 0, "failures": 2}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    execution.call("fetch_data", retry_policy=POLICY)
    first, second, third = execution.state.tool_calls[0].attempts

    assert first.scheduled_retry.attempt == 2
    assert first.scheduled_retry.delay == pytest.approx(0.1)
    assert second.scheduled_retry.attempt == 3
    assert second.scheduled_retry.delay == pytest.approx(0.2)
    assert third.scheduled_retry is None


def test_the_attempt_history_survives_a_json_round_trip(flaky_runtime):
    counter = {"calls": 0, "failures": 1}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")
    execution.call("fetch_data", retry_policy=POLICY)

    restored = ExecutionState.from_dict(json.loads(json.dumps(execution.state.to_dict())))

    assert restored == execution.state
    assert restored.tool_calls[0].attempts[0].scheduled_retry.attempt == 2


def test_a_call_that_exhausts_its_attempts_is_failed_on_the_last_one(flaky_runtime):
    counter = {"calls": 0, "failures": 99}
    register_flaky(flaky_runtime, counter, error=RetryableToolError)
    execution = flaky_runtime.start(goal="fetch")

    with pytest.raises(ToolInvocationError):
        execution.call("fetch_data", retry_policy=RetryPolicy(max_attempts=2))

    call = execution.state.tool_calls[0]
    assert call.status is ToolCallStatus.FAILED
    assert call.attempt == 2
    assert call.error["message"] == "attempt 2 failed"


def test_a_pending_retry_is_visible_while_the_call_is_still_open(db_path):
    """A scheduled-but-not-started retry is a decision, not an ambiguity."""
    counter = {"calls": 0, "failures": 1}
    observed: dict = {}

    class ObservingSleeper(RecordingSleeper):
        """Captures the state at the exact moment of the backoff."""

        def sleep(self, delay):
            super().sleep(delay)
            observed["call"] = execution.state.tool_calls[0]

    with Runtime(db_path, sleeper=ObservingSleeper()) as runtime:
        register_flaky(runtime, counter, error=RetryableToolError)
        execution = runtime.start(goal="fetch")
        execution.call("fetch_data", retry_policy=POLICY)

        call = observed["call"]
        assert call.status is ToolCallStatus.RETRYING
        assert call.attempt == 1
        assert call.pending_retry is not None
        assert call.pending_retry.attempt == 2
        assert call.pending_retry.failed_attempt == 1
        assert call.pending_retry.delay == pytest.approx(0.1)
        assert call.pending_retry.sequence == call.attempts[0].scheduled_retry.sequence
        # Not ambiguity, so it does not make the execution RECOVERY_REQUIRED.
        assert call.status.is_incomplete is False
        assert execution.state.status.value == "RUNNING"


def test_a_pending_retry_does_not_make_an_execution_need_recovery(db_path):
    counter = {"calls": 0, "failures": 1}
    with Runtime(db_path, sleeper=RecordingSleeper()) as runtime:
        register_flaky(runtime, counter, error=RetryableToolError)
        execution = runtime.start(goal="fetch")
        execution.call("fetch_data", retry_policy=POLICY)

        assert execution.pending_retries == ()
        assert execution.incomplete_tools == ()
        assert execution.needs_recovery is False




