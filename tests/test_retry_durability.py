"""Milestone 4A, part 3: retries that have to survive a crash.

The invariant under test:

    A retry decision is durable. A recovered execution knows which attempt was
    scheduled, still holds the attempt history, and reconstructs exactly what
    folding the whole journal would.

The crash cases run in a real child process that calls ``os._exit`` during the
backoff -- no ``finally``, no ``atexit``, no close -- so only what SQLite
committed survives, and only that is asserted.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from agent_runtime import (
    ExecutionStatus,
    InvalidRecoveryActionError,
    InvalidStateTransitionError,
    PermanentToolError,
    RecordingSleeper,
    RetryPolicy,
    RetryableToolError,
    Runtime,
    ToolCallStatus,
    ToolInvocationError,
    UnknownToolCallError,
)
from conftest import PROJECT_ROOT

POLICY = RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=2.0, max_delay=10.0)


@pytest.fixture
def sleeper() -> RecordingSleeper:
    """A sleeper that records delays instead of spending them."""
    return RecordingSleeper()

#: Runs in a child process that fails, schedules a retry and then dies in the
#: middle of the backoff -- the narrowest window in which retry state could be
#: lost, because the decision is journalled and the attempt has not begun.
#:
#: ``FAILURES`` is how many attempts fail before the tool succeeds, and
#: ``CRASH_ON_WAIT`` is which backoff the process dies in.
CRASH_CHILD = r'''
import os, sys

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from agent_runtime import RecordingSleeper, RetryPolicy, RetryableToolError, Runtime


class CrashDuringBackoff(RecordingSleeper):
    """Records the delay, then kills the process instead of waiting."""

    def __init__(self, crash_on_wait):
        super().__init__()
        self.crash_on_wait = crash_on_wait
        self.waits = 0

    def sleep(self, delay):
        super().sleep(delay)
        self.waits += 1
        if self.waits >= self.crash_on_wait:
            print(f"crashing during backoff {self.waits}, delay={delay}", flush=True)
            os._exit(70)


counter = {"calls": 0}


def register(runtime):
    @runtime.tool(retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.1))
    def fetch_data(url: str = "https://example.com") -> dict:
        counter["calls"] += 1
        if counter["calls"] <= int(os.environ["FAILURES"]):
            raise RetryableToolError(f"attempt {counter['calls']} failed")
        return {"url": url, "attempt": counter["calls"]}


with Runtime(
    os.environ["CRASH_DB"],
    sleeper=CrashDuringBackoff(int(os.environ["CRASH_ON_WAIT"])),
) as runtime:
    register(runtime)
    execution = runtime.resume(os.environ["CRASH_EXECUTION_ID"])
    execution.call("fetch_data", url=os.environ["CRASH_URL"])
    print("the retry loop finished, which it must not", flush=True)
    os._exit(0)
'''


def run_crashing_child(
    db_path: str,
    execution_id: str,
    *,
    failures: int = 1,
    crash_on_wait: int = 1,
    url: str = "https://example.com",
) -> str:
    """Run the child that dies during the backoff; return what it printed."""
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CRASH_DB": db_path,
        "CRASH_EXECUTION_ID": execution_id,
        "FAILURES": str(failures),
        "CRASH_ON_WAIT": str(crash_on_wait),
        "CRASH_URL": url,
    }
    result = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0, (
        "the child was supposed to die during the backoff but exited cleanly\n"
        f"{result.stdout}{result.stderr}"
    )
    return result.stdout


def register_fetch(runtime: Runtime, marker: str):
    """A tool that succeeds once the marker file exists.

    The marker is what lets a second process -- the one that recovers the
    execution -- make the retry succeed, with no shared memory involved.
    """

    @runtime.tool(retry_policy=POLICY)
    def fetch_data(url: str) -> dict:
        if not os.path.exists(marker):
            raise RetryableToolError("upstream is down")
        return {"url": url, "attempt": "recovered"}

    return fetch_data


# ---------------------------------------------------------------------------
# Checkpoints across retry events (§10)
# ---------------------------------------------------------------------------


def test_a_checkpoint_taken_mid_retry_sequence_recovers_exactly(db_path, sleeper):
    """checkpoint @ 10 + the retry tail == folding the whole journal."""
    counter = {"calls": 0}
    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool
        def flaky() -> str:
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise RetryableToolError("down")
            return "settled"

        execution = runtime.start(goal="flaky work")

        # A call that fails for good, snapshotted: the retry events all follow it.
        with pytest.raises(Exception):
            execution.call("flaky", retry_policy=RetryPolicy(max_attempts=1))
        checkpoint = execution.checkpoint()

        assert checkpoint.sequence == execution.state.last_sequence

        # The work continues, this time with a policy that retries.
        counter["calls"] = 0
        execution.call("flaky", retry_policy=POLICY)
        execution.complete()

        recovered_state = runtime.resume(execution.id).state
        assert recovered_state == runtime.reconstruct_state(execution.id)
        assert recovered_state.status is ExecutionStatus.COMPLETED


def test_a_checkpoint_before_the_retry_tail_reconstructs_the_attempts(db_path, sleeper):
    counter = {"calls": 0}
    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool
        def flaky(label: str) -> str:
            if label == "second" and counter["calls"] == 0:
                counter["calls"] += 1
                raise RetryableToolError("down")
            return f"{label}@{counter['calls']}"

        execution = runtime.start(goal="flaky work")
        execution.call("flaky", label="first")
        checkpoint = execution.checkpoint()  # before any retry event exists

        execution.call("flaky", label="second", retry_policy=POLICY)
        execution.complete()

        assert runtime.latest_checkpoint(execution.id).sequence == checkpoint.sequence
        state = runtime.resume(execution.id).state
        assert state == runtime.reconstruct_state(execution.id)
        retried = state.tool_calls[1]
        assert retried.attempt == 2
        assert retried.status is ToolCallStatus.COMPLETED
        assert [a.status for a in retried.attempts] == [
            ToolCallStatus.FAILED,
            ToolCallStatus.COMPLETED,
        ]


def test_a_checkpoint_written_between_attempts_still_holds_the_history(db_path, sleeper):
    """The snapshot may be taken at any sequence, including a scheduled retry."""
    counter = {"calls": 0}
    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool(retry_policy=POLICY)
        def flaky() -> str:
            counter["calls"] += 1
            raise RetryableToolError("always down")

        execution = runtime.start(goal="doomed")
        with pytest.raises(ToolInvocationError):
            execution.call("flaky")
        checkpoint = execution.checkpoint()
        execution.fail("gave up")

        # ExecutionStarted, then three attempts, each with a start, a failure
        # and (except after the last) a scheduled retry.
        assert checkpoint.sequence == 10
        assert [str(e.event_type) for e in runtime.get_events(execution.id)].count(
            "ToolRetryScheduled"
        ) == 2
        state = runtime.resume(execution.id).state
        assert state == runtime.reconstruct_state(execution.id)
        call = state.tool_calls[0]
        assert call.attempt == 3
        assert [a.attempt for a in call.attempts] == [1, 2, 3]
        assert call.status is ToolCallStatus.FAILED


# ---------------------------------------------------------------------------
# A retry that outlives its process (§13)
# ---------------------------------------------------------------------------


def test_retry_events_survive_a_process_restart(db_path, sleeper):
    """A whole retry sequence, written by one process and read by the next."""
    counter = {"calls": 0}
    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool(retry_policy=POLICY)
        def flaky() -> str:
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise RetryableToolError("down")
            return "settled"

        execution = runtime.start(goal="flaky work")
        execution.call("flaky")
        execution.complete()
        execution_id = execution.id

    # A brand new Runtime over the same file: no shared memory with the run above.
    with Runtime(db_path, sleeper=RecordingSleeper()) as restarted:
        recovered = restarted.resume(execution_id)

        assert recovered.state == restarted.reconstruct_state(execution_id)
        call = recovered.state.tool_calls[0]
        assert call.attempt == 2
        assert call.status is ToolCallStatus.COMPLETED
        assert [a.attempt for a in call.attempts] == [1, 2]
        assert call.attempts[0].error["message"] == "down"
        assert call.attempts[0].scheduled_retry.attempt == 2
# ---------------------------------------------------------------------------
# A real crash in the middle of the backoff (§13)
# ---------------------------------------------------------------------------


def test_a_process_that_dies_during_the_backoff_keeps_its_retry_state(
    runtime, db_path, tmp_path
):
    """attempt 1 fails -> retry scheduled -> PROCESS CRASH -> new runtime."""
    execution = runtime.start(goal="Fetch data that is flaky")
    output = run_crashing_child(db_path, execution.id, failures=1, crash_on_wait=1)

    assert "crashing during backoff 1, delay=0.1" in output
    assert "the retry loop finished" not in output

    # Everything below is read by this process from what the child's SQLite
    # committed: the decision, the failed attempt, and nothing invented.
    with Runtime(db_path, sleeper=RecordingSleeper()) as recovered_runtime:
        recovered = recovered_runtime.resume(execution.id)
        info = recovered.recovery_info()

        assert [
            str(event.event_type) for event in recovered_runtime.get_events(execution.id)
        ] == [
            "ExecutionStarted",
            "ToolRequested",
            "ToolStarted",
            "ToolFailed",
            "ToolRetryScheduled",
        ]

        # The recovered execution knows a retry was scheduled...
        (pending,) = recovered.pending_retries
        assert pending.attempt == 2
        assert pending.failed_attempt == 1
        assert pending.delay == pytest.approx(0.1)
        assert info.has_pending_retries is True
        # ... and it is not an ambiguity, so recovery is not asking for a decision.
        assert recovered.status is ExecutionStatus.RUNNING
        assert info.needs_resolution is False
        assert recovered.incomplete_tools == ()

        # The attempt history is intact: the failure is still there.
        call = recovered.state.tool_calls[0]
        assert call.status is ToolCallStatus.RETRYING
        assert call.attempt == 1
        assert [a.attempt for a in call.attempts] == [1]
        assert call.attempts[0].error["message"] == "attempt 1 failed"

        # The invariant: recovery == folding the whole journal.
        assert recovered.state == recovered_runtime.reconstruct_state(execution.id)
        assert info.state == recovered.state


def test_a_recovered_execution_can_carry_the_scheduled_retry_on(db_path, tmp_path):
    """The decision is durable, so the next process can act on it."""
    marker = str(tmp_path / "upstream-is-back")
    with Runtime(db_path) as runtime:
        execution = runtime.start(goal="Fetch data that is flaky")
        execution_id = execution.id

    run_crashing_child(db_path, execution_id, failures=1, crash_on_wait=1)

    # The upstream recovers while nothing is running.
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write("back")

    with Runtime(db_path, sleeper=RecordingSleeper()) as recovered_runtime:
        register_fetch(recovered_runtime, marker)
        recovered = recovered_runtime.resume(execution_id)
        (pending,) = recovered.pending_retries

        result = recovered.continue_pending_retry(
            next(c.call_id for c in recovered.tool_calls)
        )

        assert result == {"url": "https://example.com", "attempt": "recovered"}
        call = recovered.state.tool_calls[0]
        assert call.status is ToolCallStatus.COMPLETED
        assert call.attempt == pending.attempt == 2
        # The failed first attempt is still in the history, not overwritten.
        assert [a.status for a in call.attempts] == [
            ToolCallStatus.FAILED,
            ToolCallStatus.COMPLETED,
        ]
        assert recovered.pending_retries == ()
        assert recovered.state == recovered_runtime.reconstruct_state(execution_id)

        # The continuation is journalled too, and the journal is the source of truth.
        assert [
            str(event.event_type)
            for event in recovered_runtime.get_events(execution_id)
        ][-2:] == ["ToolStarted", "ToolCompleted"]


def test_a_crash_between_two_attempts_keeps_both_attempts(runtime, db_path):
    """Crash after the second attempt failed and its retry was scheduled."""
    execution = runtime.start(goal="Twice flaky")

    output = run_crashing_child(db_path, execution.id, failures=2, crash_on_wait=2)

    assert "crashing during backoff 2, delay=0.2" in output

    with Runtime(db_path, sleeper=RecordingSleeper()) as recovered_runtime:
        recovered = recovered_runtime.resume(execution.id)
        call = recovered.state.tool_calls[0]

        assert [a.attempt for a in call.attempts] == [1, 2]
        assert [a.status for a in call.attempts] == [
            ToolCallStatus.FAILED,
            ToolCallStatus.FAILED,
        ]
        assert call.attempt == 2
        (pending,) = recovered.pending_retries
        assert pending.attempt == 3
        assert pending.delay == pytest.approx(0.2)
        assert recovered.state == recovered_runtime.reconstruct_state(execution.id)


# ---------------------------------------------------------------------------
# Continuing a pending retry in process (§13)
# ---------------------------------------------------------------------------


def test_a_scheduled_retry_is_not_recovery_work_to_resolve(db_path, sleeper):
    """Milestone 4A did not move Milestone 2's line: only *unknown* work is
    resolved by the application, and a scheduled retry is neither ambiguous nor
    a cancellation."""
    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool(name="fetch_data")
        def fetch_data() -> str:
            raise PermanentToolError("400 from upstream")

        execution = runtime.start(goal="doomed")
        with pytest.raises(ToolInvocationError):
            execution.call("fetch_data")
        execution_id = execution.id
        call_id = execution.tool_calls[0].call_id
        # Hand-journalled: a retry the runtime decided on and never took.
        runtime.journal.append_event(
            execution_id,
            "ToolRetryScheduled",
            {
                "call_id": call_id,
                "tool": "fetch_data",
                "attempt": 2,
                "failed_attempt": 1,
                "delay": 0.1,
                "reason": "RETRYABLE",
                "error": {"type": "RetryableToolError", "message": "down"},
            },
        )
        recovered = runtime.resume(execution_id)

        assert recovered.pending_retries, "it is surfaced, not resolved"
        with pytest.raises(InvalidRecoveryActionError) as excinfo:
            recovered.resolve_recovery(call_id, "mark_failed")

    # The message points at the API that does apply.
    assert "continue_pending_retry" in str(excinfo.value)


def test_continuing_a_retry_that_is_not_pending_is_refused(db_path, sleeper):
    """The policy comes from the journal, because the caller is a new process."""
    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool(retry_policy=POLICY)
        def flaky() -> str:
            raise RetryableToolError("down")

        execution = runtime.start(goal="doomed")
        with pytest.raises(ToolInvocationError):
            execution.call("flaky")
        execution_id = execution.id
        call_id = execution.tool_calls[0].call_id

    with Runtime(db_path, sleeper=sleeper) as restarted:
        recovered = restarted.resume(execution_id)

        assert recovered.pending_retries == (), "it exhausted its attempts instead"
        with pytest.raises(InvalidStateTransitionError):
            recovered.continue_pending_retry(call_id)
        with pytest.raises(UnknownToolCallError):
            recovered.continue_pending_retry("call_never_made")


        assert recovered.pending_retries == ()
