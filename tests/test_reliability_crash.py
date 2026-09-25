"""Milestone 4C, part 4: the crash boundaries that only a real process shows.

Each test here runs a child that calls ``os._exit`` at a precise point in the
reliability story -- no ``finally``, no ``atexit``, no clean close -- and then
asserts only what SQLite committed. Four boundaries, one per §17 case:

============================  ==========================================
retry scheduled               the decision is durable; nothing is lost
timeout occurred              a stopped attempt is still a recorded attempt
cancellation requested        the decision survives; nothing resumes it
side effect, then crash       RECOVERY_REQUIRED, and no second effect
============================  ==========================================

The last one is the important one. Its fake payment provider is a *file*, and
the file is the only trace of the side effect that outlives the process -- so
"the runtime did not charge twice" is asserted against the outside world rather
than against the runtime's own opinion of itself.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from agent_runtime import (
    AmbiguousTimeoutError,
    ExecutionStatus,
    IdempotencyRecoveryRequiredError,
    IdempotencyStatus,
    InvalidStateTransitionError,
    RecoveryState,
    Runtime,
    ToolCallStatus,
)
from conftest import PROJECT_ROOT
from failure_tools import read_effects

#: A child that dies at a named point, driving the whole reliability story.
#: ``CRASH_POINT`` selects where, and ``CRASH_EFFECTS`` is the file that stands
#: in for the outside world.
CRASH_CHILD = r'''
import asyncio, os, sys, threading, time

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from agent_runtime import (
    Execution, RecordingSleeper, RetryPolicy, RetryableToolError, Runtime,
    ToolCancelledError, ToolTimedOutError,
)

point = os.environ["CRASH_POINT"]
db = os.environ["CRASH_DB"]
effects_path = os.environ["CRASH_EFFECTS"]
execution_id = os.environ["CRASH_EXECUTION_ID"]


class KillOnSleep(RecordingSleeper):
    """Records the delay, then kills the process instead of waiting."""

    def __init__(self, crash_on_wait=1):
        super().__init__()
        self.crash_on_wait = crash_on_wait
        self.waits = 0

    def interruptible_sleep(self, delay, token=None):
        super().interruptible_sleep(delay, token)
        self.waits += 1
        if self.waits >= self.crash_on_wait:
            print(f"crashing during backoff {self.waits}, delay={delay}", flush=True)
            os._exit(70)


def die(code):
    print("dying", flush=True)
    os._exit(code)


def flaky():
    """A tool that always fails retryably."""

    def tool():
        raise RetryableToolError("upstream is down")

    return tool


def slow_async():
    """A coroutine that outlives any deadline the test sets."""

    async def tool():
        await asyncio.sleep(30)
        return "never"

    return tool


def worker():
    """A cooperative tool that only stops when its token is flipped."""

    def tool(cancel_token):
        while not cancel_token.is_cancelled():
            time.sleep(0.005)
        return "stopped"

    return tool


def charge_and_ignore():
    """§16's hardest tool: the effect happens, then the tool will not stop.

    The token is flipped at the deadline and ignored, so the runtime cannot
    prove the charge was stopped -- which is exactly the ambiguity the
    milestone says must become RECOVERY_REQUIRED rather than a retry. Under the
    ``side_effect`` point it dies the instant the effect is durable, so nothing
    in the database ever learns that the money moved.
    """

    def tool(to, cancel_token):
        with open(effects_path, "a") as handle:
            handle.write(to + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if point == "side_effect":
            # The money moved; nothing durable says so. Die right here.
            os._exit(72)
        time.sleep(30)
        return {"to": to}

    return tool



# -- the four boundaries ---------------------------------------------------

if point == "retry_scheduled":
    # Fail once, journal the retry, then die *in the backoff*: the decision is
    # committed and the attempt it named has not begun.
    with Runtime(db, sleeper=KillOnSleep(1)) as runtime:
        runtime.register_tool(flaky(), name="flaky")
        execution = runtime.start(goal="die in the backoff", execution_id=execution_id)
        execution.call("flaky", retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.1))
    os._exit(0)


if point == "after_timeout":
    # The deadline cancels the coroutine for real and the attempt is journalled
    # ToolTimedOut; then the process dies before anything else happens to it.
    def die_after_the_timeout(self):
        print("dying right after the recorded timeout", flush=True)
        os._exit(73)

    Execution.checkpoint = die_after_the_timeout
    with Runtime(db) as runtime:
        runtime.register_tool(slow_async(), name="slow")
        execution = runtime.start(goal="die after a timeout", execution_id=execution_id)
        try:
            execution.call("slow", timeout=0.05)
        except ToolTimedOutError:
            die(73)
    os._exit(0)


if point == "after_cancel":
    # The cancellation is journalled; the process dies before it can do anything
    # else with it.
    with Runtime(db) as runtime:
        runtime.register_tool(worker(), name="worker")
        execution = runtime.start(goal="die after a cancel", execution_id=execution_id)
        execution.cancel("operator stop")
        die(74)


if point == "side_effect":
    # The charge happens, the deadline expires, the tool will not stop, and the
    # process dies in the middle of it.
    with Runtime(db) as runtime:
        runtime.register_tool(charge_and_ignore(), name="charge")
        execution = runtime.start(goal="charge then die", execution_id=execution_id)
        try:
            execution.call("charge", to="card-1", timeout=0.05, idempotency_key="pay-1")
        except Exception:
            pass
    os._exit(0)


print(f"unhandled crash point: {point}", flush=True)
os._exit(75)
'''


def run_crashing_child(
    db_path: str, effects: str, point: str
) -> subprocess.CompletedProcess:
    """Run the child that dies at ``point``; it must not exit cleanly."""
    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CRASH_DB": db_path,
        "CRASH_EFFECTS": effects,
        "CRASH_POINT": point,
        "CRASH_EXECUTION_ID": "exec_crashed",
    }
    completed = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD], env=env, capture_output=True, text=True
    )
    assert completed.returncode != 0, (
        f"the child was supposed to die at {point!r} but exited cleanly\n"
        f"{completed.stdout}{completed.stderr}"
    )
    return completed


# ---------------------------------------------------------------------------
# §17 case 1: retry scheduled -> crash -> recover
# ---------------------------------------------------------------------------


def test_a_crash_during_a_backoff_keeps_the_scheduled_retry(db_path, tmp_path):
    """The retry decision is durable; the backoff is not re-waited."""
    effects = str(tmp_path / "effects.txt")
    run_crashing_child(db_path, effects, "retry_scheduled")

    with Runtime(db_path) as runtime:
        info = runtime.recovery_info("exec_crashed")

        # Nothing is ambiguous: the journal says exactly which attempt is next.
        assert info.recovery_state is RecoveryState.RETRYABLE
        assert info.has_pending_retries is True
        assert info.needs_resolution is False
        assert info.status is ExecutionStatus.RUNNING

        call = info.state.tool_calls[0]
        assert call.attempt == 1
        assert call.attempts[0].status is ToolCallStatus.FAILED
        assert call.pending_retry is not None
        assert call.pending_retry.attempt == 2

        # And the recovered state is exactly what folding the journal gives.
        assert info.state == runtime.reconstruct_state("exec_crashed")


def test_a_recovered_execution_can_still_carry_the_scheduled_retry_on(db_path, tmp_path):
    """§12: a retryable state is resumable by the new process."""
    effects = str(tmp_path / "effects.txt")
    run_crashing_child(db_path, effects, "retry_scheduled")

    with Runtime(db_path) as runtime:
        runtime.register_tool(_succeed_now(), name="flaky")
        execution = runtime.resume("exec_crashed")
        call_id = execution.tool_calls[0].call_id

        assert execution.call is not None
        result = execution.continue_pending_retry(call_id)

        assert result == "recovered"
        assert execution.tool_calls[0].status is ToolCallStatus.COMPLETED
        # The failure that was retried is still in the attempt history.
        assert execution.tool_calls[0].attempts[0].status is ToolCallStatus.FAILED


def _succeed_now():
    """A tool that succeeds, so a recovered retry has somewhere to land."""

    def tool():
        return "recovered"

    return tool


# ---------------------------------------------------------------------------
# §17 case 2: timeout occurred -> crash -> recover
# ---------------------------------------------------------------------------


def test_a_crash_right_after_a_timeout_keeps_the_timeout(db_path, tmp_path):
    """A stopped attempt is still a *recorded* attempt."""
    effects = str(tmp_path / "effects.txt")
    run_crashing_child(db_path, effects, "after_timeout")

    with Runtime(db_path) as runtime:
        info = runtime.recovery_info("exec_crashed")

        # Not an incompleteness: the journal recorded that the attempt stopped.
        assert info.needs_resolution is False
        assert info.status is ExecutionStatus.RUNNING

        call = info.state.tool_calls[0]
        assert call.status is ToolCallStatus.TIMED_OUT
        assert call.timeout == 0.05
        assert call.timeout_mode == "ASYNC"
        assert call.timeout_enforced is True
        assert call.attempts[0].status is ToolCallStatus.TIMED_OUT

        assert info.state == runtime.reconstruct_state("exec_crashed")

# ---------------------------------------------------------------------------
# §17 case 3: cancellation requested -> crash -> recover
# ---------------------------------------------------------------------------


def test_a_crash_after_a_cancellation_leaves_it_cancelled(db_path, tmp_path):
    """§4: a cancelled execution stays cancelled, with nothing to resume."""
    effects = str(tmp_path / "effects.txt")
    run_crashing_child(db_path, effects, "after_cancel")

    with Runtime(db_path) as restarted:
        info = restarted.recovery_info("exec_crashed")

        assert info.recovery_state is RecoveryState.CANCELLED
        assert info.status is ExecutionStatus.CANCELLED
        # A cancellation is not an incompleteness and not a pending retry.
        assert info.needs_resolution is False
        assert info.has_pending_retries is False
        assert info.cancelled_calls == ()

        execution = restarted.resume("exec_crashed")
        assert execution.status is ExecutionStatus.CANCELLED
        # Nothing can continue, and nothing new may start.
        with pytest.raises(InvalidStateTransitionError):
            execution.call("worker")

    # The original journal still says CANCELLED, twice over if you count the
    # tool and the execution -- a durable decision, not a memory.
    with Runtime(db_path) as restarted:
        types = [str(e.event_type) for e in restarted.get_events("exec_crashed")]
        assert types == ["ExecutionStarted", "ExecutionCancelled"]


# ---------------------------------------------------------------------------
# §17 case 4: side effect, then crash before any durable result
# ---------------------------------------------------------------------------


def test_a_crash_between_the_side_effect_and_its_result_is_recovery_required(
    db_path, tmp_path
):
    """§17's hardest case, and the milestone's reason for existing.

    The fake provider appended the charge and fsync'd it -- the outside world
    really was changed -- and the process then died with nothing durable saying
    so. The second process has to answer exactly one question: does it charge
    again? The answer asserted here is no, and not because it knows.
    """
    effects = str(tmp_path / "effects.txt")
    run_crashing_child(db_path, effects, "side_effect")

    # The side effect is real, and the file is the only record of it.
    assert read_effects(effects) == ["card-1"]

    with Runtime(db_path) as restarted:
        runtime_calls: list[str] = []
        restarted.register_tool(_counting(runtime_calls), name="charge")
        info = restarted.recovery_info("exec_crashed")

        # Two separate ambiguities, and the recovery reports both.
        assert info.status is ExecutionStatus.RECOVERY_REQUIRED
        assert info.recovery_state is RecoveryState.RECOVERY_REQUIRED
        assert info.needs_resolution is True  # the call was still open
        assert [r.idempotency_key for r in info.pending_idempotency] == ["pay-1"]

        execution = restarted.resume("exec_crashed")
        assert [r.idempotency_key for r in execution.unresolved_idempotency] == ["pay-1"]

        # The open call blocks new work, so the tool cannot run at all.
        with pytest.raises(InvalidStateTransitionError):
            execution.call("charge", to="card-1", idempotency_key="pay-1")
        assert runtime_calls == []

        # Settle the open call the way an operator would...
        _settle_open_call(execution)

        # ...and the idempotency key still refuses, because nobody has said
        # whether the charge happened. The tool is still not reached.
        with pytest.raises(IdempotencyRecoveryRequiredError):
            execution.call("charge", to="card-1", idempotency_key="pay-1")
        assert runtime_calls == []

    assert read_effects(effects) == ["card-1"]


def _settle_open_call(execution) -> None:
    """Resolve the tool call a crash left open, as an operator would."""
    if execution.incomplete_tools:
        execution.resolve_recovery(
            execution.incomplete_tools[0].call_id, "mark_failed"
        )


def test_an_operator_resolves_the_ambiguous_key_without_charging_again(
    db_path, tmp_path
):
    """Scenario 2's explicit resolution, after a crash mid side effect."""
    effects = str(tmp_path / "effects.txt")
    run_crashing_child(db_path, effects, "side_effect")

    with Runtime(db_path) as restarted:
        runtime_calls: list[str] = []
        restarted.register_tool(_counting(runtime_calls), name="charge")
        execution = restarted.resume("exec_crashed")
        _settle_open_call(execution)

        # "I checked the provider: it really was charged."
        record = execution.resolve_idempotency(
            "pay-1", "mark_completed", result={"to": "card-1"}
        )
        assert record.status is IdempotencyStatus.COMPLETED

        # The duplicate is answered from the recorded result and runs nothing.
        assert execution.call("charge", to="card-1", idempotency_key="pay-1") == {
            "to": "card-1"
        }
        assert runtime_calls == []

    assert read_effects(effects) == ["card-1"]



def test_a_recovered_timeout_is_not_silently_retried(db_path, tmp_path):
    """Recovery reports the timeout; running anything again is the app's call."""
    effects = str(tmp_path / "effects.txt")
    run_crashing_child(db_path, effects, "after_timeout")

    with Runtime(db_path) as runtime:
        calls: list[str] = []
        runtime.register_tool(_counting_slow(calls), name="slow")
        execution = runtime.resume("exec_crashed")

        # The resumed execution is RUNNING, so new work is allowed -- and the
        # runtime still refuses to re-run the timed-out call on its own.
        assert execution.status is ExecutionStatus.RUNNING
        assert calls == []


def _counting_slow(sink: list[str]):
    """A stand-in for a slow tool that records if it is ever reached."""

    def tool():
        sink.append("ran")
        return "ran"

    return tool


def _counting(sink: list[str]):
    """A stand-in for a side-effecting tool that records if it is ever reached."""

    def tool(to: str, cancel_token=None) -> dict:
        sink.append(to)
        return {"to": to}

    return tool
