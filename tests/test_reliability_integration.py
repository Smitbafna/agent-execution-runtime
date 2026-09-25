"""Milestone 4C, part 3: how the pieces behave *together*.

Retries, idempotency, checkpoints, recovery and replay, all with deadlines and
cancellation in the picture. The single invariant these tests exist to defend::

    an ambiguous side effect is never silently duplicated, and a reliability
    event survives a restart and a replay without changing what it says

Organised as the milestone's own scenarios (§18), plus the state invariants of
§19 stated as assertions. Scenario 3 is the fullest of them -- timeout, retry,
success, replay -- and the one that would fail first if any layer started
inventing an outcome of its own.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent_runtime import (
    AmbiguousTimeoutError,
    ExecutionStatus,
    IdempotencyStatus,
    InvalidStateTransitionError,
    RecordingSleeper,
    RecoveryState,
    RetryPolicy,
    Runtime,
    ToolCallStatus,
    ToolCancelledError,
    ToolInvocationError,
    ToolTimedOutError,
)
from failure_tools import always_fails, async_fails_once, async_sleeps_forever

#: Retries timeouts too -- otherwise a flaky slow service could never recover.
TIMEOUT_POLICY = RetryPolicy(
    max_attempts=3, initial_delay=0.01, multiplier=1.0, retry_on_timeout=True
)
#: The default policy deliberately does *not* retry timeouts.
DEFAULT_POLICY = RetryPolicy(max_attempts=3, initial_delay=0.01, multiplier=1.0)


def event_types(execution) -> list[str]:
    return [str(event.event_type) for event in execution.events]


def record_effect(path: str, line: str) -> None:
    """Append one line to the file that stands in for the outside world."""
    with open(path, "a") as handle:
        handle.write(line + "\n")
        handle.flush()


# ---------------------------------------------------------------------------
# §5: timeout + retry
# ---------------------------------------------------------------------------


def test_a_timeout_is_not_retried_by_default(db_path):
    """The documented default, asserted rather than described."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow")
        execution = runtime.start(goal="the default")

        with pytest.raises(ToolTimedOutError):
            execution.call("slow", timeout=0.05, retry_policy=DEFAULT_POLICY)

        recorded = event_types(execution)
        assert recorded.count("ToolStarted") == 1
        assert "ToolRetryScheduled" not in recorded
        assert recorded[-1] == "ToolTimedOut"


def test_a_timeout_is_retried_when_the_policy_says_so(db_path):
    """§5's sequence: attempt 1 times out, the policy retries, attempt 2 wins."""
    with Runtime(db_path) as runtime:
        tool = async_fails_once(failures=1, value="settled")
        runtime.register_tool(tool, name="flaky_slow")
        execution = runtime.start(goal="retry a timeout")

        result = execution.call("flaky_slow", timeout=0.05, retry_policy=TIMEOUT_POLICY)

        assert result == "settled"
        assert event_types(execution) == [
            "ExecutionStarted",
            "ToolRequested",
            "ToolStarted",
            "ToolTimedOut",
            "ToolRetryScheduled",
            "ToolStarted",
            "ToolCompleted",
        ]


def test_a_retried_call_reports_its_success_not_its_earlier_timeout(db_path):
    """§19: intermediate stops do not overwrite a later success."""
    with Runtime(db_path) as runtime:
        tool = async_fails_once(failures=1, value="settled")
        runtime.register_tool(tool, name="flaky_slow")
        execution = runtime.start(goal="the last attempt wins")

        execution.call("flaky_slow", timeout=0.05, retry_policy=TIMEOUT_POLICY)

        call = execution.tool_calls[0]
        assert call.status is ToolCallStatus.COMPLETED
        assert call.attempt == 2
        # The timeout is still in the history -- visible, not erased.
        assert call.attempts[0].status is ToolCallStatus.TIMED_OUT
        assert call.attempts[1].status is ToolCallStatus.COMPLETED


def test_exhausting_the_timeout_retries_reports_the_last_timeout(db_path):
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow")
        execution = runtime.start(goal="every attempt times out")

        with pytest.raises(ToolTimedOutError) as caught:
            execution.call("slow", timeout=0.02, retry_policy=TIMEOUT_POLICY)

        assert caught.value.attempts == 3
        assert execution.tool_calls[0].status is ToolCallStatus.TIMED_OUT


# ---------------------------------------------------------------------------
# §6: timeout + idempotency
# ---------------------------------------------------------------------------


def test_a_keyed_call_whose_stop_could_not_be_enforced_is_never_retried(
    db_path, tmp_path
):
    """§6's headline: a timeout must not bypass the idempotency guarantee.

    The tool records its effect and then ignores the token, so the runtime
    cannot prove the side effect did not happen. Retrying would repeat it, so
    the runtime stops, leaves the key ``PENDING`` and reports the ambiguity.
    """
    effects = str(tmp_path / "effects.txt")
    released = threading.Event()

    def charges_then_ignores(to: str, cancel_token) -> dict:
        record_effect(effects, to)
        released.wait(10)  # the token is flipped; this tool does not care
        return {"to": to}

    with Runtime(db_path) as runtime:
        runtime.register_tool(charges_then_ignores, name="charge")
        execution = runtime.start(goal="an ambiguous charge")

        try:
            with pytest.raises(AmbiguousTimeoutError) as caught:
                execution.call(
                    "charge",
                    to="card-1",
                    timeout=0.05,
                    idempotency_key="payment-1",
                    retry_policy=TIMEOUT_POLICY,
                )
        finally:
            released.set()

        # The error says what is unknown and what to do about it.
        assert caught.value.idempotency_key == "payment-1"
        assert "may already have happened" in str(caught.value)
        assert "resolve_idempotency" in str(caught.value)

        # Not retried, and the key is left unresolved rather than marked FAILED.
        assert event_types(execution).count("ToolStarted") == 1
        record = runtime.idempotency_record("payment-1")
        assert record.status is IdempotencyStatus.PENDING
        assert execution.status is ExecutionStatus.RECOVERY_REQUIRED

        with open(effects) as handle:
            assert handle.read().split() == ["card-1"]


def test_an_unresolved_ambiguous_timeout_refuses_to_run_the_tool_again(
    db_path, tmp_path
):
    """The second half of §6: an explicit resolution is the only way on."""
    effects = str(tmp_path / "effects.txt")
    released = threading.Event()

    def charges_then_ignores(to: str, cancel_token) -> dict:
        record_effect(effects, to)
        released.wait(10)
        return {"to": to}

    with Runtime(db_path) as runtime:
        runtime.register_tool(charges_then_ignores, name="charge")
        execution = runtime.start(goal="no second charge")

        try:
            with pytest.raises(AmbiguousTimeoutError):
                execution.call(
                    "charge", to="card-1", timeout=0.05, idempotency_key="payment-1"
                )
        finally:
            released.set()

        # "I checked the provider: it really was charged."
        execution.resolve_idempotency(
            "payment-1", "mark_completed", result={"to": "card-1"}
        )
        assert (
            execution.call("charge", to="card-1", idempotency_key="payment-1")
            == {"to": "card-1"}
        )
        with open(effects) as handle:
            assert handle.read().split() == ["card-1"]


def test_an_enforced_timeout_on_a_keyed_call_settles_a_known_outcome(db_path):
    """A timeout the runtime *can* prove stopped is not ambiguous.

    The coroutine really was cancelled at its await, so the side effect did not
    complete and the key is settled ``FAILED`` -- there is nothing unknown to
    preserve, and pretending otherwise would block the call forever.
    """

    async def slow(to: str) -> dict:
        await asyncio.sleep(30)
        return {"to": to}  # pragma: no cover - the deadline prevents it

    with Runtime(db_path) as runtime:
        runtime.register_tool(slow, name="send")
        execution = runtime.start(goal="a provably stopped call")

        with pytest.raises(ToolTimedOutError):
            execution.call("send", to="user-1", timeout=0.05, idempotency_key="email-1")

        record = runtime.idempotency_record("email-1")
        assert record.status is IdempotencyStatus.FAILED
        assert execution.status is not ExecutionStatus.RECOVERY_REQUIRED


def test_a_cancellation_on_a_keyed_call_leaves_the_key_unresolved(db_path):
    """Nobody said the effect did not happen, so the key stays PENDING."""
    started = threading.Event()

    def charges(to: str, cancel_token) -> dict:
        started.set()
        while not cancel_token.is_cancelled():
            time.sleep(0.005)
        return {"to": to}

    with Runtime(db_path) as runtime:
        runtime.register_tool(charges, name="charge")
        execution = runtime.start(goal="cancel a charge")

        errors: list[BaseException] = []

        def call_it() -> None:
            try:
                execution.call("charge", to="card-1", idempotency_key="payment-1")
            except BaseException as exc:  # noqa: BLE001 - handed back to the test
                errors.append(exc)

        caller = threading.Thread(target=call_it)
        caller.start()
        assert started.wait(2)
        execution.cancel("operator stop")
        caller.join(5)

        assert isinstance(errors[0], ToolCancelledError)
        record = runtime.idempotency_record("payment-1")
        assert record.status is IdempotencyStatus.PENDING

# ---------------------------------------------------------------------------
# §12: recovery distinguishes every reliability state
# ---------------------------------------------------------------------------


def test_recovery_classifies_a_scheduled_retry_as_retryable(db_path):
    """§12: a scheduled retry is unambiguous, so a restart can carry it on."""
    with Runtime(db_path) as runtime:
        runtime.journal.append_event(
            "exec_pending", "ExecutionStarted", {"goal": "died in the backoff"}
        )
        runtime.journal.append_event(
            "exec_pending", "ToolRequested",
            {"call_id": "call_x", "tool": "flaky", "arguments": {}},
        )
        runtime.journal.append_event(
            "exec_pending", "ToolStarted",
            {"call_id": "call_x", "tool": "flaky", "attempt": 1},
        )
        runtime.journal.append_event(
            "exec_pending", "ToolFailed",
            {"call_id": "call_x", "tool": "flaky", "attempt": 1, "error": {}},
        )
        runtime.journal.append_event(
            "exec_pending", "ToolRetryScheduled",
            {
                "call_id": "call_x", "tool": "flaky", "attempt": 2,
                "failed_attempt": 1, "delay": 0.1, "reason": "RETRYABLE",
                "error": {},
            },
        )

        info = runtime.recovery_info("exec_pending")
        assert info.recovery_state is RecoveryState.RETRYABLE
        assert info.has_pending_retries is True
        assert info.needs_resolution is False
        # A pending retry is a decision, not an incompleteness: the status
        # stays RUNNING rather than becoming RECOVERY_REQUIRED.
        assert info.status is ExecutionStatus.RUNNING


def test_recovery_keeps_a_cancelled_execution_cancelled(db_path):
    """§12/§4: cancelled stays cancelled, and nothing is resumed."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "ok", name="quick")
        execution = runtime.start(goal="cancel and restart")
        execution.cancel("operator stop")
        execution_id = execution.id

    with Runtime(db_path) as restarted:
        info = restarted.recovery_info(execution_id)
        assert info.recovery_state is RecoveryState.CANCELLED
        assert info.status is ExecutionStatus.CANCELLED
        assert info.has_pending_retries is False
        assert info.needs_resolution is False

        resumed = restarted.resume(execution_id)
        assert resumed.status is ExecutionStatus.CANCELLED
        # A cancelled execution takes no new work at all.
        with pytest.raises(InvalidStateTransitionError):
            resumed.call("quick")


def test_recovery_reports_a_terminal_timeout_as_its_own_kind(db_path):
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow")
        execution = runtime.start(goal="a timeout survives a restart")
        with pytest.raises(ToolTimedOutError):
            execution.call("slow", timeout=0.05)
        execution_id = execution.id

    with Runtime(db_path) as restarted:
        info = restarted.recovery_info(execution_id)
        assert len(info.timed_out_calls) == 1
        assert info.timed_out_calls[0].timeout == 0.05
        assert info.timed_out_calls[0].timeout_enforced is True
        # Not a failure, and not an ambiguity: the runtime knows what happened.
        assert info.recovery_state is RecoveryState.RUNNING
        assert info.needs_resolution is False


def test_recovery_reports_an_unenforced_timeout_as_an_ambiguity(db_path, tmp_path):
    """§12: an ambiguous side effect is ``RECOVERY_REQUIRED``, not a failure."""
    effects = str(tmp_path / "effects.txt")
    released = threading.Event()

    def charges_then_ignores(to: str, cancel_token) -> dict:
        record_effect(effects, to)
        released.wait(10)
        return {"to": to}

    with Runtime(db_path) as runtime:
        runtime.register_tool(charges_then_ignores, name="charge")
        execution = runtime.start(goal="the side effect may have happened")
        try:
            with pytest.raises(AmbiguousTimeoutError):
                execution.call(
                    "charge", to="card-1", timeout=0.05, idempotency_key="payment-1"
                )
        finally:
            released.set()
        execution_id = execution.id

    with Runtime(db_path) as restarted:
        info = restarted.recovery_info(execution_id)
        assert info.recovery_state is RecoveryState.RECOVERY_REQUIRED
        assert len(info.unenforced_timeouts) == 1
        assert info.unenforced_timeouts[0].timeout_enforced is False
        assert [r.idempotency_key for r in info.pending_idempotency] == ["payment-1"]

    with open(effects) as handle:
        assert handle.read().split() == ["card-1"]

# ---------------------------------------------------------------------------
# §13: checkpoints hold every reliability state
# ---------------------------------------------------------------------------


def test_a_checkpoint_before_a_timeout_tail_reconstructs_it_exactly(db_path):
    """§13's worked example: checkpoint @ N, then the reliability events."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "warm", name="warm")
        execution = runtime.start(goal="checkpoint then a timeout")

        execution.call("warm")
        checkpoint = execution.checkpoint()
        assert checkpoint.sequence == execution.last_event_sequence

        runtime.register_tool(async_fails_once(failures=1, value="done"), name="flaky")
        execution.call("flaky", timeout=0.05, retry_policy=TIMEOUT_POLICY)
        execution.complete()
        execution_id = execution.id

        # checkpoint + tail == folding the whole journal, exactly.
        assert runtime.recover_state(execution_id) == runtime.reconstruct_state(
            execution_id
        )


def test_a_checkpoint_after_a_cancellation_reconstructs_cancellation(db_path):
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "ok", name="quick")
        execution = runtime.start(goal="checkpoint the cancellation")
        execution.cancel("stop")
        execution.checkpoint()
        execution_id = execution.id

    with Runtime(db_path) as restarted:
        recovered = restarted.recover_state(execution_id)
        assert recovered == restarted.reconstruct_state(execution_id)
        assert recovered.status is ExecutionStatus.CANCELLED


def test_a_checkpointed_timeout_state_round_trips(db_path):
    """The call's deadline and mode survive the JSON round trip."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow")
        execution = runtime.start(goal="round trip the timeout")
        with pytest.raises(ToolTimedOutError):
            execution.call("slow", timeout=0.05)
        stored = execution.checkpoint()

        with Runtime(db_path) as restarted:
            recovered = restarted.checkpoints.get(execution.id, stored.sequence)
            call = recovered.state.tool_calls[0]
            assert call.status is ToolCallStatus.TIMED_OUT
            assert call.timeout == 0.05
            assert call.timeout_mode == "ASYNC"
            assert call.timeout_enforced is True


# ---------------------------------------------------------------------------
# §14 / §15: replay reproduces every reliability event without running anything
# ---------------------------------------------------------------------------


def test_a_timeout_retried_into_a_success_replays_to_the_same_state(db_path):
    """Scenario 3: timeout -> retry -> success -> replay -> same final state."""
    with Runtime(db_path) as runtime:
        tool = async_fails_once(failures=1, value="settled")
        runtime.register_tool(tool, name="flaky_slow")
        execution = runtime.start(goal="timeout, retry, success")
        execution.call("flaky_slow", timeout=0.05, retry_policy=TIMEOUT_POLICY)
        execution.complete()
        execution_id = execution.id

    with Runtime(db_path) as runtime:
        result = runtime.replay(execution_id)

        assert result.matched
        assert result.final_state == result.original_state
        # The replay reproduced the timeout, the retry and the success.
        assert result.retries_replayed == 1
        # And it read the timeout back rather than inventing one.
        call = result.final_state.tool_calls[0]
        assert call.attempts[0].status is ToolCallStatus.TIMED_OUT
        assert call.attempts[0].error["timeout"] == 0.05


def test_a_replayed_timeout_costs_no_time_and_runs_no_tool(db_path):
    """§14: a recorded timeout must not be waited for again.

    The replaying runtime registers no tool at all, so it could not enforce the
    deadline even if it tried; the elapsed-time assertion is the backup.
    """
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow")
        execution = runtime.start(goal="a recorded timeout")
        with pytest.raises(ToolTimedOutError):
            execution.call("slow", timeout=0.05)
        execution_id = execution.id

    with Runtime(db_path) as runtime:
        started = time.perf_counter()
        result = runtime.replay(execution_id)
        elapsed = time.perf_counter() - started

        assert result.matched
        assert elapsed < 1.0
        assert result.final_state.tool_calls[0].status is ToolCallStatus.TIMED_OUT
        assert result.final_state.tool_calls[0].timeout == 0.05
        assert result.journal_unchanged is True


def test_a_cancelled_execution_replays_to_cancelled(db_path):
    """§15: ToolStarted -> ToolCancelled -> ExecutionCancelled, reproduced."""
    ran: list[str] = []

    def worker(cancel_token) -> str:
        ran.append("ran")
        while not cancel_token.is_cancelled():
            time.sleep(0.005)
        return "stopped"

    with Runtime(db_path) as runtime:
        runtime.register_tool(worker, name="worker")
        execution = runtime.start(goal="cancel then replay")

        errors: list[BaseException] = []

        def call_it() -> None:
            try:
                execution.call("worker")
            except BaseException as exc:  # noqa: BLE001 - handed back to the test
                errors.append(exc)

        caller = threading.Thread(target=call_it)
        caller.start()
        for _ in range(400):
            if ran:
                break
            time.sleep(0.005)
        assert ran, "the tool never started"
        execution.cancel("operator stop")
        caller.join(5)
        execution_id = execution.id

    with Runtime(db_path) as runtime:
        # Re-registered on purpose: a replay must not reach it, and the counter
        # staying at one is the proof.
        runtime.register_tool(worker, name="worker")
        result = runtime.replay(execution_id)

        assert result.matched
        assert result.final_state.status is ExecutionStatus.CANCELLED
        assert result.final_state.tool_calls[0].status is ToolCallStatus.CANCELLED
        assert ran == ["ran"]
        # The recorded cancellation is a fact to reproduce, not one to re-apply.
        assert [str(e.event_type) for e in runtime.get_events(execution_id)].count(
            "ToolCancelled"
        ) == 1

# ---------------------------------------------------------------------------
# §19: the invariants, stated directly
# ---------------------------------------------------------------------------


def test_a_replay_of_every_reliability_event_reproduces_the_same_state(db_path):
    """One history containing a timeout, a retry, a failure and a success."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "warm", name="warm")
        runtime.register_tool(async_fails_once(failures=1, value="ok"), name="flaky")
        runtime.register_tool(always_fails(), name="broken")

        execution = runtime.start(goal="everything at once")
        execution.call("warm")
        execution.call("flaky", timeout=0.05, retry_policy=TIMEOUT_POLICY)
        with pytest.raises(ToolInvocationError):
            execution.call("broken")
        execution.complete()
        execution_id = execution.id

    with Runtime(db_path) as runtime:
        result = runtime.replay(execution_id)
        assert result.matched
        assert result.final_state == runtime.reconstruct_state(execution_id)
        statuses = [call.status for call in result.final_state.tool_calls]
        assert statuses == [
            ToolCallStatus.COMPLETED,
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
        ]


def test_replaying_twice_is_stable_across_reliability_events(db_path):
    """A replay is a pure function of the journal -- twice is the same once."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_fails_once(failures=1, value="ok"), name="flaky")
        execution = runtime.start(goal="replay twice")
        execution.call("flaky", timeout=0.05, retry_policy=TIMEOUT_POLICY)
        execution_id = execution.id

    with Runtime(db_path) as runtime:
        first = runtime.replay(execution_id)
        second = runtime.replay(execution_id)
        assert first.final_state == second.final_state
        assert first.retries_replayed == second.retries_replayed


def test_a_cancelled_execution_never_silently_resumes_retrying(db_path):
    """§19: cancellation is final, across a restart and a replay.

    The call is left in a state that *looks* resumable -- a ``ToolRetryScheduled``
    is in the journal behind it -- and the reducer still refuses to continue it.
    """

    class CancelOnFirstWait(RecordingSleeper):
        def interruptible_sleep(self, delay, token=None):
            super().interruptible_sleep(delay, token)
            token.cancel("operator stop")
            return True

    with Runtime(db_path, sleeper=CancelOnFirstWait()) as runtime:
        runtime.register_tool(always_fails(), name="flaky")
        execution = runtime.start(goal="cancel, then look for work to resume")

        with pytest.raises(ToolCancelledError):
            execution.call(
                "flaky",
                retry_policy=RetryPolicy(max_attempts=5, initial_delay=0.01),
            )
        call_id = execution.tool_calls[0].call_id
        execution_id = execution.id

    with Runtime(db_path) as restarted:
        info = restarted.recovery_info(execution_id)
        # Whatever the history contains, there is nothing to carry on.
        assert info.has_pending_retries is False
        assert info.pending_retries == ()
        # And the scheduled retry is still in the journal -- visible, not
        # silently dropped -- it simply is not a pending retry any more.
        assert any(
            str(e.event_type) == "ToolRetryScheduled"
            for e in restarted.get_events(execution_id)
        )

        resumed = restarted.resume(execution_id)
        assert resumed.pending_retries == ()
        with pytest.raises(InvalidStateTransitionError):
            resumed.continue_pending_retry(call_id)


def test_recovery_is_deterministic_for_the_same_journal_and_store(db_path, tmp_path):
    """§19: same journal + same persisted idempotency state -> same recovery."""
    effects = str(tmp_path / "effects.txt")
    released = threading.Event()

    def charges_then_ignores(to: str, cancel_token) -> dict:
        record_effect(effects, to)
        released.wait(10)
        return {"to": to}

    with Runtime(db_path) as runtime:
        runtime.register_tool(charges_then_ignores, name="charge")
        execution = runtime.start(goal="two recoveries, one answer")
        try:
            with pytest.raises(AmbiguousTimeoutError):
                execution.call(
                    "charge", to="card-1", timeout=0.05, idempotency_key="payment-1"
                )
        finally:
            released.set()
        execution_id = execution.id

    with Runtime(db_path) as restarted:
        first = restarted.recovery_info(execution_id)
        second = restarted.recovery_info(execution_id)
        assert first.state == second.state
        assert first.recovery_state is second.recovery_state
        assert first.state == restarted.reconstruct_state(execution_id)


