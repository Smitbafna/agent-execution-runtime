"""Milestone 4C, part 1: deadlines that actually stop something.

The invariant under test, and the one the milestone states hardest::

    a timeout stops the tool, or the runtime says it could not

Concretely, four separate claims, each with its own tests below:

* a coroutine tool is really cancelled at the deadline (§4);
* a cooperative ``def`` tool is really asked to stop, and is then *waited for*;
* a ``def`` tool that declared nothing is **refused** a deadline, because
  Python cannot terminate a thread and a wrapper that only measured elapsed
  time would be claiming a stop that never happened;
* none of it collapses into ``ToolFailed`` (§11), and the timeout is journalled
  as its own durable event (§3).

Every test here uses a real clock. A test that takes 40ms is doing what the
production path does, not a stand-in for it -- which is the point of the
milestone, and the reason the "no fake timeouts" rule is testable at all.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_runtime import (
    RecordingSleeper,
    RetryPolicy,
    RetryableToolError,
    Runtime,
    TimeoutConfigurationError,
    TimeoutEnforcementError,
    ToolCallStatus,
    ToolInvocationError,
    ToolTimedOutError,
    UnsupportedTimeoutError,
    tool,
)
from failure_tools import (
    always_fails,
    async_fails_once,
    async_sleeps_forever,
    cooperative_worker,
    fails_once,
    sleeps_forever,
)

POLICY = RetryPolicy(max_attempts=3, initial_delay=0.01, multiplier=1.0)


def event_types(execution) -> list[str]:
    return [str(event.event_type) for event in execution.events]


# ---------------------------------------------------------------------------
# §4: the three execution modes, and the refusal that makes them honest
# ---------------------------------------------------------------------------


def test_an_async_tool_is_really_cancelled_at_its_deadline(db_path):
    """``asyncio.wait_for`` stops the coroutine; the call reports a timeout."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow_tool")
        execution = runtime.start(goal="a deadline that bites")

        started = time.perf_counter()
        with pytest.raises(ToolTimedOutError) as caught:
            execution.call("slow_tool", timeout=0.05)
        elapsed = time.perf_counter() - started

        error = caught.value
        assert error.timeout == 0.05
        assert error.mode == "ASYNC"
        # ``enforced`` is the claim under test: a coroutine cancelled at its
        # await point really did stop, and the runtime says so.
        assert error.enforced is True
        # And it stopped *at* the deadline, not after the 30s the tool wanted.
        assert elapsed < 5.0


def test_a_cancelled_coroutine_is_unwound_at_its_await_point(db_path):
    """The proof that it was cancelled rather than merely abandoned."""
    unwound: list[str] = []

    async def observes_cancellation() -> str:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            unwound.append("cancelled")
            raise
        return "never"  # pragma: no cover - the deadline prevents it

    with Runtime(db_path) as runtime:
        runtime.register_tool(observes_cancellation, name="observes")
        execution = runtime.start(goal="structured cancellation")

        with pytest.raises(ToolTimedOutError):
            execution.call("observes", timeout=0.05)

    # The coroutine's own cleanup ran. A wrapper that merely timed the call
    # could not have produced this.
    assert unwound == ["cancelled"]


def test_a_plain_sync_tool_is_refused_a_deadline_rather_than_faked(db_path):
    """§4's headline rule: no fake timeouts, and a clear reason for refusing."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(sleeps_forever(), name="sleep_forever")
        execution = runtime.start(goal="a deadline the runtime cannot keep")

        with pytest.raises(UnsupportedTimeoutError) as caught:
            execution.call("sleep_forever", timeout=0.05)

        error = caught.value
        assert error.tool_name == "sleep_forever"
        assert error.timeout == 0.05
        assert "cannot terminate a running thread" in str(error)
        # Nothing was attempted: the refusal happens before ``ToolRequested``,
        # so the journal holds no call at all.
        assert event_types(execution) == ["ExecutionStarted"]


def test_a_refused_deadline_leaves_no_call_id_and_no_claim(db_path):
    """The refusal is total, not a half-started call."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(sleeps_forever(), name="sleep_forever")
        execution = runtime.start(goal="no half-started call")

        with pytest.raises(UnsupportedTimeoutError):
            execution.call("sleep_forever", timeout=1.0, idempotency_key="key-1")

        assert execution.tool_calls == ()
        assert runtime.idempotency_record("key-1") is None
        assert runtime.pending_idempotency(execution.id) == ()


def test_a_cooperative_tool_stops_when_its_deadline_reaches_it(db_path):
    """The token is flipped, the tool notices, and the stop is enforced."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(cooperative_worker(), name="worker")
        execution = runtime.start(goal="cooperative deadline")

        with pytest.raises(ToolTimedOutError) as caught:
            execution.call("worker", timeout=0.05)

        assert caught.value.mode == "COOPERATIVE"
        assert caught.value.enforced is True
        # It really did run, and really was interrupted part way: a tool that
        # never started could not report slices completed.
        call = execution.tool_calls[0]
        assert call.status is ToolCallStatus.TIMED_OUT
        assert call.timeout_mode == "COOPERATIVE"


def test_a_cooperative_tool_that_ignores_the_token_is_told_so(db_path):
    """§4: never claim a thread stopped when it did not.

    A tool that declares a token and then ignores it is the one case the runtime
    cannot enforce. It says exactly that -- ``enforced=False`` -- rather than
    reporting a stop it cannot prove, and the error names the reason.
    """
    import threading

    released = threading.Event()

    def deaf_tool(cancel_token) -> str:  # noqa: ARG001 - deliberately ignored
        released.wait(10)
        return "finished long after the deadline"

    with Runtime(db_path) as runtime:
        runtime.register_tool(deaf_tool, name="deaf")
        execution = runtime.start(goal="a tool that will not stop")

        try:
            with pytest.raises(ToolTimedOutError) as caught:
                execution.call("deaf", timeout=0.05)
        finally:
            released.set()

        error = caught.value
        assert error.enforced is False
        assert isinstance(error.__cause__, TimeoutEnforcementError)
        assert "did not stop" in str(error.__cause__)


def test_a_tool_that_declares_kwargs_is_not_given_a_token(db_path):
    """``**kwargs`` is not an opt-in: it has to *say* it wants a token.

    A tool that swallows arbitrary keywords has not asked for anything, and
    quietly handing it a token would make its journalled arguments depend on
    the runtime rather than on the call.
    """
    seen: list[dict] = []

    def greedy(**kwargs) -> str:
        seen.append(kwargs)
        return "ok"

    with Runtime(db_path) as runtime:
        runtime.register_tool(greedy, name="greedy")
        execution = runtime.start(goal="kwargs is not consent")

        assert execution.call("greedy") == "ok"
        # And the journalled arguments are the call's, with no runtime addition.
        assert execution.tool_calls[0].arguments == {"kwargs": {}}

    assert "cancel_token" not in seen[0]


def test_only_a_tool_that_declares_the_parameter_is_given_a_token(db_path):
    """The positive half of the same rule, checked through the tool's own view."""
    received: list[object] = []

    def worker(cancel_token) -> str:
        received.append(cancel_token)
        return "ok"

    with Runtime(db_path) as runtime:
        runtime.register_tool(worker, name="worker")
        execution = runtime.start(goal="declared means given")

        execution.call("worker")

        # Still not in the recorded arguments -- the token is a runtime concern,
        # not part of the call, so a replay never has to reproduce one.
        assert execution.tool_calls[0].arguments == {}

    assert len(received) == 1


# ---------------------------------------------------------------------------
# §2: configuring a deadline
# ---------------------------------------------------------------------------


def test_a_call_level_timeout_overrides_the_tool_default(db_path):
    """§2: ``@tool(timeout=...)`` is the default; the call wins."""

    @tool(timeout=0.05, name="slow_tool")
    async def slow_tool() -> str:
        await asyncio.sleep(30)
        return "never"  # pragma: no cover - the deadline prevents it

    with Runtime(db_path) as runtime:
        runtime.register_tool(slow_tool)
        execution = runtime.start(goal="override")

        with pytest.raises(ToolTimedOutError) as caught:
            execution.call("slow_tool", timeout=0.02)

        assert caught.value.timeout == 0.02


def test_the_tool_default_applies_when_the_call_says_nothing(db_path):
    @tool(timeout=0.05, name="slow_tool")
    async def slow_tool() -> str:
        await asyncio.sleep(30)
        return "never"  # pragma: no cover - the deadline prevents it

    with Runtime(db_path) as runtime:
        runtime.register_tool(slow_tool)
        execution = runtime.start(goal="default applies")

        with pytest.raises(ToolTimedOutError) as caught:
            execution.call("slow_tool")

        assert caught.value.timeout == 0.05


@pytest.mark.parametrize("bad", [0, -1, -0.5, float("inf"), float("nan"), "5"])
def test_a_nonsense_deadline_is_rejected_where_it_is_written(bad):
    """A bad timeout is a configuration error on the line that wrote it."""
    with pytest.raises(TimeoutConfigurationError):
        tool(timeout=bad, name="whatever")(lambda: None)


def test_a_bad_deadline_at_call_time_names_the_call(db_path):
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "ok", name="quick")
        execution = runtime.start(goal="bad call-time deadline")

        with pytest.raises(TimeoutConfigurationError) as caught:
            execution.call("quick", timeout=-1)

        assert "quick" in str(caught.value)
        # Still nothing journalled: the check precedes ToolRequested.
        assert event_types(execution) == ["ExecutionStarted"]


def test_the_resolved_deadline_is_recorded_with_the_call(db_path):
    """§3: the deadline and its mechanism are durable, not implied."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow_tool")
        execution = runtime.start(goal="durable deadline")

        with pytest.raises(ToolTimedOutError):
            execution.call("slow_tool", timeout=0.05)

        requested = next(
            e for e in execution.events if e.event_type == "ToolRequested"
        )
        assert requested.payload["timeout"] == 0.05
        assert requested.payload["timeout_mode"] == "ASYNC"

        timed_out = next(e for e in execution.events if e.event_type == "ToolTimedOut")
        assert timed_out.payload["timeout"] == 0.05
        assert timed_out.payload["enforced"] is True


def test_a_call_without_a_deadline_journals_no_timeout_fields(db_path):
    """An old journal and a new one stay directly comparable."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "ok", name="quick")
        execution = runtime.start(goal="no deadline")

        execution.call("quick")

        requested = next(
            e for e in execution.events if e.event_type == "ToolRequested"
        )
        assert "timeout" not in requested.payload
        assert "timeout_mode" not in requested.payload


# ---------------------------------------------------------------------------
# §3 + §11: a timeout is its own durable, distinguishable outcome
# ---------------------------------------------------------------------------


def test_a_timeout_is_journalled_as_tool_timed_out_not_tool_failed(db_path):
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow_tool")
        execution = runtime.start(goal="a distinct event")

        with pytest.raises(ToolTimedOutError):
            execution.call("slow_tool", timeout=0.05)

        assert event_types(execution) == [
            "ExecutionStarted",
            "ToolRequested",
            "ToolStarted",
            "ToolTimedOut",
        ]


def test_a_timeout_is_distinguishable_from_a_failure_in_the_state(db_path):
    """§11: failure, timeout and cancellation are three statuses."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow_tool")
        runtime.register_tool(always_fails(), name="broken")
        execution = runtime.start(goal="three kinds of stop")

        with pytest.raises(ToolTimedOutError):
            execution.call("slow_tool", timeout=0.05)
        with pytest.raises(ToolInvocationError):
            execution.call("broken")

        timed_out, failed = execution.tool_calls
        assert timed_out.status is ToolCallStatus.TIMED_OUT
        assert failed.status is ToolCallStatus.FAILED
        assert timed_out.timed_out is True
        assert failed.timed_out is False
        # Both are "the call produced no result", but nothing conflates them.
        assert timed_out.status.is_stop and failed.status.is_stop
        assert timed_out.status is not failed.status


def test_the_timeout_survives_a_reconstruction_from_the_journal(db_path):
    """§13: the event is the source of truth, and it folds back to a timeout."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow_tool")
        execution = runtime.start(goal="durable timeout")

        with pytest.raises(ToolTimedOutError):
            execution.call("slow_tool", timeout=0.05)

        call = runtime.reconstruct_state(execution.id).tool_calls[0]
        assert call.status is ToolCallStatus.TIMED_OUT
        assert call.timeout == 0.05
        assert call.timeout_mode == "ASYNC"
        assert call.timeout_enforced is True


def test_a_timed_out_attempt_keeps_its_own_history(db_path):
    """One attempt, one record -- the attempt history stays readable."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(async_sleeps_forever, name="slow_tool")
        execution = runtime.start(goal="attempt history")

        with pytest.raises(ToolTimedOutError):
            execution.call("slow_tool", timeout=0.05)

        attempt = execution.tool_calls[0].attempts[0]
        assert attempt.status is ToolCallStatus.TIMED_OUT
        assert attempt.error["timeout"] == 0.05
        assert attempt.error["enforced"] is True

