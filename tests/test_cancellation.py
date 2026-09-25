"""Milestone 4C, part 2: cancellation, and the three things it must not be.

The invariant under test::

    ToolStarted -> execution.cancel() -> ToolCancelled -> ExecutionCancelled

and, just as importantly, what cancellation is *not*:

* it is not a failure -- a cancelled attempt gets its own status and event (§11);
* it is not a retry -- no policy, however generous, re-issues it (§10);
* it does not sleep out a pending backoff first -- the wait is interrupted (§9).

The cooperative token (§8) is exercised through a real tool on a real thread,
because a token that is only ever tested in-process would not prove the thing
that matters: that a *different* thread can interrupt a *blocked* one.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent_runtime import (
    CancellationToken,
    ExecutionStatus,
    InvalidStateTransitionError,
    RecordingSleeper,
    RetryPolicy,
    RetryableToolError,
    Runtime,
    ToolCallStatus,
    ToolCancelledError,
    ToolInvocationError,
    ToolTimedOutError,
)
from failure_tools import always_fails, cooperative_worker


#: A backoff long enough that "did not sleep the full delay" is a real claim.
LONG_BACKOFF = RetryPolicy(max_attempts=3, initial_delay=30.0, multiplier=1.0)


def event_types(execution) -> list[str]:
    return [str(event.event_type) for event in execution.events]


# ---------------------------------------------------------------------------
# §8: the cooperative token
# ---------------------------------------------------------------------------


def test_a_token_starts_uncancelled_and_can_be_cancelled():
    token = CancellationToken()
    assert token.is_cancelled() is False
    assert token.reason is None

    assert token.cancel("stop") is True
    assert token.is_cancelled() is True
    assert token.reason == "stop"


def test_cancelling_a_token_twice_is_a_no_op():
    token = CancellationToken()
    assert token.cancel("first") is True
    # The second call reports that it did nothing, so a second ``cancel()`` on
    # the execution cannot journal a second cancellation.
    assert token.cancel("second") is False
    assert token.reason == "first"


def test_raise_if_cancelled_raises_only_after_cancellation():
    token = CancellationToken()
    token.raise_if_cancelled()  # a no-op while live

    token.cancel("user asked")
    with pytest.raises(ToolCancelledError) as caught:
        token.raise_if_cancelled()
    assert caught.value.reason == "user asked"


def test_a_token_is_readable_from_another_thread():
    """The whole point of the token: the reader is not the canceller."""
    token = CancellationToken()
    observed: list[bool] = []
    ready = threading.Event()

    def reader() -> None:
        ready.set()
        while not token.is_cancelled():
            time.sleep(0.001)
        observed.append(token.is_cancelled())

    thread = threading.Thread(target=reader)
    thread.start()
    ready.wait(2)
    token.cancel("from the main thread")
    thread.join(2)

    assert observed == [True]


def test_on_cancel_fires_at_most_once_and_never_misses_the_edge():
    token = CancellationToken()
    calls: list[str] = []
    token.on_cancel(lambda: calls.append("first"))

    token.cancel("now")
    # Registering after the fact still fires: a listener cannot miss the edge.
    token.on_cancel(lambda: calls.append("late"))

    assert calls == ["first", "late"]
# ---------------------------------------------------------------------------
# §7: execution.cancel() -- durable, and shaped as the milestone says
# ---------------------------------------------------------------------------


def test_cancelling_an_execution_journals_tool_cancelled_then_execution_cancelled(
    db_path,
):
    """§7's exact sequence, and nothing collapsed into a failure."""
    started = threading.Event()

    def worker(cancel_token) -> str:  # noqa: ARG001 - the token is flipped for it
        started.set()
        while not cancel_token.is_cancelled():
            time.sleep(0.005)
        return "stopped"

    with Runtime(db_path) as runtime:
        runtime.register_tool(worker, name="worker")
        execution = runtime.start(goal="cancel me")

        errors: list[BaseException] = []

        def call_it() -> None:
            try:
                execution.call("worker")
            except BaseException as exc:  # noqa: BLE001 - handed back to the test
                errors.append(exc)

        caller = threading.Thread(target=call_it)
        caller.start()
        assert started.wait(2)

        execution.cancel("operator stopped it")
        caller.join(5)

        assert event_types(execution) == [
            "ExecutionStarted",
            "ToolRequested",
            "ToolStarted",
            "ToolCancelled",
            "ExecutionCancelled",
        ]
        assert execution.status is ExecutionStatus.CANCELLED
        assert len(errors) == 1
        assert isinstance(errors[0], ToolCancelledError)


def test_the_cancellation_reason_is_recorded(db_path):
    """The reason reaches both events that carry it."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "ok", name="quick")
        execution = runtime.start(goal="record the reason")

        execution.cancel("the queue is being drained")

        # No call was open, so there is no ``ToolCancelled`` to attach it to --
        # the execution-level event is the one that carries the reason here.
        assert event_types(execution) == ["ExecutionStarted", "ExecutionCancelled"]
        finished = next(
            e for e in execution.events if e.event_type == "ExecutionCancelled"
        )
        assert finished.payload["result"] == "the queue is being drained"


def test_the_reason_reaches_tool_cancelled_when_a_call_is_open(db_path):
    started = threading.Event()

    def worker(cancel_token) -> str:  # noqa: ARG001
        started.set()
        while not cancel_token.is_cancelled():
            time.sleep(0.005)
        return "stopped"

    with Runtime(db_path) as runtime:
        runtime.register_tool(worker, name="worker")
        execution = runtime.start(goal="record the reason on the call")

        caller = threading.Thread(
            target=_swallow, args=(lambda: execution.call("worker"),)
        )
        caller.start()
        assert started.wait(2)

        execution.cancel("the queue is being drained")
        caller.join(5)

        cancelled = next(e for e in execution.events if e.event_type == "ToolCancelled")
        assert cancelled.payload["reason"] == "the queue is being drained"


def _swallow(callable_) -> None:
    """Run ``callable_``, discarding whatever it raises."""
    try:
        callable_()
    except BaseException:  # noqa: BLE001 - the test asserts on the journal
        pass


# ---------------------------------------------------------------------------
# §9: cancellation interrupts the retry backoff
# ---------------------------------------------------------------------------


def test_a_cancellation_during_a_backoff_ends_the_wait_immediately(db_path):
    """§9: a 30-second backoff must not be slept out after ``cancel()``.

    The delay is 30 seconds, so a test that returns promptly can only be
    reporting a real interruption -- there is no way to pass this by waiting.
    """
    entered = threading.Event()

    class InterruptingSleeper(RecordingSleeper):
        """Records the delay, then waits on the token instead of sleeping."""

        def interruptible_sleep(self, delay, token=None):
            super().interruptible_sleep(delay, token)
            entered.set()
            return bool(token is not None and token.wait(delay))

    with Runtime(db_path, sleeper=InterruptingSleeper()) as runtime:
        runtime.register_tool(always_fails(), name="flaky")
        execution = runtime.start(goal="cancel during a backoff")

        errors: list[BaseException] = []

        def call_it() -> None:
            try:
                execution.call("flaky", retry_policy=LONG_BACKOFF)
            except BaseException as exc:  # noqa: BLE001 - handed back to the test
                errors.append(exc)

        caller = threading.Thread(target=call_it)
        caller.start()

        assert entered.wait(2), "the retry backoff never started"
        started = time.perf_counter()
        execution.cancel("stop the queue")
        caller.join(5)
        elapsed = time.perf_counter() - started

        assert elapsed < 5.0, "the backoff was slept out instead of interrupted"
        assert len(errors) == 1
        assert isinstance(errors[0], ToolCancelledError)


def test_a_cancelled_retry_schedules_no_further_attempt(db_path):
    """The retry was journalled, then cancelled: the call ends CANCELLED.

    Both facts are durable. The reducers read the second as the final word, so
    the call settles ``CANCELLED`` with the scheduled attempt never started.
    """

    class InterruptSleeper(RecordingSleeper):
        def interruptible_sleep(self, delay, token=None):
            super().interruptible_sleep(delay, token)
            token.cancel("cancelled while waiting")
            return True

    with Runtime(db_path, sleeper=InterruptSleeper()) as runtime:
        runtime.register_tool(always_fails(), name="flaky")
        execution = runtime.start(goal="cancel the scheduled retry")

        with pytest.raises(ToolCancelledError):
            execution.call(
                "flaky", retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.01)
            )

        call = execution.tool_calls[0]
        assert call.status is ToolCallStatus.CANCELLED
        # The attempt the cancelled decision named never started, and nothing is
        # left pending for a resume to pick up.
        assert call.attempt == 1
        assert call.pending_retry is None
        assert execution.pending_retries == ()


def test_a_recording_sleeper_still_records_every_interrupted_wait(db_path):
    """Interruption must not hide the schedule from the tests that assert it."""

    class InterruptSleeper(RecordingSleeper):
        def interruptible_sleep(self, delay, token=None):
            super().interruptible_sleep(delay, token)
            token.cancel("stop")
            return True

    sleeper = InterruptSleeper()
    with Runtime(db_path, sleeper=sleeper) as runtime:
        runtime.register_tool(always_fails(), name="flaky")
        execution = runtime.start(goal="the schedule is still recorded")

        with pytest.raises(ToolCancelledError):
            execution.call(
                "flaky", retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.5)
            )

    assert sleeper.delays == (0.5,)


# ---------------------------------------------------------------------------
# §10: a cancellation is never a retry
# ---------------------------------------------------------------------------


def test_no_policy_retries_a_cancellation(db_path):
    """§10: even a maximally generous policy re-issues nothing.

    A cancellation is a decision by the application; a policy that retried it
    would be repeating a decision nobody asked to repeat.
    """

    class CancelOnceSleeper(RecordingSleeper):
        """Cancels on the first wait, so the tool stops after attempt 1."""

        def interruptible_sleep(self, delay, token=None):
            super().interruptible_sleep(delay, token)
            token.cancel("stop after the first failure")
            return True

    generous = RetryPolicy(
        max_attempts=10, initial_delay=0.01, multiplier=1.0, retry_on_unknown=True
    )

    with Runtime(db_path, sleeper=CancelOnceSleeper()) as runtime:
        runtime.register_tool(always_fails(), name="flaky")
        execution = runtime.start(goal="a generous policy changes nothing")

        with pytest.raises(ToolCancelledError):
            execution.call("flaky", retry_policy=generous)

        recorded = [str(e.event_type) for e in execution.events]
        assert recorded.count("ToolCancelled") == 1
        # One attempt only, and no second ToolStarted for the attempt the
        # cancelled decision had scheduled.
        assert recorded.count("ToolStarted") == 1


def test_a_cancelled_call_is_reported_as_cancelled_not_failed(db_path):
    """§11 again, from the caller's side: the exception says which."""

    class CancelOnFirstWait(RecordingSleeper):
        def interruptible_sleep(self, delay, token=None):
            super().interruptible_sleep(delay, token)
            token.cancel("stop")
            return True

    with Runtime(db_path, sleeper=CancelOnFirstWait()) as runtime:
        runtime.register_tool(always_fails(), name="flaky")
        execution = runtime.start(goal="which error is it")

        with pytest.raises(ToolCancelledError) as caught:
            execution.call(
                "flaky", retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.01)
            )

        # Still a ToolInvocationError for anyone catching the older, broader
        # type -- but the specific type is what distinguishes it.
        assert isinstance(caught.value, ToolInvocationError)
        assert not isinstance(caught.value, ToolTimedOutError)
        assert caught.value.attempts == 1


def test_cancelling_before_a_call_starts_it_at_all(db_path):
    """An already-cancelled execution makes no new call attempt."""
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "ran", name="quick")
        execution = runtime.start(goal="cancel first")
        execution.cancel("do not start")

        with pytest.raises(InvalidStateTransitionError):
            execution.call("quick")

        assert execution.tool_calls == ()


# ---------------------------------------------------------------------------
# Two threads, one connection
# ---------------------------------------------------------------------------


def test_the_journal_survives_two_threads_writing_at_once(db_path):
    """The storage guarantee ``cancel()`` depends on.

    ``execution.cancel()`` is documented as callable from another thread while
    a tool is running, so two threads reach the journal through one SQLite
    connection. Without serialization the second ``BEGIN`` fails outright, and
    the caller sees a storage error instead of a cancelled execution -- which is
    exactly the failure mode this test exists to rule out.
    """
    barrier = threading.Barrier(2)
    written: list[str] = []

    def writer(tag: str) -> None:
        barrier.wait()  # release both threads at the same instant
        with Runtime(db_path) as runtime:
            runtime.register_tool(lambda: "ok", name="quick")
            execution = runtime.start(goal=f"concurrent {tag}", execution_id=f"exec_{tag}")
            execution.call("quick")
            written.append(tag)

    threads = [threading.Thread(target=writer, args=(tag,)) for tag in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert sorted(written) == ["a", "b"]

    # Both histories are intact, gapless, and independently reconstructable.
    with Runtime(db_path) as runtime:
        for tag in ("a", "b"):
            events = runtime.get_events(f"exec_{tag}")
            assert [e.sequence for e in events] == list(range(1, len(events) + 1))
            assert str(events[-1].event_type) == "ToolCompleted"
            assert runtime.reconstruct_state(f"exec_{tag}") == runtime.recover_state(
                f"exec_{tag}"
            )


def test_a_transaction_nested_inside_another_joins_it(db_path):
    """Nesting is a join, not a second ``BEGIN``.

    The store's per-thread depth exists so a caller that opens a transaction
    inside one does not blow up; the outer block still owns the commit, so a
    rollback in the inner one must not half-commit the outer one either.
    """
    with Runtime(db_path) as runtime:
        with runtime.store.transaction() as conn:
            conn.execute(
                "INSERT INTO events (event_id, execution_id, sequence, event_type,"
                " payload, timestamp) VALUES ('e1', 'exec_n', 1, 'ExecutionStarted',"
                " '{}', '2024-01-01T00:00:00+00:00')"
            )
            with runtime.store.transaction() as inner:
                inner.execute(
                    "INSERT INTO events (event_id, execution_id, sequence, event_type,"
                    " payload, timestamp) VALUES ('e2', 'exec_n', 2, 'ExecutionStarted',"
                    " '{}', '2024-01-01T00:00:01+00:00')"
                )

        assert runtime.journal.count_events("exec_n") == 2

    # And a failure anywhere inside rolls the whole thing back.
    with Runtime(db_path) as runtime:
        with pytest.raises(ValueError):
            with runtime.store.transaction() as conn:
                conn.execute(
                    "INSERT INTO events (event_id, execution_id, sequence, event_type,"
                    " payload, timestamp) VALUES ('e3', 'exec_n', 3, 'ExecutionStarted',"
                    " '{}', '2024-01-01T00:00:02+00:00')"
                )
                with runtime.store.transaction():
                    raise ValueError("boom")

        assert runtime.journal.count_events("exec_n") == 2

def test_cancelling_twice_journals_one_cancellation(db_path):
    with Runtime(db_path) as runtime:
        runtime.register_tool(cooperative_worker(), name="worker")
        execution = runtime.start(goal="cancel twice")

        execution.cancel("first")
        execution.cancel("second")

        assert event_types(execution).count("ExecutionCancelled") == 1
        assert execution.status is ExecutionStatus.CANCELLED


def test_cancelling_a_finished_execution_is_refused(db_path):
    with Runtime(db_path) as runtime:
        runtime.register_tool(lambda: "ok", name="quick")
        execution = runtime.start(goal="already done")
        execution.call("quick")
        execution.complete()

        with pytest.raises(InvalidStateTransitionError):
            execution.cancel("too late")


def test_a_failing_listener_cannot_break_the_cancellation():
    """One bad callback must not leave the flag unset."""

    def explode() -> None:
        raise RuntimeError("listener bug")

    token = CancellationToken()
    token.on_cancel(explode)
    token.cancel("regardless")

    assert token.is_cancelled() is True
