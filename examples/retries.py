"""Milestone 4A example: retries, attempts and a crash in the middle of one.

    python examples/retries.py [db_path]

(Without an argument it uses a throwaway database in a temp directory.)

Five scenes, in order:

1. one logical call, three attempts -- the journal and the state;
2. what the backoff would have cost, without spending it (a fake sleeper);
3. the three failure classifications, side by side;
4. a real child process that dies during the backoff, and the new process that
   recovers the scheduled retry and carries it on;
5. replaying all of it without executing a tool again.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running this file directly from a checkout, without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_runtime import (  # noqa: E402
    PermanentToolError,
    RecordingSleeper,
    RetryPolicy,
    RetryableToolError,
    Runtime,
    ToolInvocationError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Three attempts, 0.1 doubling, capped at 10 seconds. Five numbers, nothing more.
POLICY = RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=2.0, max_delay=10.0)

#: A counter only a real invocation moves: the proof that a replay ran nothing.
REAL_CALLS = {"fetch_data": 0}


def scene_one_call_three_attempts(db_path: str, sleeper: RecordingSleeper) -> str:
    """Scene 1: fail, fail, succeed -- one call id, three attempts."""
    print("=" * 72)
    print("1. One logical call, three attempts")
    print("=" * 72)

    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool(name="fetch_data", retry_policy=POLICY)
        def fetch_data(url: str) -> dict:
            """Fails twice with a *retryable* error, then succeeds."""
            REAL_CALLS["fetch_data"] += 1
            if REAL_CALLS["fetch_data"] < 3:
                raise RetryableToolError(
                    f"upstream unavailable (call {REAL_CALLS['fetch_data']})"
                )
            return {"url": url, "rows": 42}

        execution = runtime.start(goal="Fetch the data")

        print(f"  execution {execution.id}")
        print(f"  fetch_data -> {execution.call('fetch_data', url='https://example.com')}")

        print()
        print("  the journal, verbatim:")
        for event in execution.events:
            attempt = event.payload.get("attempt")
            suffix = f"  attempt={attempt}" if attempt is not None else ""
            if event.event_type == "ToolRetryScheduled":
                suffix += (
                    f" after {event.payload['failed_attempt']},"
                    f" delay={event.payload['delay']}s"
                )
            print(f"    {event.sequence:>2}  {event.event_type}{suffix}")

        call = execution.tool_calls[0]
        print()
        print(f"  call_id        : {call.call_id}   (unchanged by the retries)")
        print(f"  status         : {call.status}")
        print(f"  final attempt  : {call.attempt}")
        print("  attempt history:")
        for attempt in call.attempts:
            print(f"    {attempt}")

        execution.complete()
        execution_id = execution.id

    return execution_id


def scene_backoff_without_waiting(sleeper: RecordingSleeper) -> None:
    """Scene 2: the schedule is recorded, never spent."""
    print()
    print("=" * 72)
    print("2. The backoff, without waiting for it")
    print("=" * 72)

    print(f"  recorded delays        : {list(sleeper.delays)}")
    print(f"  total the runtime asked for: {sleeper.total_delay:.1f}s")
    print("  wall clock spent       : ~0s   (the sleeper is injectable)")

    patient = RetryPolicy(max_attempts=6, initial_delay=60.0, max_delay=300.0)
    print()
    print(
        "  a patient policy would have waited: "
        f"{[patient.delay(n) for n in range(1, 6)]}"
    )


def scene_error_classification(db_path: str, sleeper: RecordingSleeper) -> None:
    """Scene 3: what is retried, and what is not."""
    print()
    print("=" * 72)
    print("3. Three failure kinds, one policy")
    print("=" * 72)

    with Runtime(db_path, sleeper=sleeper) as runtime:

        @runtime.tool(name="flaky_upstream", retry_policy=POLICY)
        def flaky_upstream() -> str:
            raise RetryableToolError("503 from upstream")

        @runtime.tool(name="bad_request", retry_policy=POLICY)
        def bad_request() -> str:
            raise PermanentToolError("400: the URL is malformed")

        @runtime.tool(name="buggy", retry_policy=POLICY)
        def buggy() -> str:
            raise ValueError("a bug in the tool itself")

        execution = runtime.start(goal="Three kinds of failure")

        for name in ("flaky_upstream", "bad_request", "buggy"):
            before = len(sleeper.delays)
            try:
                execution.call(name)
            except ToolInvocationError as exc:
                waited = len(sleeper.delays) - before
                verdict = "retried" if waited else "NOT retried"
                print(
                    f"  {name:<15} -> {exc.error_type:<22}"
                    f" attempts={exc.attempts}  {verdict}"
                )

        print()
        print("  A permanent error and an unexpected one are both recorded once and")
        print("  raised. Only RetryableToolError says 'running this again might help'.")
# ---------------------------------------------------------------------------
# Scene 4: a real crash between "retry scheduled" and "attempt started"
# ---------------------------------------------------------------------------

#: Run by a child process that fails, schedules a retry, and dies instead of
#: waiting. ``os._exit``: no ``finally``, no ``atexit``, no close -- so only
#: what SQLite committed survives.
CRASH_CHILD = r'''
import os, sys

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from agent_runtime import RecordingSleeper, RetryPolicy, RetryableToolError, Runtime


class CrashDuringBackoff(RecordingSleeper):
    """Records the scheduled delay, then kills the process instead of waiting."""

    def sleep(self, delay):
        super().sleep(delay)
        print(f"    [child] retry after {delay}s scheduled; process dies here", flush=True)
        os._exit(70)


calls = {"n": 0}


def register(runtime):
    @runtime.tool(retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.1))
    def fetch_data(url: str) -> dict:
        calls["n"] += 1
        # Fails in this process, succeeds in the one that recovers it.
        if os.path.exists(os.environ["UPSTREAM_BACK"]):
            return {"url": url, "rows": 42}
        raise RetryableToolError("upstream unavailable")

    return fetch_data


with Runtime(os.environ["CRASH_DB"], sleeper=CrashDuringBackoff()) as runtime:
    register(runtime)
    execution = runtime.resume(os.environ["CRASH_EXECUTION_ID"])
    execution.call("fetch_data", url="https://example.com")
    print("    [child] the retry loop finished, which it must not", flush=True)
'''


def scene_crash_during_the_backoff(db_path: str, upstream_back: str) -> str:
    """Scene 4: the decision survives the process that made it."""
    print()
    print("=" * 72)
    print("4. Crash during the backoff, then carry the retry on")
    print("=" * 72)

    with Runtime(db_path) as runtime:
        execution = runtime.start(goal="Fetch data across a crash")
        execution_id = execution.id

    print("  child process:")
    result = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD],
        env={
            **os.environ,
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "CRASH_DB": db_path,
            "CRASH_EXECUTION_ID": execution_id,
            "UPSTREAM_BACK": upstream_back,
        },
        capture_output=True,
        text=True,
    )
    print(result.stdout.rstrip())
    print(f"    [child] exited with code {result.returncode} (no unwinding, no cleanup)")

    # The upstream recovers while nothing is running.
    Path(upstream_back).write_text("back", encoding="utf-8")

    print()
    print("  a new process, the same database:")
    with Runtime(db_path, sleeper=RecordingSleeper()) as recovered_runtime:
        # The recovering process registers the tool again -- a retry is a new
        # invocation, so somebody has to be able to run it.
        @recovered_runtime.tool(name="fetch_data", retry_policy=POLICY)
        def fetch_data(url: str) -> dict:
            REAL_CALLS["fetch_data"] += 1
            return {"url": url, "rows": 42}

        recovered = recovered_runtime.resume(execution_id)
        print(f"    status           : {recovered.status}")
        for retry in recovered.pending_retries:
            print(f"    scheduled retry  : {retry}  (at sequence {retry.sequence})")
        call = recovered.tool_calls[0]
        print(f"    attempts so far  : {[str(a) for a in call.attempts]}")
        print(
            "    still ambiguous  : "
            f"{bool(recovered.incomplete_tools)}   (a decision, not a gap)"
        )

        recovered.continue_pending_retry(call.call_id)

        call = recovered.tool_calls[0]
        print(f"    after continuing : status={call.status} attempt={call.attempt}")
        print(f"    attempt history  : {[str(a) for a in call.attempts]}")
        print(
            "    recovery == full fold of the journal: "
            f"{recovered.state == recovered_runtime.reconstruct_state(execution_id)}"
        )
        recovered.complete()

    return execution_id


def scene_replay(db_path: str, execution_id: str) -> None:
    """Scene 5: the recorded attempts replay without the tool running again."""
    print()
    print("=" * 72)
    print("5. Replay it -- the recorded attempts, no tool executed")
    print("=" * 72)

    before = REAL_CALLS["fetch_data"]
    with Runtime(db_path, register_default_tools=False) as runtime:
        result = runtime.replay(
            execution_id, on_step=lambda step: print(f"    ✓ {step.tool}  {step}")
        )
        print()
        print(f"    matched          : {result.matched}")
        print(
            "    retries replayed : "
            f"{result.retries_replayed}  delays {list(result.delays_replayed)}"
        )

    print(
        f"    real tool calls  : {before} before the replay, "
        f"{REAL_CALLS['fetch_data']} after"
    )


def main(db_path: str = "agent.db", marker_dir: str = ".") -> None:
    sleeper = RecordingSleeper()
    scene_one_call_three_attempts(db_path, sleeper)
    scene_backoff_without_waiting(sleeper)
    scene_error_classification(db_path, sleeper)
    upstream_back = os.path.join(marker_dir, "upstream-is-back")
    crashed = scene_crash_during_the_backoff(db_path, upstream_back)
    scene_replay(db_path, crashed)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], tempfile.gettempdir())
    else:
        # Default to a throwaway database so running the example is side-effect free.
        with tempfile.TemporaryDirectory() as tmp:
            main(str(Path(tmp) / "agent.db"), tmp)
