"""Milestone 4C example: timeouts, cancellation and what survives a crash.

    python examples/reliability.py [db_path]

(Without an argument it uses a throwaway database in a temp directory.)

Nine scenes, in the order §22 asks for them:

1.  a flaky tool            -- one call, three attempts, one ``call_id``;
2.  retry                   -- the backoff schedule, recorded and not spent;
3.  successful completion   -- the attempt history that outlives the run;
4.  persistent history      -- the journal, read back by a *new* runtime;
5.  idempotent side effects -- a duplicate runs nothing;
6.  timeout handling        -- a real deadline, and a real refusal;
7.  cancellation            -- a long tool, stopped on purpose, durably;
8.  crash recovery          -- a child that dies mid-flight;
9.  deterministic replay    -- the same final state, no tool run twice.

The tools are the deterministic failure-injection set from §16, defined here
rather than imported: an example that needs the test suite to be importable is
not an example.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Allow running this file directly from a checkout, without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_runtime import (  # noqa: E402
    AgentRuntimeError,
    CancellationToken,
    RecordingSleeper,
    RetryPolicy,
    RetryableToolError,
    Runtime,
    ToolTimedOutError,
    UnsupportedTimeoutError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Retries a flaky tool. Five numbers, nothing more.
POLICY = RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=2.0, max_delay=10.0)

#: A counter only a real invocation moves: the proof a replay ran nothing.
REAL_CALLS: dict[str, int] = {"fetch_data": 0}


def show_journal(execution, title: str) -> None:
    """Print the journal verbatim, with the payload that explains each event."""
    print(f"  {title}:")
    for event in execution.events:
        payload = dict(event.payload)
        attempt = payload.get("attempt")
        note = f"  attempt={attempt}" if attempt is not None else ""
        if event.event_type == "ToolRetryScheduled":
            note += f"  delay={payload['delay']}s  reason={payload['reason']}"
        elif event.event_type == "ToolTimedOut":
            note += (
                f"  timeout={payload['timeout']}s  mode={payload['timeout_mode']}"
                f"  enforced={payload['enforced']}"
            )
        elif event.event_type == "ToolCancelled":
            note += f"  reason={payload.get('reason')!r}"
        print(f"    {event.sequence:>2}  {event.event_type}{note}")


# ---------------------------------------------------------------------------
# 1-3: a flaky tool, retry, successful completion
# ---------------------------------------------------------------------------


def scene_flaky_tool_retry_completion(db_path: str, sleeper: RecordingSleeper) -> str:
    print("=" * 72)
    print("1-3. A flaky tool, retried into a successful completion")
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
            return {"url": url, "rows": 42, "attempts": REAL_CALLS["fetch_data"]}

        execution = runtime.start(goal="Fetch the data")

        result = execution.call("fetch_data", url="https://example.com")
        print(f"  result      : {result}")
        print(f"  real calls  : {REAL_CALLS['fetch_data']}")

        show_journal(execution, "the journal")

        call = execution.tool_calls[0]
        print()
        print(f"  call_id     : {call.call_id}   (unchanged by the retries)")
        print(f"  status      : {call.status}")
        print(f"  final attempt: {call.attempt}")
        print("  attempt history:")
        for attempt in call.attempts:
            print(f"    {attempt}")

        execution.complete()
        return execution.id


# ---------------------------------------------------------------------------
# 4: the history persists
# ---------------------------------------------------------------------------


def scene_persistent_history(db_path: str, execution_id: str) -> None:
    print()
    print("=" * 72)
    print("4. The attempt history persists -- read by a new runtime")
    print("=" * 72)

    with Runtime(db_path) as runtime:
        info = runtime.recovery_info(execution_id)
        call = info.state.tool_calls[0]

        print(f"  status      : {info.status}")
        print(f"  classified  : {info.recovery_state}")
        print(f"  attempts    : {call.attempt_count} recorded")
        for attempt in call.attempts:
            print(f"    {attempt}")
        print(
            "\n  None of that lived in the first process's memory. It is all in\n"
            "  SQLite, which is why a restart can show it."
        )


# ---------------------------------------------------------------------------
# 5: idempotent side effects
# ---------------------------------------------------------------------------


def scene_idempotency(db_path: str) -> None:
    print()
    print("=" * 72)
    print("5. Idempotent side effects: a duplicate runs nothing")
    print("=" * 72)

    effects: list[str] = []

    def charge_card(amount: int) -> dict:
        effects.append(f"charged {amount}")
        return {"transaction_id": f"txn-{len(effects)}", "amount": amount}

    with Runtime(db_path) as runtime:
        runtime.register_tool(charge_card, name="charge_card")
        execution = runtime.start(goal="Charge a card once")

        first = execution.call("charge_card", amount=100, idempotency_key="payment-1")
        second = execution.call("charge_card", amount=100, idempotency_key="payment-1")

        print(f"  first  call : {first}")
        print(f"  second call : {second}   (answered from the stored result)")
        print(f"  real charges: {len(effects)}")
        print(f"  effects     : {effects}")

        record = runtime.idempotency_record("payment-1")
        print(f"  record      : status={record.status} attempts={record.attempts}")

        print(
            "\n  The claim is committed before the tool runs and the outcome after,\n"
            "  so a crash in between leaves PENDING -- visible, never guessed."
        )

    """Print the journal verbatim, with the payload that explains each event."""

# ---------------------------------------------------------------------------
# 6: timeout handling
# ---------------------------------------------------------------------------


def scene_timeouts(db_path: str) -> None:
    print()
    print("=" * 72)
    print("6. Timeout handling: a real deadline, and a real refusal")
    print("=" * 72)

    unwound: list[str] = []

    async def slow_query(rows: int) -> dict:
        """An async tool: a deadline here really cancels it, at its await."""
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            unwound.append("cancelled at the await point")
            raise
        return {"rows": rows}  # pragma: no cover - the deadline prevents it

    def unstoppable(rows: int) -> dict:
        """A sync tool with no token: the runtime will not fake a timeout here."""
        while True:  # pragma: no cover - refused before it ever runs
            time.sleep(3600)

    with Runtime(db_path) as runtime:
        runtime.register_tool(slow_query, name="slow_query")
        runtime.register_tool(unstoppable, name="unstoppable")
        execution = runtime.start(goal="Respect a deadline")

        started = time.perf_counter()
        try:
            execution.call("slow_query", rows=100, timeout=0.05)
        except ToolTimedOutError as exc:
            elapsed = time.perf_counter() - started
            print(f"  ToolTimedOutError after {elapsed:.3f}s (deadline was 0.05s)")
            print(f"    timeout  : {exc.timeout}s")
            print(f"    mode     : {exc.mode}")
            print(f"    enforced : {exc.enforced}")

        print(f"  coroutine unwound : {unwound}")
        print(
            "    asyncio.wait_for cancelled the task, so the tool's own cleanup\n"
            "    ran. A wrapper that merely timed the call could not do that."
        )

        show_journal(execution, "the journal")
        print(f"  final call status  : {execution.tool_calls[0].status}")

        print()
        print("  And a deadline the runtime cannot keep is refused, not faked:")
        try:
            execution.call("unstoppable", rows=1, timeout=1.0)
        except UnsupportedTimeoutError as exc:
            print(f"    {type(exc).__name__}: {str(exc).splitlines()[0]}")
            print(f"    tool   : {exc.tool_name}")
            print(f"    timeout: {exc.timeout}s")
        print(
            "    Python cannot terminate a running thread, so the runtime says so\n"
            "    rather than running it and measuring how long it took."
        )


# ---------------------------------------------------------------------------
# 7: cancellation
# ---------------------------------------------------------------------------


def scene_cancellation(db_path: str) -> str:
    print()
    print("=" * 72)
    print("7. Cancellation: a long tool, stopped on purpose")
    print("=" * 72)

    def sync_records(cancel_token: CancellationToken, limit: int) -> dict:
        """A cooperative tool: it opts in by declaring the token."""
        done = 0
        while done < limit:
            if cancel_token.is_cancelled():
                return {"records": done, "stopped_early": True}
            time.sleep(0.01)
            done += 1
        return {"records": done, "stopped_early": False}

    with Runtime(db_path) as runtime:
        runtime.register_tool(sync_records, name="sync_records")
        execution = runtime.start(goal="Sync a lot of records")

        errors: list[BaseException] = []

        def run_call() -> None:
            try:
                execution.call("sync_records", limit=10_000)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        caller = threading.Thread(target=run_call)
        caller.start()
        time.sleep(0.1)  # let it get properly stuck in the loop
        execution.cancel("user pressed stop")
        caller.join(5)

        print(f"  raised       : {type(errors[0]).__name__ if errors else 'nothing'}")
        print(f"  reason       : {errors[0].reason if errors else ''}")
        show_journal(execution, "the journal")
        print(f"  status       : {execution.status}")
        print(
            "\n  ToolStarted -> ToolCancelled -> ExecutionCancelled, all durable,\n"
            "  and no policy retried it: cancelling is a decision, not a fault."
        )
        return execution.id


# ---------------------------------------------------------------------------
# 8: crash recovery, in a real child process
# ---------------------------------------------------------------------------


#: A child that starts an execution, claims an idempotency key, performs the
#: side effect -- the only trace of which is a file -- and then dies with
#: ``os._exit`` before anything durable records the outcome.
CRASH_CHILD = r'''
import os, sys

sys.path.insert(0, os.environ["PROJECT_ROOT"])
from agent_runtime import Runtime

db = os.environ["CRASH_DB"]
effects = os.environ["CRASH_EFFECTS"]


def send_welcome_email(to: str) -> dict:
    # The "external" side effect. Its only trace outliving the crash is this
    # file, which is exactly why it is a file.
    with open(effects, "a") as handle:
        handle.write(to + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os._exit(70)          # no ToolCompleted, no stored outcome, no close
    return {}             # pragma: no cover


with Runtime(db) as runtime:
    runtime.register_tool(send_welcome_email, name="send_email")
    execution = runtime.start(goal="welcome the user", execution_id="exec_crashed")
    execution.call("send_email", to="user@example.com", idempotency_key="welcome-1")
'''


def _recording_sender(sink: list[str]):
    """A stand-in sender that records if it is ever reached."""

    def send_email(to: str) -> dict:
        sink.append(to)
        return {"message_id": "SHOULD-NOT-HAPPEN"}

    return send_email


def scene_crash_recovery(db_path: str, effects_path: str) -> None:
    print()
    print("=" * 72)
    print("8. Crash recovery: a real child that dies mid-side-effect")
    print("=" * 72)

    env = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CRASH_DB": db_path,
        "CRASH_EFFECTS": effects_path,
    }
    crashed = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD], env=env, capture_output=True, text=True
    )
    print(f"  child exit code : {crashed.returncode}   (killed with os._exit)")

    def sent_emails() -> list[str]:
        with open(effects_path) as handle:
            return [line for line in handle.read().splitlines() if line]

    print(f"  the side effect : {sent_emails()}   -- it really happened")

    with Runtime(db_path) as runtime:
        info = runtime.recovery_info("exec_crashed")
        print()
        print(f"  status          : {info.status}")
        print(f"  classified      : {info.recovery_state}")
        print(f"  open calls      : {[c.tool for c in info.incomplete_tools]}")
        print(f"  unresolved keys : {[r.idempotency_key for r in info.pending_idempotency]}")

        execution = runtime.resume("exec_crashed")
        calls: list[str] = []
        runtime.register_tool(_recording_sender(calls), name="send_email")
        try:
            execution.call(
                "send_email", to="user@example.com", idempotency_key="welcome-1"
            )
        except AgentRuntimeError as exc:
            print(f"  a retry is refused: {type(exc).__name__}")

        # Settle the open call, then the key -- both are explicit decisions.
        execution.resolve_recovery(execution.incomplete_tools[0].call_id, "mark_failed")
        print("  -> settled the open call, and now settled the key:")
        execution.resolve_idempotency(
            "welcome-1", "mark_completed", result={"message_id": "msg-1"}
        )
        result = execution.call(
            "send_email", to="user@example.com", idempotency_key="welcome-1"
        )
        print(f"  duplicate now    : {result}  (answered from the record)")
        print(f"  tool invocations : {len(calls)}")

    print(f"  emails sent      : {len(sent_emails())}   -- the user got exactly one")


# ---------------------------------------------------------------------------
# 9: deterministic replay
# ---------------------------------------------------------------------------


def scene_replay(db_path: str, execution_id: str) -> None:
    print()
    print("=" * 72)
    print("9. Deterministic replay: the same final state, no tool run twice")
    print("=" * 72)

    before = REAL_CALLS["fetch_data"]
    with Runtime(db_path) as runtime:
        # Deliberately *not* registering fetch_data: a replay cannot need it.
        result = runtime.replay(execution_id)

    print(f"  matched        : {result.matched}")
    print(f"  events         : {result.events_replayed}")
    print(f"  tools replayed : {result.tools_replayed}")
    print(
        f"  retries        : {result.retries_replayed}"
        f"  (delays {list(result.delays_replayed)})"
    )
    print(f"  journal intact : {result.journal_unchanged}")
    print(f"  final status   : {result.final_state.status}")
    print(f"  real calls     : {REAL_CALLS['fetch_data'] - before} during the replay")
    print(
        "\n  The replay re-ran the runtime's own logic -- the same call path, the\n"
        "  same reducers -- against the recorded outcomes. It never reached a\n"
        "  tool function, which is why the counter did not move."
    )


# ---------------------------------------------------------------------------
# The reliability model, in one picture
# ---------------------------------------------------------------------------


def scene_summary() -> None:
    print()
    print("=" * 72)
    print("The reliability model")
    print("=" * 72)
    print(
        """
Tool Call
   |
   +-- attempt
   |     +-- success      -> ToolCompleted
   |     +-- failure      -> ToolFailed      -> retry, per the policy
   |     +-- timeout      -> ToolTimedOut    -> retry only if retry_on_timeout,
   |     |                                           and never for a keyed call
   |     |                                           whose stop was unenforceable
   |     +-- cancellation -> ToolCancelled   -> never retried
   |
   +-- idempotency
          +-- COMPLETED -> reuse the stored result, run nothing
          +-- PENDING   -> RECOVERY_REQUIRED, an explicit decision
"""
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "agent.db")
        effects_path = str(Path(tmp) / "effects.txt")
        sleeper = RecordingSleeper()

        execution_id = scene_flaky_tool_retry_completion(db_path, sleeper)
        print()
        print(f"  recorded backoff delays   : {list(sleeper.delays)}")
        print(f"  a real sleeper would spend: {sleeper.total_delay:.1f}s")
        print("  (the sleeper is injectable, so the suite never waits)")

        scene_persistent_history(db_path, execution_id)
        scene_idempotency(db_path)
        scene_timeouts(db_path)
        scene_cancellation(db_path)
        scene_crash_recovery(db_path, effects_path)
        scene_replay(db_path, execution_id)
        scene_summary()

    print()


if __name__ == "__main__":
    main()

