"""Milestone 2 example: checkpoint an execution, crash, recover, decide, finish.

    python examples/checkpoint_recovery.py [db_path]

(Without an argument it uses a throwaway database in a temp directory.)

Three scenes, in order:

1. run some work, checkpoint it, "restart", and resume from the checkpoint;
2. crash *inside* a tool call in a child process that really dies;
3. inspect the recovery, decide what to do, and finish the execution.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running this file directly from a checkout, without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_runtime import ExecutionStatus, Runtime  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A child process that journals the start of a tool call and then dies hard --
# os._exit runs no cleanup, no close, no atexit -- so only what SQLite committed
# survives. That is the crash this example recovers from.
CRASH_CHILD = r"""
import os, sys

sys.path.insert(0, sys.argv[1])
from agent_runtime import Runtime

db, execution_id = sys.argv[2], sys.argv[3]
with Runtime(db) as runtime:
    execution = runtime.resume(execution_id)
    journal = runtime.journal
    journal.append_event(
        execution.id,
        "ToolRequested",
        {"call_id": "call_tests", "tool": "run_tests", "arguments": {"suite": "unit"}},
    )
    journal.append_event(
        execution.id, "ToolStarted", {"call_id": "call_tests", "tool": "run_tests"}
    )
    print("   (child: ToolStarted is committed, now dying mid-tool)")
    sys.stdout.flush()
    os._exit(137)
"""


def run_and_checkpoint(db_path: str) -> str:
    """Scene 1: do some work, snapshot it, and resume in a "new process"."""
    print("=" * 72)
    print("1. Run, checkpoint, restart, resume")
    print("=" * 72)

    with Runtime(db_path) as runtime:
        execution = runtime.start(goal="Perform calculations")
        print(f"started {execution.id}")

        print("  add(2, 3)       ->", execution.call("add", a=2, b=3))
        checkpoint = execution.checkpoint()
        print(f"  checkpoint      -> {checkpoint}")

        print("  multiply(5, 10) ->", execution.call("multiply", a=5, b=10))
        execution_id = execution.id
        print(f"  journal now ends at sequence {execution.last_event_sequence}")

    # A brand new Runtime over the same file is all "restarting" means here.
    with Runtime(db_path) as runtime:
        execution = runtime.resume(execution_id)
        info = execution.recovery_info()

        print(f"\nresumed {execution.id} [{execution.status}]")
        print(f"  recovered from  : {info.source} @ sequence {info.checkpoint_sequence}")
        print(f"  events replayed : {info.events_replayed} (not all {info.last_sequence})")
        print(
            "  same as a full replay: "
            f"{execution.state == runtime.reconstruct_state(execution_id)}"
        )
        print("\nrecovered state:")
        print(execution)

    return execution_id


def crash_mid_tool(db_path: str, execution_id: str) -> None:
    """Scene 2: a child process dies between ToolStarted and the outcome."""
    print()
    print("=" * 72)
    print("2. The process dies inside a tool call")
    print("=" * 72)

    result = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD, str(PROJECT_ROOT), db_path, execution_id]
    )
    print(f"  child exited with {result.returncode} (killed, no cleanup ran)")


def decide_and_finish(db_path: str, execution_id: str) -> None:
    """Scene 3: report the ambiguity, resolve it, and carry on."""
    print()
    print("=" * 72)
    print("3. Inspect the recovery, decide, finish")
    print("=" * 72)

    with Runtime(db_path) as runtime:
        execution = runtime.resume(execution_id)

        print("recovery_info():")
        print(execution.recovery_info())
        print()
        print(f"state.incomplete_tools -> {execution.incomplete_tools}")
        print(f"status is RECOVERY_REQUIRED: {execution.status is ExecutionStatus.RECOVERY_REQUIRED}")
        print("the runtime will NOT re-run run_tests: the journal cannot say whether it ran")

        # The application owns the decision. Here: the test run did finish, so the
        # call is recorded as completed and the execution becomes runnable again.
        stuck = execution.incomplete_tools[0]
        execution.resolve_recovery(stuck.call_id, "mark_completed", result="42 passed")
        print(f"\nresolved {stuck.call_id} -> status is now {execution.status}")

        execution.call("multiply", a=5, b=10)
        execution.complete(result="all done")
        execution.checkpoint()

    with Runtime(db_path) as runtime:
        execution = runtime.resume(execution_id)
        print("\nafter another restart:")
        print(execution)
        info = execution.recovery_info()
        print(f"needs_resolution: {info.needs_resolution}")
        print(f"checkpoints stored: {[c.sequence for c in runtime.get_checkpoints(execution_id)]}")

    print()
    print("=" * 72)
    print("event journal and checkpoints")
    print("=" * 72)
    with Runtime(db_path) as runtime:
        for event in runtime.get_events(execution_id):
            print(f"  {event.sequence}. {str(event.event_type):<20} {dict(event.payload)}")
        for checkpoint in runtime.get_checkpoints(execution_id):
            print(
                f"  checkpoint @ {checkpoint.sequence}: "
                f"{len(checkpoint.state.tool_calls)} tool call(s), "
                f"status {checkpoint.state.status}"
            )


def main(db_path: str = "agent.db") -> None:
    execution_id = run_and_checkpoint(db_path)
    crash_mid_tool(db_path, execution_id)
    decide_and_finish(db_path, execution_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default to a throwaway database so running the example is side-effect free.
        with tempfile.TemporaryDirectory() as tmp:
            main(str(Path(tmp) / "agent.db"))
