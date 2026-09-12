"""Milestone 3 example: replay an execution without running its tools.

    python examples/replay.py [db_path]

(Without an argument it uses a throwaway database in a temp directory.)

Three scenes, in order:

1. run some work, including a tool with a real side effect;
2. replay it, and watch the side effect *not* happen a second time;
3. replay it again from a checkpoint, and get the same state either way.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running this file directly from a checkout, without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_runtime import (  # noqa: E402
    ReplayMismatchError,
    Runtime,
    ToolInvocationError,
)

# A counter that only a real tool call moves. If replay were executing tools,
# this would end up higher than 1.
SIDE_EFFECTS = {"create_github_issue": 0}


def scene_run_and_replay(db_path: str) -> str:
    """Scene 1+2: do some work, then replay it without repeating the side effect."""
    print("=" * 72)
    print("1. Run an execution that creates a GitHub issue")
    print("=" * 72)

    with Runtime(db_path) as runtime:

        @runtime.tool(name="create_github_issue")
        def create_github_issue(title: str) -> dict:
            """A tool with an obvious side effect: it would really file an issue."""
            SIDE_EFFECTS["create_github_issue"] += 1
            return {"url": f"https://github.com/example/repo/issues/{SIDE_EFFECTS['create_github_issue']}"}

        @runtime.tool(name="run_tests")
        def run_tests() -> str:
            """A tool that fails, so replay has a failure to reproduce."""
            raise RuntimeError("3 tests failed")

        execution = runtime.start(goal="File a bug and run the tests")

        print(f"started {execution.id}")
        print("  add(2, 3)      ->", execution.call("add", a=2, b=3))
        issue = execution.call("create_github_issue", title="Login button does nothing")
        print(f"  create_github_issue -> {issue['url']}")
        print(f"  issues really created so far: {SIDE_EFFECTS['create_github_issue']}")

        try:
            execution.call("run_tests")
        except ToolInvocationError as exc:
            print(f"  run_tests      !! {exc.error_type}: {exc}")

        execution.checkpoint()  # a snapshot partway through
        print("  multiply(5, 10)->", execution.call("multiply", a=5, b=10))
        execution.complete(result="done")

        execution_id = execution.id
        original_state = execution.state
        checkpoint = execution.latest_checkpoint

    print()
    print("=" * 72)
    print("2. Replay it -- in a brand new process with no tools registered")
    print("=" * 72)

    # No tools are registered here at all. Replay reads results from the journal,
    # so `create_github_issue` is never called again and no second issue exists.
    with Runtime(db_path, register_default_tools=False) as runtime:
        events_before = runtime.journal.count_events(execution_id)

        result = runtime.replay(execution_id, on_step=lambda step: print(f"  ✓ {step.tool}"))

        print()
        print(result)
        print()
        print(f"issues really created after replay: {SIDE_EFFECTS['create_github_issue']}")
        print(f"journal events before replay: {events_before}")
        print(f"journal events after  replay: {runtime.journal.count_events(execution_id)}")

    assert SIDE_EFFECTS["create_github_issue"] == 1, "the replay re-ran a tool!"
    assert result.final_state == original_state, "the replay did not match"
    assert result.journal_unchanged, "the replay modified the original history"

    print()
    print("=" * 72)
    print("3. Replay from the checkpoint instead of from the beginning")
    print("=" * 72)

    with Runtime(db_path, register_default_tools=False) as runtime:
        full = runtime.replay(execution_id)
        partial = runtime.replay(execution_id, from_sequence=checkpoint.sequence)

    print(f"  full replay          : {full.events_replayed:>3} events, "
          f"{full.tools_replayed} tools replayed")
    print(f"  from checkpoint @{checkpoint.sequence:<3}: {partial.events_replayed:>3} events, "
          f"{partial.tools_replayed} tools replayed")
    print(f"  same final state     : {full.final_state == partial.final_state}")

    return execution_id


def scene_mismatch(db_path: str, execution_id: str) -> None:
    """What a divergence looks like: the replay is asked for something else."""
    print()
    print("=" * 72)
    print("4. A replay that does not match the history")
    print("=" * 72)

    with Runtime(db_path, register_default_tools=False) as runtime:
        replayed = runtime.replay_engine(execution_id).prepare()

        # Ask for the recorded call correctly...
        replayed.call("add", a=2, b=3)

        # ...then ask for the same tool with different arguments.
        try:
            replayed.call("create_github_issue", title="Something else entirely")
        except ReplayMismatchError as exc:
            print(exc)
            print()
            print(f"divergence kind: {exc.kind} (at sequence {exc.sequence})")


def main(db_path: str = "agent.db") -> None:
    execution_id = scene_run_and_replay(db_path)
    scene_mismatch(db_path, execution_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default to a throwaway database so running the example is side-effect free.
        with tempfile.TemporaryDirectory() as tmp:
            main(str(Path(tmp) / "agent.db"))
