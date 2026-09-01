"""Milestone 1 example: run an execution, then recover it from SQLite.

    python examples/basic_usage.py [db_path]

(Without an argument it uses a throwaway database in a temp directory.)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow running this file directly from a checkout, without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_runtime import Runtime, ToolInvocationError  # noqa: E402


def main(db_path: str = "agent.db") -> None:
    # 1. Start an execution and record everything it does.
    with Runtime(db_path) as runtime:
        execution = runtime.start(goal="Perform some calculations")

        print(f"started {execution.id} (status={execution.status})")
        print("  add(2, 3)       ->", execution.call("add", a=2, b=3))
        print("  multiply(5, 10) ->", execution.call("multiply", a=5, b=10))

        # A failing tool is recorded as ToolFailed and then re-raised.
        try:
            execution.call("divide", a=1, b=0)
        except ToolInvocationError as exc:
            print(f"  divide(1, 0)    !! {exc.error_type}: {exc}")

        execution.complete(result="done")
        execution_id = execution.id

        print("\nlive state:")
        print(execution)

    # 2. "Process restart": a brand new Runtime over the same file.
    with Runtime(db_path) as runtime:
        recovered = runtime.resume(execution_id)

        print("\nrecovered state from SQLite:")
        print(recovered)

        events = runtime.get_events(execution_id)
        print("\nevent journal:")
        for event in events:
            print(f"  {event.sequence}. {event.event_type:<18} {json.dumps(dict(event.payload))[:88]}")

        print("\nreconstructed state as JSON:")
        print(json.dumps(recovered.to_dict(), indent=2))
        print("\nall executions in the journal:", runtime.list_executions())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default to a throwaway database so running the example is side-effect free.
        with tempfile.TemporaryDirectory() as tmp:
            main(str(Path(tmp) / "agent.db"))
