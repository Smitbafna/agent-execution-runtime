"""``agent-runtime`` -- the command line entry point.

    agent-runtime replay <execution-id>
    agent-runtime list

``replay`` is the interesting one: it re-runs an execution from its journal and
prints what the replay served, one line per recorded tool call. Nothing is
executed -- every result comes from the journal -- and the original history is
not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence, TextIO

from .exceptions import AgentRuntimeError, ReplayMismatchError
from .replay import ReplayStep
from .runtime import Runtime

__all__ = ["main", "build_parser"]

DEFAULT_DB = "agent.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-runtime",
        description="Inspect and deterministically replay agent executions.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"SQLite database to read (default: {DEFAULT_DB})",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    replay = subcommands.add_parser(
        "replay",
        help="Replay a recorded execution without executing any tool.",
        description=(
            "Replay an execution from its journal. Tool results are read from the "
            "recorded history rather than produced by running the tools again."
        ),
    )
    replay.add_argument("execution_id", help="Execution id to replay (exec_...)")
    replay.add_argument(
        "--from-sequence",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Replay only what came after sequence N, starting from the state there "
            "(0, the default, replays the whole execution)"
        ),
    )
    replay.add_argument(
        "--json",
        action="store_true",
        help="Print the replay result as JSON instead of a report.",
    )
    replay.set_defaults(handler=_cmd_replay)

    listing = subcommands.add_parser(
        "list", help="List the execution ids present in the journal."
    )
    listing.set_defaults(handler=_cmd_list)

    return parser


# -- commands -------------------------------------------------------------


def _cmd_replay(runtime: Runtime, args: argparse.Namespace, out: TextIO) -> int:
    execution_id: str = args.execution_id
    as_json: bool = bool(args.json)

    def report(step: ReplayStep) -> None:
        # A recorded failure is marked as such: it was served from the journal
        # too, just an unsuccessful outcome rather than a result.
        print(f"  {'✓' if step.succeeded else '✗'} {step.tool}", file=out)

    if as_json:
        # In JSON mode the output has to be nothing but JSON, so the progress
        # report and the banner are suppressed rather than mixed into it.
        on_step = None
    else:
        on_step = report
        print(f"Execution: {execution_id}", file=out)
        print(file=out)
        print("Replaying...", file=out)
        print(file=out)

    try:
        result = runtime.replay(
            execution_id, from_sequence=args.from_sequence, on_step=on_step
        )
    except ReplayMismatchError as exc:
        if as_json:
            print(
                json.dumps(
                    {"execution_id": execution_id, "matched": False, "error": str(exc)},
                    indent=2,
                ),
                file=out,
            )
            return 1
        print("Replay failed", file=out)
        print(file=out)
        _print_mismatch(exc, out)
        return 1

    if as_json:
        print(json.dumps(result.to_dict(), indent=2, default=str), file=out)
        return 0

    print(file=out)
    print("Replay completed", file=out)
    print(file=out)
    print(f"Events replayed: {result.events_replayed}", file=out)
    print(f"Tools replayed: {result.tools_replayed}", file=out)
    if result.from_sequence:
        print(f"Resumed from sequence: {result.from_sequence}", file=out)
    print(f"State: {result.status}", file=out)
    if result.errors:
        print(file=out)
        print(f"Recorded failures reproduced: {len(result.errors)}", file=out)
        for error in result.errors:
            message = (error.get("error") or {}).get("message", "failed")
            print(f"  ✗ {error['tool']}: {message}", file=out)
    return 0


def _cmd_list(runtime: Runtime, args: argparse.Namespace, out: TextIO) -> int:
    execution_ids = runtime.list_executions()
    if not execution_ids:
        print("No executions in the journal.", file=out)
        return 0
    for execution_id in execution_ids:
        events = runtime.journal.count_events(execution_id)
        state = runtime.reconstruct_state(execution_id)
        checkpoints = runtime.checkpoints.count(execution_id)
        print(
            f"{execution_id}  {state.status:<18} {events:>4} events  "
            f"{checkpoints} checkpoint(s)",
            file=out,
        )
    return 0
# -- mismatch report ------------------------------------------------------


def _print_mismatch(exc: ReplayMismatchError, out: TextIO) -> None:
    """Print the diagnosis: where replay diverged, and what each side said.

    A mismatch is the one case where the most useful thing to show is the
    difference itself -- the sequence it happened at, the arguments the journal
    recorded, and the arguments the replay produced -- so that is what leads.
    """
    if exc.sequence is not None:
        print(f"Sequence: {exc.sequence}", file=out)
    expected_args = _arguments_of(exc.expected)
    if expected_args is not None:
        print(f"Tool: {(exc.expected or {}).get('tool')}", file=out)
        print(file=out)
        print("Expected arguments:", file=out)
        print(f"    {expected_args}", file=out)
        print(file=out)
        print("Received:", file=out)
        print(f"    {_arguments_of(exc.received)}", file=out)
        return

    print(file=out)
    print(f"Reason: {exc}", file=out)
    if exc.original_state is not None and exc.replayed_state is not None:
        print(file=out)
        print("Original state:", file=out)
        print(f"    status = {exc.original_state.status}", file=out)
        print(f"    tool_calls = {len(exc.original_state.tool_calls)}", file=out)
        print(file=out)
        print("Replayed state:", file=out)
        print(f"    status = {exc.replayed_state.status}", file=out)
        print(f"    tool_calls = {len(exc.replayed_state.tool_calls)}", file=out)
        return
    if exc.expected:
        print(f"Expected: {exc.expected}", file=out)
    if exc.received:
        print(f"Received: {exc.received}", file=out)


def _arguments_of(payload: Any) -> str | None:
    """The ``arguments`` of an expected/received payload, as JSON."""
    if not isinstance(payload, dict) or "arguments" not in payload:
        return None
    return json.dumps(payload["arguments"], sort_keys=True)


# -- entry point ----------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    """Run the CLI; returns the process exit code.

    ``0`` on a successful replay, ``1`` on a replay mismatch, ``2`` on any other
    runtime error (unknown execution, unreadable database, ...).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = out if out is not None else sys.stdout

    try:
        with Runtime(args.db) as runtime:
            return int(args.handler(runtime, args, stream))
    except AgentRuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

