"""``agent-runtime`` -- the command line entry point.

    agent-runtime replay <execution-id>
    agent-runtime list
    agent-runtime cancel <execution-id> [--reason TEXT]
    agent-runtime idempotency list
    agent-runtime idempotency show <key>
    agent-runtime idempotency resolve <key> --action retry

``replay`` is the interesting one: it re-runs an execution from its journal and
prints what the replay served, one line per recorded tool call. Nothing is
executed -- every result comes from the journal -- and the original history is
not modified.

``cancel`` is Milestone 4C's operator window. It journals a cancellation for a
running execution from another process, which is durable: the execution stays
``CANCELLED`` afterwards and nothing retries it.

``idempotency`` is the operator's window onto Milestone 4B. After a crash it
shows which keys the runtime committed to and never learned the outcome of, and
it accepts the one thing the runtime refuses to do by itself: an explicit
decision (``retry``, ``mark_completed``, ``mark_failed``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence, TextIO

from .exceptions import AgentRuntimeError, ReplayMismatchError
from .idempotency import IDEMPOTENCY_ACTIONS, IdempotencyRecord, IdempotencyStatus
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

    keys = subcommands.add_parser(
        "idempotency",
        help="Inspect and resolve idempotency keys (Milestone 4B).",
        description=(
            "Show the claims this database holds for side-effecting tools. A "
            "PENDING key is one the runtime committed to and never learned the "
            "outcome of -- it will not run that tool again until you resolve it."
        ),
    )
    key_commands = keys.add_subparsers(dest="idempotency_command", required=True)

    key_list = key_commands.add_parser(
        "list", help="List idempotency records (default: the unresolved ones)."
    )
    key_list.add_argument(
        "--status",
        choices=[str(status) for status in IdempotencyStatus],
        help="Only records in this status (default: PENDING).",
    )
    key_list.add_argument(
        "--all",
        action="store_true",
        help="List every record, not just the PENDING ones.",
    )
    key_list.add_argument(
        "--execution", metavar="ID", help="Only keys claimed by this execution."
    )
    key_list.add_argument(
        "--json", action="store_true", help="Print the records as JSON."
    )
    key_list.set_defaults(handler=_cmd_idempotency_list)

    key_show = key_commands.add_parser("show", help="Show one idempotency record.")
    key_show.add_argument("key", help="The idempotency key to look up.")
    key_show.add_argument(
        "--json", action="store_true", help="Print the record as JSON."
    )
    key_show.set_defaults(handler=_cmd_idempotency_show)

    key_resolve = key_commands.add_parser(
        "resolve",
        help="Settle a key explicitly, after checking what actually happened.",
        description=(
            "retry           allow exactly one more execution of this key\n"
            "mark_completed  record an externally known successful result\n"
            "mark_failed     record a known failure"
        ),
    )
    key_resolve.add_argument("key", help="The idempotency key to settle.")
    key_resolve.add_argument(
        "--action",
        required=True,
        choices=list(IDEMPOTENCY_ACTIONS),
        help="What to record.",
    )
    key_resolve.add_argument(
        "--result",
        metavar="JSON",
        help="The known result for --action mark_completed (JSON).",
    )
    key_resolve.add_argument(
        "--error", metavar="TEXT", help="The known failure for --action mark_failed."
    )
    key_resolve.add_argument(
        "--note", metavar="TEXT", help="A note recorded with the decision."
    )
    key_resolve.set_defaults(handler=_cmd_idempotency_resolve)

    cancel = subcommands.add_parser(
        "cancel",
        help="Cancel a running execution (Milestone 4C).",
        description=(
            "Request cancellation of a running execution and journal it. The "
            "decision is durable: a cancelled execution stays CANCELLED after a "
            "restart, and nothing retries it."
        ),
    )
    cancel.add_argument("execution_id", help="Execution id to cancel (exec_...).")
    cancel.add_argument(
        "--reason", metavar="TEXT", help="Why. Recorded with the cancellation."
    )
    cancel.set_defaults(handler=_cmd_cancel)

    return parser


# -- commands -------------------------------------------------------------


def _cmd_replay(runtime: Runtime, args: argparse.Namespace, out: TextIO) -> int:
    execution_id: str = args.execution_id
    as_json: bool = bool(args.json)

    def report(step: ReplayStep) -> None:
        # A recorded failure is marked as such: it was served from the journal
        # too, just an unsuccessful outcome rather than a result. A call that
        # the journal shows being retried says which attempt finally settled it.
        mark = "✓" if step.succeeded else "✗"
        attempts = f" (attempt {step.attempt}/{step.attempts})" if step.retried else ""
        print(f"  {mark} {step.tool}{attempts}", file=out)

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
    if result.retries_replayed:
        # A retried call is still one tool call; the attempts and the backoff it
        # waited are what say so.
        print(
            f"Retries replayed: {result.retries_replayed} "
            f"(delays: {list(result.delays_replayed)})",
            file=out,
        )
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
        unresolved = len(runtime.unresolved_idempotency(execution_id))
        print(
            f"{execution_id}  {state.status:<18} {events:>4} events  "
            f"{checkpoints} checkpoint(s)"
            + (f"  {unresolved} unresolved key(s)" if unresolved else ""),
            file=out,
        )
    return 0


# -- idempotency -----------------------------------------------------------


def _cmd_idempotency_list(runtime: Runtime, args: argparse.Namespace, out: TextIO) -> int:
    """Show the claims this database holds, unresolved ones first by default."""
    if args.all and args.status is None:
        records = runtime.idempotency_records(execution_id=args.execution)
    else:
        status = args.status or IdempotencyStatus.PENDING
        records = runtime.idempotency_records(execution_id=args.execution, status=status)

    if args.json:
        print(json.dumps([record.to_dict() for record in records], indent=2), file=out)
        return 0

    if not records:
        print("No matching idempotency records.", file=out)
        return 0
    for record in records:
        print(_record_line(record), file=out)
    unresolved = [record for record in records if record.is_unresolved]
    if unresolved:
        print(file=out)
        print(
            f"{len(unresolved)} key(s) need an explicit decision; the runtime will not "
            "run those tools again on its own:",
            file=out,
        )
        for record in unresolved:
            print(
                f"    agent-runtime idempotency resolve {record.idempotency_key} "
                "--action {retry|mark_completed|mark_failed}",
                file=out,
            )
    return 0


def _cmd_idempotency_show(runtime: Runtime, args: argparse.Namespace, out: TextIO) -> int:
    record = runtime.idempotency_record(args.key)
    if record is None:
        print(f"No idempotency record for key {args.key!r}.", file=out)
        return 2
    if args.json:
        print(json.dumps(record.to_dict(), indent=2), file=out)
        return 0
    for key, value in record.to_dict().items():
        print(f"{key}: {value}", file=out)
    return 0


def _cmd_idempotency_resolve(runtime: Runtime, args: argparse.Namespace, out: TextIO) -> int:
    """Record the decision an operator made about one key.

    Nothing here is inferred. The runtime refused to guess whether the side effect
    happened, and this is the caller saying what it found out.
    """
    result = _parse_json_option(args.result, "--result") if args.result else None
    record = runtime.resolve_idempotency(
        args.key,
        args.action,
        result=result,
        error=args.error,
        note=args.note,
    )
    print(f"{record.idempotency_key}: {record.resolution} -> {_record_line(record)}", file=out)
    return 0


def _record_line(record: IdempotencyRecord) -> str:
    """One human-readable line per record, honest about what is unresolved."""
    return (
        f"{record.idempotency_key:<28} {str(record.status):<10} "
        f"attempts={record.attempts}  {record.tool_name}({record.call_id})"
        + (f"  retry authorized x{record.retry_authorized}" if record.retry_authorized else "")
    )


def _cmd_cancel(runtime: Runtime, args: argparse.Namespace, out: TextIO) -> int:
    """Cancel an execution, and report what recovery now makes of it."""
    runtime.cancel(args.execution_id, reason=args.reason)
    info = runtime.recovery_info(args.execution_id)

    print(f"Cancelled {args.execution_id}", file=out)
    if args.reason:
        print(f"  reason     : {args.reason}", file=out)
    print(f"  status     : {info.status}", file=out)
    print(f"  classified : {info.recovery_state}", file=out)
    for call in info.cancelled_calls:
        print(f"  {call}", file=out)
    print(
        "\nA cancelled execution is not resumed, and nothing retries its calls.",
        file=out,
    )
    return 0


def _parse_json_option(raw: str, flag: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise AgentRuntimeError(f"{flag} must be valid JSON: {exc}") from exc
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

