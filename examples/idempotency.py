"""Idempotency & crash-safe side effects -- a runnable walk through Milestone 4B.

Run it::

    python3 examples/idempotency.py

It performs, in order:

1. a keyed call, and a duplicate of it (which does not send twice);
2. a simulated crash *after* the email was sent but *before* the outcome was
   stored -- in a child process that dies with ``os._exit``, unwinding nothing;
3. the restart, which refuses to send again and reports the ambiguity;
4. the three explicit resolutions, one after another;
5. a failed keyed call, which is not retried behind your back either;
6. a replay, which runs no tool and moves no ledger row.

The "mail server" is a text file. It is the only record of the side effects
that exists outside the database -- which is the point: the runtime cannot see
it, so the runtime cannot know what happened.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_runtime import (  # noqa: E402
    IdempotencyKeyFailedError,
    IdempotencyRecoveryRequiredError,
    PermanentToolError,
    Runtime,
)

#: Run by a child process that sends the email and then dies mid-call.
CRASH_CHILD = r'''
import os, sys

sys.path.insert(0, os.environ["DEMO_ROOT"])
from agent_runtime import Runtime

db = os.environ["DEMO_DB"]
outbox = os.environ["DEMO_OUTBOX"]


def send_email(to: str, body: str = "") -> dict:
    """The mail server: append to a file that outlives the process."""
    with open(outbox, "a") as handle:
        handle.write(f"{to}: {body}\n")
    print("   ...the mail server accepted it; the process dies here")
    os._exit(70)  # no ToolCompleted, no stored outcome, no unwinding, no close


with Runtime(db) as runtime:
    runtime.register_tool(send_email, name="send_email")
    execution = runtime.start(
        goal="welcome the second user", execution_id="exec_crashed"
    )
    execution.call(
        "send_email",
        to="second@example.com",
        body="welcome!",
        idempotency_key="welcome-crashed-user",
    )

os._exit(0)
'''


def heading(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(3, 62 - len(title)))


def outbox_lines(outbox: Path) -> list[str]:
    return outbox.read_text().splitlines() if outbox.exists() else []


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="idempotency-demo-"))
    db_path = str(workdir / "agent.db")
    outbox = workdir / "outbox.txt"

    def send(to: str, body: str = "") -> dict:
        """This process's mail server, sharing the file the child writes to."""
        with open(outbox, "a") as handle:
            handle.write(f"{to}: {body}\n")
        return {"message_id": f"msg-{len(outbox_lines(outbox))}", "to": to}

    heading("1. a keyed call, then exactly the same call again")
    with Runtime(db_path) as runtime:
        runtime.register_tool(send, name="send_email")
        execution = runtime.start(
            goal="welcome the new user", execution_id="exec_demo"
        )

        first = execution.call(
            "send_email",
            to="new@example.com",
            body="welcome!",
            idempotency_key="welcome-new-user",
        )
        second = execution.call(
            "send_email",
            to="new@example.com",
            body="welcome!",
            idempotency_key="welcome-new-user",
        )
        print(f"   first  -> {first}")
        print(f"   second -> {second}")
        print("   the second call returned the stored result without sending again")
        print(f"   outbox: {outbox_lines(outbox)}")
        print(f"   record: {runtime.idempotency_record('welcome-new-user')}")

    heading("2. a crash between the claim and the outcome")
    env = {
        **os.environ,
        "DEMO_ROOT": str(PROJECT_ROOT),
        "DEMO_DB": db_path,
        "DEMO_OUTBOX": str(outbox),
    }
    code = subprocess.run([sys.executable, "-c", CRASH_CHILD], env=env).returncode
    print(f"   child exited with {code}, unwinding nothing")
    print(f"   outbox: {outbox_lines(outbox)}   <- the mail really went out")

    heading("3. the restart does not send it again")
    with Runtime(db_path) as runtime:
        runtime.register_tool(send, name="send_email")
        recovered = runtime.resume("exec_crashed")

        print(f"   status:   {recovered.status}")
        for record in recovered.unresolved_idempotency:
            print(f"   key:      {record}")
        print(f"   journal:  {recovered.incomplete_tools[0]} (still open)")

        # The journal's open call and the ledger's PENDING key are two separate
        # unresolved things, and each has its own explicit answer.
        [stuck] = recovered.incomplete_tools
        recovered.resolve_recovery(
            stuck.call_id, "mark_completed", result={"message_id": "msg-2"}
        )

        try:
            recovered.call(
                "send_email",
                to="second@example.com",
                body="welcome!",
                idempotency_key="welcome-crashed-user",
            )
        except IdempotencyRecoveryRequiredError as exc:
            print(f"\n   refused:  {str(exc).splitlines()[2]}")
        print(f"   outbox:   {outbox_lines(outbox)}   <- unchanged, which is the point")

        heading("4. the three explicit resolutions")
        authorized = recovered.resolve_idempotency(
            "welcome-crashed-user", "retry", note="the provider says it did NOT send"
        )
        print(f"   retry          -> {authorized}")
        print("                   (the key stays PENDING: still one claim allowed)")
        recovered.call(
            "send_email",
            to="second@example.com",
            body="welcome!",
            idempotency_key="welcome-crashed-user",
        )
        print(f"   outbox: {outbox_lines(outbox)}   <- the authorized send happened")

        crashed = runtime.start(goal="another crash", execution_id="exec_crashed_2")
        # Stand-ins for two more crashes, claimed but never resolved, so all
        # three resolutions can be shown on keys that never completed either.
        runtime.idempotency.claim(
            "welcome-third-user",
            execution_id=crashed.id,
            call_id="call_crashed_2",
            tool_name="send_email",
            arguments={"to": "third@example.com"},
        )
        print(
            "   mark_completed ->",
            runtime.resolve_idempotency(
                "welcome-third-user",
                "mark_completed",
                result={"message_id": "checked-by-hand"},
            ),
        )
        runtime.idempotency.claim(
            "welcome-fourth-user",
            execution_id=crashed.id,
            call_id="call_crashed_3",
            tool_name="send_email",
            arguments={"to": "fourth@example.com"},
        )
        print(
            "   mark_failed    ->",
            runtime.resolve_idempotency(
                "welcome-fourth-user", "mark_failed", error="the provider has no record"
            ),
        )

        heading("5. a failed keyed call is not retried behind your back")

        @runtime.tool
        def charge(amount: int) -> dict:
            raise PermanentToolError("card declined")

        paying = runtime.start(goal="pay the invoice")
        try:
            paying.call("charge", amount=4200, idempotency_key="invoice-2026-04")
        except Exception as exc:  # ToolInvocationError
            print(f"   charge failed: {exc}")
        try:
            paying.call("charge", amount=4200, idempotency_key="invoice-2026-04")
        except IdempotencyKeyFailedError as exc:
            print(f"   refused:       {exc}")

        heading("6. replay: no tool runs, no ledger row moves")
        before = runtime.idempotency_records()
        for execution_id in ("exec_demo", "exec_crashed"):
            result = runtime.replay(execution_id)
            print(
                f"   {execution_id}: matched={result.matched} "
                f"calls={result.tools_replayed} status={result.final_state.status}"
            )
        print(f"   ledger unchanged: {before == runtime.idempotency_records()}")
        print(f"   outbox unchanged: {len(outbox_lines(outbox))} messages")

    print(f"\nTotal emails actually sent: {len(outbox_lines(outbox))}")
    print(f"Database: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
