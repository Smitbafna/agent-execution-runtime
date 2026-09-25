"""Deterministic failure-injection tools for the Milestone 4C suite.

§16 asks for five shapes of misbehaving tool, and this module is all of them.
They live in ``tests/`` rather than in ``agent_runtime`` on purpose: a runtime
that ships tools whose job is to hang forever is shipping a hazard, and the
milestone's "do not introduce production-specific test hacks" is the reason.
Nothing in the package imports this file.

Every tool here is deterministic. The only source of variation is a counter or a
value the test passes in, so a test can assert on the exact journal a tool
produced rather than on how long something took.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable

from agent_runtime import CancellationToken, RetryableToolError

__all__ = [
    "always_fails",
    "fails_once",
    "sleeps_forever",
    "cooperative_worker",
    "side_effect_then_fail",
    "effect_recorder",
    "read_effects",
    "async_sleeps_forever",
    "async_fails_once",
]


def always_fails(message: str = "the upstream is down") -> Callable[..., Any]:
    """A tool that never succeeds: every attempt raises ``RetryableToolError``.

    ::

        runtime.register_tool(always_fails(), name="flaky")
        execution.call("flaky", retry_policy=RetryPolicy(max_attempts=3))

    Three attempts, three ``ToolFailed``, three ``ToolRetryScheduled``.
    """

    def tool() -> None:
        raise RetryableToolError(message)

    tool.__name__ = "always_fails"
    return tool


def fails_once(failures: int = 1, value: Any = "settled") -> Callable[..., Any]:
    """A tool that fails ``failures`` times and then succeeds.

    The counter is closed over, so it survives being registered under any name
    and its history is per-tool-instance rather than global.
    """
    state = {"calls": 0}

    def tool() -> Any:
        state["calls"] += 1
        if state["calls"] <= failures:
            raise RetryableToolError(f"attempt {state['calls']} failed")
        return value

    tool.__name__ = "fails_once"
    tool.attempts = state  # type: ignore[attr-defined]
    return tool


def sleeps_forever() -> Callable[..., Any]:
    """A ``def`` tool with no way to stop it: ``time.sleep`` in a loop.

    This is the tool §4 is about. It declares no ``cancel_token``, so the runtime
    **refuses** a deadline for it rather than running it and measuring how long
    it took -- which is the difference between a timeout and a stopwatch.
    """

    def tool() -> None:  # pragma: no cover - never invoked by a compliant caller
        while True:
            time.sleep(3600)

    tool.__name__ = "sleeps_forever"
    return tool


def cooperative_worker(
    slice_seconds: float = 0.01, *, raises: bool = False
) -> Callable[..., Any]:
    """A tool that polls its cancellation token and stops when asked (§8).

    The canonical cooperative tool: it checks ``is_cancelled()`` in a loop and
    returns a partial result, or raises if ``raises=True`` so the
    ``raise_if_cancelled()`` form is exercised too::

        def worker(cancel_token: CancellationToken) -> dict: ...
    """

    def tool(cancel_token: CancellationToken) -> dict:
        worked = 0
        while True:
            if cancel_token.is_cancelled():
                if raises:
                    cancel_token.raise_if_cancelled()
                return {"stopped_early": True, "slices_done": worked}
            time.sleep(slice_seconds)
            worked += 1

    tool.__name__ = "cooperative_worker"
    return tool


def effect_recorder(path: str) -> Callable[[str], Callable[..., Any]]:
    """Build a tool whose side effect is *only* the line it appends to ``path``.

    The file is fsync'd on every write, so the trace of the effect outlives an
    ``os._exit`` -- which is the whole point of the crash tests: after the crash
    the database and the file are the only two things that can be believed.
    """

    def make_tool(name: str = "side_effect") -> Callable[..., Any]:
        def tool(to: str) -> dict:
            with open(path, "a") as handle:
                handle.write(to + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return {"to": to}

        tool.__name__ = name
        return tool

    return make_tool


def side_effect_then_fail(path: str) -> Callable[..., Any]:
    """§16's ``side effect then fail``: the effect happens, *then* it raises.

    The dangerous order, and the one idempotency exists for. A retry of this tool
    performs the effect again, which is why a keyed call that stops this way
    leaves its key ``PENDING`` instead of being retried.
    """

    def tool(to: str) -> dict:
        with open(path, "a") as handle:
            handle.write(to + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        raise RetryableToolError(f"{to}: recorded, then the call failed anyway")

    tool.__name__ = "side_effect_then_fail"
    return tool


def read_effects(path: str) -> list[str]:
    """Every side effect a crash-test tool recorded, in order."""
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [line for line in handle.read().splitlines() if line]


async def async_sleeps_forever() -> None:
    """An ``async def`` tool that never returns on its own (§4's ASYNC mode).

    ``asyncio.wait_for`` cancels this at the deadline and delivers
    ``CancelledError`` at its ``await``, which is a real stop rather than a
    measurement.
    """

    while True:  # pragma: no cover - cancelled by the deadline, never reached
        await asyncio.sleep(3600)


def async_fails_once(failures: int = 1, value: Any = "settled") -> Callable[..., Any]:
    """An ``async def`` tool that outlives its deadline, then succeeds.

    Used for §5 and scenario 3: a timeout, a retry, a success, and a replay of
    all three. The first attempts sleep far past any deadline a test sets, so
    the *deadline* is what stops them -- the runtime is cancelling a running
    coroutine, not guessing from a return value.
    """
    state = {"calls": 0}

    async def tool() -> Any:
        state["calls"] += 1
        if state["calls"] <= failures:
            await asyncio.sleep(30)
        return value

    tool.__name__ = "async_fails_once"
    tool.attempts = state  # type: ignore[attr-defined]
    return tool

