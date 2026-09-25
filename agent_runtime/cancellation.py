"""Cooperative cancellation: the one thing a tool can do to stop itself.

``execution.cancel()`` writes the cancellation to the journal -- that is the
durable part, and it is what recovery and replay read. But a journal entry
cannot reach into a function that is already running. This module holds the
part that can: a :class:`CancellationToken` the runtime hands to any tool that
asks for one, which flips a flag the tool is free to look at.

    @tool
    def sync_records(cancel_token: CancellationToken, limit: int):
        for row in stream(limit):
            if cancel_token.is_cancelled():
                return {"stopped_early": True}
            handle(row)
        return {"stopped_early": False}

The rule is narrow on purpose:

* a tool opts in by *declaring* a ``cancel_token`` parameter. Tools that do not
  declare one are called exactly as before, with no extra argument and no
  behaviour change;
* a token is a *request*, not a kill. Whether a tool honours it is the tool's
  choice, which is why the runtime never reports a thread as terminated on the
  strength of one (see :mod:`agent_runtime.timeout`);
* cancelling is idempotent, thread-safe and safe from another thread -- which is
  the whole point, since the thread being cancelled is the one inside the tool.
"""

from __future__ import annotations

import threading
from typing import Any, Callable


class CancellationToken:
    """A cooperative stop signal a tool may observe while it runs.

    Thread-safe, because the thread that cancels is almost never the thread
    running the tool::

        token = CancellationToken()
        thread = Thread(target=tool, args=(token,))
        thread.start()
        token.cancel("user pressed stop")   # from the main thread
        token.is_cancelled()                # True, inside the tool

    Attributes:
        reason: Why the cancellation happened, or ``None`` while it has not.
            Free-form, and journalled with ``ToolCancelled`` so a reader can
            tell a deliberate stop from a timeout that tripped the same token.
    """

    __slots__ = ("_event", "_lock", "_reason", "_callbacks")

    def __init__(self, reason: Any = None) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = reason
        self._callbacks: list[Callable[[], None]] = []

    # -- state ----------------------------------------------------------------

    @property
    def cancelled(self) -> bool:
        """Alias of :meth:`is_cancelled`, for ``if token.cancelled:`` sites."""
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested.

        Cheap enough for a tight loop -- an :class:`threading.Event` read, not
        a lock acquisition.
        """
        return self._event.is_set()

    @property
    def reason(self) -> Any:
        """Why this token was cancelled, or ``None``."""
        with self._lock:
            return self._reason

    def cancel(self, reason: Any = None) -> bool:
        """Request cancellation. ``True`` if this call was the one that did it.

        Idempotent: the first call flips the token and notifies listeners, and
        every later call is a no-op reporting ``False``. Nothing is unwound
        here -- this only raises the flag.
        """
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            callbacks = list(self._callbacks)
        # Set outside the lock so a listener that calls back into the token
        # cannot deadlock against it.
        self._event.set()
        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001 - a listener must not break the signal
                pass
        return True

    # -- waiting --------------------------------------------------------------

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled or ``timeout`` elapses; ``True`` if cancelled.

        The real interruption primitive: it returns the moment :meth:`cancel`
        is called from another thread, so a backoff built on it does not have
        to finish its wait first.
        """
        return self._event.wait(timeout)

    # -- the tool-facing API --------------------------------------------------

    def raise_if_cancelled(self) -> None:
        """Raise :class:`~agent_runtime.exceptions.ToolCancelledError` if cancelled.

        The raising form, for a tool that would rather bail out than return a
        partial result::

            for batch in batches:
                cancel_token.raise_if_cancelled()
                send(batch)
        """
        if self._event.is_set():
            raise ToolCancelledError(
                f"Cancelled: {self.reason}" if self.reason is not None else "Cancelled",
                reason=self.reason,
            )

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register ``callback``, called once when cancellation is requested.

        The runtime uses this to hand an :mod:`asyncio` task its ``cancel()``
        from a thread that is not running the loop. A token that is already
        cancelled fires the callback immediately, so a listener can never miss
        the edge it registered for.
        """
        with self._lock:
            already = self._event.is_set()
            if not already:
                self._callbacks.append(callback)
        if already:
            callback()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        state = "cancelled" if self._event.is_set() else "live"
        return f"CancellationToken({state}, reason={self._reason!r})"


from .exceptions import ToolCancelledError

__all__ = ["CancellationToken", "CANCEL_TOKEN_PARAMETER"]

#: The parameter name a tool declares to receive a :class:`CancellationToken`.
#: A name, not a type check, on purpose: annotations are optional in Python, and
#: a tool that wants cancellation should not have to import the type to get one.
CANCEL_TOKEN_PARAMETER = "cancel_token"
