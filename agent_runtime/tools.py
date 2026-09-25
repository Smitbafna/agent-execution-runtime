"""A minimal tool abstraction.

Tools are plain Python functions. The :func:`tool` decorator turns one into a
:class:`Tool`, and :class:`ToolRegistry` resolves names to tools for the
execution runtime.

Since Milestone 4A a tool may also declare how its failures should be retried::

    @tool(retry_policy=RetryPolicy(max_attempts=3))
    def fetch_data(url: str): ...

That is the *default* for calls of this tool; a single call may override it
with ``execution.call("fetch_data", retry_policy=RetryPolicy(max_attempts=5))``.

Scope note: there are no idempotency keys, timeouts or cancellation here, and
no configuration language for retries -- a :class:`~agent_runtime.retry.RetryPolicy`
is five numbers and nothing more.
"""

from __future__ import annotations

import inspect
import json
import traceback as _traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .cancellation import CANCEL_TOKEN_PARAMETER, CancellationToken
from .exceptions import (
    RetryConfigurationError,
    ToolAlreadyRegisteredError,
    ToolArgumentError,
    ToolInvocationError,
    ToolNotFoundError,
)
from .retry import RetryPolicy
from .timeout import check_timeout, declares_cancel_token, is_async_callable

__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "make_jsonable",
    "check_retry_policy",
    "check_timeout",
    "default_registry",
]


def make_jsonable(value: Any) -> Any:
    """Convert ``value`` into something ``json.dumps`` accepts.

    JSON-native values pass through untouched (tuples become lists, so a value
    looks the same whether read from memory or from SQLite). Anything else is
    represented by its ``repr`` -- lossy, but it keeps the journal writable and
    state reconstruction faithful to what was recorded.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [make_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


@dataclass(frozen=True, slots=True)
class Tool:
    """A callable exposed to executions under a stable name."""

    name: str
    func: Callable[..., Any]
    description: str = ""
    #: The retry policy calls of this tool use unless the call overrides it
    #: (Milestone 4A). ``None`` means "whatever the caller says", which for a
    #: call that says nothing is :attr:`RetryPolicy.none`.
    retry_policy: RetryPolicy | None = None
    #: The default deadline for calls of this tool (Milestone 4C), in seconds.
    #: ``None`` means the call decides. A call-level ``timeout=`` always wins.
    timeout: float | None = None
    #: Whether this tool declared a ``cancel_token`` parameter, i.e. whether it
    #: opted into cooperative cancellation (Milestone 4C). Resolved from the
    #: function once, at registration, and recorded on the call.
    cooperative: bool = False

    def __post_init__(self) -> None:
        check_retry_policy(self.retry_policy, tool_name=self.name)
        object.__setattr__(
            self, "timeout", check_timeout(self.timeout, where=f"timeout for tool {self.name!r}")
        )
        object.__setattr__(self, "cooperative", declares_cancel_token(self.func))

    @property
    def signature(self) -> inspect.Signature:
        return inspect.signature(self.func)

    @property
    def is_async(self) -> bool:
        """Whether the tool is a coroutine function, and so cancellable."""
        return is_async_callable(self.func)

    def accepts_cancel_token(self) -> bool:
        """Whether the runtime may pass this tool a cancellation token."""
        return self.cooperative

    def bind(
        self, args: tuple[Any, ...] = (), kwargs: Mapping[str, Any] | None = None
    ) -> inspect.BoundArguments:
        """Validate and bind call arguments against the function signature.

        The ``cancel_token`` a cooperative tool declares is satisfied here with
        a *placeholder*, because this binding exists to produce the arguments
        that get journalled -- and a token is a runtime concern, not part of
        the call. :meth:`bind_with_token` performs the real binding for
        invocation, with the live token. Both agree on every other parameter.
        """
        placeholders = dict(kwargs or {})
        if self.cooperative and CANCEL_TOKEN_PARAMETER not in placeholders:
            placeholders[CANCEL_TOKEN_PARAMETER] = None
        try:
            bound = inspect.signature(self.func).bind(*args, **placeholders)
        except TypeError as exc:
            raise ToolArgumentError(
                f"Invalid arguments for tool {self.name!r}: {exc}", tool_name=self.name
            ) from exc
        bound.apply_defaults()
        return bound

    def bind_with_token(
        self,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        token: CancellationToken | None = None,
    ) -> inspect.BoundArguments:
        """Bind for invocation, injecting ``token`` only if the tool declared it.

        The injection is the whole of Milestone 4C's opt-in: a tool that does
        not ask for a token is called with exactly the arguments it was given,
        so its recorded arguments -- and every replay of them -- are unchanged.
        """
        merged = dict(kwargs or {})
        if self.cooperative:
            merged[CANCEL_TOKEN_PARAMETER] = (
                token if token is not None else CancellationToken()
            )
        return self.bind(args, merged)

    def arguments_for(self, bound: inspect.BoundArguments) -> dict[str, Any]:
        """Normalized, JSON-safe arguments -- this is what gets journalled.

        The injected ``cancel_token`` placeholder is dropped: a replay has to
        reproduce the recorded arguments exactly, and it has no token to
        substitute. What the caller asked for is the whole argument list.
        """
        arguments = {
            key: value
            for key, value in bound.arguments.items()
            if not (key == CANCEL_TOKEN_PARAMETER and value is None and self.cooperative)
        }
        return make_jsonable(arguments)

    def invoke(self, bound: inspect.BoundArguments) -> Any:
        return self.func(*bound.args, **bound.kwargs)


def check_retry_policy(
    policy: RetryPolicy | None, *, tool_name: str | None = None
) -> RetryPolicy | None:
    """Validate a retry policy where it is supplied, and return it unchanged.

    A policy that is not a :class:`~agent_runtime.retry.RetryPolicy` is a
    programming error, so it is reported where it was written rather than at the
    first failing tool call.
    """
    if policy is None or isinstance(policy, RetryPolicy):
        return policy
    where = f" for tool {tool_name!r}" if tool_name else ""
    raise RetryConfigurationError(
        f"retry_policy{where} must be a RetryPolicy or None, got "
        f"{type(policy).__name__}"
    )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    retry_policy: RetryPolicy | None = None,
    timeout: float | None = None,
) -> Any:
    """Register a function as a tool. Works bare or called::

        @tool
        def add(a: int, b: int): ...

        @tool(name="sum_two")
        def add(a: int, b: int): ...

        @tool(retry_policy=RetryPolicy(max_attempts=3))
        def fetch_data(url: str): ...

        @tool(timeout=5.0)                      # every call gets 5 seconds
        async def slow_tool(): ...
    """

    def decorator(target: Callable[..., Any]) -> Tool:
        doc = description
        if doc is None:
            doc = (inspect.getdoc(target) or "").strip().split("\n")[0]
        return Tool(
            name=name or target.__name__,
            func=target,
            description=doc or "",
            retry_policy=check_retry_policy(retry_policy, tool_name=name or target.__name__),
            timeout=timeout,
        )

    if func is None:
        return decorator
    return decorator(func)



# -- default tools -----------------------------------------------------------
#
# Registered by default so that `execution.call("add", a=2, b=3)` works out of
# the box, exactly as the examples in the spec use it.


def _add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def _subtract(a: float, b: float) -> float:
    """Return the difference of two numbers."""
    return a - b


def _multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def _divide(a: float, b: float) -> float:
    """Return the quotient of two numbers."""
    return a / b


BUILTIN_TOOLS: tuple[Callable[..., Any], ...] = (_add, _subtract, _multiply, _divide)


def _register_builtins(registry: "ToolRegistry") -> None:
    """Register the built-in tools under their public names (add, multiply, ...)."""
    for func in BUILTIN_TOOLS:
        registry.register(func, name=func.__name__.lstrip("_"))


def default_registry() -> "ToolRegistry":
    """A registry pre-populated with the built-in arithmetic tools."""
    registry = ToolRegistry(register_defaults=False)
    _register_builtins(registry)
    return registry


class ToolRegistry:
    """Name -> :class:`Tool` lookup used by the execution runtime."""

    def __init__(
        self,
        tools: Iterable[Tool | Callable[..., Any]] | None = None,
        *,
        register_defaults: bool = True,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        if register_defaults:
            _register_builtins(self)
        for item in tools or ():
            self.register(item)

    def register(self, item: Tool | Callable[..., Any], name: str | None = None) -> Tool:
        """Register a :class:`Tool` or a plain function; returns the stored tool."""
        if not isinstance(item, Tool):
            item = tool(item, name=name)
        elif name is not None and name != item.name:
            # Renaming must not drop what describes the tool: its retry policy
            # and its default timeout are properties of the function, not of
            # the name it happens to be filed under.
            item = Tool(
                name=name,
                func=item.func,
                description=item.description,
                retry_policy=item.retry_policy,
                timeout=item.timeout,
            )
        if item.name in self._tools:
            raise ToolAlreadyRegisteredError(f"Tool {item.name!r} is already registered")
        self._tools[item.name] = item
        return item

    def unregister(self, name: str) -> Tool:
        return self._tools.pop(name)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(self.names()) or "<none>"
            raise ToolNotFoundError(f"Unknown tool {name!r}. Registered tools: {known}") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a tool directly, without journaling (handy in tests/examples)."""
        tool_ = self.get(name)
        bound = tool_.bind(args, kwargs)
        try:
            return tool_.invoke(bound)
        except Exception as exc:
            raise ToolInvocationError(
                f"Tool {name!r} failed: {exc}",
                tool_name=name,
                error_type=type(exc).__name__,
                traceback_text="".join(
                    _traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            ) from exc
