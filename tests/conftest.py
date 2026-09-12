"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime import EventJournal, ExecutionStatus, Runtime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def db_path(tmp_path) -> str:
    """Path to a throwaway SQLite file for a single test."""
    return str(tmp_path / "agent.db")


@pytest.fixture
def runtime(db_path):
    """An open :class:`Runtime` backed by a temporary database."""
    with Runtime(db_path) as runtime_:
        yield runtime_


@pytest.fixture
def journal(runtime) -> EventJournal:
    return runtime.journal


@pytest.fixture
def execution(runtime):
    """A freshly started execution."""
    return runtime.start(goal="Test the execution runtime")


@pytest.fixture
def restarted(db_path):
    """A second :class:`Runtime` over the same file, as a new process would open it.

    Used to prove that what recovery reads came out of SQLite rather than out of
    an object still holding the state in memory.
    """
    with Runtime(db_path) as runtime_:
        yield runtime_


def crash_mid_tool(
    runtime, tool: str = "run_tests", *, start: bool = True, execution_id: str | None = None
) -> str:
    """Write the events of a tool call that never finished, and return its call id.

    ``start=False`` stops after ``ToolRequested``, which is what a crash between
    the request and the start looks like on disk. Without ``execution_id`` the
    events go to a freshly started execution.
    """
    journal = runtime.journal
    if execution_id is None:
        execution_id = runtime.start(goal="crash between events").id
    call_id = f"call_{tool}"
    journal.append_event(
        execution_id,
        "ToolRequested",
        {"call_id": call_id, "tool": tool, "arguments": {"suite": "unit"}},
    )
    if start:
        journal.append_event(execution_id, "ToolStarted", {"call_id": call_id, "tool": tool})
    return call_id


@pytest.fixture
def crashed_after_tool_started(runtime):
    """Id of an execution whose journal ends with an unresolved ``ToolStarted``."""
    crash_mid_tool(runtime, "run_tests", start=True)
    return runtime.list_executions()[0]


@pytest.fixture
def crashed_after_tool_requested(runtime):
    """Id of an execution whose journal ends with an unresolved ``ToolRequested``."""
    crash_mid_tool(runtime, "run_tests", start=False)
    return runtime.list_executions()[0]


def assert_recovery_required(execution) -> str:
    """Assert the execution is waiting on a decision; return the stuck call id."""
    assert execution.status is ExecutionStatus.RECOVERY_REQUIRED
    assert len(execution.incomplete_tools) == 1
    return execution.incomplete_tools[0].call_id

