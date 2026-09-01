"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from agent_runtime import EventJournal, Runtime


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
