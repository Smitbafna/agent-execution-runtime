"""Journal tests: persistence, ordering, isolation of sequences, immutability."""

from __future__ import annotations

import dataclasses
import json
import sqlite3

import pytest

from agent_runtime import (
    DuplicateSequenceError,
    Event,
    EventType,
    SequenceError,
)
from agent_runtime.exceptions import EventNotFoundError


def test_events_are_persisted(journal):
    event = journal.append(Event.create("exec_1", 1, EventType.EXECUTION_STARTED, {"goal": "g"}))

    rows = journal.store.query_all("SELECT * FROM events WHERE event_id = ?", (event.event_id,))

    assert len(rows) == 1
    assert rows[0]["execution_id"] == "exec_1"
    assert rows[0]["sequence"] == 1
    assert rows[0]["event_type"] == "ExecutionStarted"
    assert json.loads(rows[0]["payload"]) == {"goal": "g"}


def test_events_retain_sequence_order(journal):
    for sequence in range(1, 6):
        journal.append(Event.create("exec_1", sequence, EventType.TOOL_STARTED, {"n": sequence}))

    events = journal.get_events("exec_1")

    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
    assert [event.payload["n"] for event in events] == [1, 2, 3, 4, 5]


def test_events_can_be_loaded_by_execution_id(journal):
    journal.append_event("exec_a", EventType.EXECUTION_STARTED, {"goal": "a"})
    journal.append_event("exec_b", EventType.EXECUTION_STARTED, {"goal": "b"})

    events = journal.get_events("exec_a")

    assert len(events) == 1
    assert events[0].payload["goal"] == "a"
    assert journal.get_events("unknown") == []


def test_multiple_executions_have_independent_sequences(journal):
    journal.append_event("exec_a", EventType.EXECUTION_STARTED, {"goal": "a"})
    journal.append_event("exec_b", EventType.EXECUTION_STARTED, {"goal": "b"})
    journal.append_event("exec_a", EventType.EXECUTION_COMPLETED, {"result": 1})

    assert journal.get_last_sequence("exec_a") == 2
    assert journal.get_last_sequence("exec_b") == 1
    assert journal.count_events("exec_a") == 2
    assert journal.count_events("exec_b") == 1
    assert journal.list_execution_ids() == ["exec_a", "exec_b"]

    assert [event.sequence for event in journal.get_events("exec_a")] == [1, 2]
    assert [event.sequence for event in journal.get_events("exec_b")] == [1]


def test_get_last_sequence_is_zero_for_unknown_execution(journal):
    assert journal.get_last_sequence("never-seen") == 0
    assert journal.has_execution("never-seen") is False


def test_get_event_by_sequence(journal):
    journal.append_event("exec_1", EventType.EXECUTION_STARTED, {"goal": "g"})

    assert journal.get_event("exec_1", 1).event_type is EventType.EXECUTION_STARTED
    with pytest.raises(EventNotFoundError):
        journal.get_event("exec_1", 99)


def test_append_rejects_out_of_order_sequence(journal):
    journal.append_event("exec_1", EventType.EXECUTION_STARTED, {"goal": "g"})

    with pytest.raises(SequenceError):
        journal.append(Event.create("exec_1", 5, EventType.EXECUTION_COMPLETED, {}))

    # The failed append left nothing behind.
    assert journal.get_last_sequence("exec_1") == 1


def test_append_rejects_non_positive_sequence(journal):
    with pytest.raises(SequenceError):
        journal.append(Event.create("exec_1", 0, EventType.EXECUTION_STARTED, {}))


def test_database_prevents_duplicate_sequences(journal):
    event = journal.append_event("exec_1", EventType.EXECUTION_STARTED, {"goal": "g"})

    # Bypass the journal API to prove the UNIQUE constraint also protects us.
    with pytest.raises(sqlite3.IntegrityError):
        journal.store.execute(
            "INSERT INTO events (event_id, execution_id, sequence, event_type, payload, timestamp)"
            " VALUES ('evt_other', 'exec_1', 1, 'ExecutionStarted', '{}', 'now')"
        )

    # ... and the primary key rejects a reused event id at an otherwise free
    # sequence.
    with pytest.raises(sqlite3.IntegrityError):
        journal.store.execute(
            "INSERT INTO events (event_id, execution_id, sequence, event_type, payload, timestamp)"
            " VALUES (?, 'exec_1', 2, 'ExecutionStarted', '{}', 'now')",
            (event.event_id,),
        )


def test_duplicate_event_id_is_rejected(journal):
    event = journal.append_event("exec_1", EventType.EXECUTION_STARTED, {"goal": "g"})
    clone = Event(
        event_id=event.event_id,
        execution_id="exec_1",
        sequence=2,
        event_type=EventType.EXECUTION_COMPLETED,
        timestamp=event.timestamp,
        payload={},
    )

    with pytest.raises(DuplicateSequenceError):
        journal.append(clone)


def test_events_are_immutable(journal):
    event = journal.append_event("exec_1", EventType.EXECUTION_STARTED, {"goal": "g"})

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.sequence = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        event.payload["goal"] = "tampered"  # type: ignore[index]

    assert journal.get_events("exec_1")[0] == event


def test_event_round_trips_through_json(journal):
    event = journal.append_event(
        "exec_1", EventType.TOOL_COMPLETED, {"call_id": "call_1", "result": {"nested": [1, 2]}}
    )

    reloaded = Event.from_dict(json.loads(event.to_json()))

    assert reloaded == event
    assert reloaded.event_type is EventType.TOOL_COMPLETED


def test_payloads_are_stored_as_json_text(journal):
    journal.append_event("exec_1", EventType.TOOL_COMPLETED, {"result": {"a": [1, None, True]}})

    raw = journal.store.query_all("SELECT payload FROM events")[0]["payload"]

    assert isinstance(raw, str)
    assert json.loads(raw) == {"result": {"a": [1, None, True]}}
