"""Milestone 4A, part 1: the retry policy, error classification and the sleeper.

The invariant under test:

    A RetryPolicy is five immutable numbers, it decides only from an explicit
    error classification, and its backoff is deterministic -- so a test can
    assert the whole schedule without waiting for it.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from agent_runtime import (
    NO_RETRY,
    ErrorKind,
    PermanentToolError,
    RealSleeper,
    RecordingSleeper,
    RetryConfigurationError,
    RetryableToolError,
    RetryPolicy,
    Sleeper,
    ToolError,
    classify_error,
)


# ---------------------------------------------------------------------------
# The policy itself (§3)
# ---------------------------------------------------------------------------


def test_the_example_policy_is_the_documented_one():
    policy = RetryPolicy(max_attempts=3, initial_delay=0.1, multiplier=2.0, max_delay=10.0)

    assert policy.max_attempts == 3
    assert policy.initial_delay == 0.1
    assert policy.multiplier == 2.0
    assert policy.max_delay == 10.0


def test_a_policy_is_immutable():
    policy = RetryPolicy(max_attempts=3)

    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_attempts = 9  # type: ignore[misc]


def test_the_default_policy_never_retries():
    """A call is never retried unless it was configured to be."""
    assert RetryPolicy().max_attempts == 1
    assert RetryPolicy.none() is NO_RETRY
    assert NO_RETRY.should_retry(RetryableToolError("down"), 1) is False


def test_a_policy_round_trips_through_its_dict_form():
    policy = RetryPolicy(max_attempts=5, initial_delay=0.25, multiplier=3.0, max_delay=2.0)

    restored = RetryPolicy.from_dict(policy.to_dict())

    assert restored == policy


def test_a_missing_policy_reads_back_as_no_retries():
    assert RetryPolicy.from_dict(None) is NO_RETRY


def test_an_unreadable_policy_is_reported_rather_than_guessed():
    with pytest.raises(RetryConfigurationError) as excinfo:
        RetryPolicy.from_dict({"max_attempts": "lots"})

    assert "RetryPolicy" in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": -1},
        {"initial_delay": -0.1},
        {"multiplier": 0.5},
        {"max_delay": -1.0},
    ],
)
def test_an_impossible_policy_is_refused_when_it_is_defined(kwargs):
    with pytest.raises(RetryConfigurationError):
        RetryPolicy(**kwargs)


@pytest.mark.parametrize("attempt", [0, -1])
def test_attempt_numbers_are_one_based(attempt):
    policy = RetryPolicy(max_attempts=3)

    with pytest.raises(RetryConfigurationError):
        policy.should_retry(RetryableToolError("down"), attempt)
    with pytest.raises(RetryConfigurationError):
        policy.delay(attempt)


def test_a_policy_describes_itself():
    assert str(NO_RETRY) == "no retries"
    assert "max_attempts=3" in str(RetryPolicy(max_attempts=3))


# ---------------------------------------------------------------------------
# Error classification (§4)
# ---------------------------------------------------------------------------


def test_a_retryable_error_is_classified_as_retryable():
    assert classify_error(RetryableToolError("upstream is down")) is ErrorKind.RETRYABLE


def test_a_permanent_error_is_classified_as_permanent():
    assert classify_error(PermanentToolError("400 Bad Request")) is ErrorKind.PERMANENT


@pytest.mark.parametrize(
    "error",
    [
        ValueError("something odd"),
        RuntimeError("3 tests failed"),
        KeyError("missing"),
        None,
        "a string, not an exception",
    ],
)
def test_anything_else_is_unexpected(error):
    """Nothing is inferred: an unknown failure is never "transient enough"."""
    assert classify_error(error) is ErrorKind.UNEXPECTED


def test_the_classification_errors_are_tool_errors():
    """So a tool can raise them and an application's own handlers still work."""
    assert issubclass(RetryableToolError, ToolError)
    assert issubclass(PermanentToolError, ToolError)
    assert isinstance(RetryableToolError("x"), Exception)
    assert isinstance(PermanentToolError("x"), Exception)


def test_a_permanent_failure_is_never_retried_however_large_the_policy():
    policy = RetryPolicy(max_attempts=10, retry_on_unknown=True)

    assert policy.should_retry(PermanentToolError("no"), 1) is False
    assert policy.should_retry(PermanentToolError("no"), 9) is False


def test_an_unexpected_failure_is_not_retried_by_default():
    policy = RetryPolicy(max_attempts=5)

    assert policy.should_retry(RuntimeError("boom"), 1) is False


def test_an_unexpected_failure_can_be_retried_only_by_asking_for_it():
    """Opting in is explicit, because it is usually a way of hiding a bug."""
    policy = RetryPolicy(max_attempts=5, retry_on_unknown=True)

    assert policy.should_retry(RuntimeError("boom"), 1) is True
    # ... but it never overrides the exhaustion of the policy.
    assert policy.should_retry(RuntimeError("boom"), 5) is False


@pytest.mark.parametrize("attempt, expected", [(1, True), (2, True), (3, False), (4, False)])
def test_should_retry_stops_when_the_policy_runs_out_of_attempts(attempt, expected):
    policy = RetryPolicy(max_attempts=3)

    assert policy.should_retry(RetryableToolError("down"), attempt) is expected


def test_exhausted_reports_the_last_allowed_attempt():
    policy = RetryPolicy(max_attempts=3)

    assert policy.exhausted(2) is False
    assert policy.exhausted(3) is True


# ---------------------------------------------------------------------------
# Backoff (§7)
# ---------------------------------------------------------------------------


def test_backoff_is_exponential():
    policy = RetryPolicy(max_attempts=5, initial_delay=0.1, multiplier=2.0, max_delay=10.0)

    assert policy.delay(1) == pytest.approx(0.1)
    assert policy.delay(2) == pytest.approx(0.2)
    assert policy.delay(3) == pytest.approx(0.4)
    assert policy.delay(4) == pytest.approx(0.8)


def test_backoff_is_bounded_by_max_delay():
    policy = RetryPolicy(max_attempts=10, initial_delay=0.1, multiplier=10.0, max_delay=1.0)

    assert policy.delay(1) == pytest.approx(0.1)
    assert policy.delay(2) == pytest.approx(1.0)
    # 0.1 * 10 ** 5 would be 1000 seconds; the ceiling wins instead.
    assert policy.delay(6) == 1.0


def test_backoff_with_a_multiplier_of_one_is_flat():
    policy = RetryPolicy(max_attempts=3, initial_delay=0.5, multiplier=1.0, max_delay=10.0)

    assert [policy.delay(n) for n in (1, 2, 3)] == [0.5, 0.5, 0.5]


def test_backoff_is_deterministic():
    """Same inputs, same waits -- which is what lets a replay reproduce them."""
    policy = RetryPolicy(max_attempts=4, initial_delay=0.1, multiplier=2.0, max_delay=10.0)

    assert [policy.delay(n) for n in (1, 2, 3)] == [policy.delay(n) for n in (1, 2, 3)]


def test_a_zero_initial_delay_never_waits():
    policy = RetryPolicy(max_attempts=3, initial_delay=0.0)

    assert [policy.delay(n) for n in (1, 2)] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# The sleeper seam (§8)
# ---------------------------------------------------------------------------


def test_the_recording_sleeper_records_what_it_was_asked_to_wait():
    sleeper = RecordingSleeper()

    sleeper.sleep(0.1)
    sleeper.sleep(0.2)

    assert sleeper.delays == (0.1, 0.2)
    assert sleeper.total_delay == pytest.approx(0.3)
    assert len(sleeper) == 2


def test_the_recording_sleeper_can_be_cleared():
    sleeper = RecordingSleeper()
    sleeper.sleep(1.0)

    sleeper.clear()

    assert sleeper.delays == ()


def test_the_recording_sleeper_returns_immediately():
    """The whole point: a test asserts a schedule without spending it."""
    sleeper = RecordingSleeper()

    started = time.perf_counter()
    sleeper.sleep(30.0)
    elapsed = time.perf_counter() - started

    assert sleeper.delays == (30.0,)
    assert elapsed < 1.0


def test_the_real_sleeper_does_not_wait_when_there_is_nothing_to_wait_for():
    """A zero delay (what the default policy uses) costs nothing at all."""
    started = time.perf_counter()

    RealSleeper().sleep(0.0)

    assert time.perf_counter() - started < 1.0


def test_both_sleepers_satisfy_the_sleeper_protocol():
    assert isinstance(RealSleeper(), Sleeper)
    assert isinstance(RecordingSleeper(), Sleeper)
