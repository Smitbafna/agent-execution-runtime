## Milestone 4A design decisions worth your review

1. **A logical call keeps one `call_id`; the attempt number is payload, not identity.** The alternative -- a new `call_id` per attempt -- would make "attempt 3 of the same call" unrepresentable, and the milestone's own example (`call_123` with three attempts) rules it out. So `ToolRequested` is written once per call and `ToolStarted`/`ToolFailed`/`ToolCompleted` carry `attempt=N`. Every reducer defaults a missing `attempt` to 1, which is what keeps pre-Milestone-4A journals readable.
2. **The retry decision is journalled *before* the wait.** `ToolRetryScheduled` is committed, then the sleeper is asked to wait, then the next attempt starts. That is the whole reason a crash mid-backoff is recoverable: the process that died already made the decision durable, so a new process can read it instead of having to guess whether the retry was intentional.
3. **A scheduled retry is a decision, not an ambiguity, so it gets a new `ToolCallStatus.RETRYING` and is *not* `is_incomplete`.** Making it incomplete would have dragged the execution into `RECOVERY_REQUIRED` and made `Runtime.resume` refuse to continue -- for a state the journal describes exactly. `Execution.pending_retries` and `RecoveryInfo.has_pending_retries` surface it instead, and `Execution.continue_pending_retry` carries it on using the policy read back out of the call's `ToolRequested`.
4. **The retry decision is stored on the *attempt* that made it, not only on the call's pending slot.** The pending slot is consumed by the next `ToolStarted`, so a call that succeeded on attempt 3 would otherwise have no record that attempts 1 and 2 had been retried -- and a replay would have had no way to know to replay them. `ToolAttempt.scheduled_retry` keeps it.
5. **Retry decisions are made by the `ToolRunner`, not by `Execution`.** `Execution.call` asks the runner `begin_call`/`can_attempt`/`retry_decision`, exactly as it already asked it to `run` the attempt. NORMAL answers from the live exception plus the policy; the REPLAY runner answers from the recorded attempt history. A replay therefore never re-decides whether to retry -- which would be a second run, not a reproduction -- and the `assert sleeper.delays == (0.1, 0.2)` test covers NORMAL while `result.delays_replayed` covers REPLAY.
6. **`can_attempt` exists so replay can stop where the journal stops.** A history that ends after `ToolRetryScheduled` has nothing recorded for the attempt it scheduled. Milestone 3's rule ("a call the journal never settled is replayed as unsettled, not invented") applies unchanged: the replay emits the decision and stops, leaving the same pending retry the original had, rather than inventing an outcome for attempt 2.
7. **`RetryPolicy.max_attempts` defaults to 1.** Not to 3, and not to "whatever the tool said" -- one attempt is the behaviour every earlier milestone had, so nothing changes unless a caller opts in. It also means an ordinary `ValueError` from a tool is still recorded once and raised, exactly as in Milestone 1.
8. **Only `RetryableToolError` is retried; `retry_on_unknown` is the explicit opt-in for everything else.** Classifying failures by guessing -- by exception name, by message, by "it looks transient" -- is how retry loops turn bugs into timeouts. An unknown exception is `UNEXPECTED`, and a policy that wants those retried has to say so.
9. **`RetryPolicy.delay(n)` is the wait *after* attempt `n` failed**, so `delay(1) == initial_delay`. The milestone's formula `initial_delay * multiplier ** (attempt - 1)` fixes the indexing; with `max_attempts=3` that yields waits of 0.1 and 0.2 before attempts 2 and 3, which is what the tests assert and what the journal records.
10. **Waiting is a parameter, not a monkey-patch.** `Sleeper` is a one-method protocol with `RealSleeper` for production and `RecordingSleeper` for tests, threaded through `Runtime(sleeper=...)` into `Execution`. Consistent with Milestone 3's refusal to intercept `time` globally, and it means the backoff schedule is asserted rather than approximated.
11. **The resolved policy is journalled in `ToolRequested`, but kept out of `ToolCall`.** Configuration is not status, and a replay re-emits `ToolRequested` from the *recorded* policy (`ReplayEngine._recorded_policies`) so it stays faithful in a process where the tool is not registered at all. Keeping it out of the state also keeps checkpoint equality and replay equality unaffected by which policy this process happens to hold.
12. **`ToolInvocationError.attempts` reports how many were made.** The milestone asks for max-attempts behaviour to be observable; the exception is where a caller learns their call gave up, so it carries the count.
13. **`resolve_recovery` still refuses to re-run an unfinished call.** Retrying and resuming are different: a policy repeats failures the runtime recorded itself, while an open call's outcome is unknown to everyone. Milestone 4A did not move that line, and the recovery report now says so.

## Milestone 3 design decisions worth your review

1. **Replay re-runs the runtime rather than loading a stored answer.** A `ReplayExecution` is a real `Execution` over an in-memory journal and a REPLAY tool runner, so the same `Execution.call` path and the same reducers run again. The replayed state is therefore *derived* — which is the only way a divergence is detectable at all. A replay that merely re-read the checkpointed state could not tell a matching run from a wrong one.
2. **`Execution.call` now dispatches through a `ToolRunner`.** That was a small refactor of a Milestone 1 hot path, and it is what makes the guarantee structural: the REPLAY runner overrides the single method that would invoke a function, so there is no branch anywhere that can reach a tool during replay. `ReplayToolRunner.registry` is `None` on purpose — holding a registry would be the only way to accidentally run something.
3. **The replay journal mirrors the recorded events' `event_id` and `timestamp`.** Everything else — event type and payload — is produced by the re-running code. Mirroring identity is what makes `ToolCall.started_at` (taken from the event timestamp) line up, and it is why `replayed.state == original.state` holds field for field rather than only in shape. A replay that invented fresh timestamps would need that field excluded from comparison, which would weaken the check.
4. **A replayed call reuses its recorded `call_id`**, via a `_new_call_id` seam on `Execution`. Same reason as (3): identity is part of the state being compared.
5. **Replay validates *before* journalling.** The first version matched the request inside the runner, i.e. after `ToolRequested`/`ToolStarted` had been written, so a mismatch left a half-written tool call and an execution stuck in `RECOVERY_REQUIRED`. `ReplayExecution.call` now pre-checks with `ReplayToolRunner.check()`, mirroring the NORMAL path, which also validates arguments before writing anything.
6. **A call the journal never settled is replayed as unsettled, not invented.** `ToolRequested`/`ToolStarted` are re-emitted, no outcome is made up, and the replay ends in `RECOVERY_REQUIRED` — the same status the original had. Inventing a result would be the one thing replay must never do.
7. **A start point with an unresolved call is refused outright.** Replaying from mid-call would mean resolving a `ToolCompleted` whose `ToolRequested` is behind the start point. That is Milestone 2's "the journal does not say, so do not guess" applied to a start point, so it raises `ReplayMismatchError(kind="sequence")` rather than a confusing `StateReconstructionError` from deep inside the reducers.
8. **`ReplayMismatchError` always raises; `matched` is informational.** The spec asked not to return a bare `False`, so `run()` raises on divergence and the returned `ReplayResult.matched` is the post-condition check. `ReplayMismatchError.kind` is a stable string (`tool_name`, `arguments`, `sequence`, `unexpected_tool_call`, …) so callers can branch without parsing the message.
9. **`__init__.py` had a duplicated import block** (two `from .exceptions import ...` statements listing overlapping names). Left as-is rather than tidied, to keep the diff about Milestone 3 — worth a cleanup pass.
10. **No global interception of `time`/`random`/`uuid`.** Deliberate, per the milestone. Replay substitutes recorded *tool outputs*; anything else that varies is an external input the execution should expose as a tool. The determinism tests demonstrate this by using tools that read the clock, the RNG, `uuid` and `os.environ` directly and then replaying them successfully.

## Milestone 2 design decisions worth your review

1. **`Execution.state` is now cached** (invalidated on every write). Milestone 1 rebuilt it on each access. The cache is what makes §2's "state + sequence atomically" achievable — a rebuild mid-checkpoint could snapshot a state whose sequence had moved on. `checkpoint()` additionally re-reads and rebuilds if the DB moved, and `create()` re-validates in-transaction anyway.
2. **`RECOVERY_REQUIRED` is never persisted.** A checkpoint stores `RUNNING`; recovery re-derives the diagnosis from events. Otherwise two sources would own one field.
3. **Added `ExecutionCancelled`/`ToolCancelled` events.** `CANCELLED` is in the required status list but needed a journal event, otherwise the status would be a value the journal can't produce.
4. **A corrupt checkpoint raises instead of falling back** to a full replay — falling back would hide broken stored data behind a working recovery.

## Also

- `tests/test_retry_policy.py` (40 tests), `tests/test_retries.py` (36),
  `tests/test_retry_durability.py` (9) and `tests/test_retry_replay.py` (13)
  cover Milestone 4A: the policy and its backoff, the classification rules, the
  attempt loop and its journal, checkpoint and process-restart recovery across
  retry events, and replay of a retried run -- including the two interrupted
  shapes (a crash *during* a retried attempt, and a crash *during* the backoff),
  where replay must reproduce the prefix without inventing the missing outcome.
- `agent_runtime/retry.py` is new and holds only what the milestone asked for:
  `RetryPolicy`, `RetryDecision`, `ErrorKind`/`classify_error`, and the
  `Sleeper` protocol with its two implementations.
- `agent_runtime/execution.py` now has an attempt loop (`_attempt_until_settled`,
  `_schedule_retry`) behind the same public `call`, plus `continue_pending_retry`
  for the retry a crash interrupted.
- `agent_runtime/state.py` gained `ToolAttempt` and `PendingRetry`, the
  `RETRYING` status, and attempt-aware reducers. Old checkpoints and journals
  still read back: every new field defaults.
- `examples/retries.py` walks through all of it, including a child process that
  dies during a backoff.
- **Note:** `__pycache__/*.pyc` files are tracked in git in this repo. I left that alone, but you'll want a `.gitignore` and `git rm -r --cached` at some point.