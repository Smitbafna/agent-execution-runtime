# Agent Execution Runtime

A durable execution runtime for AI agents that makes agent workflows **persistent, recoverable, replayable, and debuggable**.
It records tool calls and execution state as an append-only event journal, allowing agents to resume after crashes without losing progress.
It also supports deterministic replay, retries, idempotency, and eventually time-travel debugging and execution branching.

## Features

* [x] Durable execution journal
* [x] Persistent agent and tool state
* [x] Crash recovery and resume
* [x] Deterministic execution replay
* [x] Checkpoints and state reconstruction
* [x] Tool retries with exponential backoff
* [x] Idempotent tool execution
* [ ] Tool timeouts and cancellation
* [ ] Execution inspection and debugging
* [ ] Time-travel debugging
* [ ] Execution branching

## Setup

### Requirements

* Python 3.12+
* `uv` or `pip`

### Installation

```bash
git clone https://github.com/Smitbafna/agent-execution-runtime.git
cd agent-execution-runtime

uv sync
```

Or:

```bash
pip install -e .
```

## Run

```bash
agent-runtime --help
```

## Milestone 4B: idempotency & crash-safe side effects

The invariant this milestone is built around:

> A side-effecting tool never runs twice for one idempotency key unless the
> application explicitly asks for it, and every key whose outcome the runtime
> does not know is reported rather than guessed at.

### The problem

```text
execution.call("send_email", ..., idempotency_key="email-123")
    |
    +-- claim the key                 committed to SQLite
    +-- the email is sent             outside SQLite, not undoable
    +-- store the outcome             <-- the process dies HERE
    |
    +-- recovery: "did the email go out?"
```

The claim commits and the side effect happens in different systems, and **no
implementation can make those two writes atomic**:

```text
BEGIN
<external API call>     <-- not in the transaction, not reversible
INSERT outcome
COMMIT
```

If the answer were assumed to be "no", every recovery would send the email
again. If it were assumed to be "yes", a crash *before* the request would lose
it. Neither is knowledge. So this milestone does not offer exactly-once
execution — it offers a durable claim, a visible ambiguity, and an explicit
decision.

### Using it

```python
execution.call(
    "send_email",
    to="user@example.com",
    body="hello",
    idempotency_key="welcome-user-123",   # names the intended side effect
)

execution.call(
    "send_email",
    to="user@example.com",
    body="hello",
    idempotency_key="welcome-user-123",   # returns the stored result,
)                                         # does NOT send again

runtime.idempotency_record("welcome-user-123")
# IdempotencyRecord(status=COMPLETED, result={'message_id': 'msg-1'}, attempts=1)
```

The lifecycle of a key:

```text
no record  -> claim -> PENDING -> tool runs -> COMPLETED / FAILED
                                     ^
                                     |  a crash here leaves it here
```

| | key state | what `call` does |
| --- | --- | --- |
| first use | *no record* | claims the key, runs the tool, stores the outcome |
| duplicate | `COMPLETED` | returns the stored result, **runs nothing** |
| after a crash | `PENDING` | raises `IdempotencyRecoveryRequiredError` |
| after a recorded failure | `FAILED` | raises `IdempotencyKeyFailedError` |
| after `resolve_idempotency(..., "retry")` | `PENDING` + authorized | runs the tool, once |

### After a crash

```python
with Runtime("agent.db") as runtime:
    execution = runtime.resume(execution_id)

    execution.status                   # RECOVERY_REQUIRED
    execution.unresolved_idempotency   # the keys it cannot answer for
    execution.recovery_info()          # the journal's view, plus those keys

    # "I checked the provider: it really was sent."
    execution.resolve_idempotency(
        "welcome-user-123", action="mark_completed", result={"message_id": "msg-1"}
    )
    # or: "it did not go out, send it again"
    execution.resolve_idempotency("welcome-user-123", action="retry")
    # or: "it definitely did not happen"
    execution.resolve_idempotency("welcome-user-123", action="mark_failed", error="...")

    execution.complete()          # ...and only now can the execution close
```

An execution with an unresolved key reports `RECOVERY_REQUIRED` and cannot be
completed — closing out an execution whose side effect is of unknown outcome
would hide the very thing the key exists to expose. `fail()` and
`mark_cancelled()` remain available, because giving up *is* a decision.

```
Execution: exec_123
Status: RECOVERY_REQUIRED
...
Unresolved idempotency keys:

    welcome-user-123 [PENDING] send_email(call_456) claimed by exec_123 (attempt 1, outcome unknown)
    claimed at: 2026-04-01T10:00:03+00:00
    tool: send_email(call_456)

A PENDING key means the runtime committed the intent to run the side effect and
never recorded an outcome -- the process may have died between the two. It will
not run the tool again on its own; settle each key with
execution.resolve_idempotency(key, ...).
```

`retry` is deliberately **one** authorization, not a standing permission: the
record stays `PENDING` (nobody has said the effect happened) and the next claim
consumes it. A second call without another `resolve_idempotency` is refused
again.

### Keys, calls, and attempts

A key belongs to the **logical call**, not to an attempt, so Milestone 4A's
retries share one claim — three attempts of `call_123` are still one claim of
`payment-456`, and the record reports `attempts=1`:

```text
ToolRequested  idempotency_key=payment-456
ToolStarted      attempt=1 -> ToolFailed
ToolRetryScheduled             attempt=2
ToolStarted      attempt=2 -> ToolCompleted
```

### What the database guarantees, and what it cannot

| | |
| --- | --- |
| uniqueness | `idempotency_key` is the PRIMARY KEY: two local executions cannot both hold a claim |
| atomicity | claiming, resolving and consuming an authorization each happen in one `IMMEDIATE` transaction |
| concurrency | two local writers serialize on SQLite's write lock; one wins, the other sees the claim |
| *not* covered | anything about the external world — a `PENDING` key after a crash may or may not correspond to a request that succeeded |

There is no distributed lock here, and no claim about two machines. The
concurrency guarantee is exactly "one local writer per key".

### Replay is read-only

A replay answers a keyed call from the recorded history instead of the ledger:

```text
idempotency key -> the recorded ToolRequested -> the recorded result
```

`ReplayIdempotencyGuard` has no store behind it, so a replay cannot claim a
key, insert a `PENDING` record, overwrite a result, or execute a tool — there is
no code path from a replay to the database. `tests/test_idempotency_replay.py`
compares every column of every row before and after.

### CLI

```bash
agent-runtime idempotency list                      # the PENDING keys
agent-runtime idempotency list --all --json         # every record
agent-runtime idempotency show welcome-user-123
agent-runtime idempotency resolve welcome-user-123 --action retry
agent-runtime idempotency resolve welcome-user-123 --action mark_completed \
    --result '{"message_id": "msg-1"}'
```

### API

| | |
| --- | --- |
| `execution.call(..., idempotency_key=...)` | claim, execute, store the outcome |
| `execution.idempotency_record(key)` | the stored claim, or `None` |
| `execution.pending_idempotency` | every claim still in flight |
| `execution.unresolved_idempotency` | the ones needing a decision |
| `execution.needs_idempotency_resolution` | bool |
| `execution.resolve_idempotency(key, action, ...)` | `retry` / `mark_completed` / `mark_failed` |
| `runtime.idempotency` | the `IdempotencyStore` itself |
| `runtime.idempotency_records(...)`, `runtime.pending_idempotency(...)` | queries |
| `runtime.resolve_idempotency(key, action, ...)` | the operator's API, no execution handle needed |
| `recovery_info.pending_idempotency` | what a crash left unresolved |

## Milestone 4A: retries and attempt semantics

The invariant this milestone is built around:

> A logical tool call keeps one stable `call_id` while it may make several
> numbered attempts; every attempt is journalled, and the state reports the
> final one.

```python
from agent_runtime import RetryPolicy, RetryableToolError, Runtime

runtime = Runtime("agent.db")

@runtime.tool(retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.1))
def fetch_data(url: str):
    if upstream_is_down():
        raise RetryableToolError("503")   # eligible for retry
    if bad_request(url):
        raise PermanentToolError("400")   # never retried
    return download(url)

execution = runtime.start(goal="Fetch the data")
execution.call("fetch_data", url="https://example.com/data")

call = execution.tool_calls[0]
call.call_id      # call_123 -- the same id for every attempt
call.status       # COMPLETED
call.attempt      # 3 -- the attempt that settled it
call.attempts     # (attempt 1 !! 503, attempt 2 !! 503, attempt 3 -> {...})
```

Three failures of one call and a success is a **COMPLETED call on attempt 3**,
not a failed call -- and the two failures are still in the journal, in full.

### Logical calls versus attempts

| | |
| --- | --- |
| `execution.call("fetch_data", ...)` | one **logical call**, one stable `call_id` |
| one invocation of that tool | one **attempt**, numbered from 1 |
| `call.attempt` | the final attempt number |
| `call.attempts` | every attempt, folded from the journal |

### What gets retried

Only a tool saying so. `RetryableToolError` is the one thing that makes an
attempt eligible for a retry, and nothing is inferred from how transient a
failure looks:

| | |
| --- | --- |
| `RetryableToolError` | retried while attempts remain |
| `PermanentToolError` | never retried, whatever the policy says |
| anything else | **not** retried by default (`retry_on_unknown=True` opts in) |

### Configuring the policy

```python
# per tool
@tool(retry_policy=RetryPolicy(max_attempts=3))
def fetch_data(url: str): ...

# per call -- this one wins
execution.call("fetch_data", retry_policy=RetryPolicy(max_attempts=5))
```

Call-level, then tool-level, then *no retries*. With nothing configured, a call
gets exactly one attempt: the behaviour every milestone before this one had.

### Backoff, and not waiting for it in tests

`delay(n) = min(initial_delay * multiplier ** (n - 1), max_delay)` -- so with
the policy above, the waits after attempts 1, 2 and 3 are `0.1`, `0.2`, `0.4`,
capped at `max_delay`. No jitter, so the schedule is deterministic.

The runtime waits through an injectable `Sleeper`, and production is only one
implementation of it:

```python
from agent_runtime import RecordingSleeper, Runtime

sleeper = RecordingSleeper()
runtime = Runtime("agent.db", sleeper=sleeper)     # tests never wait
...
assert sleeper.delays == (0.1, 0.2)                # they assert the schedule
```

Nothing monkey-patches `time.sleep`.

### The retry is durable, and it is journalled

`ToolRetryScheduled` is committed **before** the wait and before the attempt, so
a process that dies mid-backoff leaves the decision behind:

```text
ToolRequested
ToolStarted          attempt=1
ToolFailed           attempt=1
ToolRetryScheduled   attempt=2   ← committed here
ToolStarted          attempt=2
ToolCompleted        attempt=2
```

A scheduled-but-unstarted retry is a *decision*, not an ambiguity, so recovery
surfaces it rather than asking the application to resolve it:

```python
runtime = Runtime("agent.db")                 # a brand new process
execution = runtime.resume(execution_id)

execution.status           # RUNNING -- not RECOVERY_REQUIRED
execution.pending_retries  # (attempt 1 failed; retry 2 scheduled after 0.1s,)

if execution.pending_retries:
    execution.continue_pending_retry(call_id)  # makes attempt 2, using the
                                               # policy the call was journalled with
```

Checkpoints and replay both work across retry events, and the crash tests kill a
real child process with `os._exit` in the middle of a backoff to prove it.

### Replaying a retried run

Replay substitutes *recorded attempts*, through the same REPLAY runner as
Milestone 3: attempt 1's recorded failure, attempt 2's recorded failure,
attempt 3's recorded result, with no tool executing and no backoff spent. The
replayed state equals the original's field for field, attempt numbers included.

```bash
agent-runtime replay <execution-id>
```

```text
✓ fetch_data (attempt 3/3)

Replay completed

Events replayed: 11
Tools replayed: 1
Retries replayed: 2 (delays: [0.1, 0.2])
State: MATCHED
```

### API

| | |
| --- | --- |
| `RetryPolicy(max_attempts, initial_delay, multiplier, max_delay)` | immutable; `should_retry(error, attempt)`, `delay(attempt)` |
| `RetryPolicy.none()` / `NO_RETRY` | the default: exactly one attempt |
| `RetryableToolError` / `PermanentToolError` | the explicit classification |
| `retry_on_unknown=True` | opt into retrying exceptions the tool did not classify |
| `execution.call(name, retry_policy=...)` | call-level policy, beats the tool's |
| `execution.pending_retries` | retries scheduled but not started |
| `execution.continue_pending_retry(call_id)` | carry one on after a crash |
| `Runtime(..., sleeper=...)` | how backoff waits; `RecordingSleeper` in tests |
| `ToolCall.attempt` / `.attempts` / `.pending_retry` | the attempt semantics in the state |

## Milestone 3: deterministic replay

The invariant this milestone is built around:

> Given the same recorded execution history, replay must reproduce the same
> execution state without repeating external side effects.

```python
from agent_runtime import Runtime

runtime = Runtime("agent.db")

execution = runtime.start(goal="Perform calculations")
execution.call("add", a=2, b=3)
execution.call("multiply", a=5, b=10)
execution.complete()

result = runtime.replay(execution.id)

assert result.final_state == execution.state
assert result.matched
```

A `create_github_issue`, a `send_email` or a `database_write` from that
execution does **not** happen a second time: every tool result is read out of
the journal instead of being produced by running the tool.

### How a replay works

Replay is not a shortcut that loads the stored final answer. It re-executes the
runtime's own logic — the same `Execution.call` path, the same reducers — with
two substitutions:

| | original run | replay |
| --- | --- | --- |
| tool runner | `NORMAL` — invokes the function | `REPLAY` — returns the recorded outcome |
| journal | SQLite | in memory, so the original is never touched |

Because the logic really re-runs, the replayed state is *derived* rather than
copied — which is what makes a divergence visible at all.

### API

| | |
| --- | --- |
| `runtime.replay(id)` | replay a whole execution; returns a `ReplayResult` |
| `runtime.replay(id, from_sequence=N)` | replay only what came after `N` |
| `runtime.replay_engine(id)` | drive the replay yourself, one call at a time |
| `result.matched` | the replayed state equals the original |
| `result.final_state` / `result.state` | the state the replay derived |
| `result.events_replayed` / `result.tools_replayed` | how much was replayed |
| `result.steps` | the in-memory trace of what was served |
| `result.journal_unchanged` | replay is read-only, and says so |

### Replay from a checkpoint

```python
full    = runtime.replay(id)
partial = runtime.replay(id, from_sequence=checkpoint.sequence)

assert full.final_state == partial.final_state   # both routes agree
```

`from_sequence` uses the stored checkpoint when there is one at that sequence
and folds the event prefix otherwise; the two must agree, and the tests assert
that they do.

### When replay does not match

A replay that diverges raises `ReplayMismatchError` — it is never reported as a
bare `False` — carrying the sequence and both sides of the divergence:

```text
ReplayMismatchError

Reason: replay called 'search_code' with different arguments than the recorded call #1
Execution: exec_123
Sequence: 7

Expected:
    tool: "search_code"
    arguments: {"query": "authentication"}

Received:
    tool: "search_code"
    arguments: {"query": "database"}
```

The `kind` field is stable and machine-readable: `tool_name`, `arguments`,
`sequence`, `unexpected_tool_call`, `missing_recorded_call`,
`missing_recorded_outcome`, `state`, `journal_mutated`.

A call the journal never settled is not replayed into an invented result: the
request and start are re-emitted, no outcome is made up, and the replay ends in
the same `RECOVERY_REQUIRED` the original did.

### Determinism, and what replay does *not* do

Replay substitutes **recorded tool outputs**. It deliberately does not sandbox
the rest of Python — there is no monkey-patching of `time` or `random` here.
Values such as the clock, random numbers, generated UUIDs, environment
variables, network responses and LLM responses are **external inputs**: if one
of them influenced the execution, expose it as a tool, so its value lands in the
journal and replay can serve it back. `tests/test_replay.py` pins this down with
tools that read `time`, `random`, `uuid` and `os.environ` directly.

### CLI

```bash
agent-runtime replay <execution-id>
agent-runtime replay <execution-id> --from-sequence 500
agent-runtime replay <execution-id> --json
agent-runtime list
```

```text
Execution: exec_1827

Replaying...

✓ search_code
✓ read_file
✗ run_tests

Replay completed

Events replayed: 27
Tools replayed: 8
State: MATCHED
```

Exit codes: `0` matched, `1` replay mismatch, `2` any other runtime error.

## Milestone 2: checkpoints and crash recovery

The invariant this milestone is built around:

> A recovered execution must represent a valid state derived from a consistent
> checkpoint plus the events that were durably persisted after it.

```python
from agent_runtime import Runtime

runtime = Runtime("agent.db")

execution = runtime.start(goal="Perform calculations")
execution.call("add", a=2, b=3)
execution.checkpoint()                        # a snapshot of the state so far
execution.call("multiply", a=5, b=10)

# ... process dies, and a new process picks the work up:
runtime = Runtime("agent.db")
execution = runtime.resume(execution.id)      # checkpoint @ 4 + events 5..7

print(execution.state)                        # identical to a full replay
```

### What recovery does

| | |
| --- | --- |
| `runtime.resume(id)` | rebuilds the state from the latest checkpoint plus the events after it; with no checkpoint it replays the whole journal |
| `runtime.reconstruct_state(id)` | folds the **entire** journal — the ground truth a checkpointed recovery is checked against |
| `runtime.recovery_info(id)` | what recovery used, what it replayed, and what a crash left unfinished |
| `execution.checkpoint()` | persists the current state with the sequence of the latest event, in one transaction |
| `execution.incomplete_tools` | tool calls the journal never recorded an outcome for |
| `execution.resolve_recovery(call_id, action)` | records the application's decision: `mark_completed`, `mark_failed` or `cancel` |
| `execution.mark_cancelled(reason)` | ends an execution deliberately, with `CANCELLED` |

### Execution states

`RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, and `RECOVERY_REQUIRED` — the last
one meaning the journal ends with tool work whose outcome is unknown:

```text
ToolRequested → ToolStarted → 💥 process crash
```

Recovery surfaces that ambiguity instead of guessing:

```text
Execution: exec_123
Status: RECOVERY_REQUIRED

Incomplete operations:

Tool: run_tests
Started at sequence: 17
Status: STARTED
Arguments:
    suite='unit'
```

The runtime does **not** re-run work whose outcome the journal does not
record. Milestone 4A retries *recorded failures* -- a `ToolFailed` it wrote
itself -- and never a call whose outcome is unknown, so this decision still
belongs to the application, and it is journalled so the next resume does not
ask again. A retry that was *scheduled* before the crash is a different thing
entirely: see `execution.pending_retries`.

### Consistency

A checkpoint is written by a single `INSERT` whose `WHERE` clause checks the
journal *inside the same transaction*, and the table's foreign key pins its
sequence to a real event. A snapshot that would claim a sequence the history has
not reached — or that disagrees with the state stored beside it — is refused with
`InconsistentCheckpointError` rather than stored, and an unreadable one is
reported with `CorruptCheckpointError` rather than silently ignored.

## Examples

```bash
python examples/basic_usage.py          # journal, tools, recovery
python examples/checkpoint_recovery.py  # checkpoint, real crash, recovery, decision
python examples/replay.py               # replay without re-running tools, and a mismatch
python examples/retries.py              # attempts, backoff, classification, a crash mid-retry
python examples/idempotency.py          # keys, a crash between claim and outcome, resolution
```

## Tests

```bash
python -m pytest
```

The suite includes crash scenarios in which a child process is killed with
`os._exit` at each boundary — after `ToolRequested`, after `ToolStarted`, after
`ToolCompleted`, and between a checkpoint's `INSERT` and its `COMMIT` — and every
one of them asserts that the recovered state equals a full reconstruction.

`tests/test_replay.py` adds the replay suite: completed and interrupted
executions, failed tools, checkpoint replay, every mismatch kind, and a
counter-based tool proving the real function never runs twice.

`tests/test_retry_policy.py`, `tests/test_retries.py`,
`tests/test_retry_durability.py` and `tests/test_retry_replay.py` are the
Milestone 4A suite: the policy and its backoff, the classification rules, the
attempt loop and its journal, checkpoint and process-restart recovery across
retry events, and replay of a retried run. One of them kills a child process
with `os._exit` in the middle of a backoff — after `ToolRetryScheduled` was
committed, before the attempt it scheduled — and asserts the recovered
execution still knows the retry was scheduled and still holds the attempt
history. No test waits: they all inject a `RecordingSleeper`, so the whole suite
runs in seconds.

`tests/test_idempotency.py`, `tests/test_idempotency_crash.py`,
`tests/test_idempotency_concurrency.py`, `tests/test_idempotency_replay.py` and
`tests/test_idempotency_cli.py` are the Milestone 4B suite: the lifecycle of a
key, duplicates that do not execute, failures that are not retried behind your
back, pending records that make an execution `RECOVERY_REQUIRED`, the three
explicit resolutions, checkpoint interaction, and the transaction boundaries
between the claim and the outcome. Two of them are worth calling out:

* the crash tests kill a child process with `os._exit` at two points — after the
  side effect but before `ToolCompleted`, and after `ToolCompleted` but before
  the outcome is stored — and assert that a second process refuses to send the
  email again and reports the key as unresolved;
* the replay tests compare every column of every row of `idempotency_records`
  before and after a replay, so "read-only" is checked, not assumed.

## License

MIT
