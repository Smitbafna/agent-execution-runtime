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
* [ ] Tool retries with exponential backoff
* [ ] Idempotent tool execution
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

The runtime does **not** retry. The application decides, and the decision is
journalled so the next resume does not ask again.

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

## License

MIT
