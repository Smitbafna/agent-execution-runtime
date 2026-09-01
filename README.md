# Agent Execution Runtime

A durable execution runtime for AI agents that makes agent workflows **persistent, recoverable, replayable, and debuggable**.
It records tool calls and execution state as an append-only event journal, allowing agents to resume after crashes without losing progress.
It also supports deterministic replay, checkpoints, retries, idempotency, and eventually time-travel debugging and execution branching.

## Features

* [ ] Durable execution journal
* [ ] Persistent agent and tool state
* [ ] Crash recovery and resume
* [ ] Deterministic execution replay
* [ ] Checkpoints and state reconstruction
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

### Run

```bash
agent-runtime --help
```

## License

MIT
