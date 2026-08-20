# Agentic_loop

An explicit, auditable agentic loop over the Claude Messages API.

The loop owns its own **plan → act → observe** cycle rather than delegating it to
a helper, so every request, tool call, stop condition and token is visible in
code you can read. That is the point: in a regulated context, a loop you can
audit is worth more than a loop that is three lines shorter.

- **One engine, in Python** — `src/agentic_loop/`
- **A typed TypeScript boundary over it** — `ts/`, which runs the Python CLI and
  types its JSON contract. There is no second implementation of the loop.

## What the loop does

```
                ┌───────────────────────────────────────────────┐
                │  budget check — may we issue another request?  │
                └───────────────────┬───────────────────────────┘
                                    │ yes
                     ┌──────────────▼──────────────┐
        ┌───────────▶│  PLAN: send history + tools │
        │            └──────────────┬──────────────┘
        │                           │
        │            ┌──────────────▼──────────────┐
        │            │  OBSERVE: read stop_reason  │
        │            └──────────────┬──────────────┘
        │                           │
        │   tool_use ┌───────────────┴──────────────┬─ end_turn ──▶ completed
        │            │                              ├─ refusal ───▶ reported
        │  ┌─────────▼─────────┐                    ├─ max_tokens ▶ reported
        └──┤  ACT: run tools,  │                    └─ pause_turn ▶ resume
           │  return ALL       │
           │  results in ONE   │
           │  user message     │
           └───────────────────┘
```

Every branch is handled explicitly. A run that stops early — budget exhausted,
refused, truncated — reports *why*; it is never presented as a finished answer.

## Requirements

- Python **≥ 3.11**
- Node **≥ 20** (only for the TypeScript wrapper)
- API credentials: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an
  `ant auth login` profile. The SDK resolves them; no code in this repository
  handles a raw secret.

## Install

```bash
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"
```

## Commands

| Task | Command |
| --- | --- |
| Install (with dev tools) | `uv pip install -e ".[dev]"` |
| Test | `python -m pytest` |
| Test with coverage | `python -m pytest --cov=agentic_loop --cov-report=term-missing` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Typecheck | `mypy` |
| Run | `agentic-loop "your prompt"` |
| Inspect the tool catalogue | `agentic-loop --print-tools` |
| TypeScript: typecheck | `cd ts && npm run typecheck` |
| TypeScript: test | `cd ts && npm test` |
| TypeScript: everything | `cd ts && npm run check` |

The full test suite runs with **no API key and no network** — the loop takes its
client by injection and the tests supply a fake that reproduces the SDK surface
it uses.

## Use it from Python

```python
import anthropic
from agentic_loop import AgenticLoop, Budget, JsonLogTracer, LoopConfig

loop = AgenticLoop(
    client=anthropic.Anthropic(),
    config=LoopConfig(model="claude-opus-5", effort="high"),
    budget=Budget(max_iterations=8, max_total_tokens=200_000),
    tracer=JsonLogTracer(),
)

result = loop.run("How many words are in the loop notes?")

if result.completed:
    print(result.final_text)
else:
    # Never present an early stop as a finished answer.
    print(f"stopped early: {result.stop_reason} — {result.detail}")
```

## Use it from the CLI

```bash
# Human-readable: streamed answer on stdout, run summary on stderr
agentic-loop "What is 1200 * 1.22?"

# Machine-readable: exactly one JSON object on stdout
agentic-loop --json --trace "Search the loop notes for 'budget'"

# Bound the run
agentic-loop --max-iterations 5 --max-total-tokens 100000 "…"
```

Exit codes: `0` completed, `1` ran but stopped early, `2` could not run.

## Use it from TypeScript

```typescript
import { runAgenticLoop } from "@vitanuova/agentic-loop";

const run = await runAgenticLoop({
  prompt: "What is 6 * 7?",
  maxIterations: 5,
  maxTotalTokens: 100_000,
  onStderr: (chunk) => process.stderr.write(chunk),
});

if (run.result.completed) {
  console.log(run.result.final_text);
} else {
  console.warn(`stopped early: ${run.result.stop_reason} — ${run.result.detail}`);
}
```

An early stop **resolves** (it is a result about the run); only a failure to
produce a result rejects. Point the wrapper at the right interpreter with the
`python` option or `AGENTIC_LOOP_PYTHON`.

## How termination is guaranteed

Three independent hard ceilings, all checked *before* a request is issued so a
run never pays for a turn it cannot follow up on:

| Ceiling | Default | Why |
| --- | --- | --- |
| `max_iterations` | 10 | A model that never emits `end_turn` would otherwise run until the account is empty. |
| `max_total_tokens` | unset | Caps the spend of one run. |
| `max_tool_calls` | unset | Caps tool fan-out. |

Separately, `LoopConfig.task_budget_tokens` sets the API's **advisory**
server-side task budget: the model is told how much it has left so it can finish
gracefully. It stops nothing on its own — use it *with* the hard ceilings, not
instead of them.

## The demo tools

Three pure, deterministic tools, so a failing test points at the loop rather
than at the environment:

| Tool | Does |
| --- | --- |
| `calculator` | Evaluates arithmetic via an AST whitelist — **never** `eval`. No names, calls, or attribute access reach an interpreter. |
| `word_stats` | Exact character / word / unique-word / line counts, plus a stable frequency ranking. |
| `document_search` | Literal substring search over a fixed in-memory corpus. No path, no glob, no filesystem — nothing to traverse to. |

Register your own with `Tool` and `ToolRegistry`; registration order is
preserved because the tool list is part of the prompt-cache prefix.

## Observability

Each run emits structured events (`run_start`, `request`, `response`,
`tool_call`, `tool_result`, `budget_exhausted`, `run_end`) correlated by a single
`run_id`. `JsonLogTracer` writes one JSON line per event; `MemoryTracer` keeps
them for inspection and tests.

Recorded payloads are **redacted and truncated** before they leave the process:
keys matching credentials or regulated identifiers are masked, and long strings
are cut. A trace is an operational artefact that may be shipped off-host, so it
must not become an exfiltration path.

## Layout

```
src/agentic_loop/
  loop.py        the plan → act → observe cycle and stop-reason handling
  budget.py      hard client-side ceilings and token accounting
  tools.py       Tool, ToolRegistry, the safe evaluator, the demo tools
  telemetry.py   trace events, redaction, sinks
  errors.py      exception hierarchy
  cli.py         entry point; owns the JSON contract and API-error translation
tests/           141 Python tests, fake client in tests/fakes.py
ts/
  src/           types, errors, and the child-process runner
  test/          20 tests; fixtures/ hold fake CLIs so the real spawn path is tested
```

## Conventions

See [CLAUDE.md](./CLAUDE.md) for the full engineering conventions this
repository follows, and for what is verified versus assumed about it.
