# CLAUDE.md

Operating guide for AI assistants (Claude Code and equivalents) working in this
repository.

> **Verification status** — Last verified against the working tree of the
> `claude/first-agentic-loop` branch on 2026-08-20. Everything under "Verified
> state" was read from the repository or produced by running the commands
> listed. Everything under "Assumptions" is *not* verified and must be confirmed
> with a human before it is relied upon.

---

## 1. Verified state of the repository

The repository implements **one agentic loop engine in Python**, with a
TypeScript package that wraps it. There is no second implementation of the loop.

| Fact | Value |
| --- | --- |
| Remote | `https://github.com/marcopollara-vitanuova/Agentic_loop` |
| Default branch | `main` |
| Language / runtime | Python ≥ 3.11 (engine), Node ≥ 20 (wrapper only) |
| Package manifest | `pyproject.toml` (hatchling), `ts/package.json` |
| Runtime dependency | `anthropic>=1.0,<2` — the only one. The TypeScript package has zero runtime dependencies. |
| Tests | 153 Python (`pytest`), 20 TypeScript (`node --test`) |
| Lint / format / types | `ruff`, `ruff format`, `mypy` (strict) for Python; `tsc` strict for TypeScript |
| CI / workflows | **none** (`.github/` does not exist) — the checks below are run by hand |
| Containerisation / IaC | none |

### Architecture

```
src/agentic_loop/
  loop.py        the plan → act → observe cycle; explicit stop_reason handling
  budget.py      hard client-side ceilings (iterations, tokens, tool calls)
  tools.py       Tool, ToolRegistry, AST-whitelist evaluator, demo tools
  telemetry.py   trace events, redaction, sinks
  errors.py      exception hierarchy
  cli.py         entry point; owns the JSON contract and API-error translation
tests/           fake Anthropic client in tests/fakes.py — no key, no network
ts/src/          types, errors, child-process runner over the Python CLI
ts/test/         fixtures/ hold fake CLI modules so the real spawn path is tested
```

### Invariants — do not break these without an explicit decision

These are not style preferences; each one exists because its absence is a
defect that reaches production quietly.

1. **The loop stays explicit.** Do not replace the manual cycle in `loop.py`
   with the SDK's `tool_runner`. Owning the loop is the point of this project:
   auditability of every request, tool call and stop condition.
2. **Every `stop_reason` is handled by name.** `tool_use`, `end_turn`,
   `pause_turn`, `refusal`, `max_tokens`, `stop_sequence`, and an explicit
   fallback. A new value must get its own branch, never a silent default.
3. **An early stop is never presented as an answer.** Check
   `LoopResult.completed` / `stop_reason` before using `final_text`.
4. **All tool results for one assistant turn go back in a single user
   message.** Splitting them teaches the model to stop making parallel calls,
   and an unanswered `tool_use` block is rejected by the API.
5. **A failing tool returns `is_error: true`, it does not raise.** The model
   must be able to read the failure and correct itself.
6. **The client is injected, never constructed inside the loop.** This is what
   keeps the suite runnable with no API key and no network.
7. **Budgets are checked before a request is issued**, so a run never pays for
   a turn it cannot follow up on.
8. **The request never carries `temperature`, `top_p`, `top_k`, or
   `thinking.budget_tokens`.** These are rejected with a 400 on the current
   model family. A test asserts their absence — keep it.
9. **Tool registration order is stable.** The tool list is part of the
   prompt-cache prefix; a varying prefix silently destroys cache hits.
10. **Traces are redacted and truncated** before leaving the process. A trace
    may be shipped off-host.
11. **No secret ever passes through this code.** Credentials are resolved by
    the SDK from the environment.
12. **The CLI reports errors, it never tracebacks.** SDK exceptions are
    translated in `cli.py::_translate_api_error`. Note the SDK validates
    credentials on the *first request*, raising a bare `TypeError` — not a
    typed error — which is why translation matches on message text too.

### What follows from this

- **Re-verify before acting.** Start a task with `git log --oneline -10` and
  `git ls-files`, and trust what you see over what is written here.
- **Run the checks in section 5 before every push.** There is no CI to catch
  what you miss.

---

## 2. Assumptions (unverified — do not treat as requirements)

These are inferences from the repository name and organisational context only.
They are recorded so they can be confirmed or corrected, not so they can be
built on.

- ~~**A1** — the repository is for work on agentic loops.~~ **Confirmed** by
  the implementation: the loop is the deliverable.
- **Assumption A2** *(open)* — as a repository under the
  `marcopollara-vitanuova` account, it is expected to follow Vitanuova
  engineering standards (quality, security, traceability, documentation,
  governance) and to be coherent with the **Axieme** platform as the
  organisation's reference technology platform. *No configuration in this
  repository enforces or references this, and there is no Axieme integration.*
- **Assumption A3** *(open, interpretive)* — "core Python + TypeScript wrapper"
  was implemented as **one engine invoked over a CLI JSON contract**, rather
  than as the loop ported to both languages. The reasoning: two ports would mean
  two copies of the stop-reason handling, budget accounting and tool-result
  protocol to keep in step. *This reading was stated when the choice was made
  and was not contradicted, but it was not explicitly confirmed either.*
- **Assumption A4** *(open)* — the three demo tools (`calculator`,
  `word_stats`, `document_search`) exist to exercise the loop engine. *No real
  use case, data set or integration has been specified.*

Nothing here defines scope, requirements, deadlines, regulatory obligations or a
target architecture beyond the loop itself. If a task depends on any of those,
**ask** — do not infer.

### Known gaps, stated plainly

- **No CI.** Nothing runs the checks automatically; a broken push is only caught
  by whoever runs them next.
- **The loop has never been executed against the live API in this repository.**
  Every behavioural test runs against a fake client. The request shape follows
  the documented Messages API contract and is asserted in tests, but the
  round-trip is unverified. Verifying it needs credentials.
- **No real use case.** The tools are demonstrative.

---

## 3. Git and branch workflow

Verified from the current session configuration and repository state.

- Development happens on **feature branches**; the branch in use for
  documentation work is `claude/claude-md-docs-0buy9g`. The observed convention
  is `claude/<short-topic>-<suffix>` for assistant-authored branches.
- **Never push directly to `main`** without explicit permission.
- Create the designated branch locally if it does not exist, commit with clear
  descriptive messages, and push with:

  ```bash
  git push -u origin <branch-name>
  ```

- On network failure, retry the push up to 4 times with exponential backoff
  (2s, 4s, 8s, 16s). Do not retry on non-network failures — diagnose instead.
- Prefer fetching a specific branch: `git fetch origin <branch-name>`.
- **Do not open a pull request unless it is explicitly requested.** When one is
  requested, check for a PR template (`.github/pull_request_template.md`,
  `.github/PULL_REQUEST_TEMPLATE.md`, root `PULL_REQUEST_TEMPLATE.md`,
  `docs/PULL_REQUEST_TEMPLATE.md`) and populate its sections. None exists today.
- If the pull request for a designated branch has already been merged, restart
  that branch from the latest `main` rather than stacking new commits on merged
  history.
- Never rewrite history on a branch someone else may have checked out (no
  rebase, amend or force-push there). Resolve divergence with a merge commit.
- Do not include model identifiers or internal session references in commit
  messages, PR bodies, code comments or any other pushed artefact.

---

## 4. Working conventions

Derived from Vitanuova engineering directives. These are the standards to apply
to work in this repository; they are organisational, not inferred from code.

### Process

Structure technical work as: **analysis → design → implementation → test →
review → release → deploy → monitoring**, with quality, security,
documentation and validation checks at each phase. Scale the ceremony to the
size of the change, but do not skip test and documentation.

### Engineering priorities

Simplicity, standardisation, automation, reuse, observability, documentation,
continuous improvement. Prefer the boring, documentable option over the clever
one. Outputs should be reusable as a project, operational or documentation
baseline.

### Factual discipline

Do not invent facts, requirements, decisions, data, architectures, regulations,
deadlines or references. If something is unavailable or unverifiable, say so
explicitly. Flag assumptions as assumptions — in code comments, commit
messages, documentation and chat alike. This repository operates in a regulated
(insurance brokerage) context, where a plausible-sounding invention is a
liability.

### Secrets and data

- Never commit credentials, tokens, API keys, connection strings or personal
  data. Use environment variables and document the required variable *names*
  only.
- Do not put internal hostnames, credentials or environment variables in PR
  descriptions or public artefacts.

### Documentation

Every meaningful change updates the documentation it invalidates. When the
first code lands, `README.md` must state what the project is, how to install
it, how to run it and how to test it.

---

## 5. Commands

Every command below was run in this repository and reports what it reports here.
There is **no CI**, so run the full set before pushing.

### Python (from the repository root)

```bash
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"

python -m pytest                                          # 153 tests
python -m pytest --cov=agentic_loop --cov-report=term-missing   # ~96% coverage
ruff check .                                              # lint
ruff format .                                             # format
mypy                                                      # strict, clean

agentic-loop "your prompt"                                # run (needs credentials)
agentic-loop --print-tools                                # tool catalogue, no API call
agentic-loop --json --trace "…"                           # machine-readable + events
```

### TypeScript (from `ts/`)

```bash
npm install
npm run typecheck
npm test          # builds, then runs 20 tests
npm run check     # typecheck + test
```

### Notes that will save you a debugging session

- The Python suite needs **no API key and no network**. If a test starts
  requiring either, that is a defect in the test, not a missing credential.
- `pytest` resolves the package via `pythonpath = ["src"]` in `pyproject.toml`;
  running it from outside the repository root will not work.
- The TypeScript tests spawn a real `python3` and resolve their fake
  `agentic_loop.cli` from each fixture's directory. Override the interpreter
  with `AGENTIC_LOOP_TEST_PYTHON`.
- Exit codes: `0` completed, `1` ran but stopped early, `2` could not run. The
  TypeScript wrapper depends on this distinction — `1` resolves, `2` rejects.
- `--debug` re-raises the original exception instead of reporting it.

---

## 6. Keeping this file current

`CLAUDE.md` is the contract between the repository and its AI collaborators; a
stale one is worse than none.

Update it in the **same commit** as the change when any of the following happen:

- the first code, package manifest or dependency set is added;
- the language, framework or runtime is chosen or changed;
- build, test, lint or run commands change;
- CI workflows, deployment or environments are introduced;
- directory layout or module boundaries change materially;
- a convention in section 4 is superseded by a project-specific rule;
- an assumption in section 2 is confirmed, corrected or dropped.

When you update it, also refresh the **Verification status** line at the top
with the commit you verified against, and keep the verified/assumed split
intact — never promote an assumption to a fact without evidence in the
repository or an explicit decision from a human.

### Still to be done

- [ ] **CI** — run the section 5 checks on every push and pull request
- [ ] **A live round-trip against the API**, once credentials are available
- [ ] Real use case and tools to replace the demonstrative ones
- [ ] Confirm or correct assumptions A2, A3 and A4 in section 2
- [ ] Multi-turn / conversation persistence beyond the `history` argument
- [ ] Deployment, packaging and release process
- [ ] Any Axieme platform integration, once its requirements are stated
