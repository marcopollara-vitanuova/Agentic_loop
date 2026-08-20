# CLAUDE.md

Operating guide for AI assistants (Claude Code and equivalents) working in this
repository.

> **Verification status** — Last verified against commit `f660653` on
> 2026-08-20. Everything under "Verified state" was read directly from the
> repository. Everything under "Assumptions" is *not* verified and must be
> confirmed with a human before it is relied upon.

---

## 1. Verified state of the repository

This repository is currently an **empty scaffold**. Read this section before
planning any work: there is no application code to reason about yet.

| Fact | Value |
| --- | --- |
| Remote | `https://github.com/marcopollara-vitanuova/Agentic_loop` |
| Default branch | `main` |
| History | a single commit, `f660653` ("Initial commit") |
| Tracked files | `README.md` (2 lines: title + `Agentic loop`), `CLAUDE.md` (this file) |
| Source code | none |
| Build system / package manifest | none (`no package.json`, `pyproject.toml`, `go.mod`, `Makefile`, …) |
| Dependencies / lockfiles | none |
| Tests | none |
| CI / workflows | none (`.github/` does not exist) |
| Linter / formatter config | none |
| Containerisation / IaC | none |
| `.gitignore` | none |

### What follows from this

- **Do not assume a language, framework, runtime or architecture.** There is
  nothing in the repository that establishes one.
- **Re-verify before acting.** This file describes a snapshot. Always start a
  task with `git log --oneline -10` and `git ls-files` and trust what you see
  over what is written here.
- **The first substantive commit is a design decision, not a chore.** Choosing
  the language, package manager, test runner and CI is architectural. Propose
  options and get an explicit decision from the CTO (Marco Pollara) rather than
  picking defaults silently.

---

## 2. Assumptions (unverified — do not treat as requirements)

These are inferences from the repository name and organisational context only.
They are recorded so they can be confirmed or corrected, not so they can be
built on.

- **Assumption A1** — the name `Agentic_loop` suggests the repository is
  intended for work on agentic loops (LLM-driven plan → act → observe →
  iterate cycles). *No file in the repository states this.*
- **Assumption A2** — as a repository under the `marcopollara-vitanuova`
  account, it is expected to follow Vitanuova engineering standards (quality,
  security, traceability, documentation, governance) and to be coherent with
  the **Axieme** platform as the organisation's reference technology platform.
  *No configuration in this repository enforces or references this.*

Nothing here defines scope, requirements, deadlines, integrations or a target
architecture. If a task depends on any of those, **ask** — do not infer.

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

**There are no build, test, lint or run commands in this repository yet.** Any
command an assistant might suggest today would be fabricated.

When tooling is introduced, replace this section with the real, verified
commands — each one actually executed once before it is documented here:

```text
Install:   <to be filled in>
Build:     <to be filled in>
Test:      <to be filled in>
Lint:      <to be filled in>
Typecheck: <to be filled in>
Run:       <to be filled in>
```

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

### Sections still to be written (blocked on code existing)

- [ ] Codebase structure and module boundaries
- [ ] Architecture and key data flows
- [ ] Local development setup
- [ ] Testing strategy and how to run a single test
- [ ] CI pipeline and release/deploy process
- [ ] Code style, naming and error-handling conventions
- [ ] Observability: logging, metrics, tracing
- [ ] Integration points (including any Axieme platform integration, once confirmed)
