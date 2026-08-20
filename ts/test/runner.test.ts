/**
 * Tests for the TypeScript wrapper.
 *
 * Argument mapping is tested directly against `buildArgs`. Everything else runs
 * a real child process against the Python fixtures in `test/fixtures/`: each
 * fixture provides an `agentic_loop.cli` module that `python -m` resolves from
 * the child's cwd. That exercises the actual spawn, exit-code and JSON-parsing
 * paths without an API key or a network call.
 */

import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { before, describe, it } from "node:test";

import {
  InvalidOptionsError,
  LoopProcessError,
  LoopProtocolError,
  buildArgs,
  runAgenticLoop,
} from "../src/index.js";
import type { RunOptions } from "../src/index.js";

// `__dirname` does not exist in ES modules — derive it from import.meta.url.
const here = path.dirname(fileURLToPath(import.meta.url));
const fixtures = path.resolve(here, "../../test/fixtures");

const fixtureCwd = (name: string): string => path.join(fixtures, name);

/** The interpreter used for the fixtures; overridable in constrained CI. */
const PYTHON = process.env["AGENTIC_LOOP_TEST_PYTHON"] ?? "python3";

const run = (name: string, options: Partial<RunOptions> = {}) =>
  runAgenticLoop({
    prompt: "hello",
    python: PYTHON,
    cwd: fixtureCwd(name),
    timeoutMs: 20_000,
    ...options,
  });

describe("buildArgs", () => {
  it("always requests the JSON contract", () => {
    const args = buildArgs({ prompt: "hi" });
    assert.deepEqual(args.slice(0, 3), ["-m", "agentic_loop.cli", "--json"]);
  });

  it("puts the prompt last, after an argument terminator", () => {
    // Without `--`, a prompt starting with `-` would be parsed as a flag.
    const args = buildArgs({ prompt: "--not-a-flag" });
    assert.deepEqual(args.slice(-2), ["--", "--not-a-flag"]);
  });

  it("maps every option to its flag", () => {
    const args = buildArgs({
      prompt: "hi",
      model: "claude-sonnet-5",
      effort: "low",
      maxTokens: 2048,
      system: "Be terse.",
      thinkingDisplay: "omitted",
      maxIterations: 4,
      maxTotalTokens: 50_000,
      maxToolCalls: 3,
      strictBudget: true,
      trace: true,
      stream: false,
    });
    const joined = args.join(" ");
    assert.match(joined, /--model claude-sonnet-5/);
    assert.match(joined, /--effort low/);
    assert.match(joined, /--max-tokens 2048/);
    assert.match(joined, /--system Be terse\./);
    assert.match(joined, /--thinking-display omitted/);
    assert.match(joined, /--max-iterations 4/);
    assert.match(joined, /--max-total-tokens 50000/);
    assert.match(joined, /--max-tool-calls 3/);
    assert.match(joined, /--strict-budget/);
    assert.match(joined, /--trace/);
    assert.match(joined, /--no-stream/);
  });

  it("omits flags for options that were not set", () => {
    const args = buildArgs({ prompt: "hi" });
    assert.equal(args.includes("--model"), false);
    assert.equal(args.includes("--no-stream"), false);
    assert.equal(args.includes("--trace"), false);
    assert.equal(args.includes("--strict-budget"), false);
  });

  it("rejects an empty prompt", () => {
    assert.throws(() => buildArgs({ prompt: "   " }), InvalidOptionsError);
  });

  it("rejects non-integer and out-of-range numbers", () => {
    assert.throws(() => buildArgs({ prompt: "hi", maxTokens: 1.5 }), InvalidOptionsError);
    assert.throws(() => buildArgs({ prompt: "hi", maxIterations: 0 }), InvalidOptionsError);
    assert.throws(() => buildArgs({ prompt: "hi", maxToolCalls: -1 }), InvalidOptionsError);
  });

  it("allows a zero tool-call cap", () => {
    // Zero is meaningful: answer without touching a tool.
    assert.match(buildArgs({ prompt: "hi", maxToolCalls: 0 }).join(" "), /--max-tool-calls 0/);
  });

  it("enforces the task budget minimum", () => {
    assert.throws(
      () => buildArgs({ prompt: "hi", taskBudgetTokens: 100 }),
      /20000/,
    );
    assert.match(
      buildArgs({ prompt: "hi", taskBudgetTokens: 20_000 }).join(" "),
      /--task-budget 20000/,
    );
  });

  it("rejects a task budget combined with streaming disabled", () => {
    assert.throws(
      () => buildArgs({ prompt: "hi", taskBudgetTokens: 20_000, stream: false }),
      /requires streaming/,
    );
  });
});

describe("runAgenticLoop", () => {
  before(() => {
    // Fail loudly and early if the fixtures are not where the tests expect.
    assert.ok(fixtures.endsWith("fixtures"), `unexpected fixture path: ${fixtures}`);
  });

  it("returns the result of a completed run", async () => {
    const outcome = await run("ok");

    assert.equal(outcome.ok, true);
    assert.equal(outcome.result.completed, true);
    assert.equal(outcome.result.stop_reason, "completed");
    assert.equal(outcome.result.final_text, "42");
    assert.equal(outcome.result.iterations, 2);
    assert.equal(outcome.result.usage.total_tokens, 120);
    assert.equal(outcome.result.tool_invocations[0]?.name, "calculator");
    assert.equal(outcome.trace[0]?.kind, "run_start");
  });

  it("resolves — not rejects — when a run stops early", async () => {
    // Stopping on a budget is a result about the run, not a failure to run.
    const outcome = await run("incomplete");

    assert.equal(outcome.ok, false);
    assert.equal(outcome.result.completed, false);
    assert.equal(outcome.result.stop_reason, "budget_exhausted");
    assert.match(outcome.result.detail ?? "", /iteration cap/);
  });

  it("forwards stderr to the callback as it arrives", async () => {
    const chunks: string[] = [];
    await run("ok", { onStderr: (chunk) => chunks.push(chunk) });
    assert.match(chunks.join(""), /streamed text/);
  });

  it("rejects with the reported reason when the loop cannot run", async () => {
    await assert.rejects(run("failure"), (error: unknown) => {
      assert.ok(error instanceof LoopProcessError);
      assert.match(error.message, /no credentials/);
      return true;
    });
  });

  it("rejects when stdout is not the agreed contract", async () => {
    await assert.rejects(run("garbage"), (error: unknown) => {
      assert.ok(error instanceof LoopProtocolError);
      assert.match(error.message, /not valid JSON/);
      assert.match(error.stdout, /not JSON/);
      return true;
    });
  });

  it("rejects when the interpreter does not exist", async () => {
    await assert.rejects(
      run("ok", { python: "definitely-not-a-real-interpreter" }),
      (error: unknown) => {
        assert.ok(error instanceof LoopProcessError);
        assert.match(error.message, /could not start/);
        return true;
      },
    );
  });

  it("kills the child and rejects on timeout", async () => {
    await assert.rejects(run("slow", { timeoutMs: 500 }), (error: unknown) => {
      assert.ok(error instanceof LoopProcessError);
      assert.match(error.message, /timed out after 500ms/);
      return true;
    });
  });

  it("stops the run when the signal is aborted", async () => {
    const controller = new AbortController();
    const pending = run("slow", { signal: controller.signal });
    setTimeout(() => controller.abort(), 200);

    await assert.rejects(pending, (error: unknown) => {
      assert.ok(error instanceof LoopProcessError);
      assert.match(error.message, /aborted/);
      return true;
    });
  });

  it("passes the mapped arguments through to the process", async () => {
    const outcome = await run("echo", {
      prompt: "the prompt",
      model: "claude-sonnet-5",
      maxIterations: 7,
      stream: false,
    });

    const received = outcome.result.final_text;
    assert.match(received, /--json/);
    assert.match(received, /--model claude-sonnet-5/);
    assert.match(received, /--max-iterations 7/);
    assert.match(received, /--no-stream/);
    assert.ok(received.endsWith("the prompt"));
  });

  it("passes a prompt with shell metacharacters through untouched", async () => {
    // No shell is used, so this is inert rather than escaped.
    const hostile = "x; rm -rf / && echo $(whoami) `id`";
    const outcome = await run("echo", { prompt: hostile });
    assert.ok(outcome.result.final_text.endsWith(hostile));
  });

  it("validates options before spawning anything", async () => {
    await assert.rejects(
      runAgenticLoop({ prompt: "", python: "definitely-not-a-real-interpreter" }),
      InvalidOptionsError,
    );
  });
});
