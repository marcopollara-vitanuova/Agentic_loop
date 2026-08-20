/**
 * Runs the Python agentic loop as a child process and returns its result.
 *
 * There is deliberately no second implementation of the loop here. The loop is
 * one engine, in Python; this module is a typed boundary over its JSON
 * contract. Reimplementing the cycle in TypeScript would mean two versions of
 * the stop-reason handling, the budget accounting and the tool-result protocol
 * to keep in step — the expensive kind of duplication.
 */

import { spawn } from "node:child_process";
import {
  InvalidOptionsError,
  LoopProcessError,
  LoopProtocolError,
} from "./errors.js";
import type { LoopResult, LoopRun, RunOptions, TraceEvent } from "./types.js";

/** Exit codes the Python CLI uses. */
const EXIT_OK = 0;
const EXIT_INCOMPLETE = 1;

const DEFAULT_TIMEOUT_MS = 600_000;

/** The API's minimum for an advisory task budget. */
const MIN_TASK_BUDGET_TOKENS = 20_000;

/**
 * Translate options into CLI arguments.
 *
 * Arguments are passed to `spawn` as an array with no shell, so a prompt
 * containing shell metacharacters is inert. Validation here is only about
 * catching mistakes early with a clearer message than the Python side gives.
 */
export function buildArgs(options: RunOptions): string[] {
  if (typeof options.prompt !== "string" || options.prompt.trim() === "") {
    throw new InvalidOptionsError("prompt must be a non-empty string");
  }

  const args = ["-m", "agentic_loop.cli", "--json"];

  const pushInteger = (flag: string, value: number | undefined, min = 1): void => {
    if (value === undefined) return;
    if (!Number.isInteger(value) || value < min) {
      throw new InvalidOptionsError(`${flag} must be an integer >= ${min} (got ${value})`);
    }
    args.push(flag, String(value));
  };

  if (options.model !== undefined) args.push("--model", options.model);
  if (options.effort !== undefined) args.push("--effort", options.effort);
  if (options.system !== undefined) args.push("--system", options.system);
  if (options.thinkingDisplay !== undefined) {
    args.push("--thinking-display", options.thinkingDisplay);
  }
  pushInteger("--max-tokens", options.maxTokens);
  pushInteger("--max-iterations", options.maxIterations);
  pushInteger("--max-total-tokens", options.maxTotalTokens);
  pushInteger("--max-tool-calls", options.maxToolCalls, 0);

  if (options.taskBudgetTokens !== undefined) {
    if (options.stream === false) {
      throw new InvalidOptionsError("taskBudgetTokens requires streaming; do not set stream: false");
    }
    pushInteger("--task-budget", options.taskBudgetTokens, MIN_TASK_BUDGET_TOKENS);
  }

  if (options.strictBudget === true) args.push("--strict-budget");
  if (options.trace === true) args.push("--trace");
  if (options.stream === false) args.push("--no-stream");

  // The prompt goes last so a value starting with `-` cannot be read as a flag.
  args.push("--", options.prompt);
  return args;
}

/** Narrow parsed JSON to the CLI's success envelope. */
function parseEnvelope(stdout: string): LoopRun {
  const trimmed = stdout.trim();
  if (trimmed === "") {
    throw new LoopProtocolError("the loop produced no output on stdout", stdout);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (cause) {
    throw new LoopProtocolError(
      "stdout was not valid JSON; something other than the result was written to stdout",
      stdout,
      { cause },
    );
  }

  if (typeof parsed !== "object" || parsed === null) {
    throw new LoopProtocolError("expected a JSON object on stdout", stdout);
  }

  const envelope = parsed as Record<string, unknown>;
  if ("error" in envelope && envelope["error"] !== undefined) {
    const error = envelope["error"] as Record<string, unknown>;
    throw new LoopProcessError(
      `the loop reported an error: ${String(error["message"] ?? "unknown error")}`,
      { exitCode: null, signal: null, stderr: "" },
    );
  }

  const result = envelope["result"];
  if (typeof result !== "object" || result === null) {
    throw new LoopProtocolError("the JSON envelope has no 'result' object", stdout);
  }
  const candidate = result as Record<string, unknown>;
  for (const field of ["run_id", "stop_reason", "final_text", "iterations"]) {
    if (!(field in candidate)) {
      throw new LoopProtocolError(`the result is missing the '${field}' field`, stdout);
    }
  }

  return {
    ok: envelope["ok"] === true,
    result: result as unknown as LoopResult,
    trace: (Array.isArray(envelope["trace"]) ? envelope["trace"] : []) as readonly TraceEvent[],
  };
}

/**
 * Run the loop and return its outcome.
 *
 * A run that stops early — budget exhausted, refused, truncated — resolves
 * rather than rejecting: that is a result about the run, not a failure to run.
 * Check {@link LoopResult.completed} before using `final_text`. Rejections are
 * reserved for the loop not producing a result at all.
 *
 * @throws {InvalidOptionsError} the options cannot produce a valid call.
 * @throws {LoopProcessError} the process failed, timed out, or was killed.
 * @throws {LoopProtocolError} stdout was not the agreed JSON contract.
 *
 * @example
 * const run = await runAgenticLoop({ prompt: "What is 6 * 7?", maxIterations: 5 });
 * if (run.result.completed) console.log(run.result.final_text);
 * else console.warn(`stopped early: ${run.result.stop_reason} — ${run.result.detail}`);
 */
export async function runAgenticLoop(options: RunOptions): Promise<LoopRun> {
  const args = buildArgs(options);
  const python = options.python ?? process.env["AGENTIC_LOOP_PYTHON"] ?? "python3";
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  return new Promise<LoopRun>((resolve, reject) => {
    const child = spawn(python, args, {
      cwd: options.cwd ?? process.cwd(),
      env: options.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      // No shell: arguments are passed through verbatim, so nothing in the
      // prompt can be interpreted as a command.
      shell: false,
    });

    let stdout = "";
    let stderr = "";
    let settled = false;
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);
    // Do not hold the event loop open just for this timer.
    timer.unref?.();

    const onAbort = (): void => {
      child.kill("SIGTERM");
    };
    options.signal?.addEventListener("abort", onAbort, { once: true });

    const finish = (action: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
      action();
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      options.onStderr?.(chunk);
    });

    child.on("error", (cause) => {
      finish(() =>
        reject(
          new LoopProcessError(
            `could not start '${python}': ${cause.message}. ` +
              "Set the python option or AGENTIC_LOOP_PYTHON to a interpreter with agentic-loop installed.",
            { exitCode: null, signal: null, stderr },
            { cause },
          ),
        ),
      );
    });

    child.on("close", (code, signal) => {
      finish(() => {
        if (timedOut) {
          reject(
            new LoopProcessError(`the loop timed out after ${timeoutMs}ms`, {
              exitCode: code,
              signal,
              stderr,
            }),
          );
          return;
        }
        if (options.signal?.aborted) {
          reject(
            new LoopProcessError("the run was aborted", { exitCode: code, signal, stderr }),
          );
          return;
        }

        // Exit 1 means "ran, stopped early" — a result, not a failure.
        if (code === EXIT_OK || code === EXIT_INCOMPLETE) {
          try {
            resolve(parseEnvelope(stdout));
          } catch (error) {
            reject(error);
          }
          return;
        }

        // Any other code: the error envelope carries the reason when present.
        try {
          parseEnvelope(stdout);
        } catch (error) {
          if (error instanceof LoopProcessError) {
            reject(new LoopProcessError(error.message, { exitCode: code, signal, stderr }));
            return;
          }
        }
        reject(
          new LoopProcessError(
            `the loop exited with code ${code}${signal ? ` (signal ${signal})` : ""}`,
            { exitCode: code, signal, stderr },
          ),
        );
      });
    });
  });
}
