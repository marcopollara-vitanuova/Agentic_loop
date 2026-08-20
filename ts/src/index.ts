/**
 * A typed TypeScript boundary over the Python agentic loop.
 *
 * The loop itself lives in Python (`src/agentic_loop`); this package runs it
 * and types its JSON contract. There is one engine, not two.
 *
 * @example
 * import { runAgenticLoop } from "@vitanuova/agentic-loop";
 *
 * const run = await runAgenticLoop({
 *   prompt: "How many words are in the loop notes?",
 *   maxIterations: 6,
 *   maxTotalTokens: 200_000,
 *   onStderr: (chunk) => process.stderr.write(chunk),
 * });
 *
 * if (run.result.completed) {
 *   console.log(run.result.final_text);
 * } else {
 *   console.warn(`stopped early: ${run.result.stop_reason} — ${run.result.detail}`);
 * }
 */

export {
  AgenticLoopError,
  InvalidOptionsError,
  LoopProcessError,
  LoopProtocolError,
} from "./errors.js";
export { buildArgs, runAgenticLoop } from "./runner.js";
export type {
  Effort,
  LoopResult,
  LoopRun,
  RunOptions,
  StopReason,
  ToolInvocation,
  TraceEvent,
  Usage,
} from "./types.js";
