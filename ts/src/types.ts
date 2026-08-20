/**
 * The wire contract between the Python CLI (`agentic-loop --json`) and this
 * wrapper.
 *
 * These types mirror `LoopResult.to_dict()` and the CLI's JSON envelope. They
 * are the coupling point between the two languages: when the Python side
 * changes its payload, these declarations and the parser in `runner.ts` are
 * what must change with it.
 */

/** Reasoning effort. Higher means deeper thinking and more tokens. */
export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

/**
 * Why a run ended. `completed` is the only value that means the model finished
 * on its own terms — every other value is a ceiling or a rejection.
 */
export type StopReason =
  | "completed"
  | "budget_exhausted"
  | "refusal"
  | "max_tokens"
  | "stop_sequence"
  | "unexpected_stop_reason";

/** One tool call, as executed by the loop. */
export interface ToolInvocation {
  readonly iteration: number;
  readonly tool_use_id: string;
  readonly name: string;
  readonly input: Record<string, unknown>;
  readonly result: string;
  /** True when the tool failed and the error was handed back to the model. */
  readonly is_error: boolean;
}

/** Cumulative token usage for a run. */
export interface Usage {
  readonly input_tokens: number;
  readonly output_tokens: number;
  readonly cache_creation_input_tokens: number;
  readonly cache_read_input_tokens: number;
  readonly total_tokens: number;
}

/** A single trace event. `kind` identifies the transition. */
export interface TraceEvent {
  readonly run_id: string;
  readonly kind: string;
  readonly iteration: number;
  readonly timestamp: string;
  readonly [key: string]: unknown;
}

/** The outcome of one run. */
export interface LoopResult {
  readonly run_id: string;
  readonly stop_reason: StopReason;
  /** Equivalent to `stop_reason === "completed"`. Check it before using the text. */
  readonly completed: boolean;
  readonly final_text: string;
  readonly iterations: number;
  /** Why the run stopped early, when it did. */
  readonly detail: string | null;
  readonly usage: Usage;
  readonly tool_invocations: readonly ToolInvocation[];
}

/** What a successful (exit 0 or 1) invocation returns. */
export interface LoopRun {
  /** Mirrors `result.completed`. */
  readonly ok: boolean;
  readonly result: LoopResult;
  /** Populated when tracing is requested; empty otherwise. */
  readonly trace: readonly TraceEvent[];
}

/** Options for {@link runAgenticLoop}. */
export interface RunOptions {
  /** The user prompt. Required and non-empty. */
  readonly prompt: string;

  // -- model ---------------------------------------------------------------
  /** Model id. Defaults to the Python side's default (`claude-opus-5`). */
  readonly model?: string;
  readonly effort?: Effort;
  /** Per-response output ceiling. */
  readonly maxTokens?: number;
  readonly system?: string;
  /** Whether to return a summary of the model's reasoning. */
  readonly thinkingDisplay?: "summarized" | "omitted";
  /**
   * Stream the response. On by default — a non-streaming request with a large
   * `maxTokens` can exceed the HTTP timeout.
   */
  readonly stream?: boolean;

  // -- budget --------------------------------------------------------------
  /** Hard cap on model requests. */
  readonly maxIterations?: number;
  /** Hard cap on billed tokens for the run. */
  readonly maxTotalTokens?: number;
  /** Hard cap on tool invocations for the run. */
  readonly maxToolCalls?: number;
  /**
   * Advisory server-side budget so the model paces itself. Minimum 20000, and
   * requires streaming.
   */
  readonly taskBudgetTokens?: number;
  /** Treat budget exhaustion as an error instead of a partial result. */
  readonly strictBudget?: boolean;

  // -- process -------------------------------------------------------------
  /** Include trace events in the response. */
  readonly trace?: boolean;
  /** Python interpreter. Defaults to `AGENTIC_LOOP_PYTHON` or `python3`. */
  readonly python?: string;
  /** Working directory for the child process. */
  readonly cwd?: string;
  /** Environment for the child process. Defaults to the parent's. */
  readonly env?: NodeJS.ProcessEnv;
  /** Kill the child after this many milliseconds. Defaults to 600000 (10 min). */
  readonly timeoutMs?: number;
  /** Abort the run early. */
  readonly signal?: AbortSignal;
  /**
   * Receives the child's stderr as it arrives: streamed model text, trace
   * lines, and diagnostics.
   */
  readonly onStderr?: (chunk: string) => void;
}
