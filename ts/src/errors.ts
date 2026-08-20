/** Errors raised by the wrapper. */

/** Base class, so callers can catch the whole surface at once. */
export class AgenticLoopError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** The options passed to {@link runAgenticLoop} could not produce a valid call. */
export class InvalidOptionsError extends AgenticLoopError {}

/**
 * The Python process failed: it could not be spawned, was killed, timed out, or
 * exited reporting an error rather than a result.
 */
export class LoopProcessError extends AgenticLoopError {
  constructor(
    message: string,
    readonly detail: {
      readonly exitCode: number | null;
      readonly signal: NodeJS.Signals | null;
      readonly stderr: string;
    },
    options?: { cause?: unknown },
  ) {
    super(message, options);
  }
}

/**
 * The process succeeded but its stdout was not the agreed JSON contract.
 *
 * Almost always means something else wrote to stdout, or the two sides are on
 * incompatible versions.
 */
export class LoopProtocolError extends AgenticLoopError {
  constructor(
    message: string,
    readonly stdout: string,
    options?: { cause?: unknown },
  ) {
    super(message, options);
  }
}
