"""Command-line entry point.

Two output modes:

- **human** (default) — streamed text, then a summary of how the run ended.
- **JSON** (``--json``) — exactly one JSON object on stdout and nothing else,
  which is the contract the TypeScript wrapper depends on. In this mode all
  human-facing output goes to stderr so stdout stays machine-parsable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from . import __version__
from .budget import Budget
from .errors import AgenticLoopError, ApiError, ConfigurationError
from .loop import AgenticLoop, LoopConfig
from .telemetry import CompositeTracer, JsonLogTracer, MemoryTracer, NullTracer, Tracer
from .tools import default_registry

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-loop",
        description="Run an explicit tool-use loop against the Claude Messages API.",
    )
    parser.add_argument("prompt", nargs="?", help="The user prompt. Reads stdin when omitted.")
    parser.add_argument("--version", action="version", version=f"agentic-loop {__version__}")

    model = parser.add_argument_group("model")
    model.add_argument("--model", default="claude-opus-5", help="Model id (default: %(default)s).")
    model.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort (default: %(default)s).",
    )
    model.add_argument(
        "--max-tokens",
        type=int,
        default=16_000,
        help="Per-response output ceiling (default: %(default)s).",
    )
    model.add_argument("--system", default=None, help="System prompt.")
    model.add_argument(
        "--thinking-display",
        default="summarized",
        choices=["summarized", "omitted"],
        help="Whether to return a summary of the model's reasoning (default: %(default)s).",
    )
    model.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming. Risks HTTP timeouts on large max_tokens.",
    )

    limits = parser.add_argument_group("budget")
    limits.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Hard cap on model requests (default: %(default)s).",
    )
    limits.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Hard cap on billed tokens for the run. Unset means uncapped.",
    )
    limits.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Hard cap on tool invocations for the run. Unset means uncapped.",
    )
    limits.add_argument(
        "--task-budget",
        type=int,
        default=None,
        metavar="TOKENS",
        help=(
            "Advisory server-side token budget so the model paces itself "
            "(minimum 20000). Requires streaming."
        ),
    )
    limits.add_argument(
        "--strict-budget",
        action="store_true",
        help="Exit with an error instead of returning a partial result on exhaustion.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit one JSON object on stdout (machine-readable contract).",
    )
    output.add_argument(
        "--trace",
        action="store_true",
        help="Log one JSON trace event per line to stderr.",
    )
    output.add_argument(
        "--debug",
        action="store_true",
        help="Re-raise the original exception instead of reporting it, for diagnosis.",
    )
    output.add_argument(
        "--print-tools",
        action="store_true",
        help="Print the tool catalogue as JSON and exit without calling the API.",
    )
    return parser


def _read_prompt(value: str | None) -> str:
    """Resolve the prompt from the argument or stdin.

    Raises:
        ConfigurationError: neither source provided one.
    """
    if value is not None and value.strip():
        return value
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise ConfigurationError("no prompt given: pass it as an argument or pipe it on stdin")


def _configure_trace_logging() -> None:
    """Send trace records to stderr so stdout stays reserved for results."""
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("agentic_loop.trace")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _build_client() -> Any:
    """Construct the Anthropic client, with an actionable message when it fails.

    Credentials are resolved by the SDK (``ANTHROPIC_API_KEY``,
    ``ANTHROPIC_AUTH_TOKEN``, or an ``ant auth login`` profile) — deliberately
    not by this module, so no code path here ever handles a raw secret.
    """
    try:
        import anthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise AgenticLoopError(
            "the 'anthropic' package is not installed; run: uv sync (or pip install anthropic)"
        ) from exc
    try:
        return anthropic.Anthropic()
    except Exception as exc:
        raise AgenticLoopError(
            f"could not initialise the Anthropic client: {exc}. "
            "Set ANTHROPIC_API_KEY or run 'ant auth login'."
        ) from exc


def _translate_api_error(exc: BaseException) -> ApiError:
    """Turn whatever the SDK raised into an :class:`ApiError` with a usable message.

    The SDK does not validate credentials when the client is constructed — a
    missing key surfaces on the first request, and as a ``TypeError`` rather
    than a typed authentication error. So this maps on the exception's identity
    where it is typed, and falls back to inspecting the message where it is not.
    """
    name = type(exc).__name__
    message = str(exc)

    if name == "AuthenticationError" or "Could not resolve authentication" in message:
        return ApiError(
            "authentication failed: no usable credentials were found. "
            "Set ANTHROPIC_API_KEY or run 'ant auth login'.",
            kind="authentication",
        )
    if name == "PermissionDeniedError":
        return ApiError(
            f"the credentials lack permission for this request: {message}",
            kind="permission",
        )
    if name == "RateLimitError":
        return ApiError(f"rate limited by the API: {message}", kind="rate_limit")
    if name == "NotFoundError":
        return ApiError(
            f"the API rejected the target as unknown — check the model id: {message}",
            kind="not_found",
        )
    if name == "BadRequestError":
        return ApiError(f"the API rejected the request: {message}", kind="bad_request")
    if name == "APIConnectionError" or name == "APITimeoutError":
        return ApiError(
            f"could not reach the API ({name}): {message}. Check connectivity and any proxy.",
            kind="connection",
        )
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        kind = "server_error" if status >= 500 else "api_error"
        return ApiError(f"the API returned HTTP {status}: {message}", kind=kind)
    return ApiError(f"the request failed ({name}): {message}", kind="unexpected")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = default_registry()

    if args.print_tools:
        json.dump({"tools": registry.to_wire()}, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return EXIT_OK

    stream_enabled = not args.no_stream

    try:
        prompt = _read_prompt(args.prompt)
        config = LoopConfig(
            model=args.model,
            max_tokens=args.max_tokens,
            system=args.system,
            effort=args.effort,
            thinking_display=args.thinking_display,
            stream=stream_enabled,
            task_budget_tokens=args.task_budget,
            strict_budget=args.strict_budget,
        )
        budget = Budget(
            max_iterations=args.max_iterations,
            max_total_tokens=args.max_total_tokens,
            max_tool_calls=args.max_tool_calls,
        )
    except (AgenticLoopError, ValueError) as exc:
        return _fail(exc, json_output=args.json_output)

    tracer: Tracer = NullTracer()
    memory = MemoryTracer()
    if args.trace:
        _configure_trace_logging()
        tracer = CompositeTracer(JsonLogTracer(), memory)
    elif args.json_output:
        tracer = memory

    # In JSON mode stdout is the machine contract, so streamed text goes to
    # stderr; in human mode it is the primary output.
    def echo(delta: str) -> None:
        target = sys.stderr if args.json_output else sys.stdout
        target.write(delta)
        target.flush()

    try:
        client = _build_client()
        loop = AgenticLoop(
            client=client,
            registry=registry,
            config=config,
            budget=budget,
            tracer=tracer,
            on_text=echo if stream_enabled else None,
        )
        try:
            result = loop.run(prompt)
        except AgenticLoopError:
            raise
        except Exception as exc:
            if args.debug:
                raise
            raise _translate_api_error(exc) from exc
    except AgenticLoopError as exc:
        return _fail(exc, json_output=args.json_output)

    if args.json_output:
        payload = {
            "ok": result.completed,
            "result": result.to_dict(),
            "trace": [event.to_dict() for event in memory.events],
        }
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False, default=str)
        sys.stdout.write("\n")
    else:
        if not stream_enabled:
            sys.stdout.write(result.final_text)
        sys.stdout.write("\n")
        summary = (
            f"\n[{result.stop_reason}] iterations={result.iterations} "
            f"tool_calls={len(result.tool_invocations)} "
            f"tokens={result.usage.get('total_tokens', 0)}"
        )
        sys.stderr.write(summary + "\n")
        if result.detail:
            sys.stderr.write(f"detail: {result.detail}\n")

    return EXIT_OK if result.completed else EXIT_INCOMPLETE


def _fail(exc: Exception, *, json_output: bool) -> int:
    message = str(exc)
    if json_output:
        json.dump(
            {"ok": False, "error": {"type": exc.__class__.__name__, "message": message}},
            sys.stdout,
            indent=2,
            ensure_ascii=False,
        )
        sys.stdout.write("\n")
    else:
        sys.stderr.write(f"error: {message}\n")
    return EXIT_ERROR


def _run() -> int:  # pragma: no cover - exercised via the console script
    """Entry point wrapper that survives a closed stdout.

    Piping into `head` closes the pipe early; without this the interpreter
    reports a BrokenPipeError traceback on an otherwise successful run.
    """
    try:
        return main()
    except BrokenPipeError:
        # Prevent Python from reporting the same error again at shutdown.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_OK
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_run())
