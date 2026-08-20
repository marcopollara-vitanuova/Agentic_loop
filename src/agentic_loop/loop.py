"""The agentic loop: an explicit plan → act → observe cycle over the Messages API.

The whole point of this module is that the cycle is *visible*. Every request,
every tool call, every stop condition and every token is accounted for in code
you can read, rather than inside a helper. That is what makes it auditable, and
auditability is the reason to own the loop rather than delegate it.

The shape of one iteration:

1. **Plan** — send the conversation plus the tool catalogue to the model.
2. **Observe the stop reason** — the model either answered, asked for tools,
   paused, refused, or hit a ceiling. Each case is handled explicitly.
3. **Act** — run every requested tool and append *all* results as one user
   message, then iterate.

Termination is guaranteed by :class:`~agentic_loop.budget.Budget`: the loop
cannot issue an unbounded number of requests even if the model never stops
calling tools.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .budget import Budget, BudgetTracker
from .errors import BudgetExceededError, ConfigurationError, ToolError
from .telemetry import NullTracer, TraceEvent, Tracer, new_run_id, redact
from .tools import ToolRegistry, default_registry

#: Why a run ended. ``completed`` is the only outcome that means the model
#: finished on its own terms; everything else is a ceiling or a rejection and
#: must be surfaced to the caller rather than swallowed.
StopReason = Literal[
    "completed",
    "budget_exhausted",
    "refusal",
    "max_tokens",
    "stop_sequence",
    "unexpected_stop_reason",
]

#: Beta flag required by the server-side (advisory) task budget.
TASK_BUDGET_BETA = "task-budgets-2026-03-13"

#: The API rejects a task budget below this.
MIN_TASK_BUDGET_TOKENS = 20_000


@dataclass(frozen=True)
class LoopConfig:
    """Everything that shapes the requests a run makes.

    Attributes:
        model: Model id. Defaults to Claude Opus 5.
        max_tokens: Per-response output ceiling. This is an enforced cap the
            model is *not* aware of — hitting it truncates the response.
        system: System prompt, or ``None``.
        effort: Reasoning effort — ``low`` through ``max``. Controls thinking
            depth and overall token spend.
        thinking_display: ``"summarized"`` returns a readable summary of the
            model's reasoning; ``"omitted"`` (the API default on Opus 5) leaves
            it empty. Thinking happens and is billed either way.
        stream: Stream the response. Kept on by default: with a large
            ``max_tokens`` a non-streaming request can exceed the HTTP timeout.
        task_budget_tokens: Advisory server-side budget so the model paces
            itself. ``None`` disables it. Must be at least
            :data:`MIN_TASK_BUDGET_TOKENS`.
        cache_prompt: Cache the stable request prefix. Only worth enabling once
            the prefix (tools + system) is over ~1024 tokens.
        strict_budget: Raise :class:`BudgetExceededError` instead of returning a
            result with ``stop_reason="budget_exhausted"``.
    """

    model: str = "claude-opus-5"
    max_tokens: int = 16_000
    system: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    thinking_display: Literal["summarized", "omitted"] = "summarized"
    stream: bool = True
    task_budget_tokens: int | None = None
    cache_prompt: bool = False
    strict_budget: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ConfigurationError("max_tokens must be positive")
        if self.task_budget_tokens is not None:
            if self.task_budget_tokens < MIN_TASK_BUDGET_TOKENS:
                raise ConfigurationError(
                    "task_budget_tokens must be at least "
                    f"{MIN_TASK_BUDGET_TOKENS} (got {self.task_budget_tokens})"
                )
            if not self.stream:
                # A task budget implies a long agentic turn; a non-streaming
                # request of that size is a timeout waiting to happen.
                raise ConfigurationError("task_budget_tokens requires stream=True")


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call, as executed by the loop."""

    iteration: int
    tool_use_id: str
    name: str
    tool_input: Mapping[str, Any]
    result: str
    is_error: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "tool_use_id": self.tool_use_id,
            "name": self.name,
            "input": redact(dict(self.tool_input)),
            "result": redact(self.result),
            "is_error": self.is_error,
        }


@dataclass
class LoopResult:
    """The outcome of a run, including how it ended and what it cost.

    ``stop_reason`` is the field to check first. A caller that reads
    ``final_text`` without it can silently present a budget-truncated or refused
    run as a finished answer.
    """

    run_id: str
    stop_reason: StopReason
    final_text: str
    iterations: int
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    detail: str | None = None

    @property
    def completed(self) -> bool:
        """True only when the model finished on its own terms."""
        return self.stop_reason == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
            "completed": self.completed,
            "final_text": self.final_text,
            "iterations": self.iterations,
            "detail": self.detail,
            "usage": self.usage,
            "tool_invocations": [call.to_dict() for call in self.tool_invocations],
        }


class AgenticLoop:
    """Runs an explicit tool-use loop against the Messages API.

    The Anthropic client is injected rather than constructed here. That keeps
    credential resolution outside this class and makes the loop testable against
    a fake client with no API key and no network.

    Example:
        >>> import anthropic
        >>> loop = AgenticLoop(anthropic.Anthropic())         # doctest: +SKIP
        >>> result = loop.run("How many words are in this sentence?")  # doctest: +SKIP
        >>> result.stop_reason                                 # doctest: +SKIP
        'completed'
    """

    def __init__(
        self,
        client: Any,
        registry: ToolRegistry | None = None,
        config: LoopConfig | None = None,
        budget: Budget | None = None,
        tracer: Tracer | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        """
        Args:
            client: An ``anthropic.Anthropic`` instance, or any object exposing
                the same ``messages.create`` / ``messages.stream`` surface.
            registry: Tools the model may call. Defaults to the demo registry.
            config: Request shaping. Defaults to :class:`LoopConfig`.
            budget: Hard ceilings. Defaults to :class:`Budget`.
            tracer: Event sink. Defaults to a tracer that discards events.
            on_text: Called with each streamed text delta. Only invoked when
                ``config.stream`` is true.
        """
        self.client = client
        self.registry = registry if registry is not None else default_registry()
        self.config = config or LoopConfig()
        self.budget = budget or Budget()
        self.tracer = tracer or NullTracer()
        self.on_text = on_text

    # -- public API ---------------------------------------------------------

    def run(
        self,
        user_input: str | Sequence[Mapping[str, Any]],
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> LoopResult:
        """Run the loop to a terminal state and return the outcome.

        Args:
            user_input: The user turn. A string, or a list of content blocks.
            history: Prior conversation turns to prepend. The API is stateless,
                so multi-turn conversations replay their history.

        Returns:
            A :class:`LoopResult`. Check ``stop_reason`` before using
            ``final_text``.

        Raises:
            BudgetExceededError: only when ``config.strict_budget`` is set.
        """
        run_id = new_run_id()
        tracker = BudgetTracker(budget=self.budget)
        messages: list[dict[str, Any]] = [dict(turn) for turn in (history or [])]
        messages.append({"role": "user", "content": user_input})

        invocations: list[ToolInvocation] = []
        self._emit(
            run_id,
            "run_start",
            0,
            {
                "model": self.config.model,
                "tools": self.registry.names,
                "budget": {
                    "max_iterations": self.budget.max_iterations,
                    "max_total_tokens": self.budget.max_total_tokens,
                    "max_tool_calls": self.budget.max_tool_calls,
                },
            },
        )

        stop_reason: StopReason = "budget_exhausted"
        detail: str | None = None
        last_response: Any = None

        while True:
            exhausted = tracker.exhaustion_reason()
            if exhausted is not None:
                stop_reason, detail = "budget_exhausted", exhausted
                self._emit(run_id, "budget_exhausted", tracker.iterations, {"reason": exhausted})
                break

            tracker.start_iteration()
            iteration = tracker.iterations

            request = self._build_request(messages, tracker)
            self._emit(
                run_id,
                "request",
                iteration,
                {
                    "message_count": len(messages),
                    "max_tokens": request["max_tokens"],
                    "effort": self.config.effort,
                    "streaming": self.config.stream,
                },
            )

            response = self._send(request)
            last_response = response
            tracker.record_usage(getattr(response, "usage", None))

            response_stop = getattr(response, "stop_reason", None)
            self._emit(
                run_id,
                "response",
                iteration,
                {
                    "stop_reason": response_stop,
                    "block_types": [getattr(block, "type", None) for block in response.content],
                    "usage": tracker.usage.to_dict(),
                },
            )

            # --- observe the stop reason ----------------------------------
            if response_stop == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                results, new_invocations = self._run_tools(
                    run_id, iteration, response.content, tracker
                )
                invocations.extend(new_invocations)
                # All results for one assistant turn go back in a SINGLE user
                # message. Splitting them teaches the model to stop making
                # parallel tool calls.
                messages.append({"role": "user", "content": results})
                continue

            if response_stop == "pause_turn":
                # A server-side tool hit its iteration limit mid-turn. Append
                # the paused turn verbatim and re-send to resume it.
                messages.append({"role": "assistant", "content": response.content})
                self._emit(run_id, "pause_turn_resumed", iteration, {})
                continue

            if response_stop == "end_turn":
                stop_reason = "completed"
                messages.append({"role": "assistant", "content": response.content})
                break

            if response_stop == "refusal":
                stop_reason = "refusal"
                detail = self._describe_refusal(response)
                self._emit(run_id, "refusal", iteration, {"detail": detail})
                break

            if response_stop == "max_tokens":
                stop_reason = "max_tokens"
                detail = (
                    f"response hit the max_tokens ceiling of {self.config.max_tokens}; "
                    "the answer is truncated"
                )
                messages.append({"role": "assistant", "content": response.content})
                break

            if response_stop == "stop_sequence":
                stop_reason = "stop_sequence"
                detail = "response ended on a configured stop sequence"
                messages.append({"role": "assistant", "content": response.content})
                break

            stop_reason = "unexpected_stop_reason"
            detail = f"unhandled stop_reason: {response_stop!r}"
            self._emit(run_id, "unexpected_stop_reason", iteration, {"stop_reason": response_stop})
            break

        result = LoopResult(
            run_id=run_id,
            stop_reason=stop_reason,
            final_text=self._extract_text(last_response),
            iterations=tracker.iterations,
            tool_invocations=invocations,
            messages=messages,
            usage=tracker.usage.to_dict(),
            detail=detail,
        )
        self._emit(
            run_id,
            "run_end",
            tracker.iterations,
            {
                "stop_reason": stop_reason,
                "detail": detail,
                "tool_calls": tracker.tool_calls,
                "usage": tracker.usage.to_dict(),
            },
        )

        if stop_reason == "budget_exhausted" and self.config.strict_budget:
            raise BudgetExceededError(detail or "budget exhausted")
        return result

    # -- request construction ----------------------------------------------

    def _build_request(
        self, messages: Sequence[Mapping[str, Any]], tracker: BudgetTracker
    ) -> dict[str, Any]:
        """Assemble the request body for one iteration.

        Ordering matters for prompt caching: the API renders ``tools`` → ``system``
        → ``messages``, and a cache prefix is a byte-prefix match. Tool order is
        registration order and the system prompt is fixed for the run, so the
        prefix stays stable across iterations and only the growing message tail
        varies.
        """
        output_config: dict[str, Any] = {"effort": self.config.effort}
        if self.config.task_budget_tokens is not None:
            # Advisory: the server injects a countdown the model can see so it
            # wraps up gracefully. `remaining` is deliberately left unset — the
            # server derives spend from the history we resend, and passing a
            # client-computed value alongside full history under-reports it.
            output_config["task_budget"] = {
                "type": "tokens",
                "total": self.config.task_budget_tokens,
            }

        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [dict(message) for message in messages],
            "tools": self.registry.to_wire(),
            "output_config": output_config,
            # Adaptive thinking: the model decides when and how deeply to think.
            # A fixed `budget_tokens` is not accepted on this model family.
            "thinking": {"type": "adaptive", "display": self.config.thinking_display},
        }
        if self.config.system is not None:
            request["system"] = self.config.system
        if self.config.cache_prompt:
            request["cache_control"] = {"type": "ephemeral"}
        return request

    def _send(self, request: Mapping[str, Any]) -> Any:
        """Issue one request and return the complete message.

        Routes to the beta namespace only when a server-side task budget is in
        play, so a run that does not use it takes on no beta dependency.
        """
        kwargs = dict(request)
        uses_task_budget = "task_budget" in kwargs.get("output_config", {})
        messages_api = self.client.beta.messages if uses_task_budget else self.client.messages
        if uses_task_budget:
            kwargs["betas"] = [TASK_BUDGET_BETA]

        if not self.config.stream:
            return messages_api.create(**kwargs)

        with messages_api.stream(**kwargs) as stream:
            if self.on_text is not None:
                for delta in stream.text_stream:
                    self.on_text(delta)
            # Let the SDK assemble the message rather than reconstructing it
            # from events.
            return stream.get_final_message()

    # -- acting -------------------------------------------------------------

    def _run_tools(
        self,
        run_id: str,
        iteration: int,
        content: Sequence[Any],
        tracker: BudgetTracker,
    ) -> tuple[list[dict[str, Any]], list[ToolInvocation]]:
        """Execute every ``tool_use`` block in one assistant turn.

        A failing tool produces a ``tool_result`` with ``is_error: true`` rather
        than an exception: the model can read the error and correct itself,
        which is the whole value of the loop. Dropping the result instead would
        leave the conversation with an unanswered ``tool_use`` block, which the
        API rejects.
        """
        results: list[dict[str, Any]] = []
        invocations: list[ToolInvocation] = []

        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue

            name = getattr(block, "name", "")
            tool_use_id = getattr(block, "id", "")
            # Tool input is JSON-decoded by the SDK; never string-match on it.
            tool_input = getattr(block, "input", None) or {}
            if not isinstance(tool_input, Mapping):
                tool_input = {}

            tracker.record_tool_call()
            self._emit(
                run_id,
                "tool_call",
                iteration,
                {"tool": name, "tool_use_id": tool_use_id, "input": redact(dict(tool_input))},
            )

            try:
                output = self.registry.invoke(name, tool_input)
                is_error = False
            except ToolError as exc:
                output, is_error = f"Error: {exc}", True

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": output,
                    **({"is_error": True} if is_error else {}),
                }
            )
            invocation = ToolInvocation(
                iteration=iteration,
                tool_use_id=tool_use_id,
                name=name,
                tool_input=dict(tool_input),
                result=output,
                is_error=is_error,
            )
            invocations.append(invocation)
            self._emit(
                run_id,
                "tool_result",
                iteration,
                {
                    "tool": name,
                    "tool_use_id": tool_use_id,
                    "is_error": is_error,
                    "result": redact(output),
                },
            )

        return results, invocations

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Concatenate the text blocks of a response.

        ``content`` is a heterogeneous list — thinking, tool_use and text blocks
        all appear — so blocks are filtered by ``type`` rather than indexed.
        """
        if response is None:
            return ""
        blocks = getattr(response, "content", None) or []
        return "\n".join(
            block.text
            for block in blocks
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ).strip()

    @staticmethod
    def _describe_refusal(response: Any) -> str:
        """Render refusal details defensively.

        ``stop_details`` is populated only for a refusal and is ``None``
        otherwise, so it always needs a guard.
        """
        details = getattr(response, "stop_details", None)
        if details is None:
            return "the model declined the request; no details were returned"
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
        return f"the model declined the request (category={category!r}): {explanation}"

    def _emit(self, run_id: str, kind: str, iteration: int, data: dict[str, Any]) -> None:
        self.tracer.emit(TraceEvent(run_id=run_id, kind=kind, iteration=iteration, data=data))
