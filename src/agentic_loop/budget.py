"""Client-side budget accounting for a loop run.

Two distinct things are called a "budget" when working with the Messages API,
and conflating them is a common source of surprise:

- **This module** is a *hard*, client-enforced ceiling. The loop stops issuing
  requests when it is exhausted. The model is not aware of it.
- **A server-side task budget** (``output_config.task_budget``) is *advisory*.
  The API tells the model how much it has left so it can pace itself and finish
  gracefully. It does not stop anything.

Use both: the task budget for graceful degradation, this one as the backstop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Budget:
    """Hard ceilings for a single run.

    Attributes:
        max_iterations: Maximum number of model requests. Every agentic loop
            needs this — without it a tool-calling model that never emits
            ``end_turn`` runs until the account is empty.
        max_total_tokens: Maximum billed tokens (input + output, cache included)
            across the run. ``None`` disables the check.
        max_tool_calls: Maximum tool invocations across the run. ``None``
            disables the check.
    """

    max_iterations: int = 10
    max_total_tokens: int | None = None
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_total_tokens is not None and self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive when set")
        if self.max_tool_calls is not None and self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative when set")


@dataclass
class Usage:
    """Cumulative token usage for a run.

    Field names mirror the API's ``usage`` object so the mapping stays obvious.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Every billed token, including both cache directions."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def add_from_response(self, usage: Any) -> None:
        """Accumulate one response's ``usage`` object.

        Missing or non-integer fields are treated as zero: a usage object that
        gains fields, or a fake used in tests, must not break accounting.
        """
        for attribute in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = getattr(usage, attribute, None) if usage is not None else None
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, attribute, getattr(self, attribute) + value)

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class BudgetTracker:
    """Tracks consumption against a :class:`Budget`.

    The tracker answers one question — *may the loop issue another request?* —
    and records why not when the answer is no.
    """

    budget: Budget
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0
    tool_calls: int = 0

    def start_iteration(self) -> None:
        self.iterations += 1

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_usage(self, usage: Any) -> None:
        self.usage.add_from_response(usage)

    def exhaustion_reason(self) -> str | None:
        """Why the loop must stop, or ``None`` if it may continue.

        Checked *before* issuing a request, so a run never pays for a request it
        has already decided it cannot afford to follow up on.
        """
        if self.iterations >= self.budget.max_iterations:
            return f"iteration cap reached ({self.iterations}/{self.budget.max_iterations})"
        if (
            self.budget.max_total_tokens is not None
            and self.usage.total_tokens >= self.budget.max_total_tokens
        ):
            return (
                f"token budget exhausted ({self.usage.total_tokens}/"
                f"{self.budget.max_total_tokens} tokens)"
            )
        if self.budget.max_tool_calls is not None and self.tool_calls >= self.budget.max_tool_calls:
            return f"tool call cap reached ({self.tool_calls}/{self.budget.max_tool_calls})"
        return None

    def remaining_tokens(self) -> int | None:
        """Tokens left against ``max_total_tokens``, or ``None`` if uncapped."""
        if self.budget.max_total_tokens is None:
            return None
        return max(0, self.budget.max_total_tokens - self.usage.total_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "usage": self.usage.to_dict(),
            "remaining_tokens": self.remaining_tokens(),
        }
