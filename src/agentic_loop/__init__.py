"""An explicit, auditable agentic loop over the Claude Messages API.

The loop owns its own plan → act → observe cycle instead of delegating it, so
every request, tool call, stop condition and token is visible in code.

Typical use:

    import anthropic
    from agentic_loop import AgenticLoop, Budget, LoopConfig

    loop = AgenticLoop(
        client=anthropic.Anthropic(),
        config=LoopConfig(model="claude-opus-5", effort="high"),
        budget=Budget(max_iterations=8, max_total_tokens=200_000),
    )
    result = loop.run("How many words are in the loop notes?")
    if result.completed:
        print(result.final_text)
    else:
        print(f"stopped early: {result.stop_reason} — {result.detail}")
"""

from __future__ import annotations

from .budget import Budget, BudgetTracker, Usage
from .errors import (
    AgenticLoopError,
    BudgetExceededError,
    ConfigurationError,
    ToolError,
    ToolNotFoundError,
)
from .loop import (
    MIN_TASK_BUDGET_TOKENS,
    TASK_BUDGET_BETA,
    AgenticLoop,
    LoopConfig,
    LoopResult,
    StopReason,
    ToolInvocation,
)
from .telemetry import (
    CompositeTracer,
    JsonLogTracer,
    MemoryTracer,
    NullTracer,
    TraceEvent,
    Tracer,
)
from .tools import (
    CALCULATOR,
    SAMPLE_DOCUMENTS,
    WORD_STATS,
    Tool,
    ToolRegistry,
    default_registry,
    evaluate_arithmetic,
    make_document_search_tool,
)

__version__ = "0.1.0"

__all__ = [
    "CALCULATOR",
    "MIN_TASK_BUDGET_TOKENS",
    "SAMPLE_DOCUMENTS",
    "TASK_BUDGET_BETA",
    "WORD_STATS",
    "AgenticLoop",
    "AgenticLoopError",
    "Budget",
    "BudgetExceededError",
    "BudgetTracker",
    "CompositeTracer",
    "ConfigurationError",
    "JsonLogTracer",
    "LoopConfig",
    "LoopResult",
    "MemoryTracer",
    "NullTracer",
    "StopReason",
    "Tool",
    "ToolError",
    "ToolInvocation",
    "ToolNotFoundError",
    "ToolRegistry",
    "TraceEvent",
    "Tracer",
    "Usage",
    "__version__",
    "default_registry",
    "evaluate_arithmetic",
    "make_document_search_tool",
]
