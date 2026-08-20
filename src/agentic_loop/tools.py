"""Tool definitions and the registry the loop executes against.

A tool is a name, a JSON Schema describing its input, and a callable. The
registry owns two responsibilities the loop should not have to think about:
turning tools into the wire format the Messages API expects, and dispatching a
model-chosen name to the right callable with validated input.

The demo tools here are deliberately pure and deterministic: no filesystem, no
network, no clock. They exist to exercise the loop engine, so a failing test
points at the loop rather than at the environment.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import ToolError, ToolNotFoundError

#: Tool results longer than this are truncated before being sent back to the
#: model. An unbounded tool result is an unbounded bill and a context-window
#: hazard.
MAX_TOOL_RESULT_CHARS = 8_000


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke.

    Attributes:
        name: Wire name. Must match ``^[a-zA-Z0-9_-]{1,64}$``.
        description: What the tool does and when to use it. The model reads this
            to decide; vague descriptions are the most common cause of a tool
            being ignored or misused.
        input_schema: JSON Schema for the input object.
        handler: Receives the validated input mapping, returns a string.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[Mapping[str, Any]], str]

    def to_wire(self) -> dict[str, Any]:
        """Render the tool as a Messages API tool definition.

        ``strict`` is set so the API guarantees the input validates against the
        schema, which removes a whole class of defensive parsing from handlers.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


class ToolRegistry:
    """An ordered, name-unique collection of tools.

    Order is preserved and stable: the wire tool list is part of the prompt
    cache prefix, so a set-like iteration order would silently destroy cache
    hits between otherwise identical requests.
    """

    _NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not self._NAME_RE.match(tool.name):
            raise ValueError(f"invalid tool name {tool.name!r}: must match {self._NAME_RE.pattern}")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def to_wire(self) -> list[dict[str, Any]]:
        """Tool definitions in registration order."""
        return [tool.to_wire() for tool in self._tools.values()]

    def invoke(self, name: str, tool_input: Mapping[str, Any]) -> str:
        """Run a tool by name and return its result, truncated to a safe length.

        Raises:
            ToolNotFoundError: the name is not registered.
            ToolError: the handler rejected the input or failed.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"unknown tool {name!r}; available tools: {', '.join(self.names) or 'none'}"
            )
        try:
            result = tool.handler(tool_input)
        except ToolError:
            raise
        except Exception as exc:  # a tool bug must not kill the run
            raise ToolError(f"tool {name!r} failed: {exc.__class__.__name__}: {exc}") from exc

        if not isinstance(result, str):
            result = str(result)
        if len(result) > MAX_TOOL_RESULT_CHARS:
            omitted = len(result) - MAX_TOOL_RESULT_CHARS
            result = f"{result[:MAX_TOOL_RESULT_CHARS]}\n… [truncated {omitted} characters]"
        return result


# ---------------------------------------------------------------------------
# Safe arithmetic evaluation
# ---------------------------------------------------------------------------

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Guards against `9**9**9`-style expressions that are cheap to write and
#: expensive to evaluate.
_MAX_EXPONENT = 64
_MAX_EXPRESSION_CHARS = 200


def evaluate_arithmetic(expression: str) -> float | int:
    """Evaluate an arithmetic expression without ``eval``.

    Only numeric literals and the operators ``+ - * / // % **`` are accepted;
    names, calls, attributes, subscripts and comprehensions are rejected by
    construction. This is a whitelist walk over the AST, not a sanitiser — there
    is no input that reaches an interpreter.

    Raises:
        ToolError: the expression is malformed, too large, or uses a construct
            outside the whitelist.
    """
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ToolError(
            f"expression too long ({len(expression)} chars, max {_MAX_EXPRESSION_CHARS})"
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not a valid arithmetic expression: {exc.msg}") from exc

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError("only int and float literals are allowed")
            return node.value
        if isinstance(node, ast.UnaryOp):
            unary_op = _UNARY_OPS.get(type(node.op))
            if unary_op is None:
                raise ToolError(f"unsupported unary operator: {type(node.op).__name__}")
            return unary_op(visit(node.operand))
        if isinstance(node, ast.BinOp):
            binary_op = _BIN_OPS.get(type(node.op))
            if binary_op is None:
                raise ToolError(f"unsupported operator: {type(node.op).__name__}")
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ToolError(f"exponent {right} exceeds the maximum of {_MAX_EXPONENT}")
            try:
                return binary_op(left, right)
            except ZeroDivisionError as exc:
                raise ToolError("division by zero") from exc
            except OverflowError as exc:
                raise ToolError("numeric overflow") from exc
        raise ToolError(f"unsupported expression element: {type(node).__name__}")

    result = visit(tree)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ToolError("expression did not evaluate to a number")
    return result


# ---------------------------------------------------------------------------
# Demo tools
# ---------------------------------------------------------------------------


def _calculator_handler(payload: Mapping[str, Any]) -> str:
    expression = payload.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("'expression' must be a non-empty string")
    result = evaluate_arithmetic(expression)
    return f"{expression.strip()} = {result}"


CALCULATOR = Tool(
    name="calculator",
    description=(
        "Evaluate a single arithmetic expression over numbers. Supports "
        "+ - * / // % and ** with parentheses. Use this instead of doing "
        "arithmetic yourself whenever precision matters. Does not accept "
        "variables, function calls, or units."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(1200 * 1.22) / 3'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=_calculator_handler,
)


def _word_stats_handler(payload: Mapping[str, Any]) -> str:
    text = payload.get("text")
    if not isinstance(text, str):
        raise ToolError("'text' must be a string")
    top_n = payload.get("top_n", 5)
    if not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= 50:
        raise ToolError("'top_n' must be an integer between 1 and 50")

    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    frequencies: dict[str, int] = {}
    for word in words:
        frequencies[word] = frequencies.get(word, 0) + 1
    # Sort by descending count, then alphabetically, so the output is stable.
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:top_n]

    lines = [
        f"characters: {len(text)}",
        f"words: {len(words)}",
        f"unique_words: {len(frequencies)}",
        f"lines: {len(text.splitlines()) if text else 0}",
    ]
    if ranked:
        lines.append("most_frequent:")
        lines.extend(f"  {word}: {count}" for word, count in ranked)
    return "\n".join(lines)


WORD_STATS = Tool(
    name="word_stats",
    description=(
        "Compute exact statistics for a block of text: character count, word "
        "count, unique word count, line count, and the most frequent words. "
        "Use this instead of estimating counts by reading."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to analyse."},
            "top_n": {
                "type": "integer",
                "description": "How many of the most frequent words to return (1-50).",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    handler=_word_stats_handler,
)


def make_document_search_tool(documents: Mapping[str, str]) -> Tool:
    """Build a search tool bound to a fixed, in-memory document set.

    The corpus is captured at construction time. The model can only ever see
    what is in ``documents`` — there is no path, no glob, and no filesystem
    access, so there is nothing for a crafted query to traverse to.

    Args:
        documents: Mapping of document title to document body.
    """
    corpus = dict(documents)

    def handler(payload: Mapping[str, Any]) -> str:
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("'query' must be a non-empty string")
        max_results = payload.get("max_results", 5)
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ToolError("'max_results' must be an integer")
        max_results = max(1, min(max_results, 20))

        needle = query.strip().lower()
        hits: list[str] = []
        for title, body in corpus.items():
            for line_number, line in enumerate(body.splitlines(), start=1):
                if needle in line.lower():
                    hits.append(f"{title}:{line_number}: {line.strip()}")
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break

        if not hits:
            available = ", ".join(corpus) or "none"
            return f"No match for {query!r}. Documents available: {available}."
        return "\n".join(hits)

    return Tool(
        name="document_search",
        description=(
            "Search a fixed set of in-memory documents for a literal, "
            "case-insensitive substring. Returns matching lines as "
            "'title:line: text'. Use this to ground an answer in the documents "
            "rather than recalling their contents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Literal substring to search for (not a regex).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matching lines to return (1-20, default 5).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


#: A tiny corpus so the demo registry is useful out of the box. Content is
#: illustrative sample text, not a statement about any real product or policy.
SAMPLE_DOCUMENTS: dict[str, str] = {
    "loop-notes.md": (
        "The agentic loop runs plan, act, observe until the model stops asking for tools.\n"
        "Every iteration is bounded by a hard iteration cap and a token budget.\n"
        "Tool failures are returned to the model as errors so it can recover.\n"
    ),
    "operations.md": (
        "Each run emits structured trace events keyed by a single run_id.\n"
        "Traces are redacted and truncated before they leave the process.\n"
        "A run that stops on budget exhaustion is reported, never silently truncated.\n"
    ),
}


def default_registry(documents: Mapping[str, str] | None = None) -> ToolRegistry:
    """Return the demo registry: calculator, word_stats, document_search.

    Args:
        documents: Corpus for ``document_search``. Defaults to
            :data:`SAMPLE_DOCUMENTS`.
    """
    return ToolRegistry(
        [
            CALCULATOR,
            WORD_STATS,
            make_document_search_tool(SAMPLE_DOCUMENTS if documents is None else documents),
        ]
    )


def registry_from_tools(tools: Sequence[Tool]) -> ToolRegistry:
    """Convenience constructor for a registry from an explicit tool sequence."""
    return ToolRegistry(tools)
