"""Tests for the tool registry, the safe evaluator, and the demo tools."""

from __future__ import annotations

import pytest

from agentic_loop.errors import ToolError, ToolNotFoundError
from agentic_loop.tools import (
    CALCULATOR,
    MAX_TOOL_RESULT_CHARS,
    WORD_STATS,
    Tool,
    ToolRegistry,
    default_registry,
    evaluate_arithmetic,
    make_document_search_tool,
)


class TestEvaluateArithmetic:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1 + 1", 2),
            ("2 * (3 + 4)", 14),
            ("10 / 4", 2.5),
            ("10 // 3", 3),
            ("10 % 3", 1),
            ("2 ** 10", 1024),
            ("-5 + 2", -3),
            ("+7", 7),
            ("1200 * 1.22", pytest.approx(1464.0)),
        ],
    )
    def test_evaluates_arithmetic(self, expression: str, expected: object) -> None:
        assert evaluate_arithmetic(expression) == expected

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "print(1)",
            "x + 1",
            "[1, 2, 3]",
            "{'a': 1}",
            "lambda: 1",
            "(1).__class__",
            "1 if True else 2",
            "1 < 2",
            "'a' * 3",
            "True",
        ],
    )
    def test_rejects_everything_outside_the_whitelist(self, expression: str) -> None:
        """No input reaches an interpreter — the AST walk is a whitelist."""
        with pytest.raises(ToolError):
            evaluate_arithmetic(expression)

    def test_rejects_division_by_zero(self) -> None:
        with pytest.raises(ToolError, match="division by zero"):
            evaluate_arithmetic("1 / 0")

    def test_caps_the_exponent(self) -> None:
        """Guards against a cheap-to-write, expensive-to-evaluate expression."""
        with pytest.raises(ToolError, match="exponent"):
            evaluate_arithmetic("9 ** 999999")

    def test_rejects_overlong_expressions(self) -> None:
        with pytest.raises(ToolError, match="too long"):
            evaluate_arithmetic("1+" * 500 + "1")

    def test_rejects_malformed_input(self) -> None:
        with pytest.raises(ToolError, match="not a valid arithmetic expression"):
            evaluate_arithmetic("1 +")


class TestCalculatorTool:
    def test_returns_expression_and_result(self) -> None:
        assert CALCULATOR.handler({"expression": "2 + 2"}) == "2 + 2 = 4"

    @pytest.mark.parametrize("payload", [{}, {"expression": ""}, {"expression": 42}])
    def test_rejects_bad_input(self, payload: dict) -> None:
        with pytest.raises(ToolError):
            CALCULATOR.handler(payload)

    def test_wire_format_is_strict(self) -> None:
        wire = CALCULATOR.to_wire()
        assert wire["strict"] is True
        assert wire["input_schema"]["additionalProperties"] is False
        assert wire["input_schema"]["required"] == ["expression"]


class TestWordStatsTool:
    def test_counts_exactly(self) -> None:
        result = WORD_STATS.handler({"text": "the cat the dog"})
        assert "words: 4" in result
        assert "unique_words: 3" in result
        assert "characters: 15" in result

    def test_ranking_is_stable_for_ties(self) -> None:
        """Ties break alphabetically so identical input gives identical output."""
        first = WORD_STATS.handler({"text": "b a c", "top_n": 3})
        second = WORD_STATS.handler({"text": "b a c", "top_n": 3})
        assert first == second
        assert first.index("a:") < first.index("b:") < first.index("c:")

    def test_handles_empty_text(self) -> None:
        result = WORD_STATS.handler({"text": ""})
        assert "words: 0" in result
        assert "lines: 0" in result

    @pytest.mark.parametrize("top_n", [0, 51, "5", True])
    def test_rejects_invalid_top_n(self, top_n: object) -> None:
        with pytest.raises(ToolError, match="top_n"):
            WORD_STATS.handler({"text": "hello", "top_n": top_n})


class TestDocumentSearchTool:
    @pytest.fixture
    def tool(self) -> Tool:
        return make_document_search_tool(
            {"a.md": "alpha line\nshared line", "b.md": "beta line\nshared line"}
        )

    def test_finds_matches_with_location(self, tool: Tool) -> None:
        result = tool.handler({"query": "alpha"})
        assert result == "a.md:1: alpha line"

    def test_is_case_insensitive(self, tool: Tool) -> None:
        assert "alpha" in tool.handler({"query": "ALPHA"})

    def test_reports_no_match_with_available_documents(self, tool: Tool) -> None:
        result = tool.handler({"query": "nonexistent"})
        assert "No match" in result
        assert "a.md" in result and "b.md" in result

    def test_respects_max_results(self, tool: Tool) -> None:
        result = tool.handler({"query": "line", "max_results": 1})
        assert len(result.splitlines()) == 1

    def test_clamps_out_of_range_max_results(self, tool: Tool) -> None:
        """Out-of-range values are clamped, not rejected — the model gets an answer."""
        assert tool.handler({"query": "line", "max_results": 0})
        assert tool.handler({"query": "line", "max_results": 9999})

    def test_corpus_is_captured_not_shared(self) -> None:
        """Mutating the caller's mapping afterwards cannot change what the tool sees."""
        documents = {"a.md": "alpha"}
        tool = make_document_search_tool(documents)
        documents["b.md"] = "beta"
        result = tool.handler({"query": "beta"})
        assert "No match" in result
        assert "b.md" not in result

    @pytest.mark.parametrize("payload", [{}, {"query": "  "}, {"query": None}])
    def test_rejects_bad_query(self, tool: Tool, payload: dict) -> None:
        with pytest.raises(ToolError):
            tool.handler(payload)


class TestToolRegistry:
    def test_preserves_registration_order(self) -> None:
        """Order is part of the prompt cache prefix, so it must be stable."""
        registry = default_registry()
        assert registry.names == ["calculator", "word_stats", "document_search"]
        assert [tool["name"] for tool in registry.to_wire()] == registry.names

    def test_rejects_duplicate_names(self) -> None:
        registry = ToolRegistry([CALCULATOR])
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CALCULATOR)

    @pytest.mark.parametrize("name", ["", "has space", "has/slash", "x" * 65])
    def test_rejects_invalid_names(self, name: str) -> None:
        tool = Tool(name=name, description="d", input_schema={}, handler=lambda _: "")
        with pytest.raises(ValueError, match="invalid tool name"):
            ToolRegistry([tool])

    def test_invoke_dispatches_by_name(self) -> None:
        registry = default_registry()
        assert registry.invoke("calculator", {"expression": "3 * 3"}) == "3 * 3 = 9"

    def test_invoke_raises_for_unknown_tool_and_lists_alternatives(self) -> None:
        registry = default_registry()
        with pytest.raises(ToolNotFoundError, match="calculator"):
            registry.invoke("nope", {})

    def test_invoke_wraps_unexpected_handler_errors(self) -> None:
        """A tool bug becomes a recoverable ToolError, never a crashed run."""

        def explode(_: object) -> str:
            raise RuntimeError("boom")

        registry = ToolRegistry(
            [Tool(name="explode", description="d", input_schema={}, handler=explode)]
        )
        with pytest.raises(ToolError, match="RuntimeError: boom"):
            registry.invoke("explode", {})

    def test_invoke_truncates_oversized_results(self) -> None:
        """An unbounded tool result is an unbounded bill."""
        registry = ToolRegistry(
            [
                Tool(
                    name="firehose",
                    description="d",
                    input_schema={},
                    handler=lambda _: "x" * (MAX_TOOL_RESULT_CHARS + 500),
                )
            ]
        )
        result = registry.invoke("firehose", {})
        assert "truncated 500 characters" in result
        assert len(result) < MAX_TOOL_RESULT_CHARS + 100

    def test_invoke_coerces_non_string_results(self) -> None:
        registry = ToolRegistry(
            [Tool(name="numeric", description="d", input_schema={}, handler=lambda _: 42)]
        )
        assert registry.invoke("numeric", {}) == "42"

    def test_membership_and_length(self) -> None:
        registry = default_registry()
        assert "calculator" in registry
        assert "nope" not in registry
        assert len(registry) == 3
