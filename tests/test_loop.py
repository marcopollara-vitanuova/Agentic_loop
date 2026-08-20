"""Tests for the agentic loop engine.

Every test runs against the fake client in ``tests/fakes.py`` — no API key, no
network. What is being verified is the loop's own logic: how it reads stop
reasons, how it feeds tool results back, and where it refuses to continue.
"""

from __future__ import annotations

import pytest

from agentic_loop.budget import Budget
from agentic_loop.errors import BudgetExceededError, ConfigurationError
from agentic_loop.loop import (
    MIN_TASK_BUDGET_TOKENS,
    TASK_BUDGET_BETA,
    AgenticLoop,
    LoopConfig,
)
from agentic_loop.telemetry import MemoryTracer
from agentic_loop.tools import ToolRegistry, default_registry

from .fakes import (
    FakeAnthropic,
    FakeMessage,
    FakeStopDetails,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeUsage,
    text_message,
    tool_use_message,
)


def build_loop(
    responses: list[FakeMessage],
    *,
    config: LoopConfig | None = None,
    budget: Budget | None = None,
    registry: ToolRegistry | None = None,
    tracer: MemoryTracer | None = None,
    on_text=None,
) -> tuple[AgenticLoop, FakeAnthropic]:
    client = FakeAnthropic(responses)
    loop = AgenticLoop(
        client=client,
        registry=registry if registry is not None else default_registry(),
        config=config or LoopConfig(stream=False),
        budget=budget or Budget(),
        tracer=tracer,
        on_text=on_text,
    )
    return loop, client


class TestSingleTurn:
    def test_returns_the_answer_when_the_model_finishes(self) -> None:
        loop, client = build_loop([text_message("Rome")])
        result = loop.run("What is the capital of Italy?")

        assert result.completed
        assert result.stop_reason == "completed"
        assert result.final_text == "Rome"
        assert result.iterations == 1
        assert client.request_count == 1

    def test_joins_multiple_text_blocks_and_skips_thinking(self) -> None:
        """`content` is heterogeneous; only text blocks belong in the answer."""
        response = FakeMessage(
            content=[
                FakeThinkingBlock(thinking="internal"),
                FakeTextBlock(text="first"),
                FakeTextBlock(text="second"),
            ]
        )
        loop, _ = build_loop([response])
        result = loop.run("hello")
        assert result.final_text == "first\nsecond"
        assert "internal" not in result.final_text

    def test_accumulates_usage(self) -> None:
        response = text_message(usage=FakeUsage(input_tokens=100, output_tokens=25))
        loop, _ = build_loop([response])
        result = loop.run("hello")
        assert result.usage["input_tokens"] == 100
        assert result.usage["total_tokens"] == 125

    def test_prepends_history_for_multi_turn(self) -> None:
        loop, client = build_loop([text_message("Alice")])
        history = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice."},
        ]
        loop.run("What is my name?", history=history)

        sent = client.last_request["messages"]
        assert len(sent) == 3
        assert sent[0]["content"] == "My name is Alice."
        assert sent[-1] == {"role": "user", "content": "What is my name?"}


class TestToolUse:
    def test_executes_a_tool_and_feeds_the_result_back(self) -> None:
        loop, client = build_loop(
            [
                tool_use_message(("call_1", "calculator", {"expression": "6 * 7"})),
                text_message("42"),
            ]
        )
        result = loop.run("What is 6 * 7?")

        assert result.completed
        assert result.iterations == 2
        assert len(result.tool_invocations) == 1

        invocation = result.tool_invocations[0]
        assert invocation.name == "calculator"
        assert invocation.result == "6 * 7 = 42"
        assert invocation.is_error is False

        # Second request must carry the assistant turn and the tool result.
        follow_up = client.requests[1]["messages"]
        assert follow_up[1]["role"] == "assistant"
        assert follow_up[2]["role"] == "user"
        assert follow_up[2]["content"][0]["tool_use_id"] == "call_1"

    def test_returns_all_parallel_results_in_a_single_user_message(self) -> None:
        """Splitting them teaches the model to stop making parallel calls."""
        loop, client = build_loop(
            [
                tool_use_message(
                    ("call_1", "calculator", {"expression": "1 + 1"}),
                    ("call_2", "word_stats", {"text": "one two"}),
                ),
                text_message("both done"),
            ]
        )
        result = loop.run("do two things")

        assert len(result.tool_invocations) == 2
        user_messages = [m for m in client.requests[1]["messages"] if m["role"] == "user"]
        # The original prompt plus exactly one message carrying both results.
        assert len(user_messages) == 2
        results_block = user_messages[1]["content"]
        assert len(results_block) == 2
        assert [block["tool_use_id"] for block in results_block] == ["call_1", "call_2"]

    def test_runs_several_sequential_tool_rounds(self) -> None:
        loop, client = build_loop(
            [
                tool_use_message(("c1", "calculator", {"expression": "2 + 2"})),
                tool_use_message(("c2", "calculator", {"expression": "4 * 4"})),
                text_message("16"),
            ]
        )
        result = loop.run("chain")
        assert result.completed
        assert result.iterations == 3
        assert [call.iteration for call in result.tool_invocations] == [1, 2]
        assert client.request_count == 3

    def test_tool_failure_is_returned_to_the_model_as_an_error(self) -> None:
        """The model must be able to see the failure and correct itself."""
        loop, client = build_loop(
            [
                tool_use_message(("call_1", "calculator", {"expression": "1 / 0"})),
                text_message("cannot divide by zero"),
            ]
        )
        result = loop.run("divide by zero")

        assert result.completed
        invocation = result.tool_invocations[0]
        assert invocation.is_error is True
        assert "division by zero" in invocation.result

        result_block = client.requests[1]["messages"][2]["content"][0]
        assert result_block["is_error"] is True
        assert result_block["tool_use_id"] == "call_1"

    def test_unknown_tool_becomes_an_error_result_not_a_crash(self) -> None:
        loop, client = build_loop(
            [
                tool_use_message(("call_1", "no_such_tool", {})),
                text_message("recovered"),
            ]
        )
        result = loop.run("call a missing tool")

        assert result.completed
        assert result.tool_invocations[0].is_error is True
        assert "unknown tool" in result.tool_invocations[0].result
        # Every tool_use block must be answered or the API rejects the turn.
        assert client.requests[1]["messages"][2]["content"][0]["tool_use_id"] == "call_1"

    def test_ignores_non_tool_blocks_when_collecting_calls(self) -> None:
        mixed = FakeMessage(
            content=[
                FakeTextBlock(text="I will calculate that."),
                FakeThinkingBlock(thinking="…"),
            ],
            stop_reason="tool_use",
        )
        loop, client = build_loop([mixed, text_message("done")])
        result = loop.run("nothing to call")

        assert result.tool_invocations == []
        # An empty results list still closes the turn.
        assert client.requests[1]["messages"][2]["content"] == []


class TestStopReasons:
    def test_stops_on_refusal_and_reports_the_category(self) -> None:
        refusal = FakeMessage(
            content=[],
            stop_reason="refusal",
            stop_details=FakeStopDetails(category="cyber", explanation="declined"),
        )
        loop, client = build_loop([refusal])
        result = loop.run("something refused")

        assert not result.completed
        assert result.stop_reason == "refusal"
        assert "cyber" in (result.detail or "")
        assert client.request_count == 1

    def test_handles_a_refusal_with_no_details(self) -> None:
        loop, _ = build_loop([FakeMessage(content=[], stop_reason="refusal")])
        result = loop.run("refused")
        assert result.stop_reason == "refusal"
        assert "no details" in (result.detail or "")

    def test_flags_a_truncated_response(self) -> None:
        """max_tokens is an enforced cap — the answer is incomplete, not final."""
        truncated = FakeMessage(
            content=[FakeTextBlock(text="half an ans")], stop_reason="max_tokens"
        )
        loop, _ = build_loop([truncated])
        result = loop.run("write a lot")

        assert not result.completed
        assert result.stop_reason == "max_tokens"
        assert "truncated" in (result.detail or "")
        assert result.final_text == "half an ans"

    def test_resumes_a_paused_turn(self) -> None:
        paused = FakeMessage(content=[FakeTextBlock(text="partial")], stop_reason="pause_turn")
        loop, client = build_loop([paused, text_message("finished")])
        result = loop.run("long running")

        assert result.completed
        assert result.iterations == 2
        # The paused assistant turn is appended verbatim so the model resumes it.
        assert client.requests[1]["messages"][1]["role"] == "assistant"

    def test_reports_an_unrecognised_stop_reason(self) -> None:
        loop, _ = build_loop([FakeMessage(content=[], stop_reason="something_new")])
        result = loop.run("hello")
        assert result.stop_reason == "unexpected_stop_reason"
        assert "something_new" in (result.detail or "")

    def test_reports_a_stop_sequence(self) -> None:
        loop, _ = build_loop(
            [FakeMessage(content=[FakeTextBlock(text="cut")], stop_reason="stop_sequence")]
        )
        result = loop.run("hello")
        assert result.stop_reason == "stop_sequence"
        assert not result.completed


class TestBudgetEnforcement:
    def test_iteration_cap_terminates_a_model_that_never_stops(self) -> None:
        """The guarantee that makes the loop safe to run unattended."""
        endless = [
            tool_use_message((f"call_{i}", "calculator", {"expression": "1 + 1"}))
            for i in range(10)
        ]
        loop, client = build_loop(endless, budget=Budget(max_iterations=3))
        result = loop.run("loop forever")

        assert not result.completed
        assert result.stop_reason == "budget_exhausted"
        assert "iteration cap" in (result.detail or "")
        assert result.iterations == 3
        assert client.request_count == 3

    def test_token_budget_stops_before_issuing_another_request(self) -> None:
        """The check runs before the request, so no run pays for a turn it cannot follow up."""
        expensive = tool_use_message(
            ("call_1", "calculator", {"expression": "1 + 1"}),
            usage=FakeUsage(input_tokens=900, output_tokens=200),
        )
        loop, client = build_loop(
            [expensive, text_message("never reached")],
            budget=Budget(max_iterations=10, max_total_tokens=1_000),
        )
        result = loop.run("expensive")

        assert result.stop_reason == "budget_exhausted"
        assert "token budget exhausted" in (result.detail or "")
        assert client.request_count == 1

    def test_tool_call_cap_is_enforced(self) -> None:
        loop, _ = build_loop(
            [
                tool_use_message(("c1", "calculator", {"expression": "1 + 1"})),
                tool_use_message(("c2", "calculator", {"expression": "2 + 2"})),
            ],
            budget=Budget(max_iterations=10, max_tool_calls=2),
        )
        result = loop.run("many tools")
        assert result.stop_reason == "budget_exhausted"
        assert "tool call cap" in (result.detail or "")

    def test_strict_budget_raises_instead_of_returning_a_partial_result(self) -> None:
        loop, _ = build_loop(
            [tool_use_message(("c1", "calculator", {"expression": "1 + 1"}))],
            config=LoopConfig(stream=False, strict_budget=True),
            budget=Budget(max_iterations=1),
        )
        with pytest.raises(BudgetExceededError, match="iteration cap"):
            loop.run("hit the cap")

    def test_partial_work_is_preserved_when_the_budget_runs_out(self) -> None:
        """A truncated run still reports what it did before stopping."""
        loop, _ = build_loop(
            [
                tool_use_message(("c1", "calculator", {"expression": "8 * 8"})),
                tool_use_message(("c2", "calculator", {"expression": "9 * 9"})),
            ],
            budget=Budget(max_iterations=2),
        )
        result = loop.run("two rounds")
        assert result.stop_reason == "budget_exhausted"
        assert [call.result for call in result.tool_invocations] == ["8 * 8 = 64", "9 * 9 = 81"]


class TestRequestShape:
    def test_sends_tools_thinking_and_effort(self) -> None:
        loop, client = build_loop([text_message()], config=LoopConfig(stream=False, effort="xhigh"))
        loop.run("hello")

        request = client.last_request
        assert request["model"] == "claude-opus-5"
        assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert request["output_config"]["effort"] == "xhigh"
        assert [tool["name"] for tool in request["tools"]] == [
            "calculator",
            "word_stats",
            "document_search",
        ]

    def test_never_sends_parameters_the_model_family_rejects(self) -> None:
        """`budget_tokens` and sampling parameters are 400s on this family."""
        loop, client = build_loop([text_message()])
        loop.run("hello")

        request = client.last_request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "top_k" not in request
        assert "budget_tokens" not in request["thinking"]

    def test_omits_the_system_prompt_when_unset(self) -> None:
        loop, client = build_loop([text_message()])
        loop.run("hello")
        assert "system" not in client.last_request

    def test_includes_the_system_prompt_when_set(self) -> None:
        loop, client = build_loop(
            [text_message()], config=LoopConfig(stream=False, system="Be terse.")
        )
        loop.run("hello")
        assert client.last_request["system"] == "Be terse."

    def test_cache_control_is_opt_in(self) -> None:
        loop, client = build_loop([text_message()])
        loop.run("hello")
        assert "cache_control" not in client.last_request

        loop, client = build_loop(
            [text_message()], config=LoopConfig(stream=False, cache_prompt=True)
        )
        loop.run("hello")
        assert client.last_request["cache_control"] == {"type": "ephemeral"}

    def test_tool_prefix_is_byte_identical_across_iterations(self) -> None:
        """A varying prefix silently destroys prompt-cache hits."""
        loop, client = build_loop(
            [
                tool_use_message(("c1", "calculator", {"expression": "1 + 1"})),
                text_message("2"),
            ]
        )
        loop.run("hello")
        assert client.requests[0]["tools"] == client.requests[1]["tools"]
        assert client.requests[0]["output_config"] == client.requests[1]["output_config"]


class TestTaskBudget:
    def test_uses_the_default_namespace_without_a_task_budget(self) -> None:
        """No task budget means no beta dependency."""
        loop, client = build_loop([text_message()])
        loop.run("hello")
        assert client.namespaces == ["default"]
        assert "betas" not in client.last_request
        assert "task_budget" not in client.last_request["output_config"]

    def test_routes_to_the_beta_namespace_with_the_flag_when_enabled(self) -> None:
        loop, client = build_loop(
            [text_message()],
            config=LoopConfig(stream=True, task_budget_tokens=MIN_TASK_BUDGET_TOKENS),
        )
        loop.run("hello")

        assert client.namespaces == ["beta"]
        assert client.last_request["betas"] == [TASK_BUDGET_BETA]
        assert client.last_request["output_config"]["task_budget"] == {
            "type": "tokens",
            "total": MIN_TASK_BUDGET_TOKENS,
        }

    def test_leaves_remaining_unset(self) -> None:
        """The server derives spend from the resent history; a client value under-reports it."""
        loop, client = build_loop(
            [text_message()],
            config=LoopConfig(stream=True, task_budget_tokens=50_000),
        )
        loop.run("hello")
        assert "remaining" not in client.last_request["output_config"]["task_budget"]

    def test_rejects_a_task_budget_below_the_api_minimum(self) -> None:
        with pytest.raises(ConfigurationError, match=str(MIN_TASK_BUDGET_TOKENS)):
            LoopConfig(task_budget_tokens=100)

    def test_rejects_a_task_budget_without_streaming(self) -> None:
        with pytest.raises(ConfigurationError, match="stream=True"):
            LoopConfig(stream=False, task_budget_tokens=MIN_TASK_BUDGET_TOKENS)

    def test_rejects_a_non_positive_max_tokens(self) -> None:
        with pytest.raises(ConfigurationError, match="max_tokens"):
            LoopConfig(max_tokens=0)


class TestStreaming:
    def test_streams_text_deltas_to_the_callback(self) -> None:
        chunks: list[str] = []
        loop, client = build_loop(
            [text_message("streamed answer here")],
            config=LoopConfig(stream=True),
            on_text=chunks.append,
        )
        result = loop.run("hello")

        assert client.streamed == [True]
        assert "".join(chunks).strip() == "streamed answer here"
        assert result.final_text == "streamed answer here"

    def test_non_streaming_path_uses_create(self) -> None:
        loop, client = build_loop([text_message()], config=LoopConfig(stream=False))
        loop.run("hello")
        assert client.streamed == [False]

    def test_streaming_works_without_a_callback(self) -> None:
        loop, _ = build_loop([text_message("ok")], config=LoopConfig(stream=True))
        assert loop.run("hello").final_text == "ok"


class TestTracing:
    def test_emits_the_full_lifecycle(self) -> None:
        tracer = MemoryTracer()
        loop, _ = build_loop(
            [
                tool_use_message(("c1", "calculator", {"expression": "1 + 1"})),
                text_message("2"),
            ],
            tracer=tracer,
        )
        loop.run("hello")

        assert tracer.kinds() == [
            "run_start",
            "request",
            "response",
            "tool_call",
            "tool_result",
            "request",
            "response",
            "run_end",
        ]

    def test_every_event_shares_one_run_id(self) -> None:
        tracer = MemoryTracer()
        loop, _ = build_loop([text_message()], tracer=tracer)
        result = loop.run("hello")
        assert {event.run_id for event in tracer.events} == {result.run_id}

    def test_records_why_a_run_stopped_early(self) -> None:
        tracer = MemoryTracer()
        loop, _ = build_loop(
            [tool_use_message(("c1", "calculator", {"expression": "1 + 1"}))],
            budget=Budget(max_iterations=1),
            tracer=tracer,
        )
        loop.run("hello")

        assert "budget_exhausted" in tracer.kinds()
        assert tracer.of_kind("run_end")[0].data["stop_reason"] == "budget_exhausted"

    def test_redacts_sensitive_tool_input(self) -> None:
        """A trace is an operational artefact and may leave the host."""
        tracer = MemoryTracer()
        from agentic_loop.tools import Tool

        registry = ToolRegistry(
            [
                Tool(
                    name="echo",
                    description="d",
                    input_schema={},
                    handler=lambda payload: "ok",
                )
            ]
        )
        loop, _ = build_loop(
            [
                tool_use_message(("c1", "echo", {"api_key": "sk-secret", "note": "visible"})),
                text_message("done"),
            ],
            registry=registry,
            tracer=tracer,
        )
        loop.run("hello")

        recorded = tracer.of_kind("tool_call")[0].data["input"]
        assert recorded["api_key"] == "[redacted]"
        assert recorded["note"] == "visible"


class TestResultSerialisation:
    def test_result_round_trips_to_a_plain_dict(self) -> None:
        loop, _ = build_loop(
            [
                tool_use_message(("c1", "calculator", {"expression": "1 + 1"})),
                text_message("2"),
            ]
        )
        payload = loop.run("hello").to_dict()

        assert payload["completed"] is True
        assert payload["stop_reason"] == "completed"
        assert payload["final_text"] == "2"
        assert payload["tool_invocations"][0]["name"] == "calculator"
        assert "total_tokens" in payload["usage"]

    def test_run_ids_are_unique_per_run(self) -> None:
        loop, _ = build_loop([text_message(), text_message()])
        first = loop.run("one").run_id
        second = loop.run("two").run_id
        assert first != second
