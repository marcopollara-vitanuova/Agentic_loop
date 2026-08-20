"""Tests for client-side budget accounting."""

from __future__ import annotations

import pytest

from agentic_loop.budget import Budget, BudgetTracker, Usage

from .fakes import FakeUsage


class TestBudgetValidation:
    def test_rejects_zero_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            Budget(max_iterations=0)

    def test_rejects_non_positive_token_cap(self) -> None:
        with pytest.raises(ValueError, match="max_total_tokens"):
            Budget(max_total_tokens=0)

    def test_rejects_negative_tool_call_cap(self) -> None:
        with pytest.raises(ValueError, match="max_tool_calls"):
            Budget(max_tool_calls=-1)


class TestUsage:
    def test_total_includes_both_cache_directions(self) -> None:
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=3,
        )
        assert usage.total_tokens == 38

    def test_accumulates_across_responses(self) -> None:
        usage = Usage()
        usage.add_from_response(FakeUsage(input_tokens=10, output_tokens=5))
        usage.add_from_response(FakeUsage(input_tokens=7, output_tokens=2))
        assert usage.input_tokens == 17
        assert usage.output_tokens == 7

    def test_tolerates_missing_and_malformed_fields(self) -> None:
        """Accounting must not break on a usage object that gains or omits fields."""
        usage = Usage()
        usage.add_from_response(None)
        usage.add_from_response(object())
        usage.add_from_response(FakeUsage(input_tokens=4))
        assert usage.total_tokens == 4

    def test_ignores_booleans_masquerading_as_ints(self) -> None:
        class Weird:
            input_tokens = True
            output_tokens = 3

        usage = Usage()
        usage.add_from_response(Weird())
        assert usage.input_tokens == 0
        assert usage.output_tokens == 3


class TestBudgetTracker:
    def test_permits_work_while_under_every_cap(self) -> None:
        tracker = BudgetTracker(Budget(max_iterations=3, max_total_tokens=100))
        tracker.start_iteration()
        tracker.record_usage(FakeUsage(input_tokens=10))
        assert tracker.exhaustion_reason() is None

    def test_stops_on_iteration_cap(self) -> None:
        tracker = BudgetTracker(Budget(max_iterations=2))
        tracker.start_iteration()
        tracker.start_iteration()
        reason = tracker.exhaustion_reason()
        assert reason is not None
        assert "iteration cap" in reason

    def test_stops_on_token_cap(self) -> None:
        tracker = BudgetTracker(Budget(max_iterations=99, max_total_tokens=50))
        tracker.record_usage(FakeUsage(input_tokens=40, output_tokens=15))
        reason = tracker.exhaustion_reason()
        assert reason is not None
        assert "token budget exhausted" in reason

    def test_stops_on_tool_call_cap(self) -> None:
        tracker = BudgetTracker(Budget(max_iterations=99, max_tool_calls=1))
        tracker.record_tool_call()
        reason = tracker.exhaustion_reason()
        assert reason is not None
        assert "tool call cap" in reason

    def test_remaining_tokens_is_none_when_uncapped(self) -> None:
        assert BudgetTracker(Budget()).remaining_tokens() is None

    def test_remaining_tokens_never_goes_negative(self) -> None:
        tracker = BudgetTracker(Budget(max_total_tokens=10))
        tracker.record_usage(FakeUsage(input_tokens=100))
        assert tracker.remaining_tokens() == 0

    def test_serialises_for_reporting(self) -> None:
        tracker = BudgetTracker(Budget(max_total_tokens=100))
        tracker.start_iteration()
        tracker.record_tool_call()
        tracker.record_usage(FakeUsage(input_tokens=10))
        snapshot = tracker.to_dict()
        assert snapshot["iterations"] == 1
        assert snapshot["tool_calls"] == 1
        assert snapshot["usage"]["total_tokens"] == 10
        assert snapshot["remaining_tokens"] == 90
