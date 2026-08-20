"""Tests for redaction and the trace sinks."""

from __future__ import annotations

import json
import logging

from agentic_loop.telemetry import (
    MAX_RECORDED_CHARS,
    CompositeTracer,
    JsonLogTracer,
    MemoryTracer,
    NullTracer,
    TraceEvent,
    new_run_id,
    redact,
)


class TestRedact:
    def test_masks_sensitive_keys_case_insensitively(self) -> None:
        payload = {"API_Key": "sk-1", "secret": "s", "Authorization": "Bearer x", "note": "keep"}
        assert redact(payload) == {
            "API_Key": "[redacted]",
            "secret": "[redacted]",
            "Authorization": "[redacted]",
            "note": "keep",
        }

    def test_masks_personal_identifiers(self) -> None:
        """Redaction covers regulated identifiers, not just credentials."""
        assert redact({"codice_fiscale": "X", "iban": "Y"}) == {
            "codice_fiscale": "[redacted]",
            "iban": "[redacted]",
        }

    def test_recurses_through_nested_containers(self) -> None:
        payload = {"outer": {"token": "t", "list": [{"password": "p"}, "plain"]}}
        redacted = redact(payload)
        assert redacted["outer"]["token"] == "[redacted]"
        assert redacted["outer"]["list"][0]["password"] == "[redacted]"
        assert redacted["outer"]["list"][1] == "plain"

    def test_truncates_long_strings(self) -> None:
        result = redact("x" * (MAX_RECORDED_CHARS + 50))
        assert "truncated 50 chars" in result
        assert len(result) < MAX_RECORDED_CHARS + 60

    def test_preserves_scalars(self) -> None:
        assert redact({"n": 1, "f": 1.5, "b": True, "none": None}) == {
            "n": 1,
            "f": 1.5,
            "b": True,
            "none": None,
        }

    def test_never_raises_on_an_unexpected_type(self) -> None:
        class Odd:
            def __repr__(self) -> str:
                return "<odd>"

        assert redact({"value": Odd()}) == {"value": "<odd>"}

    def test_stringifies_non_string_keys(self) -> None:
        assert redact({1: "a"}) == {"1": "a"}


class TestTracers:
    def test_null_tracer_discards(self) -> None:
        assert NullTracer().emit(TraceEvent(run_id="r", kind="k", iteration=0)) is None

    def test_memory_tracer_records_in_order(self) -> None:
        tracer = MemoryTracer()
        tracer.emit(TraceEvent(run_id="r", kind="first", iteration=0))
        tracer.emit(TraceEvent(run_id="r", kind="second", iteration=1))
        assert tracer.kinds() == ["first", "second"]
        assert len(tracer.of_kind("first")) == 1

    def test_json_log_tracer_writes_one_parsable_line(self, caplog) -> None:
        tracer = JsonLogTracer(logger=logging.getLogger("test.trace"))
        with caplog.at_level(logging.INFO, logger="test.trace"):
            tracer.emit(TraceEvent(run_id="abc", kind="request", iteration=2, data={"n": 1}))

        assert len(caplog.records) == 1
        record = json.loads(caplog.records[0].message)
        assert record["run_id"] == "abc"
        assert record["kind"] == "request"
        assert record["iteration"] == 2
        assert record["n"] == 1

    def test_json_log_tracer_survives_unserialisable_data(self) -> None:
        tracer = JsonLogTracer(logger=logging.getLogger("test.trace2"))
        tracer.emit(TraceEvent(run_id="r", kind="k", iteration=0, data={"obj": object()}))

    def test_composite_tracer_fans_out(self) -> None:
        first, second = MemoryTracer(), MemoryTracer()
        CompositeTracer(first, second).emit(TraceEvent(run_id="r", kind="k", iteration=0))
        assert first.kinds() == second.kinds() == ["k"]


class TestTraceEvent:
    def test_flattens_data_into_the_record(self) -> None:
        event = TraceEvent(run_id="r", kind="k", iteration=1, data={"extra": "v"})
        record = event.to_dict()
        assert record["extra"] == "v"
        assert set(record) >= {"run_id", "kind", "iteration", "timestamp"}

    def test_timestamp_is_utc_iso8601(self) -> None:
        event = TraceEvent(run_id="r", kind="k", iteration=0)
        assert event.timestamp.endswith("+00:00")

    def test_run_ids_are_unique(self) -> None:
        assert new_run_id() != new_run_id()
