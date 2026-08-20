"""Structured tracing for the agentic loop.

The loop emits one event per meaningful transition (request issued, response
received, tool invoked, budget consumed, loop terminated). Events are plain
dictionaries so they can be logged as JSON, shipped to a collector, or asserted
against in tests without a mocking framework.

Design notes:

- The tracer is *not* a logger. It records events; the sink decides what to do
  with them. :class:`JsonLogTracer` is the batteries-included sink.
- Tool inputs and results are truncated and redacted before being recorded.
  A trace is an operational artefact and may be shipped off-host, so it must
  never become an exfiltration path for prompt content or secrets.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

#: Keys whose values are replaced wholesale in recorded payloads.
_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|credential|authorization|cookie"
    r"|codice[_-]?fiscale|iban)",
    re.IGNORECASE,
)

_REDACTED = "[redacted]"

#: Recorded strings longer than this are truncated. Traces are for diagnosis,
#: not for reconstructing full payloads.
MAX_RECORDED_CHARS = 2_000


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def redact(value: Any, *, max_chars: int = MAX_RECORDED_CHARS) -> Any:
    """Return ``value`` with sensitive keys masked and long strings truncated.

    Recurses through dicts and sequences. Anything that is not a container, a
    string, or a JSON scalar is rendered via ``repr`` so a tracer can never
    raise on an unexpected type.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _SENSITIVE_KEY_RE.search(key_str):
                out[key_str] = _REDACTED
            else:
                out[key_str] = redact(item, max_chars=max_chars)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item, max_chars=max_chars) for item in value]
    if isinstance(value, str):
        if len(value) > max_chars:
            return f"{value[:max_chars]}… [truncated {len(value) - max_chars} chars]"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(repr(value), max_chars=max_chars)


@dataclass(frozen=True)
class TraceEvent:
    """A single observable transition in a loop run."""

    run_id: str
    kind: str
    iteration: int
    timestamp: str = field(default_factory=_utc_now_iso)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            **self.data,
        }


class Tracer(Protocol):
    """Sink for loop events."""

    def emit(self, event: TraceEvent) -> None:  # pragma: no cover - protocol
        ...


class NullTracer:
    """Discards every event. The default, so the library stays quiet by default."""

    def emit(self, event: TraceEvent) -> None:
        return None


class MemoryTracer:
    """Keeps events in a list. Intended for tests and short-lived inspection."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        """Event kinds in emission order — convenient for asserting on flow."""
        return [event.kind for event in self.events]

    def of_kind(self, kind: str) -> list[TraceEvent]:
        return [event for event in self.events if event.kind == kind]


class JsonLogTracer:
    """Writes each event as a single-line JSON record to a stdlib logger.

    One line per event keeps the output greppable and ingestible by log
    pipelines without a parser for multi-line records.
    """

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO) -> None:
        self._logger = logger or logging.getLogger("agentic_loop.trace")
        self._level = level

    def emit(self, event: TraceEvent) -> None:
        self._logger.log(self._level, json.dumps(event.to_dict(), ensure_ascii=False, default=str))


class CompositeTracer:
    """Fans one event out to several tracers (e.g. log it and keep it in memory)."""

    def __init__(self, *tracers: Tracer) -> None:
        self._tracers = tracers

    def emit(self, event: TraceEvent) -> None:
        for tracer in self._tracers:
            tracer.emit(event)


def new_run_id() -> str:
    """Correlation id for one loop run. Used as the join key across all events."""
    return uuid.uuid4().hex
