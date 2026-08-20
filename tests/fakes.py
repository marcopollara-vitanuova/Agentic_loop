"""A fake Anthropic client.

The loop takes its client by injection and touches a deliberately small part of
the SDK surface: ``messages.create``, ``messages.stream``, and the same two
under ``beta.messages``. That is small enough to reproduce faithfully here,
which is what lets the whole suite run with no API key and no network.

Responses are scripted per test, and every request the loop issues is recorded
so tests can assert on the request shape as well as the behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeThinkingBlock:
    thinking: str = ""
    type: str = "thinking"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeStopDetails:
    category: str | None = None
    explanation: str | None = None
    type: str = "refusal"


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_details: FakeStopDetails | None = None


def text_message(text: str = "done", **kwargs: Any) -> FakeMessage:
    return FakeMessage(content=[FakeTextBlock(text=text)], stop_reason="end_turn", **kwargs)


def tool_use_message(*calls: tuple[str, str, dict[str, Any]], **kwargs: Any) -> FakeMessage:
    """Build an assistant turn requesting one or more tools.

    Args:
        *calls: ``(tool_use_id, tool_name, tool_input)`` triples. Passing more
            than one models parallel tool use.
    """
    blocks: list[Any] = [
        FakeToolUseBlock(id=call_id, name=name, input=payload) for call_id, name, payload in calls
    ]
    return FakeMessage(content=blocks, stop_reason="tool_use", **kwargs)


class _FakeStream:
    """Context manager mirroring the SDK's streaming helper."""

    def __init__(self, message: FakeMessage) -> None:
        self._message = message
        self.entered = False

    def __enter__(self) -> _FakeStream:
        self.entered = True
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    @property
    def text_stream(self):
        """Yield the message's text in chunks, as the real helper does."""
        for block in self._message.content:
            if getattr(block, "type", None) == "text":
                for chunk in block.text.split(" "):
                    yield chunk + " "

    def get_final_message(self) -> FakeMessage:
        return self._message


class _FakeMessagesResource:
    def __init__(self, client: FakeAnthropic, namespace: str) -> None:
        self._client = client
        self._namespace = namespace

    def create(self, **kwargs: Any) -> FakeMessage:
        return self._client._next(kwargs, namespace=self._namespace, streamed=False)

    def stream(self, **kwargs: Any) -> _FakeStream:
        message = self._client._next(kwargs, namespace=self._namespace, streamed=True)
        return _FakeStream(message)


class _FakeBeta:
    def __init__(self, client: FakeAnthropic) -> None:
        self.messages = _FakeMessagesResource(client, namespace="beta")


class FakeAnthropic:
    """Replays a scripted sequence of responses.

    Args:
        responses: Returned in order, one per request.
        error: Raised instead of returning the first response, to exercise
            error paths.
    """

    def __init__(self, responses: list[FakeMessage] | None = None, error: Exception | None = None):
        self._responses = list(responses or [])
        self._error = error
        self.requests: list[dict[str, Any]] = []
        self.namespaces: list[str] = []
        self.streamed: list[bool] = []
        self.messages = _FakeMessagesResource(self, namespace="default")
        self.beta = _FakeBeta(self)

    def _next(self, kwargs: dict[str, Any], *, namespace: str, streamed: bool) -> FakeMessage:
        self.requests.append(kwargs)
        self.namespaces.append(namespace)
        self.streamed.append(streamed)
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError(
                f"fake client ran out of scripted responses after {len(self.requests)} request(s); "
                "the loop issued more requests than the test expected"
            )
        return self._responses.pop(0)

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def last_request(self) -> dict[str, Any]:
        return self.requests[-1]
