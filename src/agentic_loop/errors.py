"""Exception hierarchy for the agentic loop.

Every failure the loop can raise is a subclass of :class:`AgenticLoopError`, so
callers can catch the whole surface with one `except` while still being able to
discriminate the cases that matter.
"""

from __future__ import annotations


class AgenticLoopError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(AgenticLoopError):
    """The loop was configured with values that cannot produce a valid request."""


class ToolError(AgenticLoopError):
    """A tool failed in a way the model is allowed to see and recover from.

    The loop catches this, returns it to the model as a `tool_result` with
    ``is_error: true``, and keeps iterating. Raise it from a tool when the input
    is invalid or the operation cannot be completed — not for programming bugs.
    """


class ToolNotFoundError(ToolError):
    """The model asked for a tool name that is not in the registry."""


class BudgetExceededError(AgenticLoopError):
    """A hard budget was exhausted and the loop refused to issue another request.

    Only raised when the loop is configured with ``strict_budget=True``; the
    default behaviour is to stop cleanly and report the reason in the result.
    """


class ApiError(AgenticLoopError):
    """The conversation with the API failed.

    Wraps whatever the SDK raised — authentication, rate limiting, connectivity,
    a rejected request — so callers of this package see one error surface and a
    message they can act on, instead of a raw traceback.
    """

    def __init__(self, message: str, *, kind: str = "api_error") -> None:
        super().__init__(message)
        #: Coarse classification, useful for deciding whether to retry.
        self.kind = kind
