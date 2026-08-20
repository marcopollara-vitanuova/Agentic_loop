"""Fixture: a completed run.

Resolved by `python -m agentic_loop.cli` when the child's cwd is this fixture
directory, which lets the wrapper's real spawn-and-parse path be tested without
an API key or a network call.
"""

import json
import sys

ENVELOPE = {
    "ok": True,
    "result": {
        "run_id": "run-ok",
        "stop_reason": "completed",
        "completed": True,
        "final_text": "42",
        "iterations": 2,
        "detail": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 120,
        },
        "tool_invocations": [
            {
                "iteration": 1,
                "tool_use_id": "call_1",
                "name": "calculator",
                "input": {"expression": "6 * 7"},
                "result": "6 * 7 = 42",
                "is_error": False,
            }
        ],
    },
    "trace": [
        {
            "run_id": "run-ok",
            "kind": "run_start",
            "iteration": 0,
            "timestamp": "2026-01-01T00:00:00.000+00:00",
        }
    ],
}

sys.stderr.write("streamed text")
json.dump(ENVELOPE, sys.stdout)
sys.exit(0)
