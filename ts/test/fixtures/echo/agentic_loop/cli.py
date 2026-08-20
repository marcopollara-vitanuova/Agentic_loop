"""Fixture: echoes the argv it received, so argument mapping can be asserted."""

import json
import sys

json.dump(
    {
        "ok": True,
        "result": {
            "run_id": "run-echo",
            "stop_reason": "completed",
            "completed": True,
            "final_text": " ".join(sys.argv[1:]),
            "iterations": 1,
            "detail": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_tokens": 0,
            },
            "tool_invocations": [],
        },
        "trace": [],
    },
    sys.stdout,
)
sys.exit(0)
