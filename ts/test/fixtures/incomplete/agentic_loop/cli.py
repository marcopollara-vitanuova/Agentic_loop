"""Fixture: a run that stopped on its budget. Exits 1 — a result, not a failure."""

import json
import sys

json.dump(
    {
        "ok": False,
        "result": {
            "run_id": "run-partial",
            "stop_reason": "budget_exhausted",
            "completed": False,
            "final_text": "",
            "iterations": 3,
            "detail": "iteration cap reached (3/3)",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_tokens": 15,
            },
            "tool_invocations": [],
        },
        "trace": [],
    },
    sys.stdout,
)
sys.exit(1)
