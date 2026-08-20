"""Fixture: the loop could not run at all. Exits 2 with an error envelope."""

import json
import sys

json.dump(
    {"ok": False, "error": {"type": "AgenticLoopError", "message": "no credentials"}},
    sys.stdout,
)
sys.exit(2)
