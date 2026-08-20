"""Fixture: never finishes, so the timeout and abort paths can be exercised."""

import time

time.sleep(60)
