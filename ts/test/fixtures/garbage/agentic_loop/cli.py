"""Fixture: writes something other than the contract to stdout."""

import sys

sys.stdout.write("Warning: this is not JSON")
sys.exit(0)
