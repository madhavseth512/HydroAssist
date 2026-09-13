"""The browser's setup choices must match what the server can convert.

frontend/src/setupOptions.ts repeats the pumping-rate units and the recovery
clocks from src/project/workspace.py. Offering a unit the server cannot convert
would turn a valid-looking form into a rejected request, so the two lists are
compared here rather than trusted to stay in step.
"""

import re
from pathlib import Path

import pytest

from src.project import workspace as ws

OPTIONS = Path("frontend/src/setupOptions.ts")

pytestmark = pytest.mark.skipif(not OPTIONS.is_file(),
                                reason="frontend sources not present")


def _source() -> str:
    return OPTIONS.read_text(encoding="utf-8")


def test_the_pumping_rate_units_match_the_conversions():
    match = re.search(r"Q_UNITS\s*=\s*\[([^\]]*)\]", _source())
    assert match, "Q_UNITS not found in setupOptions.ts"

    units = re.findall(r"'([^']*)'", match.group(1))
    assert units == ws.Q_UNITS


def test_the_recovery_clocks_match_the_server():
    pairs = re.findall(r"value:\s*'([^']*)',\s*label:\s*'([^']*)'", _source())
    assert dict(pairs) == ws.CLOCK_LABELS
