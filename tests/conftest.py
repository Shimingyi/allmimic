"""pytest setup: the repository root (allmimic, scripts, tools) and tests/ on sys.path.

  pytest tests          # CPU checks; every file also runs as a script: PYTHONPATH=. python tests/test_flags.py

tests/gpu/ holds GPU harnesses (scripts with arguments) and tests/test_motion_lib.py needs built libraries, so pytest
skips both; run them as scripts (see their docstrings).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
collect_ignore = ['gpu', 'test_motion_lib.py']
