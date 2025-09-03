"""Pytest configuration file for test discovery and fixture loading."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

# Import all fixtures from centralized fixtures file
from tests.fixtures import *  # noqa: F401, F403
