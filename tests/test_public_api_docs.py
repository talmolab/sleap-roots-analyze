"""Enforce the public-API introspection contract in the test suite.

Wraps ``scripts/check_public_api_docs.py`` so every ``sleap_roots_analyze.__all__``
entry is verified introspection-ready (complete type hints + parsable docstrings)
on every test run. See issue #117 — this guards bloom-mcp's autopop generation path.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_api_docs.py"


def _load_audit():
    """Import the audit script as a module.

    Returns:
        The loaded ``check_public_api_docs`` module exposing ``run_audit``.
    """
    spec = importlib.util.spec_from_file_location("check_public_api_docs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_public_api_entries_are_introspection_ready():
    """Every __all__ entry passes the introspection contract (issue #117)."""
    violations = _load_audit().run_audit()
    assert violations == [], "Public API introspection violations:\n" + "\n".join(
        f"  - {v}" for v in violations
    )
