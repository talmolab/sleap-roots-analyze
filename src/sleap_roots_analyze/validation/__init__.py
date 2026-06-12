"""Optional input-contract validation at the data-load boundary (issue #144)."""

from __future__ import annotations

from sleap_roots_analyze.validation.input_contract import (
    CONTRACTS_AVAILABLE,
    validate_entry_input,
)

__all__ = ["CONTRACTS_AVAILABLE", "validate_entry_input"]
