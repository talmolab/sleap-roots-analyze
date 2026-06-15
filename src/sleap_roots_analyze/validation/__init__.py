"""Optional input-contract validation at the data-load boundary (issue #144)."""

from __future__ import annotations

from sleap_roots_analyze.validation.input_contract import (
    CANONICAL_ROLES,
    CONTRACTS_AVAILABLE,
    VALIDATE_INPUT_MODES,
    validate_cross_platform_experiment,
    validate_entry_input,
)

__all__ = [
    "CANONICAL_ROLES",
    "CONTRACTS_AVAILABLE",
    "VALIDATE_INPUT_MODES",
    "validate_cross_platform_experiment",
    "validate_entry_input",
]
