"""Canonicalize-then-validate boundary helper for analysis input (issue #144).

This module wires the optional ``sleap-roots-contracts`` validator into analyze's
data-load boundary. It is a non-intrusive side-check: validation runs on a discarded
copy of the entry frame and never alters the data fed to the pipeline, and the whole
module degrades to a logged no-op when ``sleap-roots-contracts`` is not installed.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import pandas as pd

# Optional dependency: degrade to a logged no-op when absent (issue #144).
# Mirrors the UMAP_AVAILABLE guard in sleap_roots_analyze/umap.py.
try:
    from sleap_roots_contracts import (
        canonicalize_role_dtypes,
        validate_analysis_input,
    )

    CONTRACTS_AVAILABLE = True
except ImportError:
    canonicalize_role_dtypes = None
    validate_analysis_input = None
    CONTRACTS_AVAILABLE = False


def _build_validation_frame(
    df: pd.DataFrame,
    *,
    columns: Any,
    additional_exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build the canonical validation copy of an entry frame.

    Renames the configured role columns that are present to their contract-canonical
    names, drops non-trait metadata via ``get_trait_columns``, and casts role columns
    to string via the shared ``canonicalize_role_dtypes`` helper. Operates on a copy;
    the input frame is never modified.

    Args:
        df: The entry DataFrame (analyze-named columns).
        columns: A ``ColumnConfig`` providing the configured role column names.
        additional_exclude: Extra non-trait metadata columns to exclude.

    Returns:
        A canonicalized copy (role columns + trait columns) ready for the validator.
    """
    from sleap_roots_analyze.data_cleanup import get_trait_columns

    # Rename config role names -> canonical, only for roles actually present.
    rename_map = {}
    if columns.genotype in df.columns:
        rename_map[columns.genotype] = "genotype"
    if columns.barcode in df.columns:
        rename_map[columns.barcode] = "sample_id"
    if columns.replicate and columns.replicate in df.columns:
        rename_map[columns.replicate] = "replicate"
    image_path = getattr(columns, "image_path", None)
    if image_path and image_path in df.columns:
        rename_map[image_path] = "image_path"

    renamed = df.rename(columns=rename_map)

    # Drop non-trait metadata, keeping role columns out of the trait set.
    trait_cols = get_trait_columns(
        renamed,
        barcode_col="sample_id",
        genotype_col="genotype",
        replicate_col="replicate" if "replicate" in renamed.columns else None,
        additional_exclude=additional_exclude,
    )
    role_cols = [
        c
        for c in ("genotype", "sample_id", "replicate", "image_path")
        if c in renamed.columns
    ]
    check = renamed[role_cols + trait_cols].copy()
    return canonicalize_role_dtypes(check)


def validate_entry_input(
    df: pd.DataFrame,
    *,
    columns: Any,
    mode: str,
    additional_exclude: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Validate an analysis entry frame against the input contract (issue #144).

    A non-intrusive, optional side-check at the data-load boundary. Canonicalizes a
    *copy* of ``df`` (rename roles -> canonical, drop metadata, cast roles to string)
    and runs ``sleap-roots-contracts`` against it. The frame passed in is never
    modified, so enabling validation never changes pipeline results.

    Severity follows ``mode``:

    - ``"off"``: no validation work at all (returns immediately).
    - ``"warn"``: log non-fatal warnings; raise only on the universal structural
      errors (missing ``genotype``, no numeric trait, bad role dtype / NaN genotype).
    - ``"strict"``: raise on any contract violation, including recommended-column
      issues such as a missing ``sample_id``.

    When ``sleap-roots-contracts`` is not installed, validation degrades to a logged
    no-op for any mode (never an ``ImportError``).

    Args:
        df: The entry DataFrame to validate (analyze-named columns).
        columns: A ``ColumnConfig`` providing the configured role column names.
        mode: One of ``"off"``, ``"warn"``, or ``"strict"``.
        additional_exclude: Extra non-trait metadata columns to exclude from the
            validation copy (e.g. ``DataConfig.additional_exclude_cols``).
        logger: Logger to use; defaults to this module's logger.

    Raises:
        ValueError: If the contract validation fails for the given ``mode``.
    """
    log = logger or logging.getLogger(__name__)

    if mode == "off":
        return

    if not CONTRACTS_AVAILABLE:
        log.info(
            "sleap-roots-contracts not installed; skipping input validation "
            "(validate_input=%s).",
            mode,
        )
        return

    check = _build_validation_frame(
        df, columns=columns, additional_exclude=additional_exclude
    )
    result = validate_analysis_input(check, strict=(mode == "strict"))
    for warning in result.warnings:
        log.warning("input validation: %s: %s", warning.column, warning.message)
    result.raise_for_status()
