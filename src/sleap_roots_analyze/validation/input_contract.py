"""Canonicalize-then-validate boundary helper for analysis input (issue #144).

This module wires the optional ``sleap-roots-contracts`` validator into analyze's
data-load boundary. It is a non-intrusive side-check: validation runs on a discarded
copy of the entry frame and never alters the data fed to the pipeline, and the whole
module degrades to a logged no-op when ``sleap-roots-contracts`` is not installed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol

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

# Single source of truth for the magic strings shared across input_contract.py,
# config/utils.py, and config/components.py (issue #144 review). ``VALIDATE_INPUT_MODES``
# is ordered for stable user-facing messages; ``CANONICAL_ROLES`` is the fixed contract
# role vocabulary (analyze renames its configured role columns to these names).
VALIDATE_INPUT_MODES = ("off", "warn", "strict")
CANONICAL_ROLES = ("genotype", "sample_id", "replicate", "image_path")


class ColumnRoles(Protocol):
    """Duck-typed view of the role-column names the validator needs.

    Matches ``pipeline.config.components.ColumnConfig`` structurally without importing it
    (keeps validation free of a config dependency) and documents the minimal interface:
    a required ``genotype`` and ``barcode`` plus an optional ``replicate``. ``image_path``
    is read via ``getattr`` since not every config declares it.
    """

    genotype: str
    barcode: str
    replicate: Optional[str]


def _build_validation_frame(
    df: pd.DataFrame,
    *,
    columns: ColumnRoles,
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

    # The genotype role is the one structural requirement; surface a misconfigured name
    # here (naming configured-vs-available) rather than letting the contract emit a bare
    # "required column 'genotype' is missing" that doesn't point at the config.
    if columns.genotype not in df.columns:
        raise ValueError(
            f"configured genotype column (columns.genotype={columns.genotype!r}) "
            f"is not present in the input frame. Available columns: {list(df.columns)}"
        )

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

    # A rename target that already exists under its canonical name (as a different column)
    # would silently collide into duplicate columns; report it instead of letting pandas
    # raise a bare "duplicate column names".
    for source, canonical in rename_map.items():
        if canonical != source and canonical in df.columns:
            raise ValueError(
                f"cannot canonicalize role column {source!r} -> {canonical!r}: the input "
                f"already has a column named {canonical!r}. Rename or drop the conflicting "
                f"column before validation."
            )

    renamed = df.rename(columns=rename_map)

    # Drop non-trait metadata, keeping role columns out of the trait set.
    trait_cols = get_trait_columns(
        renamed,
        barcode_col="sample_id",
        genotype_col="genotype",
        replicate_col="replicate" if "replicate" in renamed.columns else None,
        additional_exclude=additional_exclude,
    )
    role_cols = [c for c in CANONICAL_ROLES if c in renamed.columns]
    check = renamed[role_cols + trait_cols].copy()
    return canonicalize_role_dtypes(check)


def validate_entry_input(
    df: pd.DataFrame,
    *,
    columns: ColumnRoles,
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
        ValueError: If ``mode`` is not one of ``VALIDATE_INPUT_MODES``, or if the contract
            validation fails for the given ``mode``.
    """
    log = logger or logging.getLogger(__name__)

    # Guard the documented three-value contract: a programmatic caller bypassing
    # validate_qc_config must get an explicit error, not silent warn semantics.
    if mode not in VALIDATE_INPUT_MODES:
        raise ValueError(
            f"validate_input mode must be one of "
            f"{' | '.join(VALIDATE_INPUT_MODES)}; got {mode!r}"
        )

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
