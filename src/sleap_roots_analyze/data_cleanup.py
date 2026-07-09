"""Data loading utilities for wheat trait analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np
import inspect
import json
import logging
import warnings

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .data_utils import convert_to_json_serializable, create_run_directory

# Set up module logger
logger = logging.getLogger(__name__)

# Minimum surviving samples for the entry point to return an analysis-ready frame.
# This is the mathematical runnability floor for PCA, NOT a sufficiency guarantee.
MIN_SAMPLES_FOR_ANALYSIS = 2


def load_trait_data(
    csv_path: Path | str,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
) -> pd.DataFrame:
    """Load and validate trait data from CSV file.

    Args:
        csv_path: Path to trait CSV file
        barcode_col: Name of the barcode/plant ID column (default: "Barcode")
        genotype_col: Name of the genotype column (default: "geno")
        replicate_col: Name of the replicate column if present (default: "rep", None if not needed)

    Returns:
        DataFrame with trait data

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Trait data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check for required columns
    required_cols = [barcode_col, genotype_col]
    if replicate_col:
        required_cols.append(replicate_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Try to find similar column names for helpful error message
        available_cols = df.columns.tolist()
        suggestions = []
        for missing in missing_cols:
            similar = [
                col
                for col in available_cols
                if missing.lower() in col.lower() or col.lower() in missing.lower()
            ]
            if similar:
                suggestions.append(f"{missing} (maybe: {', '.join(similar[:3])})")
            else:
                suggestions.append(missing)
        raise ValueError(
            f"Missing required columns: {suggestions}. Available columns: {available_cols[:10]}..."
        )

    return df


def get_trait_columns(
    df: pd.DataFrame,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    additional_exclude: Optional[List[str]] = None,
) -> List[str]:
    """Get list of numeric trait columns excluding metadata.

    Args:
        df: Trait dataframe
        barcode_col: Name of the barcode/plant ID column to exclude (default: "Barcode")
        genotype_col: Name of the genotype column to exclude (default: "geno")
        replicate_col: Name of the replicate column to exclude if present (default: "rep")
        additional_exclude: Additional columns to exclude (e.g., date columns)

    Returns:
        List of trait column names
    """
    # Build list of columns to exclude
    exclude_cols = [barcode_col, genotype_col]
    if replicate_col:
        exclude_cols.append(replicate_col)

    # Add any additional exclusions
    if additional_exclude:
        exclude_cols.extend(additional_exclude)

    # CRITICAL: Also exclude Plot column (root core pipeline metadata)
    # Root core pipeline uses: Plot, Rep, geno, Barcode
    if "Plot" in df.columns:
        exclude_cols.append("Plot")

    # Also exclude common metadata columns that might exist with different names
    # These are case-insensitive substring matches
    metadata_substring_patterns = [
        "date",
        "time",
        "sterilization",
        "experiment",
        "batch",
        "operator",
        "notes",
        "comments",
        "index",
        "qc_",  # QC-related columns
        "outlier",  # Outlier flags
        "wave_name",  # Experimental metadata
        "wave_number",  # Experimental metadata
        "germ_day",  # Germination day (experimental metadata)
        "plant_age",  # Plant age (experimental metadata)
        "age_days",  # Age in days (experimental metadata)
        "day_",  # Day-related columns
        "_color",  # Color coding columns
        "dot",  # Date of treatment/transplant
        "scan_",  # Scan-related metadata
        "scanner",  # Scanner information
        "phenotyper",  # Phenotyper information
        "uploaded",  # Upload timestamps
        "accession",  # Accession IDs
        "species_",  # Species information
        "plant_name",  # Plant naming
    ]
    # Suffix patterns: matched with str.endswith() to avoid false positives.
    # Bug #75: bare "id" substring matched "width", "widths", "solidity" etc.
    # All real ID columns in SLEAP Roots data use the *_id suffix pattern.
    metadata_suffix_patterns = [
        "_id",
    ]
    for col in df.columns:
        col_lower = col.lower()
        if any(meta in col_lower for meta in metadata_substring_patterns):
            exclude_cols.append(col)
        elif any(col_lower.endswith(suffix) for suffix in metadata_suffix_patterns):
            exclude_cols.append(col)

    # Remove duplicates and filter to columns that actually exist
    exclude_cols = list(set(col for col in exclude_cols if col in df.columns))

    # Get all columns not in exclude list. Ordering is taken from df.columns
    # (stable), NOT from the set() above, so trait order is deterministic.
    trait_cols = [col for col in df.columns if col not in exclude_cols]

    # Only return numeric columns
    numeric_cols = []
    for col in trait_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def get_numeric_traits_only(
    df: pd.DataFrame,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    additional_exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract only numeric trait columns for analysis.

    Args:
        df: Full trait dataframe
        barcode_col: Name of the barcode/plant ID column to exclude (default: "Barcode")
        genotype_col: Name of the genotype column to exclude (default: "geno")
        replicate_col: Name of the replicate column to exclude if present (default: "rep")
        additional_exclude: Additional columns to exclude (e.g., date columns)

    Returns:
        DataFrame with only numeric trait columns
    """
    trait_cols = get_trait_columns(
        df,
        barcode_col=barcode_col,
        genotype_col=genotype_col,
        replicate_col=replicate_col,
        additional_exclude=additional_exclude,
    )
    return df[trait_cols].copy()


def save_cleaned_data(
    df: pd.DataFrame,
    outliers: Dict,
    run_dir: Path | str,
    log_info: Optional[Dict] = None,
) -> Tuple[Path, Path]:
    """Save cleaned data and analysis logs.

    Args:
        df: Cleaned trait dataframe
        outliers: Dictionary with outlier detection results
        run_dir: Run directory path
        log_info: Additional information to log

    Returns:
        Tuple of (cleaned_data_path, log_path)
    """
    run_dir = Path(run_dir)

    # Save cleaned data
    cleaned_path = run_dir / "cleaned_traits.csv"
    df.to_csv(cleaned_path, index=False)

    # Create log dictionary
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "original_samples": len(df),
        "outlier_detection": convert_to_json_serializable(outliers),
        "cleaned_data_path": cleaned_path.as_posix(),
    }

    if log_info:
        log_data.update(convert_to_json_serializable(log_info))

    # Save log
    log_path = run_dir / "cleanup_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return cleaned_path, log_path


def remove_nan_samples(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_nan_fraction: float = 0.2,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    save_removed_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Remove samples with NaN values in trait columns and optionally save removed rows.

    This is the centralized function for NaN removal that should be called
    before any outlier detection or other analysis. It handles both removal
    and saving of removed samples in a single operation.

    Args:
        df: Original dataframe
        trait_cols: List of trait columns to check for NaN
        max_nan_fraction: Maximum fraction of NaN allowed per sample (0-1). This
            standalone default (0.2) is intentionally looser than the canonical QC
            per-sample budget (0.0, ``CleanupConfig.max_nan_fraction``); the
            orchestrator ``apply_data_cleanup_filters`` always passes an explicit
            value down, so this default only affects direct callers of this helper.
        barcode_col: Name of the barcode/plant ID column (default: "Barcode")
        genotype_col: Name of the genotype column (default: "geno")
        replicate_col: Name of the replicate column if present (default: "rep")
        save_removed_path: Optional path to save removed samples CSV. If provided,
                          removed samples will be saved with NaN information.

    Returns:
        Tuple of:
        - DataFrame with NaN samples removed
        - DataFrame of removed samples with NaN info
        - Dictionary with removal statistics (includes 'saved_path' if saved)
    """
    removal_stats = {
        "original_samples": len(df),
        "samples_with_any_nan": 0,
        "samples_removed": 0,
        "removal_details": [],
    }

    # Warn once if metadata columns are missing before iterating
    if barcode_col not in df.columns:
        logger.warning(
            "Column '%s' not found in DataFrame; barcode will be empty string "
            "in removal details. Available columns: %s",
            barcode_col,
            df.columns.tolist(),
        )
    if genotype_col not in df.columns:
        logger.warning(
            "Column '%s' not found in DataFrame; genotype will be empty string "
            "in removal details. Available columns: %s",
            genotype_col,
            df.columns.tolist(),
        )
    if replicate_col and replicate_col not in df.columns:
        logger.warning(
            "Column '%s' not found in DataFrame; rep will be empty string "
            "in removal details. Available columns: %s",
            replicate_col,
            df.columns.tolist(),
        )

    # Identify samples with NaN in trait columns
    samples_to_remove = []
    removed_details = []

    for idx in df.index:
        nan_traits = [col for col in trait_cols if pd.isna(df.loc[idx, col])]
        nan_count = len(nan_traits)
        nan_fraction = nan_count / len(trait_cols) if trait_cols else 0

        if nan_count > 0:
            removal_stats["samples_with_any_nan"] += 1

            # Remove if exceeds threshold
            if nan_fraction > max_nan_fraction:
                samples_to_remove.append(idx)
                removed_details.append(
                    {
                        "sample_index": int(idx),
                        "barcode": (
                            df.loc[idx, barcode_col]
                            if barcode_col in df.columns
                            else ""
                        ),
                        "genotype": (
                            df.loc[idx, genotype_col]
                            if genotype_col in df.columns
                            else ""
                        ),
                        "rep": (
                            df.loc[idx, replicate_col]
                            if replicate_col and replicate_col in df.columns
                            else ""
                        ),
                        "nan_count": nan_count,
                        "nan_fraction": float(nan_fraction),
                        "nan_traits": "; ".join(nan_traits),
                        "removal_reason": f"Contains {nan_count} NaN values",
                    }
                )

    # Create cleaned dataframe
    df_cleaned = df.drop(index=samples_to_remove).copy()

    # Create removed samples dataframe
    if removed_details:
        df_removed = pd.DataFrame(removed_details)
        # Add the actual data for the removed samples
        for col in df.columns:
            if col not in df_removed.columns:
                df_removed[col] = df.loc[samples_to_remove, col].values
    else:
        # Create empty dataframe with expected columns
        base_columns = [
            "sample_index",
            "barcode",
            "genotype",
            "nan_count",
            "nan_fraction",
            "nan_traits",
            "removal_reason",
        ]
        if replicate_col:
            # Insert rep column after genotype
            base_columns.insert(3, "rep")
        df_removed = pd.DataFrame(columns=base_columns)

    # Update stats
    removal_stats["samples_removed"] = len(samples_to_remove)
    removal_stats["samples_retained"] = len(df_cleaned)
    removal_stats["removal_details"] = removed_details

    # Save removed samples if path provided
    if save_removed_path and not df_removed.empty:
        save_path = Path(save_removed_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_removed.to_csv(save_path, index=False)
        removal_stats["saved_path"] = str(save_path)
    elif save_removed_path:
        # Create empty file if no rows were removed
        save_path = Path(save_removed_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=df_removed.columns).to_csv(save_path, index=False)
        removal_stats["saved_path"] = str(save_path)

    return df_cleaned, df_removed, removal_stats


def get_numeric_traits_only(
    df: pd.DataFrame,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    additional_exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract only numeric trait columns for analysis.

    Args:
        df: Full trait dataframe
        barcode_col: Name of the barcode/plant ID column to exclude (default: "Barcode")
        genotype_col: Name of the genotype column to exclude (default: "geno")
        replicate_col: Name of the replicate column to exclude if present (default: "rep")
        additional_exclude: Additional columns to exclude (e.g., date columns)

    Returns:
        DataFrame with only numeric trait columns
    """
    trait_cols = get_trait_columns(
        df,
        barcode_col=barcode_col,
        genotype_col=genotype_col,
        replicate_col=replicate_col,
        additional_exclude=additional_exclude,
    )
    return df[trait_cols].copy()


def remove_low_heritability_traits(
    df: pd.DataFrame,
    heritability_results: Dict[str, Dict],
    heritability_threshold: float = 0.3,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    additional_exclude: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Remove traits with heritability below threshold.

    Args:
        df: Trait dataframe
        heritability_results: Dictionary with heritability results for each trait
        heritability_threshold: Minimum H² value to retain trait (default: 0.3)
        barcode_col: Name of the barcode/plant ID column (default: "Barcode")
        genotype_col: Name of the genotype column (default: "geno")
        replicate_col: Name of the replicate column if present (default: "rep")
        additional_exclude: Additional columns to exclude from trait columns

    Returns:
        Tuple of:
            - DataFrame with low heritability traits removed
            - List of removed trait names
            - Dictionary with removal details
    """
    # Get current trait columns - MUST pass column parameters!
    trait_cols = get_trait_columns(
        df,
        barcode_col=barcode_col,
        genotype_col=genotype_col,
        replicate_col=replicate_col,
        additional_exclude=additional_exclude,
    )
    metadata_cols = [col for col in df.columns if col not in trait_cols]

    # Identify traits to remove
    traits_to_remove = []
    removal_details = []

    for trait in trait_cols:
        if trait in heritability_results:
            result = heritability_results[trait]
            if isinstance(result, dict) and "heritability" in result:
                h2 = result["heritability"]
                if h2 < heritability_threshold:
                    traits_to_remove.append(trait)
                    removal_details.append(
                        {
                            "trait": trait,
                            "heritability": h2,
                            "reason": f"H² = {h2:.3f} < threshold ({heritability_threshold})",
                            "genetic_variance": result.get(
                                "var_genetic", result.get("genetic_variance", np.nan)
                            ),
                            "environmental_variance": result.get(
                                "var_residual",
                                result.get("environmental_variance", np.nan),
                            ),
                        }
                    )
            else:
                # Invalid heritability result - consider removing
                traits_to_remove.append(trait)
                removal_details.append(
                    {
                        "trait": trait,
                        "heritability": np.nan,
                        "reason": "Invalid heritability estimate",
                        "genetic_variance": (
                            result.get(
                                "var_genetic", result.get("genetic_variance", np.nan)
                            )
                            if isinstance(result, dict)
                            else np.nan
                        ),
                        "environmental_variance": (
                            result.get(
                                "var_residual",
                                result.get("environmental_variance", np.nan),
                            )
                            if isinstance(result, dict)
                            else np.nan
                        ),
                    }
                )
        else:
            # No heritability result - consider removing
            traits_to_remove.append(trait)
            removal_details.append(
                {
                    "trait": trait,
                    "heritability": np.nan,
                    "reason": "No heritability estimate available",
                    "genetic_variance": np.nan,
                    "environmental_variance": np.nan,
                }
            )

    # Keep only high heritability traits
    traits_to_keep = [t for t in trait_cols if t not in traits_to_remove]
    df_cleaned = df[metadata_cols + traits_to_keep].copy()

    # Create summary
    summary = {
        "threshold": heritability_threshold,
        "original_traits": len(trait_cols),
        "removed_traits": len(traits_to_remove),
        "retained_traits": len(traits_to_keep),
        "removal_details": removal_details,
    }

    return df_cleaned, traits_to_remove, summary


def remove_zero_inflated_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_zero_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Remove traits with excessive zero values.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names to check
        max_zero_fraction: Maximum allowed fraction of zeros (0-1)

    Returns:
        Tuple of (filtered_dataframe, remaining_trait_cols, removal_details)
    """
    removed_traits = []
    removal_details = {}
    df_filtered = df.copy()

    for trait in trait_cols:
        if trait in df.columns:
            zero_fraction = (df[trait] == 0).sum() / len(df)
            if zero_fraction > max_zero_fraction:
                removed_traits.append(trait)
                removal_details[trait] = {
                    "reason": "too_many_zeros",
                    "zero_fraction": float(zero_fraction),
                    "threshold": max_zero_fraction,
                }

    if removed_traits:
        df_filtered = df_filtered.drop(columns=removed_traits)

    remaining_traits = [t for t in trait_cols if t not in removed_traits]

    return df_filtered, remaining_traits, removal_details


def remove_traits_with_many_nans(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_nan_fraction: float = 0.3,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Remove traits with excessive NaN values.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names to check
        max_nan_fraction: Maximum allowed fraction of NaNs (0-1)

    Returns:
        Tuple of (filtered_dataframe, remaining_trait_cols, removal_details)
    """
    removed_traits = []
    removal_details = {}
    df_filtered = df.copy()

    for trait in trait_cols:
        if trait in df.columns:
            nan_fraction = df[trait].isna().sum() / len(df)
            if nan_fraction > max_nan_fraction:
                removed_traits.append(trait)
                removal_details[trait] = {
                    "reason": "too_many_nans",
                    "nan_fraction": float(nan_fraction),
                    "threshold": max_nan_fraction,
                }

    if removed_traits:
        df_filtered = df_filtered.drop(columns=removed_traits)

    remaining_traits = [t for t in trait_cols if t not in removed_traits]

    return df_filtered, remaining_traits, removal_details


def remove_low_sample_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
    min_samples: int = 10,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Remove traits with insufficient valid samples.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names to check
        min_samples: Minimum number of valid (non-NaN) samples required

    Returns:
        Tuple of (filtered_dataframe, remaining_trait_cols, removal_details)
    """
    removed_traits = []
    removal_details = {}
    df_filtered = df.copy()

    for trait in trait_cols:
        if trait in df.columns:
            valid_samples = df[trait].notna().sum()
            if valid_samples < min_samples:
                removed_traits.append(trait)
                removal_details[trait] = {
                    "reason": "insufficient_samples",
                    "valid_samples": int(valid_samples),
                    "required_samples": min_samples,
                }

    if removed_traits:
        df_filtered = df_filtered.drop(columns=removed_traits)

    remaining_traits = [t for t in trait_cols if t not in removed_traits]

    return df_filtered, remaining_traits, removal_details


def remove_zero_variance_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
    min_variance: float = 0.0,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Remove constant (zero-variance) traits.

    Drops traits whose population variance ``var(ddof=0) <= min_variance``. Using
    ``ddof=0`` matches :func:`sleap_roots_analyze.pca.standardize_data`'s zero-variance
    test, so a trait that survives cleanup will not be silently dropped later by PCA.
    With ``min_variance=0.0`` (the default) this removes exactly-constant traits; set
    ``min_variance`` to a negative value to disable the filter (variance is always
    ``>= 0``).

    Mirrors the contract of the sibling trait filters
    (:func:`remove_zero_inflated_traits`, :func:`remove_traits_with_many_nans`,
    :func:`remove_low_sample_traits`): it returns ``(filtered_df, remaining_trait_cols,
    removal_details)`` and is intended to run as the **final** cleanup step, after
    sample removal, since a trait can become constant only once NaN-carrying rows are
    dropped.

    Edge cases: an empty or all-NaN frame yields ``var == NaN`` and ``NaN <= x`` is
    ``False``, so nothing is flagged (degenerate frames are handled by validation, not
    here); a single-row frame yields ``var == 0`` and is correctly flagged constant. A
    trait not present in ``df`` is skipped (never flagged) but stays in
    ``remaining_trait_cols``, matching the sibling filters. Unlike the sibling filters
    (which use dtype-agnostic ``== 0`` / ``.isna()`` tests), variance is numeric-only, so
    a **non-numeric** trait column is skipped rather than raising — the caller is
    responsible for passing numeric ``trait_cols`` (``get_trait_columns`` and
    ``clean_traits_for_analysis`` already guarantee this).

    Note:
        ``min_variance`` compares **raw, pre-standardization** variance (in squared trait
        units), so it is scale-dependent: a single non-zero threshold mixes, e.g., mm²
        with dimensionless ratios. Keep the ``0.0`` default (exactly "is constant") unless
        you have a per-analysis reason to raise it. A ``NaN`` threshold silently disables
        the filter (``var <= NaN`` is always ``False``) and ``+inf`` drops every trait.

    Args:
        df: DataFrame with trait data.
        trait_cols: List of trait column names to check. Non-numeric columns are skipped.
        min_variance: Traits with ``var(ddof=0) <= min_variance`` are removed. ``0.0``
            drops exactly-constant traits; a negative value disables the filter.

    Returns:
        Tuple of ``(filtered_dataframe, remaining_trait_cols, removal_details)`` where
        ``removal_details`` maps each removed trait to
        ``{"reason": "zero_variance", "variance": <float>, "threshold": <min_variance>}``.
    """
    removed_traits = []
    removal_details = {}
    df_filtered = df.copy()

    for trait in trait_cols:
        # Skip absent columns (kept in remaining, matching the siblings) and non-numeric
        # columns (variance is undefined for them; skip rather than raise a TypeError so a
        # hand-built trait_cols with an object column does not crash public callers).
        if trait not in df.columns or not pd.api.types.is_numeric_dtype(df[trait]):
            continue
        variance = df[trait].var(ddof=0)
        # NaN (empty/all-NaN column) is not <= any real threshold, so it is never flagged.
        if variance <= min_variance:
            removed_traits.append(trait)
            removal_details[trait] = {
                "reason": "zero_variance",
                "variance": float(variance),
                "threshold": min_variance,
            }

    if removed_traits:
        df_filtered = df_filtered.drop(columns=removed_traits)

    remaining_traits = [t for t in trait_cols if t not in removed_traits]

    return df_filtered, remaining_traits, removal_details


def apply_data_cleanup_filters(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_zeros_per_trait: float = 0.5,
    max_nans_per_trait: float = 0.2,
    max_nans_per_sample: float = 0.0,
    min_samples_per_trait: int = 10,
    min_variance: float = 0.0,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
) -> Tuple[pd.DataFrame, Dict]:
    """Apply configurable data cleanup filters to minimize sample loss.

    This function orchestrates multiple cleanup steps using modular functions:
    1. Remove zero-inflated traits
    2. Remove traits with many NaNs
    3. Remove samples with many NaNs
    4. Remove traits with insufficient samples
    5. Remove zero-variance (constant) traits

    Step 5 runs **last** (after sample removal) because a trait can become constant only
    once NaN-carrying rows are dropped; its variance must therefore be measured on the
    reduced frame.

    The signature defaults are the **QC pipeline's canonical** cleanup thresholds:
    they equal ``CleanupConfig()``'s defaults (with ``max_nans_per_sample`` mapping
    to ``CleanupConfig.max_nan_fraction``), so every default-using caller cleans the
    same way the pipeline does. A drift guard
    (``tests/test_data_cleanup.py::TestCanonicalDefaultDriftGuard``) asserts this
    equality so the two cannot silently diverge (#167).

    Args:
        df: Original dataframe
        trait_cols: List of trait column names
        max_zeros_per_trait: Maximum fraction of zeros allowed per trait (0-1).
            Default ``0.5`` (canonical QC).
        max_nans_per_trait: Maximum fraction of NaNs allowed per trait (0-1).
            Default ``0.2`` (canonical QC; drops NaN-heavier traits sooner).
        max_nans_per_sample: Maximum fraction of NaNs allowed per sample (0-1).
            Default ``0.0`` (canonical QC; drops any sample that still has *any*
            NaN in a surviving trait).
        min_samples_per_trait: Minimum number of valid samples required per trait.
            Default ``10`` (canonical QC).
        min_variance: Traits with population variance ``var(ddof=0) <= min_variance`` are
            removed as the final step (after sample removal), using ``ddof=0`` to match
            ``standardize_data``. Default ``0.0`` drops exactly-constant traits; set
            negative to disable.
        barcode_col: Name of the barcode/plant ID column (default: "Barcode")
        genotype_col: Name of the genotype column (default: "geno")
        replicate_col: Name of the replicate column if present (default: "rep")

    Returns:
        Tuple of (cleaned_dataframe, cleanup_log) where cleanup_log is a dict with keys:
        - ``original_samples``: int — sample count before cleanup
        - ``original_traits``: int — trait count before cleanup
        - ``final_samples``: int — sample count after cleanup
        - ``final_traits``: int — trait count after cleanup
        - ``removed_traits``: list[dict] — one entry per removed trait
        - ``removed_samples``: list[dict] — independent deep copy of ``removed_samples_detail``
        - ``removed_samples_detail``: list[dict] — one entry per removed sample;
          each dict has keys ``sample_index``, ``barcode``, ``genotype``, ``rep``,
          ``nan_count``, ``nan_fraction``, ``nan_traits``, ``removal_reason``.
          Populated from ``remove_nan_samples()`` via its ``"removal_details"`` key.
        - ``cleanup_steps``: list[dict] — one entry per cleanup step with step name
          and counts
    """
    # Annotate as Dict[str, Any]: the values mix int, list, and float, so without this
    # mypy infers dict[str, object] and every ``.append`` on a value errors (#177 review).
    cleanup_log: Dict[str, Any] = {
        "original_samples": len(df),
        "original_traits": len(trait_cols),
        "removed_traits": [],
        "removed_samples": [],
        "cleanup_steps": [],
        "removed_samples_detail": [],  # Detailed info about removed samples
    }

    df_clean = df.copy()
    valid_traits = trait_cols.copy()

    # Step 1: Remove traits with too many zeros
    df_clean, valid_traits, zero_removal_details = remove_zero_inflated_traits(
        df_clean, valid_traits, max_zero_fraction=max_zeros_per_trait
    )

    # Log removed traits
    for trait, details in zero_removal_details.items():
        cleanup_log["removed_traits"].append({"trait": trait, **details})

    cleanup_log["cleanup_steps"].append(
        {
            "step": "remove_high_zero_traits",
            "traits_removed": len(zero_removal_details),
            "remaining_traits": len(valid_traits),
        }
    )

    # Step 2: Remove traits with too many NaNs
    df_clean, valid_traits, nan_removal_details = remove_traits_with_many_nans(
        df_clean, valid_traits, max_nan_fraction=max_nans_per_trait
    )

    # Log removed traits
    for trait, details in nan_removal_details.items():
        cleanup_log["removed_traits"].append({"trait": trait, **details})

    cleanup_log["cleanup_steps"].append(
        {
            "step": "remove_high_nan_traits",
            "traits_removed": len(nan_removal_details),
            "remaining_traits": len(valid_traits),
        }
    )

    # Step 3: Remove samples with too many NaNs (reuse existing function)
    if valid_traits:
        # Use existing remove_nan_samples function
        df_clean, df_removed, removal_stats = remove_nan_samples(
            df_clean,
            valid_traits,
            max_nan_fraction=max_nans_per_sample,
            barcode_col=barcode_col,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
            save_removed_path=None,  # Don't save to file here
        )

        # Update cleanup log with sample removal details
        cleanup_log["removed_samples_detail"] = removal_stats.get("removal_details", [])
        cleanup_log["removed_samples"] = [
            dict(e) for e in cleanup_log["removed_samples_detail"]
        ]

        cleanup_log["cleanup_steps"].append(
            {
                "step": "remove_high_nan_samples",
                "samples_removed": removal_stats["samples_removed"],
                "remaining_samples": removal_stats["samples_retained"],
            }
        )

    # Step 4: Remove traits with insufficient samples after cleanup
    df_clean, valid_traits, low_sample_removal_details = remove_low_sample_traits(
        df_clean, valid_traits, min_samples=min_samples_per_trait
    )

    # Log removed traits
    for trait, details in low_sample_removal_details.items():
        cleanup_log["removed_traits"].append({"trait": trait, **details})

    cleanup_log["cleanup_steps"].append(
        {
            "step": "remove_low_sample_traits",
            "traits_removed": len(low_sample_removal_details),
            "remaining_traits": len(valid_traits),
        }
    )

    # Step 5: Remove zero-variance (constant) traits. Runs last so variance is measured
    # on the post-sample-removal frame — a trait can go constant only after NaN rows are
    # dropped. Does not raise when this empties the trait set; the entry point /
    # validation handle emptiness.
    df_clean, valid_traits, zero_variance_removal_details = remove_zero_variance_traits(
        df_clean, valid_traits, min_variance=min_variance
    )

    # Log removed traits
    for trait, details in zero_variance_removal_details.items():
        cleanup_log["removed_traits"].append({"trait": trait, **details})

    cleanup_log["cleanup_steps"].append(
        {
            "step": "remove_zero_variance_traits",
            "traits_removed": len(zero_variance_removal_details),
            "remaining_traits": len(valid_traits),
        }
    )

    # Final summary
    cleanup_log["final_samples"] = len(df_clean)
    cleanup_log["final_traits"] = len(valid_traits)
    cleanup_log["samples_retained_fraction"] = (
        len(df_clean) / cleanup_log["original_samples"]
        if cleanup_log["original_samples"] > 0
        else 0
    )
    cleanup_log["traits_retained_fraction"] = (
        len(valid_traits) / cleanup_log["original_traits"]
        if cleanup_log["original_traits"] > 0
        else 0
    )

    return df_clean, cleanup_log


def build_clean_validation_report(
    df: pd.DataFrame,
    trait_cols: List[str],
) -> Dict[str, Any]:
    """Build the no-NaN validation report for a cleaned trait table.

    This is the single source of truth for the validation report that QC step 03
    (``ValidateCleanStep``) emits and that :func:`clean_traits_for_analysis`
    consumes. It is pure (no I/O, no raising) so callers can save the report
    before deciding whether to raise.

    Args:
        df: Cleaned trait dataframe to inspect.
        trait_cols: Trait column names to check for NaNs. Remaining columns are
            treated as metadata and reported separately.

    Returns:
        Dict with keys: ``validation_passed`` (bool), ``total_samples`` (int),
        ``total_trait_columns`` (int), ``total_metadata_columns`` (int),
        ``nan_values_in_traits`` (int), ``nan_values_in_metadata`` (int),
        ``trait_nan_counts`` (dict of trait -> count, only > 0), and
        ``metadata_nan_counts`` (dict of column -> count, only > 0).
    """
    nan_counts = df[trait_cols].isna().sum()
    total_nans = nan_counts.sum()

    metadata_cols = [col for col in df.columns if col not in trait_cols]
    metadata_nan_counts = df[metadata_cols].isna().sum()
    total_metadata_nans = metadata_nan_counts.sum()

    return {
        "validation_passed": total_nans == 0,
        "total_samples": len(df),
        "total_trait_columns": len(trait_cols),
        "total_metadata_columns": len(metadata_cols),
        "nan_values_in_traits": int(total_nans),
        "nan_values_in_metadata": int(total_metadata_nans),
        "trait_nan_counts": {
            trait: int(count) for trait, count in nan_counts.items() if count > 0
        },
        "metadata_nan_counts": {
            col: int(count) for col, count in metadata_nan_counts.items() if count > 0
        },
    }


def _format_nan_validation_error(report: Dict[str, Any]) -> str:
    """Format the canonical no-NaN validation error message.

    Single source of truth for the message raised by both ``ValidateCleanStep``
    and :func:`validate_clean_traits` so they cannot drift.

    Args:
        report: Report dict from :func:`build_clean_validation_report`.

    Returns:
        The error message string.
    """
    return (
        f"Validation failed: {report['nan_values_in_traits']} NaN values found "
        f"in trait columns!\n"
        f"Affected traits: {list(report['trait_nan_counts'].keys())}"
    )


def validate_clean_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
) -> Dict[str, Any]:
    """Validate that a cleaned trait table has no NaNs in its trait columns.

    Builds the validation report and raises if any NaN remains in ``trait_cols``.
    This is the importable form of QC step 03's check, shared by
    ``ValidateCleanStep`` and :func:`clean_traits_for_analysis`.

    Args:
        df: Cleaned trait dataframe to validate.
        trait_cols: Trait column names that must be NaN-free.

    Returns:
        The validation report dict from :func:`build_clean_validation_report`.

    Raises:
        ValueError: If any NaN values remain in ``trait_cols``. The message names
            the affected traits.
    """
    report = build_clean_validation_report(df, trait_cols)
    if not report["validation_passed"]:
        raise ValueError(_format_nan_validation_error(report))
    return report


def clean_traits_for_analysis(
    df: pd.DataFrame,
    trait_cols: Optional[List[str]] = None,
    *,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    **cleanup_kwargs: Any,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Turn a raw wide trait table into a clean, analysis-ready table.

    Minimal-QC public entry point: composes the canonical smart cleanup
    (:func:`apply_data_cleanup_filters`, QC step 02) with the no-NaN validation
    (:func:`validate_clean_traits`, QC step 03), then adds analysis-readiness
    gates so the result is safe to hand to ``perform_pca_analysis`` / UMAP /
    clustering without silently dropping rows or raising on zero variance.

    Cleanup order (matching the QC pipeline's stricter cleaning): drop bad
    **traits** first (zero-inflated / too-many-NaN / low-sample), then drop any
    remaining **rows** that still carry NaN in the surviving traits. Dropping bad
    traits first still loses fewer samples than a naive ``df.dropna()``, but with
    the QC-canonical ``max_nans_per_sample=0.0`` default, any sample carrying a NaN
    in a surviving trait is dropped. The returned frame is guaranteed NaN-free in
    its trait columns.

    Validation runs in a fixed order, each raising a distinct, actionable error:
    (1) empty input, (2) no NaN in surviving traits, (3) at least
    ``MIN_SAMPLES_FOR_ANALYSIS`` (2) surviving samples, (4) at least one
    non-constant numeric trait (``var(ddof=0) > 0``, matching ``standardize_data``).

    Notes:
        - Default thresholds are the **QC pipeline's canonical** values
          (``max_zeros_per_trait=0.5``, ``max_nans_per_trait=0.2``,
          ``max_nans_per_sample=0.0``, ``min_samples_per_trait=10``,
          ``min_variance=0.0``), so the analysis-ready frame matches what the QC
          pipeline produces rather than a looser clean. The ``min_variance=0.0`` filter
          drops any constant trait so the returned frame is constant-free (re-checked
          after the residual-NaN row drop below). These are inherited directly from
          ``apply_data_cleanup_filters``'s signature defaults, which were aligned to
          the canonical QC values in #167 (so there is no separate copy to drift).
          Caller kwargs override; the effective thresholds are recorded in
          ``cleanup_log["effective_thresholds"]`` and logged at INFO.
        - This entry point does **NOT** apply trait-name sanitization
          (``sanitize_trait_names``) or column reordering that
          ``CleanupTraitsStep`` performs, so its output is **not** byte-equivalent
          to the pipeline's ``02_data_samples_cleaned.csv`` (trait names, identities,
          and order can differ) — byte-equivalence with the pipeline is not a goal.
        - ``MIN_SAMPLES_FOR_ANALYSIS`` is the runnability floor only; a
          ``UserWarning`` is emitted in the p > n regime (more traits than samples),
          where multivariate analysis is statistically unreliable.

    Args:
        df: Raw wide trait dataframe.
        trait_cols: Trait column names. If ``None``, resolved via
            :func:`get_trait_columns`.
        barcode_col: Barcode/plant ID column to exclude when inferring traits.
        genotype_col: Genotype column to exclude when inferring traits.
        replicate_col: Replicate column to exclude if present (``None`` if absent).
        **cleanup_kwargs: Forwarded to :func:`apply_data_cleanup_filters`
            (``max_zeros_per_trait``, ``max_nans_per_trait``,
            ``max_nans_per_sample``, ``min_samples_per_trait``, ``min_variance``).

    Returns:
        Tuple of ``(clean_df, trait_cols, cleanup_log)`` where ``trait_cols`` is
        the surviving trait columns and ``cleanup_log`` is the
        :func:`apply_data_cleanup_filters` log enriched with
        ``effective_thresholds`` and ``validation_summary``.

    Raises:
        ValueError: On empty input; duplicate column names; explicit ``trait_cols``
            that are missing from ``df`` or non-numeric; fewer than
            ``MIN_SAMPLES_FOR_ANALYSIS`` surviving samples; or no non-constant
            numeric trait remaining. (Residual NaN is dropped, not raised; the
            shared validation remains as a defensive guard.)
    """
    # Check (1): empty input — raise our own message before delegating.
    if df is None or df.shape[0] == 0:
        raise ValueError(
            "clean_traits_for_analysis: input table has no rows; nothing to clean."
        )

    # Reject duplicate column names up front: they make trait selection ambiguous
    # and otherwise degrade into a misleading "no trait columns found" error.
    if df.columns.duplicated().any():
        dups = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(
            f"clean_traits_for_analysis: duplicate column names in input: {dups}. "
            "Deduplicate columns before cleaning."
        )

    if trait_cols is None:
        trait_cols = get_trait_columns(
            df,
            barcode_col=barcode_col,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
        )
    else:
        # Validate explicit trait_cols up front so misuse yields an actionable
        # error instead of a bare pandas KeyError or a later PCA failure.
        missing = [c for c in trait_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"clean_traits_for_analysis: trait_cols not found in dataframe: "
                f"{missing}."
            )
        non_numeric = [
            c for c in trait_cols if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise ValueError(
                "clean_traits_for_analysis: trait_cols must be numeric; non-numeric "
                f"columns passed: {non_numeric}."
            )
    if not trait_cols:
        raise ValueError(
            "clean_traits_for_analysis: no trait columns found to clean. Pass "
            "trait_cols explicitly or check the metadata column names."
        )

    # Resolve effective thresholds. The entry point inherits its defaults directly
    # from apply_data_cleanup_filters' signature, which now *is* the QC pipeline's
    # canonical cleaning (its defaults equal CleanupConfig()'s; aligned in #167). So
    # the analysis-ready frame it hands to PCA/UMAP equals what the QC pipeline would
    # produce, not a looser clean — with no separate hardcoded copy to drift. Caller-
    # passed kwargs still override.
    sig = inspect.signature(apply_data_cleanup_filters)
    threshold_names = (
        "max_zeros_per_trait",
        "max_nans_per_trait",
        "max_nans_per_sample",
        "min_samples_per_trait",
        "min_variance",
    )
    # Guard against a renamed signature parameter: surface the drift here with a
    # clear message instead of as an opaque KeyError in the comprehension below.
    missing_params = set(threshold_names) - set(sig.parameters)
    assert not missing_params, (
        "threshold_names out of sync with apply_data_cleanup_filters signature: "
        f"{sorted(missing_params)}"
    )
    thresholds = {
        name: cleanup_kwargs.pop(name, sig.parameters[name].default)
        for name in threshold_names
    }

    clean_df, cleanup_log = apply_data_cleanup_filters(
        df,
        trait_cols,
        barcode_col=barcode_col,
        genotype_col=genotype_col,
        replicate_col=replicate_col,
        **thresholds,
        **cleanup_kwargs,
    )

    # Surviving traits: removed traits are dropped from the frame by the cleanup
    # helpers, so column membership is the robust derivation.
    surviving = [col for col in trait_cols if col in clean_df.columns]

    # Drop any rows that still carry NaN in the surviving traits.
    # apply_data_cleanup_filters removes NaN-heavy *traits* and NaN-heavy *samples*
    # (per the thresholds) but can leave residual NaNs below those thresholds.
    # Dropping them here delivers the promised clean frame ("drop bad traits, then
    # the remaining NaN rows") while keeping sample loss minimal — the bad traits
    # were already removed first, so far fewer rows are lost than a naive dropna().
    n_rows_before_dropna = len(clean_df)
    if surviving:
        clean_df = clean_df.dropna(subset=surviving)

    # Re-run the zero-variance filter ONLY when the dropna above actually removed rows.
    # On the canonical max_nans_per_sample=0.0 path that dropna is provably a no-op (Step 3
    # already dropped every row with any NaN in a superset of the surviving traits), so the
    # surviving traits' variances are unchanged from the orchestrator's Step 5 and the
    # re-check would be a pure duplicate full-frame scan. Only a loosened
    # max_nans_per_sample lets the dropna remove rows, which can turn a surviving trait
    # constant *after* the orchestrator's variance step; re-check then keeps the
    # analysis-ready frame constant-free and the log complete.
    if surviving and len(clean_df) != n_rows_before_dropna:
        clean_df, surviving, zero_variance_recheck = remove_zero_variance_traits(
            clean_df, surviving, min_variance=thresholds["min_variance"]
        )
        for trait, details in zero_variance_recheck.items():
            cleanup_log["removed_traits"].append({"trait": trait, **details})
        if zero_variance_recheck:
            cleanup_log["cleanup_steps"].append(
                {
                    "step": "remove_zero_variance_traits",
                    "traits_removed": len(zero_variance_recheck),
                    "remaining_traits": len(surviving),
                }
            )
            # Keep the orchestrator-set summary fields consistent with the extra drop so
            # programmatic consumers (e.g. bloom-mcp) never read a self-contradictory log.
            cleanup_log["final_traits"] = len(surviving)
            cleanup_log["traits_retained_fraction"] = (
                len(surviving) / cleanup_log["original_traits"]
                if cleanup_log["original_traits"] > 0
                else 0
            )

    # Check (2): no NaN in surviving traits. After the row drop above this holds by
    # construction; keep the shared step-03 assertion as a defensive guard against
    # silent corruption.
    validate_clean_traits(clean_df, surviving)

    # Check (3): at least MIN_SAMPLES_FOR_ANALYSIS surviving samples.
    n_samples = len(clean_df)
    if n_samples < MIN_SAMPLES_FOR_ANALYSIS:
        raise ValueError(
            f"clean_traits_for_analysis: only {n_samples} sample(s) remain after "
            f"cleanup; need at least {MIN_SAMPLES_FOR_ANALYSIS} for analysis. "
            "Meaningful multivariate analysis needs many more samples than traits."
        )

    # Check (4): at least one non-constant numeric trait (var(ddof=0) > 0),
    # matching standardize_data's zero-variance drop test.
    numeric = clean_df[surviving].select_dtypes(include=[np.number])
    n_nonconstant = int((numeric.var(ddof=0) > 0).sum()) if not numeric.empty else 0
    if n_nonconstant < 1:
        raise ValueError(
            "clean_traits_for_analysis: no non-constant numeric trait remains "
            "after cleanup; PCA/UMAP/clustering require at least one varying trait."
        )

    # Warn in the p > n regime: PCA/UMAP/clustering on more traits than samples is
    # statistically fragile even though it runs.
    if n_samples < len(surviving):
        warnings.warn(
            f"clean_traits_for_analysis: {n_samples} samples < {len(surviving)} "
            "traits (p > n); multivariate analysis results will be statistically "
            "unreliable. Consider more samples or fewer traits.",
            stacklevel=2,
        )

    cleanup_log["effective_thresholds"] = thresholds
    cleanup_log["validation_summary"] = {
        "n_samples": n_samples,
        "n_surviving_traits": len(surviving),
        "n_nonconstant_traits": n_nonconstant,
    }

    # Surface the effective thresholds for programmatic consumers (e.g. bloom-mcp),
    # which never see the docstring's note about the QC-canonical defaults and that
    # name-sanitization is not applied.
    logger.info(
        "clean_traits_for_analysis: kept %d samples x %d traits using thresholds "
        "%s (QC-canonical defaults unless overridden; trait-name sanitization NOT "
        "applied, so output is not byte-equivalent to the pipeline).",
        n_samples,
        len(surviving),
        thresholds,
    )

    return clean_df, surviving, cleanup_log


def inspect_nan_samples(
    df: pd.DataFrame,
    trait_cols: List[str],
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: str = "rep",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Inspect samples with NaN values and provide detailed information.

    This function identifies samples containing NaN values in trait columns
    and creates a detailed report including which traits have missing values
    and the fraction of missing data per sample. When verbose=True, it logs
    helpful summary statistics about the NaN distribution.

    Args:
        df: DataFrame containing trait data
        trait_cols: List of trait column names to inspect
        barcode_col: Name of the barcode/sample ID column (default: "Barcode")
        genotype_col: Name of the genotype column (default: "geno")
        replicate_col: Name of the replicate column (default: "rep")
        save_path: Optional path to save the inspection results as CSV
        verbose: Whether to log inspection summary (default: True)

    Returns:
        DataFrame with NaN inspection details including:
        - sample_index: Original DataFrame index
        - barcode: Sample barcode/ID
        - genotype: Genotype name
        - rep: Replicate number
        - nan_count: Number of traits with NaN
        - nan_fraction: Fraction of traits with NaN
        - nan_traits: Semicolon-separated list of traits with NaN
        - data_status: Status indicator ("original_data_with_nan")

    Examples:
        >>> df = pd.DataFrame({
        ...     'Barcode': ['BC1', 'BC2', 'BC3'],
        ...     'geno': ['G1', 'G2', 'G3'],
        ...     'rep': [1, 2, 3],
        ...     'trait1': [1.0, np.nan, 3.0],
        ...     'trait2': [4.0, 5.0, np.nan]
        ... })
        >>> trait_cols = ['trait1', 'trait2']
        >>> inspection_df = inspect_nan_samples(df, trait_cols)
        >>> len(inspection_df)  # 2 samples have NaN
        2
    """
    # Identify samples with NaN values in trait columns
    samples_with_nan = df[trait_cols].isna().any(axis=1)
    n_samples_with_nan = samples_with_nan.sum()

    if n_samples_with_nan > 0:
        # Get samples with NaN
        nan_samples = df[samples_with_nan].copy()
        nan_info_detailed = []

        # Create detailed information for each sample with NaN
        for idx in nan_samples.index:
            # Find which traits have NaN for this sample
            nan_traits_list = [col for col in trait_cols if pd.isna(df.loc[idx, col])]
            nan_count = len(nan_traits_list)
            nan_fraction = nan_count / len(trait_cols) if len(trait_cols) > 0 else 0

            # Build info dictionary
            info = {
                "sample_index": int(idx),
                "nan_count": nan_count,
                "nan_fraction": nan_fraction,
                "nan_traits": "; ".join(nan_traits_list),
                "data_status": "original_data_with_nan",
            }

            # Add sample metadata if columns exist
            if barcode_col in df.columns:
                info["barcode"] = nan_samples.loc[idx, barcode_col]
            if genotype_col in df.columns:
                info["genotype"] = nan_samples.loc[idx, genotype_col]
            if replicate_col in df.columns:
                info["rep"] = nan_samples.loc[idx, replicate_col]

            nan_info_detailed.append(info)

        # Create DataFrame from collected information
        inspection_df = pd.DataFrame(nan_info_detailed)

        # Reorder columns for better readability
        col_order = []
        if "sample_index" in inspection_df.columns:
            col_order.append("sample_index")
        if "barcode" in inspection_df.columns:
            col_order.append("barcode")
        if "genotype" in inspection_df.columns:
            col_order.append("genotype")
        if "rep" in inspection_df.columns:
            col_order.append("rep")
        col_order.extend(["nan_count", "nan_fraction", "nan_traits", "data_status"])

        inspection_df = inspection_df[col_order]

        # Log helpful information if verbose
        if verbose:
            logger.info(
                f"Initial NaN inspection - found {len(inspection_df)} samples with NaN values"
            )
            logger.info(
                "These samples will be handled based on your data cleaning strategy"
            )

            # Show sample distribution of NaN counts
            nan_count_distribution = (
                inspection_df["nan_count"].value_counts().sort_index()
            )
            if len(nan_count_distribution) > 0:
                logger.info("NaN distribution:")
                for count, n_samples in nan_count_distribution.items():
                    logger.info(f"  - {n_samples} sample(s) with {count} NaN trait(s)")

            # Log summary statistics
            logger.info(
                f"Summary: {len(inspection_df)} samples with NaN values out of {len(df)} total samples ({len(inspection_df) / len(df) * 100:.1f}%)"
            )

        # Save to CSV if path provided
        if save_path is not None:
            inspection_df.to_csv(save_path, index=False)
            if verbose:
                logger.info(f"Saved NaN inspection to: {save_path}")

        return inspection_df
    else:
        # No samples with NaN
        if verbose:
            logger.info("No NaN values found in the data")

        # Return empty DataFrame with expected structure
        return pd.DataFrame(
            columns=[
                "sample_index",
                "barcode",
                "genotype",
                "rep",
                "nan_count",
                "nan_fraction",
                "nan_traits",
                "data_status",
            ]
        )
