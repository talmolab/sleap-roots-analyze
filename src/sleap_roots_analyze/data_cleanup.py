"""Data loading utilities for wheat trait analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np

import json

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


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

    # Also exclude common metadata columns that might exist with different names
    # These are case-insensitive matches
    common_metadata = [
        "date",
        "time",
        "sterilization",
        "experiment",
        "batch",
        "operator",
        "notes",
        "comments",
        "id",
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
        "plant_id",  # Plant IDs
    ]
    for col in df.columns:
        col_lower = col.lower()
        if any(meta in col_lower for meta in common_metadata):
            exclude_cols.append(col)

    # Remove duplicates and filter to columns that actually exist
    exclude_cols = list(set(col for col in exclude_cols if col in df.columns))

    # Get all columns not in exclude list
    trait_cols = [col for col in df.columns if col not in exclude_cols]

    # Only return numeric columns
    numeric_cols = []
    for col in trait_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def link_images_to_samples(
    df: pd.DataFrame,
    image_dir: Path | str,
    image_types: Optional[List[str]] = None,
    barcode_col: str = "Barcode",
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Link sample barcodes to their corresponding image files.

    Args:
        df: Trait dataframe with barcode/ID column
        image_dir: Directory containing processed images
        image_types: List of image suffixes to look for (default: ['features.png', 'seg.png'])
        barcode_col: Name of the barcode/plant ID column (default: "Barcode")

    Returns:
        Dictionary mapping barcode to image paths
    """
    if image_types is None:
        image_types = ["features.png", "seg.png"]

    image_dir = Path(image_dir)
    image_links = {}

    # Check if barcode column exists
    if barcode_col not in df.columns:
        raise ValueError(
            f"Barcode column '{barcode_col}' not found in dataframe. Available columns: {df.columns.tolist()[:10]}..."
        )

    for barcode in df[barcode_col]:
        image_links[barcode] = {}

        for img_type in image_types:
            # Images follow pattern: {barcode}_c1_p1_{type}
            img_filename = f"{barcode}_c1_p1_{img_type}"
            img_path = image_dir / img_filename

            if img_path.exists():
                image_links[barcode][img_type] = img_path
            else:
                image_links[barcode][img_type] = None

    return image_links


def create_run_directory(base_dir: Path | str = "turface_analysis/runs") -> Path:
    """Create timestamped run directory for outputs.

    Args:
        base_dir: Base directory for runs

    Returns:
        Path to created run directory
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


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
        "outlier_detection": _convert_to_json_serializable(outliers),
        "cleaned_data_path": cleaned_path.as_posix(),
    }

    if log_info:
        log_data.update(_convert_to_json_serializable(log_info))

    # Save log
    log_path = run_dir / "cleanup_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return cleaned_path, log_path


def _convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types recursively."""
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return obj


def remove_nan_samples(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_nan_fraction: float = 0.2,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Remove samples with NaN values in trait columns.

    This is the centralized function for NaN removal that should be called
    before any outlier detection or other analysis.

    Args:
        df: Original dataframe
        trait_cols: List of trait columns to check for NaN
        max_nan_fraction: Maximum fraction of NaN allowed per sample (0-1)
        barcode_col: Name of the barcode/plant ID column (default: "Barcode")
        genotype_col: Name of the genotype column (default: "geno")
        replicate_col: Name of the replicate column if present (default: "rep")

    Returns:
        Tuple of:
        - DataFrame with NaN samples removed
        - DataFrame of removed samples with NaN info
        - Dictionary with removal statistics
    """
    removal_stats = {
        "original_samples": len(df),
        "samples_with_any_nan": 0,
        "samples_removed": 0,
        "removal_details": [],
    }

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

    return df_cleaned, df_removed, removal_stats


def save_nan_removed_rows(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    run_dir: Path | str,
    trait_cols: List[str],
) -> Path:
    """Save rows that were removed due to NaN values.

    Args:
        df_original: Original dataframe before NaN removal
        df_cleaned: Dataframe after NaN removal
        run_dir: Run directory path
        trait_cols: List of trait columns that were checked for NaNs

    Returns:
        Path to saved NaN-removed rows CSV
    """
    run_dir = Path(run_dir)

    # Find rows that were removed (those in original but not in cleaned)
    original_indices = set(df_original.index)
    cleaned_indices = set(df_cleaned.index)
    removed_indices = original_indices - cleaned_indices

    if removed_indices:
        # Get the removed rows
        removed_rows = df_original.loc[list(removed_indices)].copy()

        # Add information about which traits had NaNs
        nan_info = []
        for idx in removed_indices:
            nan_traits = [
                col for col in trait_cols if pd.isna(df_original.loc[idx, col])
            ]
            nan_info.append("; ".join(nan_traits) if nan_traits else "Unknown")

        removed_rows["nan_traits"] = nan_info
        removed_rows["removal_reason"] = "Contains NaN values in trait columns"

        # Save to CSV
        nan_removed_path = run_dir / "nan_removed_rows.csv"
        removed_rows.to_csv(nan_removed_path, index=False)

        return nan_removed_path
    else:
        # Create empty file if no rows were removed
        nan_removed_path = run_dir / "nan_removed_rows.csv"
        pd.DataFrame(columns=["removal_reason"]).to_csv(nan_removed_path, index=False)
        return nan_removed_path


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
