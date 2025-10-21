"""Utility functions for data processing and file management."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import json
import shutil


def create_run_directory(base_dir: Path) -> Path:
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


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types recursively."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
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


def link_rhizovision_images_to_samples(
    df: pd.DataFrame,
    image_dir: Path | str,
    image_types: Optional[List[str]] = None,
    barcode_col: str = "Barcode",
) -> Dict[str, Dict[str, Optional[Path]]]:
    """Link Rhizovision images to their corresponding sample barcodes.

    This function is specific to Rhizovision image naming conventions,
    expecting filenames in the format: {barcode}_{suffix}

    Args:
        df: Trait dataframe with barcode/ID column
        image_dir: Directory containing Rhizovision processed images
        image_types: List of Rhizovision image suffixes to look for (default: ['features.png', 'seg.png'])
        barcode_col: Name of the barcode/plant ID column (default: "Barcode")

    Returns:
        Dictionary mapping barcode to Rhizovision image paths
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


def setup_analysis_directories(
    base_dir: Union[str, Path], subdirs: Optional[List[str]] = None
) -> Dict[str, Path]:
    """Create organized directory structure for analysis outputs.

    Args:
        base_dir: Base run directory
        subdirs: List of subdirectory names to create

    Returns:
        Dictionary mapping directory names to Path objects
    """
    if subdirs is None:
        subdirs = [
            "figures",
            "publication_figures",
            "interactive_plots",
            "analysis_outputs",
        ]

    base_dir = Path(base_dir)
    directories = {"base": base_dir}

    for subdir in subdirs:
        dir_path = base_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        directories[subdir] = dir_path

    return directories


def save_notebook_snapshot(
    notebook_path: Union[str, Path],
    output_dir: Union[str, Path],
    prefix: str = "executed_notebook",
) -> Optional[Path]:
    """Save a snapshot of the current notebook with timestamp.

    Args:
        notebook_path: Path to the notebook file
        output_dir: Directory to save the snapshot
        prefix: Prefix for the saved file

    Returns:
        Path to saved notebook or None if failed
    """
    try:
        from nbformat import read, write

        notebook_path = Path(notebook_path)
        output_dir = Path(output_dir)

        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = read(f, as_version=4)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{prefix}_{timestamp}.ipynb"

        # Write the snapshot
        with open(output_path, "w", encoding="utf-8") as f:
            write(nb, f)

        return output_path

    except Exception as e:
        print(f"⚠️ Could not save notebook snapshot: {e}")
        return None


def log_analysis_summary(
    summary_dict: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "analysis_summary.json",
) -> None:
    """Save and display analysis summary.

    Args:
        summary_dict: Dictionary containing analysis summary
        output_dir: Directory to save the summary
        filename: Name of the output file
    """
    output_dir = Path(output_dir)
    output_path = output_dir / filename

    # Convert to JSON-serializable format
    serializable_summary = convert_to_json_serializable(summary_dict)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(serializable_summary, f, indent=2)

    print(f"💾 Analysis summary saved to: {output_path}")


def create_analysis_summary(
    df: pd.DataFrame,
    trait_cols: List[str],
    pca_results: Optional[Dict] = None,
    heritability_results: Optional[pd.DataFrame] = None,
    output_counts: Optional[Dict[str, int]] = None,
    genotype_col: str = "geno",
    replicate_col: str = "rep",
) -> Dict[str, Any]:
    """Create a comprehensive analysis summary.

    Args:
        df: Main dataframe
        trait_cols: List of trait column names
        pca_results: PCA analysis results
        heritability_results: Heritability results DataFrame
        output_counts: Counts of generated outputs
        genotype_col: Name of genotype column
        replicate_col: Name of replicate column

    Returns:
        Dictionary containing analysis summary
    """
    summary = {
        "data_overview": {
            "n_samples": len(df),
            "n_traits": len(trait_cols),
            "n_genotypes": (
                df[genotype_col].nunique() if genotype_col in df.columns else 0
            ),
            "n_replicates_per_genotype": (
                df.groupby(genotype_col)[replicate_col].nunique().mean()
                if genotype_col in df.columns and replicate_col in df.columns
                else 0
            ),
        },
        "timestamp": datetime.now().isoformat(),
    }

    if pca_results is not None:
        summary["pca_analysis"] = {
            "n_components_selected": pca_results.get("n_components_selected", 0),
            "total_variance_explained": float(
                pca_results.get("total_variance_explained", 0)
            ),
        }

    if heritability_results is not None:
        summary["heritability"] = {
            "mean_heritability": float(heritability_results["heritability"].mean()),
            "min_heritability": float(heritability_results["heritability"].min()),
            "max_heritability": float(heritability_results["heritability"].max()),
            "n_traits_analyzed": len(heritability_results),
        }

    if output_counts is not None:
        summary["outputs_generated"] = output_counts

    return summary
