"""Utility functions for data processing and file management."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


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
