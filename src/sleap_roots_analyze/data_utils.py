"""Utility functions for data processing and file management."""

import numpy as np
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
