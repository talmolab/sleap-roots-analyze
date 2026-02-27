"""Utilities for the /configure-run-all interactive slash command.

These helpers handle backup path generation and statistical feasibility
checks that the Claude Code slash command calls during interactive config
authoring sessions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def make_backup_path(
    source_path: str | Path,
    archive_dir: str | Path,
    timestamp: datetime | None = None,
) -> Path:
    """Return the path where a backup of source_path should be written.

    Backup filename format: ``<stem>_backup_<YYYYMMDD_HHMMSS>.yaml``

    Args:
        source_path: Existing config file to back up.
        archive_dir: Destination directory (e.g. ``configs/archive/``).
        timestamp: Timestamp to embed in the filename.  Defaults to now.

    Returns:
        Path inside archive_dir for the backup file.  The directory is not
        created; the caller must ``mkdir`` before writing.

    Example:
        >>> from datetime import datetime
        >>> p = make_backup_path("configs/active/qc.yaml", "configs/archive",
        ...                      timestamp=datetime(2026, 2, 18, 10, 30, 45))
        >>> p.name
        'qc_backup_20260218_103045.yaml'
    """
    if timestamp is None:
        timestamp = datetime.now()
    source_path = Path(source_path)
    archive_dir = Path(archive_dir)
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    backup_name = f"{source_path.stem}_backup_{ts_str}.yaml"
    return archive_dir / backup_name


def warn_mahalanobis_small_n(n_samples: int) -> str | None:
    """Return a warning string when Mahalanobis chi-squared is unreliable.

    The chi-squared approximation for Mahalanobis distances requires n >= 30
    to be statistically reliable.

    Args:
        n_samples: Number of samples in the group.

    Returns:
        Warning string if n_samples < 30, else None.

    Example:
        >>> warn_mahalanobis_small_n(25)
        'WARNING: Mahalanobis chi-squared outlier detection is unreliable ...'
        >>> warn_mahalanobis_small_n(30) is None
        True
    """
    if n_samples >= 30:
        return None
    return (
        f"WARNING: Mahalanobis chi-squared outlier detection is unreliable for "
        f"n<30. This group has only {n_samples} samples. Consider using a more "
        f"permissive chi2_percentile (e.g. 95.0 instead of 99.0) or disabling "
        f"outlier detection for this group."
    )


def warn_heritability_low_replicates(min_replicates: int) -> str | None:
    """Return a warning string when heritability estimation is unreliable.

    Heritability (H²) estimation via ANOVA requires at least 3 replicates
    per genotype to partition genetic from environmental variance.

    Args:
        min_replicates: Minimum number of replicates per genotype in any group.

    Returns:
        Warning string if min_replicates < 3, else None.

    Example:
        >>> warn_heritability_low_replicates(2)
        'WARNING: Heritability estimation requires ...'
        >>> warn_heritability_low_replicates(3) is None
        True
    """
    if min_replicates >= 3:
        return None
    return (
        f"WARNING: Heritability estimation requires ≥3 replicates per genotype. "
        f"This dataset has only {min_replicates} replicate(s). H² estimates will "
        f"be unreliable. Consider disabling heritability filtering "
        f"(heritability.enabled: false) or increasing the replicate count."
    )


def recommend_umap_n_neighbors(n_samples: int) -> tuple[int, str | None]:
    """Recommend a UMAP n_neighbors value for the given sample size.

    Recommendation: ``min(15, max(2, n_samples // 4))``.

    Args:
        n_samples: Number of samples in the smallest group.

    Returns:
        Tuple of (recommended_n_neighbors, warning_string_or_None).
        A warning is returned when n_samples < 15.

    Example:
        >>> recommend_umap_n_neighbors(100)
        (15, None)
        >>> recommend_umap_n_neighbors(20)
        (5, None)
        >>> n, w = recommend_umap_n_neighbors(8)
        >>> n
        2
        >>> w is not None
        True
    """
    recommended = min(15, max(2, n_samples // 4))
    warning = None
    if n_samples < 15:
        warning = (
            f"WARNING: UMAP with only {n_samples} samples may not produce "
            f"meaningful structure. Recommended n_neighbors={recommended} "
            f"(= max(2, {n_samples}//4))."
        )
    return recommended, warning


def inspect_dataset(csv_path: str | Path) -> dict[str, Any]:
    """Read a CSV and return metadata useful for config authoring.

    Reports sample count, column names, numeric trait count, and candidate
    group_by columns (columns with ≤20 unique values).

    Args:
        csv_path: Path to the trait CSV file.

    Returns:
        Dict with keys:
        - ``n_samples``: Total row count.
        - ``columns``: List of all column names.
        - ``n_numeric_cols``: Count of numeric columns.
        - ``group_by_candidates``: List of column names with ≤20 unique values.

    Example:
        >>> result = inspect_dataset("traits.csv")
        >>> result["n_samples"]
        120
    """
    df = pd.read_csv(csv_path)
    group_by_candidates = [col for col in df.columns if df[col].nunique() <= 20]
    return {
        "n_samples": len(df),
        "columns": list(df.columns),
        "n_numeric_cols": int(df.select_dtypes("number").shape[1]),
        "group_by_candidates": group_by_candidates,
    }
