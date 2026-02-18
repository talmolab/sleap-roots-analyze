"""Utility functions for pipeline infrastructure.

This module provides helper functions for creating run directories, getting
git information, and package versioning.
"""

from __future__ import annotations

import copy
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def create_run_directory(
    output_dir: str | Path,
    pipeline_name: str,
    timestamp: Optional[str] = None,
) -> Path:
    """Create a timestamped run directory for pipeline outputs.

    Args:
        output_dir: Base output directory.
        pipeline_name: Name of the pipeline.
        timestamp: Optional timestamp string. If not provided, uses current time
            in format YYYYMMDD_HHMMSS.

    Returns:
        Path to the created run directory.

    Example:
        >>> run_dir = create_run_directory("./outputs", "qc_pipeline")
        >>> # Creates ./outputs/qc_pipeline_20241021_143052
    """
    output_dir = Path(output_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = output_dir / f"{pipeline_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def get_git_commit_hash() -> Optional[str]:
    """Get the current git commit hash.

    Returns:
        The git commit hash as a string, or None if not in a git repo or
        git is not available.

    Example:
        >>> commit = get_git_commit_hash()
        >>> # Returns: 'abc123def456...' or None
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_branch() -> Optional[str]:
    """Get the current git branch name.

    Returns:
        The git branch name, or None if not in a git repo or git is not available.

    Example:
        >>> branch = get_git_branch()
        >>> # Returns: 'main' or 'feature/my-feature' or None
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_remote_url() -> Optional[str]:
    """Get the git remote origin URL.

    Returns:
        The remote URL, or None if not in a git repo or git is not available.

    Example:
        >>> url = get_git_remote_url()
        >>> # Returns: 'https://github.com/user/repo.git' or None
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty() -> bool:
    """Check if the git repository has uncommitted changes.

    Returns:
        True if there are uncommitted changes, False otherwise.
        Returns False if not in a git repo or git is not available.

    Example:
        >>> dirty = is_git_dirty()
        >>> # Returns: True or False
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"], capture_output=True, check=False
        )
        return result.returncode != 0
    except (FileNotFoundError, OSError):
        return False


def create_code_archive(output_path: str | Path) -> Path:
    """Create a tar.gz archive of the sleap_roots_analyze package source.

    Args:
        output_path: Path where the archive should be saved.

    Returns:
        Path to the created archive.

    Example:
        >>> archive = create_code_archive("./code_snapshot.tar.gz")
        >>> # Creates archive at ./code_snapshot.tar.gz
    """
    import tarfile

    import sleap_roots_analyze

    output_path = Path(output_path)
    package_path = Path(sleap_roots_analyze.__file__).parent

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(package_path, arcname="sleap_roots_analyze")

    return output_path


def _get_dependency_versions() -> Dict[str, str]:
    """Get versions of key scientific dependencies.

    Returns:
        Dictionary mapping package names to version strings.
        Packages that are not installed return "not installed".
    """
    key_packages = [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "umap-learn",
    ]

    versions = {}
    for pkg in key_packages:
        version = get_package_version(pkg)
        versions[pkg] = version if version else "not installed"

    return versions


def get_code_snapshot(
    run_dir: Path, create_archive_if_dirty: bool = True
) -> Dict[str, any]:
    """Get a complete code snapshot for reproducibility.

    Strategy:
    1. Try to get git information
    2. If git is dirty or unavailable, optionally create code archive
    3. Always capture package version and Python version
    4. Capture versions of key scientific dependencies

    Args:
        run_dir: Directory to save code archive if needed.
        create_archive_if_dirty: Whether to create archive if git is dirty or unavailable.

    Returns:
        Dictionary with code snapshot information:
            - package_version: Version of sleap_roots_analyze
            - git_commit: Git commit hash (if available)
            - git_branch: Git branch name (if available)
            - git_remote: Git remote URL (if available)
            - git_is_dirty: Whether there are uncommitted changes
            - code_archive: Path to code archive (if created)
            - python_version: Python version string
            - dependencies: Dictionary of key package versions

    Example:
        >>> snapshot = get_code_snapshot(Path("./run_20241021"))
        >>> # Returns: {
        >>> #     'package_version': '0.1.0',
        >>> #     'git_commit': 'abc123...',
        >>> #     'git_branch': 'main',
        >>> #     'git_remote': 'https://github.com/...',
        >>> #     'git_is_dirty': False,
        >>> #     'code_archive': None,
        >>> #     'python_version': '3.11.0 ...',
        >>> #     'dependencies': {'pandas': '2.0.0', 'numpy': '1.24.0', ...}
        >>> # }
    """
    import sys

    snapshot = {
        "package_version": get_package_version("sleap-roots-analyze") or "unknown",
        "python_version": sys.version,
        "git_commit": get_git_commit_hash(),
        "git_branch": get_git_branch(),
        "git_remote": get_git_remote_url(),
        "git_is_dirty": is_git_dirty(),
        "code_archive": None,
        "dependencies": _get_dependency_versions(),
    }

    # Create archive if git is dirty or unavailable
    should_archive = False
    if snapshot["git_commit"] is None:
        should_archive = True  # No git available
    elif snapshot["git_is_dirty"]:
        should_archive = True  # Uncommitted changes

    if should_archive and create_archive_if_dirty:
        archive_path = run_dir / "code_snapshot.tar.gz"
        create_code_archive(archive_path)
        snapshot["code_archive"] = str(archive_path)

    return snapshot


def get_package_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package.

    Args:
        package_name: Name of the package.

    Returns:
        Version string, or None if package is not installed.

    Example:
        >>> version = get_package_version("pandas")
        >>> # Returns: '2.0.0' or None
    """
    try:
        import importlib.metadata

        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_package_versions(package_names: list[str]) -> Dict[str, str]:
    """Get versions for multiple packages.

    Args:
        package_names: List of package names.

    Returns:
        Dictionary mapping package names to version strings.
        Missing packages are included with version "not installed".

    Example:
        >>> versions = get_package_versions(["pandas", "numpy", "scipy"])
        >>> # Returns: {'pandas': '2.0.0', 'numpy': '1.24.0', 'scipy': '1.10.0'}
    """
    versions = {}
    for name in package_names:
        version = get_package_version(name)
        versions[name] = version if version is not None else "not installed"
    return versions


def get_environment_info() -> Dict[str, str]:
    """Get environment information including git and package versions.

    Returns:
        Dictionary with git commit, branch, and package versions for key
        dependencies.

    Example:
        >>> env_info = get_environment_info()
        >>> # Returns: {
        >>> #     'git_commit': 'abc123...',
        >>> #     'git_branch': 'main',
        >>> #     'pandas': '2.0.0',
        >>> #     'numpy': '1.24.0',
        >>> #     ...
        >>> # }
    """
    info = {
        "git_commit": get_git_commit_hash() or "unknown",
        "git_branch": get_git_branch() or "unknown",
    }

    # Add versions for key packages
    packages = [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
    ]
    info.update(get_package_versions(packages))

    return info


def split_data_by_group(
    df: pd.DataFrame, group_by_column: str
) -> Dict[Any, pd.DataFrame]:
    """Split a DataFrame into separate DataFrames based on unique values in a column.

    Args:
        df: Input DataFrame to split.
        group_by_column: Name of column to group by.

    Returns:
        Dictionary mapping group values to DataFrames containing only rows
        with that group value. Returns empty dict for empty input DataFrame.

    Raises:
        KeyError: If group_by_column does not exist in DataFrame.

    Example:
        >>> data = {"day": [7, 7, 14, 14], "trait": [1.0, 2.0, 3.0, 4.0]}
        >>> df = pd.DataFrame(data)
        >>> groups = split_data_by_group(df, "day")
        >>> len(groups)
        2
        >>> groups[7].shape
        (2, 2)
    """
    # Validate column exists
    if group_by_column not in df.columns:
        raise KeyError(
            f"Column '{group_by_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Handle empty DataFrame
    if df.empty:
        return {}

    # Split by unique values in group column
    groups = {}
    for group_value in df[group_by_column].unique():
        group_df = df[df[group_by_column] == group_value].copy()
        groups[group_value] = group_df

    return groups


def filter_valid_groups(
    groups: Dict[Any, pd.DataFrame],
    min_samples: int,
    group_column_name: str,
) -> tuple[Dict[Any, pd.DataFrame], Dict[Any, int]]:
    """Filter groups to only keep those with sufficient samples.

    Args:
        groups: Dictionary mapping group values to DataFrames
        min_samples: Minimum number of samples required per group
        group_column_name: Name of the grouping column (for logging)

    Returns:
        Tuple of (valid_groups, skipped_groups) where:
        - valid_groups: Dict of groups with >= min_samples
        - skipped_groups: Dict mapping skipped group values to their sample counts
    """
    valid_groups = {}
    skipped_groups = {}

    for group_value, group_df in groups.items():
        sample_count = len(group_df)

        if sample_count >= min_samples:
            valid_groups[group_value] = group_df
        else:
            skipped_groups[group_value] = sample_count
            logger.warning(
                f"Skipping group {group_column_name}={group_value} "
                f"({sample_count} samples < {min_samples} minimum)"
            )

    return valid_groups, skipped_groups


def run_grouped_pipelines(
    config: Any,
    output_dir: str | Path,
    pipeline_class: type,
    **pipeline_kwargs,
) -> Dict[Any, Dict[str, Any]]:
    """Run pipelines with optional grouping by a metadata column.

    If config.data.group_by is set, splits the data into groups and runs
    separate pipeline instances for each group with isolated output directories.
    Otherwise, runs a single pipeline instance.

    Args:
        config: Pipeline configuration object (must have .data.group_by attribute)
        output_dir: Base output directory for pipeline runs
        pipeline_class: Pipeline class to instantiate (e.g., QCPipeline, VizPipeline)
        **pipeline_kwargs: Additional keyword arguments to pass to pipeline constructor

    Returns:
        Dictionary mapping group values to their results:
        - If grouped: {group_value: {"pipeline": pipeline_instance, "output_dir": path, "results": task_results}}
        - If not grouped: {None: {"pipeline": pipeline_instance, "output_dir": path, "results": task_results}}

    Example:
        >>> config = QCPipelineConfig(...)
        >>> config.data.group_by = "plant_age_days"
        >>> results = run_grouped_pipelines(
        ...     config=config,
        ...     output_dir="./outputs",
        ...     pipeline_class=QCPipeline
        ... )
        >>> # Creates: outputs/plant_age_days_7_20260213_140530/
        >>> #          outputs/plant_age_days_14_20260213_140545/
        >>> #          outputs/plant_age_days_21_20260213_140600/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if grouping is requested
    group_by_column = getattr(config.data, "group_by", None)

    if group_by_column is None:
        # No grouping - run single pipeline
        logger.info("Running single pipeline (no grouping)")
        pipeline = pipeline_class(
            config=config,
            output_dir=output_dir,
            **pipeline_kwargs,
        )
        results = pipeline.run()
        return {
            None: {
                "pipeline": pipeline,
                "output_dir": str(pipeline.run_dir),
                "results": results,
            }
        }

    # Grouping requested - load data and split into groups
    logger.info(f"Grouping pipelines by column: {group_by_column}")

    # Load data
    df = pd.read_csv(config.data.csv_path)
    logger.info(f"Loaded {len(df)} samples from {config.data.csv_path}")

    # Validate group_by column exists before attempting to split
    if group_by_column not in df.columns:
        raise ValueError(
            f"group_by column '{group_by_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    # Split by group column
    groups = split_data_by_group(df, group_by_column=group_by_column)
    logger.info(f"Split data into {len(groups)} groups: {list(groups.keys())}")

    # Filter groups by minimum sample count
    min_samples = getattr(config.cleanup, "min_samples_per_trait", 3)
    valid_groups, skipped_groups = filter_valid_groups(
        groups=groups,
        min_samples=min_samples,
        group_column_name=group_by_column,
    )
    logger.info(f"Retained {len(valid_groups)} valid groups after filtering")
    if not valid_groups:
        logger.warning(
            "No valid groups remain after filtering with min_samples_per_trait=%s. "
            "This likely indicates a configuration issue or incompatible data.",
            min_samples,
        )
        return {}
    if skipped_groups:
        logger.info(f"Skipped {len(skipped_groups)} groups: {skipped_groups}")

    # Run pipeline for each valid group
    grouped_results = {}
    for group_value in sorted(valid_groups.keys()):
        group_df = valid_groups[group_value]
        logger.info(
            f"Processing group {group_by_column}={group_value} ({len(group_df)} samples)"
        )

        # Create temporary CSV for this group
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as tmp_file:
            group_csv_path = Path(tmp_file.name)
            group_df.to_csv(tmp_file, index=False)

        # Create modified config for this group
        group_config = copy.deepcopy(config)
        group_config.data.csv_path = str(group_csv_path)
        # Remove group_by to prevent infinite recursion
        group_config.data.group_by = None

        # Modify pipeline name to include group information
        # This will create directories like: plant_age_days_7_20260213_140530/
        group_config.pipeline_name = f"{group_by_column}_{group_value}"

        # Run pipeline for this group
        try:
            pipeline = pipeline_class(
                config=group_config,
                output_dir=output_dir,
                **pipeline_kwargs,
            )

            results = pipeline.run()
            grouped_results[group_value] = {
                "pipeline": pipeline,
                "output_dir": str(pipeline.run_dir),
                "results": results,
            }
            logger.info(f"Group {group_by_column}={group_value} completed successfully")
        finally:
            # Clean up temporary CSV
            group_csv_path.unlink(missing_ok=True)

    logger.info(
        f"All grouped pipelines completed. Processed {len(grouped_results)} groups."
    )
    return grouped_results
