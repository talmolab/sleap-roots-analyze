"""Utility functions for pipeline infrastructure.

This module provides helper functions for creating run directories, getting
git information, and package versioning.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


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
