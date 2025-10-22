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


def get_code_snapshot(
    run_dir: Path, create_archive_if_dirty: bool = True
) -> Dict[str, any]:
    """Get a complete code snapshot for reproducibility.

    Strategy:
    1. Try to get git information
    2. If git is dirty or unavailable, optionally create code archive
    3. Always capture package version and Python version

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

    Example:
        >>> snapshot = get_code_snapshot(Path("./run_20241021"))
        >>> # Returns: {
        >>> #     'package_version': '0.1.0',
        >>> #     'git_commit': 'abc123...',
        >>> #     'git_branch': 'main',
        >>> #     'git_remote': 'https://github.com/...',
        >>> #     'git_is_dirty': False,
        >>> #     'code_archive': None,
        >>> #     'python_version': '3.11.0 ...'
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
