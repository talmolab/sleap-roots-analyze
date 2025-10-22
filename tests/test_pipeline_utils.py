"""Tests for pipeline utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleap_roots_analyze.pipeline.utils import (
    create_run_directory,
    get_environment_info,
    get_git_branch,
    get_git_commit_hash,
    get_package_version,
    get_package_versions,
)


def test_create_run_directory(tmp_path):
    """Test creating a run directory."""
    run_dir = create_run_directory(tmp_path, "test_pipeline")

    assert run_dir.exists()
    assert run_dir.parent == tmp_path
    assert "test_pipeline" in run_dir.name


def test_create_run_directory_with_timestamp(tmp_path):
    """Test creating a run directory with custom timestamp."""
    run_dir = create_run_directory(
        tmp_path, "test_pipeline", timestamp="20241021_120000"
    )

    assert run_dir.exists()
    assert run_dir.name == "test_pipeline_20241021_120000"


def test_create_run_directory_creates_parents(tmp_path):
    """Test that run directory creation creates parent dirs."""
    nested_path = tmp_path / "nested" / "path"
    run_dir = create_run_directory(nested_path, "test")

    assert run_dir.exists()
    assert nested_path.exists()


def test_get_git_commit_hash():
    """Test getting git commit hash."""
    commit = get_git_commit_hash()

    # Should either return a valid commit hash or None
    if commit is not None:
        # Git commit hashes are 40 characters (SHA-1)
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)


def test_get_git_branch():
    """Test getting git branch name."""
    branch = get_git_branch()

    # Should either return a valid branch name or None
    if branch is not None:
        assert isinstance(branch, str)
        assert len(branch) > 0


def test_get_package_version_installed():
    """Test getting version of an installed package."""
    # pandas should be installed in our environment
    version = get_package_version("pandas")

    assert version is not None
    assert isinstance(version, str)
    # Version should look like "x.y.z"
    assert len(version.split(".")) >= 2


def test_get_package_version_not_installed():
    """Test getting version of a package that doesn't exist."""
    version = get_package_version("nonexistent-package-12345")

    assert version is None


def test_get_package_versions():
    """Test getting versions for multiple packages."""
    packages = ["pandas", "numpy", "nonexistent-pkg-12345"]
    versions = get_package_versions(packages)

    assert isinstance(versions, dict)
    assert len(versions) == 3

    # pandas and numpy should be installed
    assert versions["pandas"] != "not installed"
    assert versions["numpy"] != "not installed"

    # nonexistent package should not be installed
    assert versions["nonexistent-pkg-12345"] == "not installed"


def test_get_package_versions_empty_list():
    """Test getting versions with empty package list."""
    versions = get_package_versions([])

    assert versions == {}


def test_get_environment_info():
    """Test getting environment information."""
    env_info = get_environment_info()

    assert isinstance(env_info, dict)

    # Should have git info (even if unknown)
    assert "git_commit" in env_info
    assert "git_branch" in env_info

    # Should have package versions
    assert "pandas" in env_info
    assert "numpy" in env_info
    assert "scipy" in env_info

    # Verify package versions are strings
    assert isinstance(env_info["pandas"], str)
    assert isinstance(env_info["numpy"], str)


def test_get_git_remote_url():
    """Test getting git remote URL."""
    from sleap_roots_analyze.pipeline.utils import get_git_remote_url

    url = get_git_remote_url()

    # Should either return a valid URL or None
    if url is not None:
        assert isinstance(url, str)
        assert len(url) > 0


def test_is_git_dirty():
    """Test checking if git repo is dirty."""
    from sleap_roots_analyze.pipeline.utils import is_git_dirty

    is_dirty = is_git_dirty()

    assert isinstance(is_dirty, bool)


def test_create_code_archive(tmp_path):
    """Test creating a code archive."""
    from sleap_roots_analyze.pipeline.utils import create_code_archive

    archive_path = tmp_path / "code_snapshot.tar.gz"
    result = create_code_archive(archive_path)

    assert result == archive_path
    assert archive_path.exists()
    assert archive_path.stat().st_size > 0

    # Verify it's a valid tar.gz
    import tarfile

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getnames()
        # Should contain sleap_roots_analyze directory
        assert any("sleap_roots_analyze" in name for name in members)


def test_get_code_snapshot_with_git(tmp_path):
    """Test getting code snapshot when git is available."""
    from sleap_roots_analyze.pipeline.utils import get_code_snapshot

    snapshot = get_code_snapshot(tmp_path, create_archive_if_dirty=False)

    assert "package_version" in snapshot
    assert "python_version" in snapshot
    assert "git_commit" in snapshot
    assert "git_branch" in snapshot
    assert "git_remote" in snapshot
    assert "git_is_dirty" in snapshot
    assert "code_archive" in snapshot

    # Verify types
    assert isinstance(snapshot["python_version"], str)
    assert isinstance(snapshot["git_is_dirty"], bool)


def test_get_code_snapshot_creates_archive_if_dirty(tmp_path):
    """Test that code snapshot creates archive when git is dirty."""
    from sleap_roots_analyze.pipeline.utils import get_code_snapshot, is_git_dirty

    # Only test archive creation if git is actually dirty
    if is_git_dirty():
        snapshot = get_code_snapshot(tmp_path, create_archive_if_dirty=True)
        assert snapshot["code_archive"] is not None
        assert Path(snapshot["code_archive"]).exists()


def test_get_code_snapshot_no_archive_if_clean(tmp_path):
    """Test that code snapshot doesn't create archive when git is clean."""
    from sleap_roots_analyze.pipeline.utils import get_code_snapshot, is_git_dirty

    # Only test if git is clean
    if not is_git_dirty():
        snapshot = get_code_snapshot(tmp_path, create_archive_if_dirty=True)
        assert snapshot["code_archive"] is None
