"""Tests for config authoring utilities.

Task coverage (add-config-authoring-command):
  4.1  Backup naming convention and archive behavior
  4.2  Statistical guardrail warning thresholds
  4.3  Integration: dataset inspection from a fixture CSV
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.config_authoring import (
    make_backup_path,
    recommend_umap_n_neighbors,
    warn_heritability_low_replicates,
    warn_mahalanobis_small_n,
    inspect_dataset,
)


# =============================================================================
# Task 4.1 — Backup naming convention
# =============================================================================


class TestMakeBackupPath:
    """Backup naming: <stem>_backup_<YYYYMMDD_HHMMSS>.yaml under archive dir."""

    def test_backup_path_uses_archive_dir(self, tmp_path):
        """Backup is written inside archive_dir."""
        source = tmp_path / "configs/active/qc_my_analysis.yaml"
        archive = tmp_path / "configs/archive"
        ts = datetime(2026, 2, 18, 10, 30, 45)
        result = make_backup_path(source, archive, timestamp=ts)
        assert result.parent == archive

    def test_backup_filename_format(self, tmp_path):
        """Backup filename is <stem>_backup_<YYYYMMDD_HHMMSS>.yaml."""
        source = tmp_path / "qc_my_analysis.yaml"
        archive = tmp_path / "archive"
        ts = datetime(2026, 2, 18, 10, 30, 45)
        result = make_backup_path(source, archive, timestamp=ts)
        assert result.name == "qc_my_analysis_backup_20260218_103045.yaml"

    def test_backup_has_yaml_extension(self, tmp_path):
        """Backup retains .yaml extension regardless of source extension."""
        source = tmp_path / "manifest.yaml"
        archive = tmp_path / "archive"
        ts = datetime(2026, 1, 1, 0, 0, 0)
        result = make_backup_path(source, archive, timestamp=ts)
        assert result.suffix == ".yaml"

    def test_two_different_timestamps_do_not_collide(self, tmp_path):
        """Two backups at different seconds produce distinct paths."""
        source = tmp_path / "qc.yaml"
        archive = tmp_path / "archive"
        ts1 = datetime(2026, 2, 18, 10, 30, 44)
        ts2 = datetime(2026, 2, 18, 10, 30, 45)
        assert make_backup_path(source, archive, ts1) != make_backup_path(
            source, archive, ts2
        )

    def test_default_timestamp_is_not_none(self, tmp_path):
        """Without explicit timestamp, returns a path (auto-uses current time)."""
        source = tmp_path / "qc.yaml"
        archive = tmp_path / "archive"
        result = make_backup_path(source, archive)
        assert result is not None
        assert result.suffix == ".yaml"
        assert "_backup_" in result.name


# =============================================================================
# Task 4.2 — Statistical guardrail thresholds
# =============================================================================


class TestWarnMahalanobisSmallN:
    """Mahalanobis chi-squared is unreliable for n < 30."""

    def test_no_warning_for_30_samples(self):
        """No warning when n == 30 (at threshold)."""
        assert warn_mahalanobis_small_n(30) is None

    def test_no_warning_for_large_n(self):
        """No warning for n > 30."""
        assert warn_mahalanobis_small_n(100) is None

    def test_warning_for_29_samples(self):
        """Warning raised when n < 30."""
        result = warn_mahalanobis_small_n(29)
        assert result is not None
        assert "29" in result

    def test_warning_for_very_small_n(self):
        """Warning raised for very small groups."""
        result = warn_mahalanobis_small_n(5)
        assert result is not None
        assert "5" in result

    def test_warning_mentions_chi_squared(self):
        """Warning text mentions chi-squared reliability issue."""
        result = warn_mahalanobis_small_n(10)
        assert result is not None
        assert "chi" in result.lower() or "mahalanobis" in result.lower()


class TestWarnHeritabilityLowReplicates:
    """Heritability estimation requires >= 3 replicates per genotype."""

    def test_no_warning_for_3_replicates(self):
        """No warning when min_replicates == 3."""
        assert warn_heritability_low_replicates(3) is None

    def test_no_warning_for_many_replicates(self):
        """No warning for n > 3."""
        assert warn_heritability_low_replicates(10) is None

    def test_warning_for_2_replicates(self):
        """Warning raised when n < 3."""
        result = warn_heritability_low_replicates(2)
        assert result is not None
        assert "2" in result

    def test_warning_for_1_replicate(self):
        """Warning raised for single replicate."""
        result = warn_heritability_low_replicates(1)
        assert result is not None

    def test_warning_mentions_heritability(self):
        """Warning text mentions heritability or replicates."""
        result = warn_heritability_low_replicates(2)
        assert "heritab" in result.lower() or "replicate" in result.lower()


class TestRecommendUmapNNeighbors:
    """UMAP n_neighbors recommendation: min(15, n_samples // 4)."""

    def test_large_dataset_recommends_15(self):
        """For n >= 60, recommend 15."""
        n, warning = recommend_umap_n_neighbors(100)
        assert n == 15
        assert warning is None

    def test_small_dataset_scales_down(self):
        """For n=20, recommend min(15, 20//4) = 5."""
        n, warning = recommend_umap_n_neighbors(20)
        assert n == 5

    def test_very_small_dataset_warns(self):
        """For n < 15, warning is issued."""
        n, warning = recommend_umap_n_neighbors(10)
        assert warning is not None

    def test_minimum_value_is_2(self):
        """For tiny n, n_neighbors is at least 2."""
        n, _ = recommend_umap_n_neighbors(4)
        assert n >= 2

    def test_boundary_at_60(self):
        """n=60 → min(15, 60//4=15) = 15, no warning."""
        n, warning = recommend_umap_n_neighbors(60)
        assert n == 15
        assert warning is None


# =============================================================================
# Task 4.3 — Dataset inspection integration
# =============================================================================


class TestInspectDataset:
    """inspect_dataset reads a CSV and returns structured metadata."""

    @pytest.fixture
    def fixture_csv(self, tmp_path):
        """Small fixture CSV with realistic column structure."""
        data = {
            "plant_qr_code": [f"p{i}" for i in range(50)],
            "accession_id": [f"GEN_{i % 5}" for i in range(50)],
            "plant_id": [i % 3 for i in range(50)],
            "plant_age_days": [7 if i < 25 else 14 for i in range(50)],
            "trait_a": [float(i) for i in range(50)],
            "trait_b": [float(i * 2) for i in range(50)],
            "trait_c": [float(i * 3) for i in range(50)],
        }
        path = tmp_path / "traits.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def test_returns_sample_count(self, fixture_csv):
        """inspect_dataset reports total row count."""
        result = inspect_dataset(fixture_csv)
        assert result["n_samples"] == 50

    def test_returns_column_names(self, fixture_csv):
        """inspect_dataset lists all columns."""
        result = inspect_dataset(fixture_csv)
        assert "plant_qr_code" in result["columns"]
        assert "trait_a" in result["columns"]

    def test_returns_numeric_trait_count(self, fixture_csv):
        """inspect_dataset counts numeric columns that look like traits."""
        result = inspect_dataset(fixture_csv)
        # trait_a, trait_b, trait_c, plant_id, plant_age_days are numeric
        assert result["n_numeric_cols"] >= 3

    def test_detects_group_by_candidates(self, fixture_csv):
        """Columns with <= 20 unique values are candidate group_by columns."""
        result = inspect_dataset(fixture_csv)
        candidates = result["group_by_candidates"]
        # plant_age_days has 2 unique values — should be a candidate
        assert "plant_age_days" in candidates

    def test_high_cardinality_column_excluded_from_candidates(self, fixture_csv):
        """Columns with many unique values are not group_by candidates."""
        result = inspect_dataset(fixture_csv)
        candidates = result["group_by_candidates"]
        # plant_qr_code has 50 unique values — not a candidate
        assert "plant_qr_code" not in candidates
