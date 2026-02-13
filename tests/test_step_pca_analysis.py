"""Tests for PCAAnalysisStep."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    PCAConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import PCAAnalysisStep


@pytest.fixture
def config():
    """Create test config with PCA settings."""
    return QCPipelineConfig(
        pipeline_name="test_pca",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="test.csv"),
        pca=PCAConfig(
            n_components=2,
            standardize=True,
            n_top_features=5,
            feature_selection_strategy="top_absolute",
        ),
    )


@pytest.fixture
def sample_data():
    """Create sample data with traits."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame(
        {
            "Barcode": [f"sample_{i}" for i in range(n_samples)],
            "geno": [f"geno_{i % 5}" for i in range(n_samples)],
            "rep": [i % 3 + 1 for i in range(n_samples)],
            "trait1": np.random.randn(n_samples),
            "trait2": np.random.randn(n_samples) * 2,
            "trait3": np.random.randn(n_samples) * 0.5,
            "trait4": np.random.randn(n_samples) + 5,
            "trait5": np.random.randn(n_samples) - 2,
            "trait6": np.random.randn(n_samples) * 3,
        }
    )


@pytest.fixture
def prev_result(sample_data):
    """Create previous step result."""
    trait_cols = ["trait1", "trait2", "trait3", "trait4", "trait5", "trait6"]
    return StepResult(
        data=sample_data,
        metadata={
            "trait_cols": trait_cols,
            "metadata_cols": ["Barcode", "geno", "rep"],
        },
    )


class TestPCAAnalysisStep:
    """Test suite for PCAAnalysisStep."""

    def test_pca_basic_execution(self, config, sample_data, prev_result, tmp_path):
        """Test basic PCA step execution."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.equals(sample_data)

        # Check metadata
        assert "pca_results" in result.metadata
        assert "top_features" in result.metadata
        assert "n_pca_components" in result.metadata
        assert "pca_explained_variance" in result.metadata

        # Check PCA results
        assert result.metadata["n_pca_components"] == 2
        assert len(result.metadata["top_features"]) == 5

    def test_pca_output_files(self, config, sample_data, prev_result, tmp_path):
        """Test that PCA outputs are saved correctly."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        pca_dir = tmp_path / "data" / "pca"
        assert pca_dir.exists()

        # Check all expected files
        assert (pca_dir / "pc_scores.csv").exists()
        assert (pca_dir / "loadings.csv").exists()
        assert (pca_dir / "explained_variance.csv").exists()
        assert (pca_dir / "top_features.csv").exists()

        # Validate file contents
        pc_scores = pd.read_csv(pca_dir / "pc_scores.csv", index_col=0)
        assert pc_scores.shape == (50, 2)  # n_samples x n_components
        assert list(pc_scores.columns) == ["PC1", "PC2"]

        loadings = pd.read_csv(pca_dir / "loadings.csv", index_col=0)
        assert loadings.shape == (6, 2)  # n_traits x n_components

        explained_var = pd.read_csv(pca_dir / "explained_variance.csv", index_col=0)
        assert explained_var.shape == (2, 3)  # n_components x 3 metrics
        assert "explained_variance" in explained_var.columns
        assert "explained_variance_ratio" in explained_var.columns
        assert "cumulative_variance_ratio" in explained_var.columns

        top_features = pd.read_csv(pca_dir / "top_features.csv")
        assert len(top_features) == 5

    def test_pca_with_variance_threshold(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test PCA with variance threshold instead of fixed components."""
        config.pca.n_components = 0.95  # 95% variance threshold

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should select components to reach 95% variance
        n_components = result.metadata["n_pca_components"]
        explained_var = result.metadata["pca_explained_variance"]

        assert n_components >= 1
        assert explained_var >= 0.95

    def test_pca_different_feature_selection_strategies(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test different feature selection strategies."""
        strategies = ["top_absolute", "extreme", "top_contribution", "top_variance"]

        for strategy in strategies:
            config.pca.feature_selection_strategy = strategy
            step = PCAAnalysisStep()

            run_dir = tmp_path / strategy
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            # All strategies should return at least the requested number (some may return more)
            # The "extreme" strategy may return all features if they all have extreme values
            assert len(result.metadata["top_features"]) >= config.pca.n_top_features
            assert all(
                f in prev_result.metadata["trait_cols"]
                for f in result.metadata["top_features"]
            )

    def test_pca_without_standardization(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test PCA without standardization."""
        config.pca.standardize = False

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should still complete successfully
        assert "pca_results" in result.metadata
        assert result.metadata["n_pca_components"] == 2

    def test_pca_metadata_propagation(self, config, sample_data, prev_result, tmp_path):
        """Test that previous metadata is propagated."""
        prev_result.metadata["custom_field"] = "custom_value"

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Previous metadata should be retained
        assert "custom_field" in result.metadata
        assert result.metadata["custom_field"] == "custom_value"
        assert "trait_cols" in result.metadata

        # New metadata should be added
        assert "pca_results" in result.metadata
        assert "top_features" in result.metadata

    def test_pca_with_more_features_than_samples(self, config, prev_result, tmp_path):
        """Test PCA when requesting more features than exist."""
        config.pca.n_top_features = 100  # More than 6 traits

        # Create data with only 3 samples (less than features)
        small_data = pd.DataFrame(
            {
                "Barcode": ["s1", "s2", "s3"],
                "geno": ["g1", "g2", "g3"],
                "rep": [1, 1, 1],
                "trait1": [1, 2, 3],
                "trait2": [2, 3, 4],
                "trait3": [3, 4, 5],
                "trait4": [4, 5, 6],
                "trait5": [5, 6, 7],
                "trait6": [6, 7, 8],
            }
        )

        prev_result.data = small_data
        step = PCAAnalysisStep()

        result = step.execute(
            data=small_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should return at most the number of available features
        assert len(result.metadata["top_features"]) <= 6

    def test_pca_different_n_components(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test PCA with different numbers of components."""
        for n_comp in [1, 3, 5]:
            config.pca.n_components = n_comp
            step = PCAAnalysisStep()

            run_dir = tmp_path / f"pca_{n_comp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            # Check correct number of components
            assert result.metadata["n_pca_components"] == n_comp

            # Check saved files have correct dimensions
            pca_dir = run_dir / "data" / "pca"
            pc_scores = pd.read_csv(pca_dir / "pc_scores.csv", index_col=0)
            assert pc_scores.shape[1] == n_comp

    def test_pca_top_features_selection(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test selecting different numbers of top features."""
        for n_features in [1, 3, 6]:
            config.pca.n_top_features = n_features
            step = PCAAnalysisStep()

            run_dir = tmp_path / f"features_{n_features}"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            assert len(result.metadata["top_features"]) == n_features

    def test_pca_preserves_data_unchanged(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test that PCA step doesn't modify input data."""
        original_data = sample_data.copy()

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Data should be unchanged
        pd.testing.assert_frame_equal(result.data, original_data)

    def test_pca_with_existing_data_pca_dir(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test that PCA step handles existing data/pca directory."""
        pca_dir = tmp_path / "data" / "pca"
        pca_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy file
        (pca_dir / "old_file.txt").write_text("old content")

        step = PCAAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should complete successfully and create new files
        assert (pca_dir / "pc_scores.csv").exists()
        assert (pca_dir / "loadings.csv").exists()


class TestPCADataOrganization:
    """Test that PCA outputs are saved to data/pca/ subdirectory (VIZ-OUTPUT-001)."""

    def test_pca_outputs_saved_to_data_pca_directory(
        self, config, sample_data, prev_result, tmp_path
    ):
        """Test that PCA step saves CSVs to data/pca/ subdirectory."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # PCA outputs should be in data/pca/, not pca/
        data_pca_dir = tmp_path / "data" / "pca"
        assert data_pca_dir.exists(), "data/pca/ directory should exist"

        # All PCA CSV files should be in data/pca/
        assert (data_pca_dir / "pc_scores.csv").exists()
        assert (data_pca_dir / "loadings.csv").exists()
        assert (data_pca_dir / "explained_variance.csv").exists()
        assert (data_pca_dir / "top_features.csv").exists()

        # Old path should NOT exist
        old_pca_dir = tmp_path / "pca"
        assert not old_pca_dir.exists(), "pca/ at root should not exist"

    def test_pca_output_files_content(self, config, sample_data, prev_result, tmp_path):
        """Test that PCA output file contents are correct in new location."""
        step = PCAAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        data_pca_dir = tmp_path / "data" / "pca"

        # Validate file contents
        pc_scores = pd.read_csv(data_pca_dir / "pc_scores.csv", index_col=0)
        assert pc_scores.shape == (50, 2)  # n_samples x n_components
        assert list(pc_scores.columns) == ["PC1", "PC2"]

        loadings = pd.read_csv(data_pca_dir / "loadings.csv", index_col=0)
        assert loadings.shape == (6, 2)  # n_traits x n_components

        explained_var = pd.read_csv(
            data_pca_dir / "explained_variance.csv", index_col=0
        )
        assert explained_var.shape == (2, 3)  # n_components x 3 metrics

        top_features = pd.read_csv(data_pca_dir / "top_features.csv")
        assert len(top_features) == 5
