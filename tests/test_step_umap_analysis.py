"""Tests for UMAPAnalysisStep (TDD - tests written before implementation)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    PCAConfig,
)
from sleap_roots_analyze.pipeline.config.components import StatisticsConfig, UMAPConfig
from sleap_roots_analyze.pipeline.config.viz_config import VizPipelineConfig
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import UMAPAnalysisStep


@pytest.fixture
def viz_config():
    """Create test Viz config with UMAP enabled."""
    return VizPipelineConfig(
        pipeline_name="test_umap",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path="test.csv"),
        pca=PCAConfig(n_components=0.95, standardize=True),
        statistics=StatisticsConfig(),
        umap=UMAPConfig(
            enabled=True,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        ),
    )


@pytest.fixture
def viz_config_umap_disabled():
    """Create test Viz config with UMAP disabled."""
    return VizPipelineConfig(
        pipeline_name="test_umap_disabled",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path="test.csv"),
        umap=UMAPConfig(enabled=False),
    )


@pytest.fixture
def sample_data():
    """Create sample data with traits."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame(
        {
            "Barcode": [f"sample_{i}" for i in range(n_samples)],
            "Genotype": [f"geno_{i % 5}" for i in range(n_samples)],
            "Replicate": [i % 3 + 1 for i in range(n_samples)],
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
    """Create previous step result with trait names and PCA results."""
    np.random.seed(42)
    n_samples = len(sample_data)
    trait_cols = ["trait1", "trait2", "trait3", "trait4", "trait5", "trait6"]

    # Mock PCA results (as would come from PCAAnalysisStep)
    mock_pca_results = {
        "transformed_data": np.random.randn(n_samples, 3),
        "explained_variance_ratio": np.array([0.5, 0.3, 0.15]),
        "cumulative_variance_ratio": np.array([0.5, 0.8, 0.95]),
        "loadings": np.random.randn(len(trait_cols), 3),
        "eigenvalues": np.array([2.5, 1.5, 0.75]),
        "n_components_selected": 3,
        "feature_names": trait_cols,
    }

    return StepResult(
        data=sample_data,
        metadata={
            "trait_names": trait_cols,
            "valid_trait_names": trait_cols,
            "n_samples": n_samples,
            "pca_results": mock_pca_results,
            "heritability_results": {"trait1": {"heritability": 0.7}},
        },
    )


class TestUMAPAnalysisStepBasic:
    """Test basic UMAPAnalysisStep functionality."""

    def test_step_initialization(self):
        """Test that UMAPAnalysisStep initializes correctly."""
        step = UMAPAnalysisStep()
        assert step.step_name == "UMAPAnalysis"
        assert (
            "umap" in step.description.lower()
            or "dimensionality" in step.description.lower()
        )

    def test_basic_execution_when_enabled(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test that UMAP step executes successfully when enabled."""
        step = UMAPAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.equals(sample_data)

        # Check UMAP results in metadata
        assert "umap_results" in result.metadata
        assert (
            "umap_status" not in result.metadata
            or result.metadata.get("umap_status") != "disabled"
        )

    def test_skip_when_disabled(
        self, viz_config_umap_disabled, sample_data, prev_result, tmp_path
    ):
        """Test that UMAP step skips when disabled."""
        step = UMAPAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=viz_config_umap_disabled,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should skip and return status
        assert result.metadata.get("umap_status") == "disabled"
        # Data should be unchanged
        assert result.data.equals(sample_data)

    def test_skip_when_umap_not_installed(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test graceful handling when umap-learn is not installed."""
        step = UMAPAnalysisStep()

        # Mock UMAP_AVAILABLE to False (patch where it's imported, not where it's defined)
        with patch(
            "sleap_roots_analyze.pipeline.steps.umap_analysis.UMAP_AVAILABLE", False
        ):
            result = step.execute(
                data=sample_data,
                config=viz_config,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        # Should skip gracefully
        assert "umap_status" in result.metadata
        assert (
            "skipped" in result.metadata["umap_status"]
            or "not_installed" in result.metadata["umap_status"]
        )


class TestUMAPMetadataPreservation:
    """Test that UMAPAnalysisStep preserves metadata correctly."""

    def test_metadata_preservation(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test that all previous metadata is preserved."""
        # Add custom metadata
        prev_result.metadata["custom_field"] = "custom_value"
        prev_result.metadata["another_field"] = 12345

        step = UMAPAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Previous metadata should be retained
        assert result.metadata.get("custom_field") == "custom_value"
        assert result.metadata.get("another_field") == 12345
        assert "trait_names" in result.metadata
        assert "pca_results" in result.metadata
        assert "heritability_results" in result.metadata

    def test_image_paths_preserved(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """CRITICAL: Test that image_paths is preserved for downstream interactive plots."""
        # Add image_paths to metadata (as would come from LoadDataAndImagesStep)
        mock_image_paths = {
            "sample_0": {"features.png": Path("/mock/sample_0_features.png")},
            "sample_1": {"features.png": Path("/mock/sample_1_features.png")},
        }
        prev_result.metadata["image_paths"] = mock_image_paths

        step = UMAPAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # CRITICAL: image_paths must be preserved
        assert "image_paths" in result.metadata, (
            "UMAPAnalysisStep must preserve image_paths from previous step metadata. "
            "Without this, GenerateInteractiveStep cannot create UMAP plots with image hover."
        )
        assert result.metadata["image_paths"] == mock_image_paths


class TestUMAPResults:
    """Test UMAP results structure and content."""

    def test_umap_results_in_metadata(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test that umap_results dict is available in output metadata."""
        step = UMAPAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert "umap_results" in result.metadata
        umap_results = result.metadata["umap_results"]

        # Check required keys
        assert "embedding" in umap_results
        assert "n_neighbors" in umap_results
        assert "min_dist" in umap_results

    def test_embedding_shape(self, viz_config, sample_data, prev_result, tmp_path):
        """Test that UMAP embedding has correct shape (n_samples, 2)."""
        step = UMAPAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        umap_results = result.metadata["umap_results"]
        embedding = umap_results["embedding"]

        # Should be 2D with n_samples rows
        assert embedding.shape == (len(sample_data), 2), (
            f"UMAP embedding should have shape ({len(sample_data)}, 2), "
            f"got {embedding.shape}"
        )

    def test_reproducibility(self, viz_config, sample_data, prev_result, tmp_path):
        """Test that same random_state produces identical UMAP results."""
        step = UMAPAnalysisStep()

        # Run twice with same config
        result1 = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path / "run1",
            prev_result=prev_result,
        )

        result2 = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path / "run2",
            prev_result=prev_result,
        )

        # Embeddings should be identical
        embedding1 = result1.metadata["umap_results"]["embedding"]
        embedding2 = result2.metadata["umap_results"]["embedding"]

        np.testing.assert_array_almost_equal(
            embedding1,
            embedding2,
            err_msg="UMAP embeddings should be identical with same random_state",
        )


class TestUMAPArtifactExport:
    """Test that UMAP artifacts are saved correctly."""

    def test_artifacts_saved_to_data_umap_directory(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test that UMAP outputs are saved to data/umap/ directory."""
        step = UMAPAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        umap_dir = tmp_path / "data" / "umap"
        assert umap_dir.exists(), "data/umap/ directory should exist"

        # Check expected files
        assert (
            umap_dir / "umap_embedding.csv"
        ).exists(), "umap_embedding.csv should exist"
        assert (
            umap_dir / "umap_parameters.json"
        ).exists(), "umap_parameters.json should exist"

    def test_embedding_csv_content(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test that embedding CSV has correct content."""
        step = UMAPAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        umap_dir = tmp_path / "data" / "umap"
        embedding_df = pd.read_csv(umap_dir / "umap_embedding.csv", index_col=0)

        # Should have correct shape and column names
        assert embedding_df.shape == (len(sample_data), 2)
        assert list(embedding_df.columns) == ["UMAP1", "UMAP2"]

    def test_parameters_json_content(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test that parameters JSON has correct content."""
        step = UMAPAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        umap_dir = tmp_path / "data" / "umap"
        with open(umap_dir / "umap_parameters.json") as f:
            params = json.load(f)

        # Check required parameters
        assert params["n_neighbors"] == 15
        assert params["min_dist"] == 0.1
        assert params["random_state"] == 42


class TestUMAPDataPreservation:
    """Test that UMAP step preserves data correctly."""

    def test_data_unchanged(self, viz_config, sample_data, prev_result, tmp_path):
        """Test that UMAP step doesn't modify input data."""
        original_data = sample_data.copy()

        step = UMAPAnalysisStep()
        result = step.execute(
            data=sample_data,
            config=viz_config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        pd.testing.assert_frame_equal(result.data, original_data)


class TestUMAPConfigParameters:
    """Test UMAP with different configuration parameters."""

    def test_different_n_neighbors(
        self, viz_config, sample_data, prev_result, tmp_path
    ):
        """Test UMAP with different n_neighbors values."""
        for n_neighbors in [5, 15, 30]:
            viz_config.umap.n_neighbors = n_neighbors
            step = UMAPAnalysisStep()

            run_dir = tmp_path / f"n_neighbors_{n_neighbors}"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=viz_config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            assert result.metadata["umap_results"]["n_neighbors"] == n_neighbors

    def test_different_min_dist(self, viz_config, sample_data, prev_result, tmp_path):
        """Test UMAP with different min_dist values."""
        for min_dist in [0.0, 0.1, 0.5]:
            viz_config.umap.min_dist = min_dist
            step = UMAPAnalysisStep()

            run_dir = tmp_path / f"min_dist_{min_dist}"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = step.execute(
                data=sample_data,
                config=viz_config,
                run_dir=run_dir,
                prev_result=prev_result,
            )

            assert result.metadata["umap_results"]["min_dist"] == min_dist
