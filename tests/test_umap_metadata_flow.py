"""Integration tests for UMAP metadata flow through the pipeline.

These tests verify that metadata (especially image_paths) flows correctly
from LoadDataAndImagesStep through UMAPAnalysisStep to GenerateInteractiveStep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    PCAConfig,
)
from sleap_roots_analyze.pipeline.config.components import (
    InteractiveVisualizationConfig,
    StatisticsConfig,
    StaticVisualizationConfig,
    UMAPConfig,
)
from sleap_roots_analyze.pipeline.config.viz_config import VizPipelineConfig
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import (
    PCAAnalysisStep,
    StatisticalAnalysisStep,
    UMAPAnalysisStep,
)


@pytest.fixture
def sample_data_with_traits():
    """Create sample data with realistic traits."""
    np.random.seed(42)
    n_samples = 30
    return pd.DataFrame(
        {
            "Barcode": [f"plant_{i:03d}" for i in range(n_samples)],
            "Genotype": [f"geno_{i % 5}" for i in range(n_samples)],
            "Replicate": [i % 3 + 1 for i in range(n_samples)],
            "primary_length_mean": np.random.randn(n_samples) * 10 + 50,
            "lateral_count_mean": np.random.randn(n_samples) * 5 + 15,
            "root_width_mean": np.random.randn(n_samples) * 2 + 8,
            "branching_angle_mean": np.random.randn(n_samples) * 15 + 45,
            "tip_depth_mean": np.random.randn(n_samples) * 8 + 30,
        }
    )


@pytest.fixture
def viz_config_with_umap(tmp_path):
    """Create Viz config with UMAP and image settings enabled."""
    return VizPipelineConfig(
        pipeline_name="test_metadata_flow",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path=str(tmp_path / "test.csv")),
        pca=PCAConfig(n_components=3, standardize=True),
        statistics=StatisticsConfig(),
        umap=UMAPConfig(
            enabled=True,
            n_neighbors=10,
            min_dist=0.1,
            random_state=42,
        ),
        interactive_viz=InteractiveVisualizationConfig(
            create_pca_plots=True,
            create_umap_plots=True,
            show_images_on_hover=True,
        ),
    )


@pytest.fixture
def mock_image_paths():
    """Create mock image_paths metadata as would come from LoadDataAndImagesStep."""
    return {
        "plant_000": {"features.png": Path("/mock/images/plant_000_features.png")},
        "plant_001": {"features.png": Path("/mock/images/plant_001_features.png")},
        "plant_002": {"features.png": Path("/mock/images/plant_002_features.png")},
        "plant_010": {"features.png": Path("/mock/images/plant_010_features.png")},
    }


class TestUMAPMetadataFlow:
    """Test that metadata flows correctly through pipeline steps."""

    def test_image_paths_preserved_through_statistical_analysis(
        self, sample_data_with_traits, viz_config_with_umap, mock_image_paths, tmp_path
    ):
        """Test that image_paths survives StatisticalAnalysisStep."""
        trait_cols = [
            "primary_length_mean",
            "lateral_count_mean",
            "root_width_mean",
            "branching_angle_mean",
            "tip_depth_mean",
        ]

        # Simulate result from LoadDataAndImagesStep
        load_result = StepResult(
            data=sample_data_with_traits,
            metadata={
                "image_paths": mock_image_paths,
                "trait_names": trait_cols,
                "valid_trait_names": trait_cols,
                "n_samples": len(sample_data_with_traits),
            },
        )

        # Run StatisticalAnalysisStep
        stats_step = StatisticalAnalysisStep()
        stats_result = stats_step.execute(
            data=sample_data_with_traits,
            config=viz_config_with_umap,
            run_dir=tmp_path / "stats",
            prev_result=load_result,
        )

        # CRITICAL: image_paths must be preserved
        assert (
            "image_paths" in stats_result.metadata
        ), "StatisticalAnalysisStep must preserve image_paths metadata"
        assert stats_result.metadata["image_paths"] == mock_image_paths

    def test_image_paths_preserved_through_pca(
        self, sample_data_with_traits, viz_config_with_umap, mock_image_paths, tmp_path
    ):
        """Test that image_paths survives PCAAnalysisStep."""
        trait_cols = [
            "primary_length_mean",
            "lateral_count_mean",
            "root_width_mean",
            "branching_angle_mean",
            "tip_depth_mean",
        ]

        # Simulate result from StatisticalAnalysisStep
        stats_result = StepResult(
            data=sample_data_with_traits,
            metadata={
                "image_paths": mock_image_paths,
                "trait_names": trait_cols,
                "valid_trait_names": trait_cols,
                "n_samples": len(sample_data_with_traits),
                "heritability_results": {"primary_length_mean": {"heritability": 0.7}},
                "anova_results": {},
            },
        )

        # Run PCAAnalysisStep
        pca_step = PCAAnalysisStep()
        pca_result = pca_step.execute(
            data=sample_data_with_traits,
            config=viz_config_with_umap,
            run_dir=tmp_path / "pca",
            prev_result=stats_result,
        )

        # CRITICAL: image_paths must be preserved
        assert (
            "image_paths" in pca_result.metadata
        ), "PCAAnalysisStep must preserve image_paths metadata"
        assert pca_result.metadata["image_paths"] == mock_image_paths
        # Also verify PCA results were added
        assert "pca_results" in pca_result.metadata

    def test_image_paths_preserved_through_umap(
        self, sample_data_with_traits, viz_config_with_umap, mock_image_paths, tmp_path
    ):
        """Test that image_paths survives UMAPAnalysisStep."""
        trait_cols = [
            "primary_length_mean",
            "lateral_count_mean",
            "root_width_mean",
            "branching_angle_mean",
            "tip_depth_mean",
        ]

        # Simulate result from PCAAnalysisStep
        np.random.seed(42)
        pca_result = StepResult(
            data=sample_data_with_traits,
            metadata={
                "image_paths": mock_image_paths,
                "trait_names": trait_cols,
                "valid_trait_names": trait_cols,
                "n_samples": len(sample_data_with_traits),
                "heritability_results": {"primary_length_mean": {"heritability": 0.7}},
                "pca_results": {
                    "transformed_data": np.random.randn(
                        len(sample_data_with_traits), 3
                    ),
                    "explained_variance_ratio": np.array([0.5, 0.3, 0.15]),
                    "cumulative_variance_ratio": np.array([0.5, 0.8, 0.95]),
                    "loadings": np.random.randn(len(trait_cols), 3),
                    "eigenvalues": np.array([2.5, 1.5, 0.75]),
                    "n_components_selected": 3,
                    "feature_names": trait_cols,
                },
                "top_features": trait_cols[:3],
            },
        )

        # Run UMAPAnalysisStep
        umap_step = UMAPAnalysisStep()
        umap_result = umap_step.execute(
            data=sample_data_with_traits,
            config=viz_config_with_umap,
            run_dir=tmp_path / "umap",
            prev_result=pca_result,
        )

        # CRITICAL: image_paths must be preserved
        assert (
            "image_paths" in umap_result.metadata
        ), "UMAPAnalysisStep must preserve image_paths metadata"
        assert umap_result.metadata["image_paths"] == mock_image_paths
        # Also verify UMAP results were added
        assert "umap_results" in umap_result.metadata


class TestUMAPMetadataEndToEnd:
    """Test end-to-end metadata flow through multiple steps."""

    def test_full_pipeline_metadata_flow(
        self, sample_data_with_traits, viz_config_with_umap, mock_image_paths, tmp_path
    ):
        """Test metadata flow: Load → Stats → PCA → UMAP."""
        trait_cols = [
            "primary_length_mean",
            "lateral_count_mean",
            "root_width_mean",
            "branching_angle_mean",
            "tip_depth_mean",
        ]

        # 1. Simulate LoadDataAndImagesStep result
        load_result = StepResult(
            data=sample_data_with_traits,
            metadata={
                "image_paths": mock_image_paths,
                "trait_names": trait_cols,
                "valid_trait_names": trait_cols,
                "n_samples": len(sample_data_with_traits),
            },
        )

        # 2. Run StatisticalAnalysisStep
        stats_step = StatisticalAnalysisStep()
        stats_result = stats_step.execute(
            data=sample_data_with_traits,
            config=viz_config_with_umap,
            run_dir=tmp_path / "run" / "stats",
            prev_result=load_result,
        )

        # 3. Run PCAAnalysisStep
        pca_step = PCAAnalysisStep()
        pca_result = pca_step.execute(
            data=sample_data_with_traits,
            config=viz_config_with_umap,
            run_dir=tmp_path / "run" / "pca",
            prev_result=stats_result,
        )

        # 4. Run UMAPAnalysisStep
        umap_step = UMAPAnalysisStep()
        umap_result = umap_step.execute(
            data=sample_data_with_traits,
            config=viz_config_with_umap,
            run_dir=tmp_path / "run" / "umap",
            prev_result=pca_result,
        )

        # Verify all critical metadata is present for GenerateInteractiveStep
        final_metadata = umap_result.metadata

        # Required for interactive visualizations
        assert (
            "image_paths" in final_metadata
        ), "image_paths required for interactive plots"
        assert "umap_results" in final_metadata, "umap_results required for UMAP plots"
        assert "pca_results" in final_metadata, "pca_results required for PCA plots"
        assert (
            "heritability_results" in final_metadata
        ), "heritability_results useful for coloring"
        assert (
            "trait_names" in final_metadata
        ), "trait_names required for feature selection"

        # Verify umap_results structure
        umap_results = final_metadata["umap_results"]
        assert "embedding" in umap_results
        assert umap_results["embedding"].shape == (len(sample_data_with_traits), 2)
        assert "n_neighbors" in umap_results
        assert "min_dist" in umap_results


class TestUMAPResultsForVisualization:
    """Test that UMAP results have correct structure for visualization functions."""

    def test_umap_results_compatible_with_interactive_viz(
        self, sample_data_with_traits, viz_config_with_umap, tmp_path
    ):
        """Test that umap_results has all fields needed by create_interactive_umap_*."""
        trait_cols = [
            "primary_length_mean",
            "lateral_count_mean",
            "root_width_mean",
            "branching_angle_mean",
            "tip_depth_mean",
        ]

        np.random.seed(42)
        pca_result = StepResult(
            data=sample_data_with_traits,
            metadata={
                "trait_names": trait_cols,
                "valid_trait_names": trait_cols,
                "n_samples": len(sample_data_with_traits),
                "pca_results": {
                    "transformed_data": np.random.randn(
                        len(sample_data_with_traits), 3
                    ),
                    "explained_variance_ratio": np.array([0.5, 0.3, 0.15]),
                    "cumulative_variance_ratio": np.array([0.5, 0.8, 0.95]),
                    "loadings": np.random.randn(len(trait_cols), 3),
                    "eigenvalues": np.array([2.5, 1.5, 0.75]),
                    "n_components_selected": 3,
                    "feature_names": trait_cols,
                },
            },
        )

        umap_step = UMAPAnalysisStep()
        umap_result = umap_step.execute(
            data=sample_data_with_traits,
            config=viz_config_with_umap,
            run_dir=tmp_path,
            prev_result=pca_result,
        )

        umap_results = umap_result.metadata["umap_results"]

        # Check all required fields for visualization
        assert "embedding" in umap_results
        assert isinstance(umap_results["embedding"], np.ndarray)
        assert umap_results["embedding"].shape[1] == 2  # 2D for scatter plots

        assert "n_neighbors" in umap_results
        assert isinstance(umap_results["n_neighbors"], int)

        assert "min_dist" in umap_results
        assert isinstance(umap_results["min_dist"], float)

        assert "random_state" in umap_results  # For reproducibility info

        assert "clean_indices" in umap_results  # For aligning with original data
        assert len(umap_results["clean_indices"]) == umap_results["embedding"].shape[0]
