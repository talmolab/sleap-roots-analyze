"""Tests for StatisticalAnalysisStep (Step 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import StatisticalAnalysisStep


@pytest.fixture
def sample_data():
    """Create sample data with standardized column names (as after CleanupTraitsStep)."""
    np.random.seed(42)
    n_samples = 30
    return pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(n_samples)],
            "Genotype": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "Replicate": [1, 2, 3] * 10,
            "trait1": np.random.randn(n_samples) * 10 + 50,
            "trait2": np.random.randn(n_samples) * 5 + 25,
            "trait3": np.random.randn(n_samples) * 3 + 15,
        }
    )


@pytest.fixture
def config():
    """Create config with standardized column names."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path="dummy.csv"),
    )


@pytest.fixture
def prev_result(sample_data):
    """Previous result."""
    return StepResult(
        data=sample_data,
        metadata={
            "valid_trait_names": ["trait1", "trait2", "trait3"],
            "samples": 30,
        },
        files_generated=[],
    )


class TestStatisticalAnalysisStepBasic:
    """Test basic functionality."""

    def test_step_initialization(self):
        """Test initialization."""
        step = StatisticalAnalysisStep()
        assert step.step_name == "StatisticalAnalysis"
        assert "statistic" in step.description.lower()

    def test_basic_execution(self, sample_data, config, prev_result, tmp_path):
        """Test basic execution."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        assert isinstance(result, StepResult)
        assert result.data.equals(sample_data)
        assert "heritability_results" in result.metadata
        assert len(result.files_generated) > 0

    def test_heritability_file_generated(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test heritability results file is created in data/ directory."""
        step = StatisticalAnalysisStep()
        step.execute(sample_data, config, tmp_path, prev_result)

        assert (tmp_path / "data" / "08_heritability_results.csv").exists()

    def test_data_unchanged(self, sample_data, config, prev_result, tmp_path):
        """Test data unchanged."""
        step = StatisticalAnalysisStep()
        original = sample_data.copy()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        pd.testing.assert_frame_equal(result.data, original)


class TestStatisticalAnalysisStepMetadata:
    """Test metadata."""

    def test_heritability_metadata(self, sample_data, config, prev_result, tmp_path):
        """Test heritability in metadata."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        h2 = result.metadata["heritability_results"]
        assert "trait1" in h2
        assert "trait2" in h2
        assert "trait3" in h2

    def test_trait_names_propagation(self, sample_data, config, prev_result, tmp_path):
        """Test trait names propagated."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        assert result.metadata["valid_trait_names"] == ["trait1", "trait2", "trait3"]

    def test_image_paths_metadata_preserved(self, sample_data, config, tmp_path):
        """TDD Test: image_paths from previous step must be preserved in output metadata.

        Bug: StatisticalAnalysisStep creates a new metadata dict without **prev_result.metadata,
        causing image_paths (set by LoadDataAndImagesStep) to be lost.

        This breaks the image-dependent interactive plots in the Viz pipeline because
        GenerateInteractiveStep receives image_paths=None from PCA step metadata.

        Flow:
            LoadDataAndImagesStep -> sets image_paths in metadata
            StatisticalAnalysisStep -> MUST preserve image_paths
            PCAAnalysisStep -> preserves with **prev_result.metadata
            GenerateInteractiveStep -> reads image_paths from metadata

        Fix: Add **prev_result.metadata to metadata dict in StatisticalAnalysisStep.
        """
        from pathlib import Path

        # Create mock image_paths dict (as returned by link_rhizovision_images_to_samples)
        mock_image_paths = {
            "plant0": {"features.png": Path("/mock/plant0_features.png")},
            "plant1": {"features.png": Path("/mock/plant1_features.png")},
        }

        # Previous result with image_paths in metadata
        prev_result_with_images = StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "image_paths": mock_image_paths,  # This MUST be preserved
                "n_samples": 30,
                "other_metadata": "should_also_be_preserved",
            },
            files_generated=[],
        )

        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result_with_images)

        # CRITICAL: image_paths must be preserved in output metadata
        assert "image_paths" in result.metadata, (
            "StatisticalAnalysisStep must preserve image_paths from previous step metadata. "
            "Fix: Add **prev_result.metadata to the metadata dict creation. "
            "Without this, GenerateInteractiveStep receives image_paths=None and "
            "skips all image-dependent plots (scatter_with_images, image_gallery, etc.)"
        )
        assert result.metadata["image_paths"] == mock_image_paths

    def test_all_previous_metadata_preserved(self, sample_data, config, tmp_path):
        """TDD Test: All metadata from previous step should be preserved.

        Ensures **prev_result.metadata is used in the metadata dict.
        """
        # Create previous result with various metadata keys
        prev_result_with_metadata = StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2", "trait3"],
                "n_samples": 30,
                "n_traits": 3,
                "barcode_col": "Barcode",
                "genotype_col": "Genotype",
                "custom_key": "custom_value",
            },
            files_generated=[],
        )

        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result_with_metadata)

        # All previous metadata keys should be preserved
        assert result.metadata.get("n_samples") == 30
        assert result.metadata.get("n_traits") == 3
        assert result.metadata.get("barcode_col") == "Barcode"
        assert result.metadata.get("genotype_col") == "Genotype"
        assert result.metadata.get("custom_key") == "custom_value"


class TestStatisticalAnalysisNoHeritabilityPlots:
    """Test that statistical_analysis step does NOT generate heritability plots.

    Per VIZ-OUTPUT-002: Each figure type MUST be generated by exactly one pipeline step.
    Heritability plots should only be generated by generate_static_figures step,
    NOT by statistical_analysis step.
    """

    def test_no_heritability_plots_in_figures_dir(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that NO heritability plots are generated in figures/ directory.

        The statistical_analysis step should only generate data files (CSVs, JSONs),
        not visualization plots. Heritability plots are generated by the
        generate_static_figures step to avoid duplication.
        """
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Check that no figures/ directory is created by this step
        figures_dir = tmp_path / "figures"
        if figures_dir.exists():
            # If figures/ exists, verify no heritability plots are in it
            heritability_plots = list(figures_dir.glob("*heritability*"))
            assert len(heritability_plots) == 0, (
                f"statistical_analysis step should NOT generate heritability plots. "
                f"Found: {[p.name for p in heritability_plots]}"
            )

    def test_files_generated_are_data_only(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that files_generated contains only data files, no figures."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # All files should be data files (CSVs, JSONs), not figures (PNGs, PDFs)
        figure_extensions = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
        for file_path in result.files_generated:
            suffix = file_path.suffix.lower()
            assert suffix not in figure_extensions, (
                f"statistical_analysis step should not generate figure files. "
                f"Found: {file_path.name}"
            )


class TestStatisticalAnalysisDataOrganization:
    """Test that statistical analysis outputs are saved to data/ subdirectory (VIZ-OUTPUT-001)."""

    def test_all_outputs_saved_to_data_directory(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that all statistical analysis outputs are saved to data/ directory."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        data_dir = tmp_path / "data"
        assert data_dir.exists(), "data/ directory should exist"

        # All expected files should be in data/
        assert (data_dir / "08_trait_statistics.json").exists()
        assert (data_dir / "08_anova_results.csv").exists()
        assert (data_dir / "08_heritability_results.csv").exists()
        assert (data_dir / "08_statistical_analysis_summary.json").exists()

    def test_outputs_not_in_root_directory(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that statistical outputs are NOT saved directly to run_dir root."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Files should NOT be at root level
        assert not (tmp_path / "08_trait_statistics.json").exists()
        assert not (tmp_path / "08_anova_results.csv").exists()
        assert not (tmp_path / "08_heritability_results.csv").exists()
        assert not (tmp_path / "08_statistical_analysis_summary.json").exists()

    def test_files_generated_paths_correct(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that files_generated paths point to data/ directory."""
        step = StatisticalAnalysisStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        data_dir = tmp_path / "data"
        for file_path in result.files_generated:
            # All files should be under data/ directory
            assert file_path.parent == data_dir, (
                f"File {file_path.name} should be in data/ directory, "
                f"but is in {file_path.parent}"
            )
