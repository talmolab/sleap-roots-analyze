"""Tests for ExploratoryAnalysisStep (Step 4)."""

from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    CleanupConfig,
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
    VisualizationConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import ExploratoryAnalysisStep

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 30
    data = {
        "Barcode": [f"plant{i}" for i in range(n_samples)],
        "geno": (["A"] * 10 + ["B"] * 10 + ["C"] * 10),
        "rep": [1, 2, 3] * 10,
        "trait1": np.random.randn(n_samples) * 10 + 50,
        "trait2": np.random.randn(n_samples) * 5 + 25,
        "trait3": np.random.randn(n_samples) * 3 + 15,
        "trait4": np.random.randn(n_samples) * 2 + 8,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_many_traits():
    """Create sample data with >16 traits for batched visualization testing."""
    np.random.seed(42)
    n_samples = 30
    data = {
        "Barcode": [f"plant{i}" for i in range(n_samples)],
        "geno": (["A"] * 10 + ["B"] * 10 + ["C"] * 10),
        "rep": [1, 2, 3] * 10,
    }
    # Add 20 traits to trigger batched plots
    for i in range(20):
        data[f"trait{i + 1}"] = np.random.randn(n_samples) * 10 + 50
    return pd.DataFrame(data)


@pytest.fixture
def config():
    """Create test configuration."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
        cleanup=CleanupConfig(
            max_zeros_per_trait=0.5,
            max_nans_per_trait=0.3,
            min_samples_per_trait=10,
        ),
        visualization=VisualizationConfig(dpi=100),  # Lower DPI for faster tests
    )


@pytest.fixture
def prev_result(sample_data):
    """Create mock previous step result."""
    trait_cols = ["trait1", "trait2", "trait3", "trait4"]
    return StepResult(
        data=sample_data,
        metadata={
            "valid_trait_names": trait_cols,
            "trait_names": trait_cols,
            "samples": len(sample_data),
            "cleanup_log": {},
        },
        files_generated=[],
    )


@pytest.fixture
def prev_result_many_traits(sample_data_many_traits):
    """Create mock previous step result with many traits."""
    trait_cols = [f"trait{i + 1}" for i in range(20)]
    return StepResult(
        data=sample_data_many_traits,
        metadata={
            "valid_trait_names": trait_cols,
            "trait_names": trait_cols,
            "samples": len(sample_data_many_traits),
            "cleanup_log": {},
        },
        files_generated=[],
    )


class TestExploratoryAnalysisStepBasic:
    """Test basic functionality of ExploratoryAnalysisStep."""

    def test_step_initialization(self):
        """Test that step initializes with correct name and description."""
        step = ExploratoryAnalysisStep()

        assert step.step_name == "ExploratoryAnalysis"
        assert (
            "eda" in step.description.lower()
            or "exploratory" in step.description.lower()
        )

    def test_basic_execution(self, sample_data, config, prev_result, tmp_path):
        """Test basic step execution generates outputs."""
        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert result.data.equals(sample_data)

        # Check metadata
        assert result.metadata["samples"] == 30
        assert result.metadata["traits"] == 4
        assert result.metadata["figures_generated"] > 0
        assert "statistics" in result.metadata
        assert "trait_names" in result.metadata

        # Check files generated
        assert len(result.files_generated) > 0

    def test_figures_directory_created(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that figures directory is created."""
        step = ExploratoryAnalysisStep()

        step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        figures_dir = tmp_path / "figures"
        assert figures_dir.exists()
        assert figures_dir.is_dir()

    def test_data_unchanged(self, sample_data, config, prev_result, tmp_path):
        """Test that data is not modified by exploratory analysis."""
        step = ExploratoryAnalysisStep()

        original_data = sample_data.copy()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        pd.testing.assert_frame_equal(result.data, original_data)


class TestExploratoryAnalysisStatistics:
    """Test statistical outputs of ExploratoryAnalysisStep."""

    def test_statistics_file_created(self, sample_data, config, prev_result, tmp_path):
        """Test that trait statistics JSON is created."""
        step = ExploratoryAnalysisStep()

        step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        stats_file = tmp_path / "trait_statistics.json"
        assert stats_file.exists()

    def test_statistics_content(self, sample_data, config, prev_result, tmp_path):
        """Test that statistics contain expected metrics."""
        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        stats = result.metadata["statistics"]

        # Check that statistics exist for each trait
        assert len(stats) == 4
        for trait in ["trait1", "trait2", "trait3", "trait4"]:
            assert trait in stats
            assert "mean" in stats[trait]
            assert "std" in stats[trait]
            assert "min" in stats[trait]
            assert "max" in stats[trait]


class TestExploratoryAnalysisFigures:
    """Test figure generation in ExploratoryAnalysisStep."""

    def test_figures_saved_as_png(self, sample_data, config, prev_result, tmp_path):
        """Test that figures are saved as PNG files."""
        step = ExploratoryAnalysisStep()

        step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        figures_dir = tmp_path / "figures"
        png_files = list(figures_dir.glob("*.png"))

        assert len(png_files) > 0
        for png_file in png_files:
            assert png_file.suffix == ".png"

    def test_batched_plots_not_created_for_few_traits(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that batched plots are NOT created when traits <= 16."""
        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should not have batched plots for only 4 traits
        figure_names = result.metadata["figure_names"]
        batch_figures = [name for name in figure_names if "batch" in name]

        assert len(batch_figures) == 0

    def test_batched_plots_created_for_many_traits(
        self, sample_data_many_traits, config, prev_result_many_traits, tmp_path
    ):
        """Test that batched plots ARE created when traits > 16."""
        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=sample_data_many_traits,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_many_traits,
        )

        # Should have batched plots for 20 traits
        figure_names = result.metadata["figure_names"]
        batch_figures = [name for name in figure_names if "batch" in name]

        assert len(batch_figures) > 0

        # Check for both histogram and boxplot batches
        hist_batches = [name for name in batch_figures if "histogram" in name]
        box_batches = [name for name in batch_figures if "boxplot" in name]

        assert len(hist_batches) > 0
        assert len(box_batches) > 0


class TestExploratoryAnalysisEdgeCases:
    """Test edge cases for ExploratoryAnalysisStep."""

    def test_single_trait(self, config, tmp_path):
        """Test with single trait."""
        np.random.seed(42)
        single_trait_data = pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(10)],
                "geno": ["A"] * 5 + ["B"] * 5,
                "rep": [1, 2] * 5,
                "trait1": np.random.randn(10) * 10 + 50,
            }
        )

        prev_result = StepResult(
            data=single_trait_data,
            metadata={
                "valid_trait_names": ["trait1"],
                "trait_names": ["trait1"],
                "samples": 10,
                "cleanup_log": {},
            },
            files_generated=[],
        )

        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=single_trait_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert result.metadata["traits"] == 1
        assert result.metadata["samples"] == 10
        assert len(result.files_generated) > 0

    def test_single_genotype(self, config, tmp_path):
        """Test with single genotype."""
        np.random.seed(42)
        single_geno_data = pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(10)],
                "geno": ["A"] * 10,
                "rep": [1, 2, 3] * 3 + [1],
                "trait1": np.random.randn(10) * 10 + 50,
                "trait2": np.random.randn(10) * 5 + 25,
            }
        )

        prev_result = StepResult(
            data=single_geno_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "trait_names": ["trait1", "trait2"],
                "samples": 10,
                "cleanup_log": {},
            },
            files_generated=[],
        )

        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=single_geno_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should still generate figures even with single genotype
        assert result.metadata["traits"] == 2
        assert result.metadata["samples"] == 10
        assert len(result.files_generated) > 0


class TestExploratoryAnalysisMetadataPropagation:
    """Test metadata propagation through ExploratoryAnalysisStep."""

    def test_trait_names_propagation(self, sample_data, config, prev_result, tmp_path):
        """Test that trait names are correctly propagated."""
        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert "trait_names" in result.metadata
        assert "valid_trait_names" in result.metadata
        assert result.metadata["trait_names"] == result.metadata["valid_trait_names"]
        assert result.metadata["trait_names"] == [
            "trait1",
            "trait2",
            "trait3",
            "trait4",
        ]

    def test_sample_count_propagation(self, sample_data, config, prev_result, tmp_path):
        """Test that sample count is correctly recorded."""
        step = ExploratoryAnalysisStep()

        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert result.metadata["samples"] == len(sample_data)
        assert result.metadata["samples"] == 30

    def test_cleanup_log_propagation(self, sample_data, config, tmp_path):
        """Test that cleanup log from previous step is used."""
        cleanup_log = {
            "removed_samples": 5,
            "removed_traits": ["bad_trait1", "bad_trait2"],
        }

        prev_result = StepResult(
            data=sample_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2", "trait3", "trait4"],
                "trait_names": ["trait1", "trait2", "trait3", "trait4"],
                "samples": 30,
                "cleanup_log": cleanup_log,
            },
            files_generated=[],
        )

        step = ExploratoryAnalysisStep()

        # Should execute without error using cleanup_log
        result = step.execute(
            data=sample_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert result.metadata["samples"] == 30
