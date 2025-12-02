"""Tests for FilterHeritabilityStep (Step 9)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    HeritabilityConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import FilterHeritabilityStep


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(20)],
            "geno": ["A"] * 10 + ["B"] * 10,
            "rep": [1, 2] * 10,
            "high_h2_trait": np.random.randn(20) * 10 + 50,
            "med_h2_trait": np.random.randn(20) * 5 + 25,
            "low_h2_trait": np.random.randn(20) * 3 + 15,
        }
    )


@pytest.fixture
def config():
    """Create config."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
        heritability=HeritabilityConfig(threshold=0.3),
    )


@pytest.fixture
def prev_result(sample_data):
    """Previous result with heritability data."""
    return StepResult(
        data=sample_data,
        metadata={
            "trait_names": [
                "high_h2_trait",
                "med_h2_trait",
                "low_h2_trait",
            ],  # Primary key
            "valid_trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
            "heritability_results": {
                "high_h2_trait": {"heritability": 0.8},
                "med_h2_trait": {"heritability": 0.4},
                "low_h2_trait": {"heritability": 0.1},
            },
            "samples": 20,
        },
        files_generated=[],
    )


class TestFilterHeritabilityStepBasic:
    """Test basic functionality."""

    def test_step_initialization(self):
        """Test initialization."""
        step = FilterHeritabilityStep()
        assert step.step_name == "FilterHeritability"
        assert "heritability" in step.description.lower()

    def test_filtering_removes_low_h2_traits(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that low H2 traits are removed."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # low_h2_trait (H2=0.1) should be removed from trait list
        # Note: DataFrame still has all columns, only valid_trait_names changes
        assert "low_h2_trait" not in result.metadata["valid_trait_names"]
        assert "high_h2_trait" in result.metadata["valid_trait_names"]
        assert "med_h2_trait" in result.metadata["valid_trait_names"]

    def test_metadata_tracks_removed_traits(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test metadata tracks removed traits."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Removed traits tracked in metadata
        assert "low_h2_trait" not in result.metadata["valid_trait_names"]

    def test_valid_trait_names_updated(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test valid trait names updated."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        assert "low_h2_trait" not in result.metadata["valid_trait_names"]
        assert "high_h2_trait" in result.metadata["valid_trait_names"]


class TestFilterHeritabilityStepEdgeCases:
    """Test edge cases."""

    def test_all_traits_pass(self, sample_data, tmp_path):
        """Test when all traits pass threshold."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.0),  # All pass
        )

        prev_result = StepResult(
            data=sample_data,
            metadata={
                "trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "valid_trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "heritability_results": {
                    "high_h2_trait": {"heritability": 0.8},
                    "med_h2_trait": {"heritability": 0.4},
                    "low_h2_trait": {"heritability": 0.1},
                },
                "samples": 20,
            },
            files_generated=[],
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # All traits should remain
        assert len(result.metadata["valid_trait_names"]) == 3


class TestFilterHeritabilityStepDiagnostics:
    """Test diagnostic mode functionality."""

    def test_diagnostics_disabled_by_default(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that diagnostics are not generated by default."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Diagnostic results should not be in metadata
        assert "diagnostic_results" not in result.metadata

        # Diagnostic files should not exist
        assert not (tmp_path / "09_heritability_diagnostics.csv").exists()
        assert not (tmp_path / "figures" / "09_variance_decomposition.png").exists()
        assert not (tmp_path / "figures" / "09_removed_traits_boxplots.png").exists()

    def test_diagnostics_enabled_generates_files(
        self, sample_data, prev_result, tmp_path
    ):
        """Test that enabling diagnostics generates expected files."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(
                threshold=0.3, generate_diagnostics=True  # Enable diagnostics
            ),
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Diagnostic results should be in metadata
        assert "diagnostic_results" in result.metadata
        assert "comparison_df" in result.metadata["diagnostic_results"]

        # Diagnostic files should exist
        assert (tmp_path / "09_heritability_diagnostics.csv").exists()
        assert (tmp_path / "figures" / "09_variance_decomposition.png").exists()
        assert (tmp_path / "figures" / "09_removed_traits_boxplots.png").exists()

        # Check that diagnostic CSV has data
        diag_df = pd.read_csv(tmp_path / "09_heritability_diagnostics.csv")
        assert len(diag_df) == 3  # All 3 traits analyzed
        assert "trait" in diag_df.columns
        assert "heritability" in diag_df.columns
        assert "pct_var_between" in diag_df.columns

    def test_diagnostics_added_to_files_generated(
        self, sample_data, prev_result, tmp_path
    ):
        """Test that diagnostic files are added to files_generated list."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.3, generate_diagnostics=True),
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Check that diagnostic files are in files_generated
        file_names = [f.name for f in result.files_generated]
        assert "09_heritability_diagnostics.csv" in file_names
        assert "09_variance_decomposition.png" in file_names
        assert "09_removed_traits_boxplots.png" in file_names

    def test_diagnostics_not_generated_if_no_removed_traits(
        self, sample_data, tmp_path
    ):
        """Test that diagnostics are not generated if no traits are removed."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(
                threshold=0.0,  # All traits pass
                generate_diagnostics=True,
            ),
        )

        prev_result = StepResult(
            data=sample_data,
            metadata={
                "trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "valid_trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "heritability_results": {
                    "high_h2_trait": {"heritability": 0.8},
                    "med_h2_trait": {"heritability": 0.4},
                    "low_h2_trait": {"heritability": 0.1},
                },
                "samples": 20,
            },
            files_generated=[],
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # No traits removed, so diagnostics should not be generated
        assert "diagnostic_results" not in result.metadata
        assert not (tmp_path / "09_heritability_diagnostics.csv").exists()

    def test_diagnostics_metadata_structure(self, sample_data, prev_result, tmp_path):
        """Test that diagnostic results in metadata have expected structure."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.3, generate_diagnostics=True),
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        diag = result.metadata["diagnostic_results"]
        assert "comparison_df" in diag
        assert "diagnostic_csv" in diag
        assert "variance_plot" in diag
        assert "boxplot" in diag
        assert "traits_analyzed" in diag
        assert "traits_removed_plotted" in diag

        # Check values
        assert diag["traits_analyzed"] == 3
        assert diag["traits_removed_plotted"] == 1  # Only low_h2_trait removed

    def test_diagnostics_with_many_removed_traits(self, tmp_path):
        """Test that boxplots are limited to top 10 when many traits are removed."""
        # Create data with 15 traits
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(20)],
                "geno": ["A"] * 10 + ["B"] * 10,
                "rep": [1, 2] * 10,
            }
        )
        # Add 15 traits, all with low heritability
        for i in range(15):
            data[f"trait_{i}"] = np.random.randn(20) * 3 + 15

        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.8, generate_diagnostics=True),
        )

        # All traits have H2 < 0.8
        h2_results = {f"trait_{i}": {"heritability": 0.1} for i in range(15)}
        prev_result = StepResult(
            data=data,
            metadata={
                "trait_names": [f"trait_{i}" for i in range(15)],
                "valid_trait_names": [f"trait_{i}" for i in range(15)],
                "heritability_results": h2_results,
                "samples": 20,
            },
            files_generated=[],
        )

        step = FilterHeritabilityStep()
        result = step.execute(data, config, tmp_path, prev_result)

        # Check that only 10 traits were plotted
        assert result.metadata["diagnostic_results"]["traits_removed_plotted"] == 10
