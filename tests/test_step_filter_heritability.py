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
