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
        """Test heritability results file is created."""
        step = StatisticalAnalysisStep()
        step.execute(sample_data, config, tmp_path, prev_result)

        assert (tmp_path / "08_heritability_results.csv").exists()

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
