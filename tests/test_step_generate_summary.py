"""Tests for GenerateSummaryStep (Step 10)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import GenerateSummaryStep


@pytest.fixture
def sample_data():
    """Create sample data with standardized column names (as after CleanupTraitsStep)."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(15)],
            "Genotype": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "Replicate": [1, 2, 3] * 5,
            "trait1": np.random.randn(15) * 10 + 50,
            "trait2": np.random.randn(15) * 5 + 25,
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
            "valid_trait_names": ["trait1", "trait2"],
            "samples": 15,
            "removed_samples": 5,
            "removed_traits": ["bad_trait1"],
        },
        files_generated=["file1.csv", "file2.png"],
    )


class TestGenerateSummaryStepBasic:
    """Test basic functionality."""

    def test_step_initialization(self):
        """Test initialization."""
        step = GenerateSummaryStep()
        assert step.step_name == "GenerateSummary"
        assert "summary" in step.description.lower()

    def test_summary_file_created(self, sample_data, config, prev_result, tmp_path):
        """Test summary JSON is created."""
        step = GenerateSummaryStep()
        step.execute(sample_data, config, tmp_path, prev_result)

        assert (tmp_path / "10_pipeline_summary.json").exists()

    def test_summary_content(self, sample_data, config, prev_result, tmp_path):
        """Test summary contains expected information."""
        step = GenerateSummaryStep()
        step.execute(sample_data, config, tmp_path, prev_result)

        with open(tmp_path / "10_pipeline_summary.json") as f:
            summary = json.load(f)

        # Should contain basic stats
        assert (
            "final_n_samples" in summary
            or "samples" in summary
            or "final_data" in summary
        )
        assert (
            "final_n_traits" in summary
            or "traits" in summary
            or "final_data" in summary
        )

    def test_data_unchanged(self, sample_data, config, prev_result, tmp_path):
        """Test data unchanged."""
        step = GenerateSummaryStep()
        original = sample_data.copy()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        pd.testing.assert_frame_equal(result.data, original)


class TestGenerateSummaryStepMetadata:
    """Test metadata."""

    def test_metadata_propagation(self, sample_data, config, prev_result, tmp_path):
        """Test metadata has summary structure."""
        step = GenerateSummaryStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Summary step has different metadata structure
        assert "summary" in result.metadata or "pipeline_complete" in result.metadata

    def test_files_generated(self, sample_data, config, prev_result, tmp_path):
        """Test files are tracked."""
        step = GenerateSummaryStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        assert len(result.files_generated) > 0
        assert any("10_pipeline_summary.json" in str(f) for f in result.files_generated)
