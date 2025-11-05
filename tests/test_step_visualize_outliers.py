"""Tests for VisualizeOutliersStep (Step 6)."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import warnings

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
    VisualizationConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import VisualizeOutliersStep

matplotlib.use("Agg")


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(20)],
            "geno": ["A"] * 10 + ["B"] * 10,
            "rep": [1, 2] * 10,
            "trait1": np.random.randn(20) * 10 + 50,
            "trait2": np.random.randn(20) * 5 + 25,
        }
    )


@pytest.fixture
def config():
    """Create config."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
        visualization=VisualizationConfig(dpi=100),
    )


@pytest.fixture
def prev_result_no_outliers(sample_data):
    """Previous result with no methods run."""
    return StepResult(
        data=sample_data,
        metadata={
            "valid_trait_names": ["trait1", "trait2"],
            "methods_run": [],
            "outlier_results": {},
            "samples": 20,
        },
        files_generated=[],
    )


class TestVisualizeOutliersStepBasic:
    """Test basic functionality."""

    def test_step_initialization(self):
        """Test initialization."""
        step = VisualizeOutliersStep()
        assert step.step_name == "VisualizeOutliers"
        assert "visualiz" in step.description.lower()

    def test_no_methods_warning(
        self, sample_data, config, prev_result_no_outliers, tmp_path
    ):
        """Test warning when no methods were run."""
        step = VisualizeOutliersStep()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = step.execute(
                sample_data, config, tmp_path, prev_result_no_outliers
            )
            assert any(
                "No outlier detection methods were run" in str(w_msg.message)
                for w_msg in w
            )

        assert len(result.files_generated) == 0

    def test_data_unchanged(
        self, sample_data, config, prev_result_no_outliers, tmp_path
    ):
        """Test data unchanged."""
        step = VisualizeOutliersStep()
        original = sample_data.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = step.execute(
                sample_data, config, tmp_path, prev_result_no_outliers
            )

        pd.testing.assert_frame_equal(result.data, original)


class TestVisualizeOutliersStepMetadata:
    """Test metadata."""

    def test_metadata_propagation(
        self, sample_data, config, prev_result_no_outliers, tmp_path
    ):
        """Test metadata is propagated."""
        step = VisualizeOutliersStep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = step.execute(
                sample_data, config, tmp_path, prev_result_no_outliers
            )

        assert "valid_trait_names" in result.metadata
        assert result.metadata["valid_trait_names"] == ["trait1", "trait2"]
