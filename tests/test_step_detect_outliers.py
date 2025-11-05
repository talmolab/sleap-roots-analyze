"""Tests for DetectOutliersStep (Step 5).

Note: Currently tests the step structure and no-methods warning case.
Individual outlier detection methods have a parameter name mismatch between
the step implementation and the underlying functions that needs to be fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import warnings

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    OutlierDetectionConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import DetectOutliersStep


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
    }
    return pd.DataFrame(data)


@pytest.fixture
def config_no_methods():
    """Create config with NO outlier detection methods."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
        outlier_detection=OutlierDetectionConfig(
            traditional_methods=[],
            clustering_methods=[],
        ),
    )


@pytest.fixture
def prev_result(sample_data):
    """Create mock previous step result."""
    trait_cols = ["trait1", "trait2", "trait3"]
    return StepResult(
        data=sample_data,
        metadata={
            "valid_trait_names": trait_cols,
            "trait_names": trait_cols,
            "samples": len(sample_data),
        },
        files_generated=[],
    )


class TestDetectOutliersStepBasic:
    """Test basic functionality of DetectOutliersStep."""

    def test_step_initialization(self):
        """Test that step initializes with correct name and description."""
        step = DetectOutliersStep()

        assert step.step_name == "DetectOutliers"
        assert "outlier" in step.description.lower()

    def test_no_methods_configured_warning(
        self, sample_data, config_no_methods, prev_result, tmp_path
    ):
        """Test that warning is raised when no methods are configured."""
        step = DetectOutliersStep()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = step.execute(
                data=sample_data,
                config=config_no_methods,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

            # Check that a warning was raised
            assert len(w) == 1
            assert "No outlier detection methods configured" in str(w[0].message)

        # Should return with empty results
        assert isinstance(result, StepResult)
        assert result.metadata["methods_run"] == []
        assert result.metadata["outlier_results"] == {}
        assert result.metadata["total_methods"] == 0

    def test_data_unchanged_no_methods(
        self, sample_data, config_no_methods, prev_result, tmp_path
    ):
        """Test that data is unchanged when no methods configured."""
        step = DetectOutliersStep()

        original_data = sample_data.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = step.execute(
                data=sample_data,
                config=config_no_methods,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        pd.testing.assert_frame_equal(result.data, original_data)

    def test_no_files_generated(
        self, sample_data, config_no_methods, prev_result, tmp_path
    ):
        """Test that no files are generated (detection-only step)."""
        step = DetectOutliersStep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = step.execute(
                data=sample_data,
                config=config_no_methods,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        assert len(result.files_generated) == 0


class TestDetectOutliersStepMetadata:
    """Test metadata handling."""

    def test_metadata_structure(
        self, sample_data, config_no_methods, prev_result, tmp_path
    ):
        """Test that metadata has expected structure."""
        step = DetectOutliersStep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = step.execute(
                data=sample_data,
                config=config_no_methods,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        # Check all expected metadata keys
        assert "samples" in result.metadata
        assert "methods_run" in result.metadata
        assert "outlier_results" in result.metadata
        assert "outlier_counts" in result.metadata
        assert "total_methods" in result.metadata
        assert "trait_names" in result.metadata
        assert "valid_trait_names" in result.metadata

    def test_trait_names_propagation(
        self, sample_data, config_no_methods, prev_result, tmp_path
    ):
        """Test that trait names are propagated."""
        step = DetectOutliersStep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = step.execute(
                data=sample_data,
                config=config_no_methods,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        assert result.metadata["trait_names"] == ["trait1", "trait2", "trait3"]
        assert result.metadata["valid_trait_names"] == ["trait1", "trait2", "trait3"]

    def test_sample_count(
        self, sample_data, config_no_methods, prev_result, tmp_path
    ):
        """Test that sample count is recorded."""
        step = DetectOutliersStep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = step.execute(
                data=sample_data,
                config=config_no_methods,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        assert result.metadata["samples"] == 30


class TestDetectOutliersStepEdgeCases:
    """Test edge cases."""

    def test_empty_methods_lists(
        self, sample_data, config_no_methods, prev_result, tmp_path
    ):
        """Test with empty method lists."""
        step = DetectOutliersStep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = step.execute(
                data=sample_data,
                config=config_no_methods,
                run_dir=tmp_path,
                prev_result=prev_result,
            )

        assert result.metadata["outlier_counts"] == {}
        assert result.metadata["methods_run"] == []
