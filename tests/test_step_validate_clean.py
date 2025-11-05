"""Tests for ValidateCleanStep (Step 3)."""

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
from sleap_roots_analyze.pipeline.steps import ValidateCleanStep


@pytest.fixture
def clean_data():
    """Create clean sample data with no NaN values."""
    np.random.seed(42)
    data = {
        "Barcode": [f"plant{i}" for i in range(10)],
        "geno": ["A"] * 5 + ["B"] * 5,
        "rep": [1, 2, 1, 2, 1] * 2,
        "trait1": np.random.randn(10) * 10 + 50,
        "trait2": np.random.randn(10) * 5 + 25,
        "trait3": np.random.randn(10) * 3 + 15,
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_with_trait_nans():
    """Create data with NaN values in trait columns."""
    np.random.seed(42)
    data = {
        "Barcode": [f"plant{i}" for i in range(10)],
        "geno": ["A"] * 5 + ["B"] * 5,
        "rep": [1, 2, 1, 2, 1] * 2,
        "trait1": [np.nan, 2.0, 3.0] + list(np.random.randn(7)),
        "trait2": np.random.randn(10) * 5 + 25,
        "trait3": [1.0, np.nan, 3.0] + list(np.random.randn(7)),
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_with_metadata_nans():
    """Create data with NaN values in metadata columns only."""
    np.random.seed(42)
    data = {
        "Barcode": [f"plant{i}" if i < 9 else None for i in range(10)],
        "geno": ["A"] * 5 + ["B"] * 5,
        "rep": [1, 2, 1, 2, 1] * 2,
        "trait1": np.random.randn(10) * 10 + 50,
        "trait2": np.random.randn(10) * 5 + 25,
        "trait3": np.random.randn(10) * 3 + 15,
    }
    return pd.DataFrame(data)


@pytest.fixture
def config():
    """Create test configuration."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
    )


@pytest.fixture
def prev_result_clean(clean_data):
    """Create mock previous step result with clean data."""
    trait_cols = ["trait1", "trait2", "trait3"]
    return StepResult(
        data=clean_data,
        metadata={
            "valid_trait_names": trait_cols,
            "trait_names": trait_cols,
            "samples": len(clean_data),
        },
        files_generated=[],
    )


@pytest.fixture
def prev_result_with_nans(data_with_trait_nans):
    """Create mock previous step result with NaNs in traits."""
    trait_cols = ["trait1", "trait2", "trait3"]
    return StepResult(
        data=data_with_trait_nans,
        metadata={
            "valid_trait_names": trait_cols,
            "trait_names": trait_cols,
            "samples": len(data_with_trait_nans),
        },
        files_generated=[],
    )


class TestValidateCleanStepBasic:
    """Test basic functionality of ValidateCleanStep."""

    def test_step_initialization(self):
        """Test that step initializes with correct name and description."""
        step = ValidateCleanStep()

        assert step.step_name == "ValidateClean"
        assert "validate" in step.description.lower()
        assert "nan" in step.description.lower()

    def test_validation_passes_with_clean_data(
        self, clean_data, config, prev_result_clean, tmp_path
    ):
        """Test that validation passes when data has no NaN values."""
        step = ValidateCleanStep()

        result = step.execute(
            data=clean_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_clean,
        )

        # Check result structure
        assert isinstance(result, StepResult)
        assert result.data.equals(clean_data)

        # Check metadata
        assert result.metadata["validation_passed"] is True
        assert result.metadata["total_nans_in_traits"] == 0
        assert result.metadata["samples"] == 10
        assert result.metadata["trait_columns"] == 3
        assert result.metadata["trait_names"] == ["trait1", "trait2", "trait3"]

        # Check files generated
        assert len(result.files_generated) == 1
        assert (tmp_path / "03_validation_report.json").exists()

    def test_validation_report_content(
        self, clean_data, config, prev_result_clean, tmp_path
    ):
        """Test that validation report contains correct information."""
        step = ValidateCleanStep()

        step.execute(
            data=clean_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_clean,
        )

        # Read validation report
        with open(tmp_path / "03_validation_report.json") as f:
            report = json.load(f)

        # JSON booleans may be saved as strings due to default=str in save_json
        assert report["validation_passed"] in [True, "True"]
        assert report["total_samples"] == 10
        assert report["total_trait_columns"] == 3
        assert report["total_metadata_columns"] == 3  # Barcode, geno, rep
        assert report["nan_values_in_traits"] == 0
        assert report["trait_nan_counts"] == {}

    def test_data_unchanged_after_validation(
        self, clean_data, config, prev_result_clean, tmp_path
    ):
        """Test that validation step doesn't modify the data."""
        step = ValidateCleanStep()

        original_data = clean_data.copy()

        result = step.execute(
            data=clean_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_clean,
        )

        # Data should be unchanged
        pd.testing.assert_frame_equal(result.data, original_data)


class TestValidateCleanStepFailures:
    """Test validation failure scenarios."""

    def test_validation_fails_with_trait_nans(
        self, data_with_trait_nans, config, prev_result_with_nans, tmp_path
    ):
        """Test that validation fails when NaN values exist in traits."""
        step = ValidateCleanStep()

        with pytest.raises(ValueError) as excinfo:
            step.execute(
                data=data_with_trait_nans,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result_with_nans,
            )

        # Check error message
        error_msg = str(excinfo.value)
        assert "Validation failed" in error_msg
        assert "NaN values found" in error_msg
        assert "trait1" in error_msg or "trait3" in error_msg

    def test_validation_report_shows_nan_traits(
        self, data_with_trait_nans, config, prev_result_with_nans, tmp_path
    ):
        """Test that validation report identifies traits with NaNs."""
        step = ValidateCleanStep()

        try:
            step.execute(
                data=data_with_trait_nans,
                config=config,
                run_dir=tmp_path,
                prev_result=prev_result_with_nans,
            )
        except ValueError:
            pass  # Expected

        # Read validation report
        with open(tmp_path / "03_validation_report.json") as f:
            report = json.load(f)

        # JSON booleans may be saved as strings due to default=str in save_json
        assert report["validation_passed"] in [False, "False"]
        assert report["nan_values_in_traits"] > 0
        assert "trait1" in report["trait_nan_counts"]
        assert "trait3" in report["trait_nan_counts"]
        assert report["trait_nan_counts"]["trait1"] == 1
        assert report["trait_nan_counts"]["trait3"] == 1

    def test_validation_passes_with_metadata_nans(
        self, data_with_metadata_nans, config, tmp_path
    ):
        """Test that validation passes even with NaN values in metadata."""
        step = ValidateCleanStep()

        trait_cols = ["trait1", "trait2", "trait3"]
        prev_result = StepResult(
            data=data_with_metadata_nans,
            metadata={
                "valid_trait_names": trait_cols,
                "trait_names": trait_cols,
                "samples": len(data_with_metadata_nans),
            },
            files_generated=[],
        )

        result = step.execute(
            data=data_with_metadata_nans,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Validation should pass (NaNs in metadata are OK)
        assert result.metadata["validation_passed"] is True
        assert result.metadata["total_nans_in_traits"] == 0
        assert result.metadata["total_nans_in_metadata"] > 0


class TestValidateCleanStepEdgeCases:
    """Test edge cases for ValidateCleanStep."""

    def test_empty_dataframe(self, config, tmp_path):
        """Test validation with empty DataFrame."""
        empty_data = pd.DataFrame(
            columns=["Barcode", "geno", "rep", "trait1", "trait2"]
        )
        prev_result = StepResult(
            data=empty_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "trait_names": ["trait1", "trait2"],
                "samples": 0,
            },
            files_generated=[],
        )

        step = ValidateCleanStep()

        result = step.execute(
            data=empty_data, config=config, run_dir=tmp_path, prev_result=prev_result
        )

        # Should pass validation with 0 samples
        assert result.metadata["validation_passed"] is True
        assert result.metadata["samples"] == 0
        assert result.metadata["total_nans_in_traits"] == 0

    def test_single_sample(self, config, tmp_path):
        """Test validation with single sample."""
        single_data = pd.DataFrame(
            {
                "Barcode": ["plant1"],
                "geno": ["A"],
                "rep": [1],
                "trait1": [10.5],
                "trait2": [5.2],
            }
        )
        prev_result = StepResult(
            data=single_data,
            metadata={
                "valid_trait_names": ["trait1", "trait2"],
                "trait_names": ["trait1", "trait2"],
                "samples": 1,
            },
            files_generated=[],
        )

        step = ValidateCleanStep()

        result = step.execute(
            data=single_data, config=config, run_dir=tmp_path, prev_result=prev_result
        )

        assert result.metadata["validation_passed"] is True
        assert result.metadata["samples"] == 1
        assert result.metadata["trait_columns"] == 2

    def test_single_trait(self, config, tmp_path):
        """Test validation with single trait column."""
        np.random.seed(42)
        single_trait_data = pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(5)],
                "geno": ["A", "A", "B", "B", "A"],
                "rep": [1, 2, 1, 2, 3],
                "trait1": np.random.randn(5) * 10 + 50,
            }
        )
        prev_result = StepResult(
            data=single_trait_data,
            metadata={
                "valid_trait_names": ["trait1"],
                "trait_names": ["trait1"],
                "samples": 5,
            },
            files_generated=[],
        )

        step = ValidateCleanStep()

        result = step.execute(
            data=single_trait_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        assert result.metadata["validation_passed"] is True
        assert result.metadata["trait_columns"] == 1
        assert result.metadata["trait_names"] == ["trait1"]

    def test_all_metadata_no_traits(self, config, tmp_path):
        """Test validation when all columns are metadata (no traits)."""
        metadata_only = pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(5)],
                "geno": ["A", "A", "B", "B", "A"],
                "rep": [1, 2, 1, 2, 3],
            }
        )
        prev_result = StepResult(
            data=metadata_only,
            metadata={
                "valid_trait_names": [],
                "trait_names": [],
                "samples": 5,
            },
            files_generated=[],
        )

        step = ValidateCleanStep()

        result = step.execute(
            data=metadata_only,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result,
        )

        # Should pass validation with 0 traits
        assert result.metadata["validation_passed"] is True
        assert result.metadata["trait_columns"] == 0
        assert result.metadata["total_nans_in_traits"] == 0


class TestValidateCleanStepMetadataPropagation:
    """Test metadata propagation through the validation step."""

    def test_trait_names_propagation(
        self, clean_data, config, prev_result_clean, tmp_path
    ):
        """Test that trait names are correctly propagated."""
        step = ValidateCleanStep()

        result = step.execute(
            data=clean_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_clean,
        )

        # Check that both trait_names and valid_trait_names are present
        assert "trait_names" in result.metadata
        assert "valid_trait_names" in result.metadata
        assert result.metadata["trait_names"] == result.metadata["valid_trait_names"]
        assert result.metadata["trait_names"] == ["trait1", "trait2", "trait3"]

    def test_sample_count_propagation(
        self, clean_data, config, prev_result_clean, tmp_path
    ):
        """Test that sample count is correctly recorded."""
        step = ValidateCleanStep()

        result = step.execute(
            data=clean_data,
            config=config,
            run_dir=tmp_path,
            prev_result=prev_result_clean,
        )

        assert result.metadata["samples"] == len(clean_data)
        assert result.metadata["samples"] == 10
