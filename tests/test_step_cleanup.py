"""Tests for CleanupTraitsStep and ValidateCleanStep."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import (
    CleanupConfig,
    ColumnConfig,
    DataConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import CleanupTraitsStep, ValidateCleanStep


@pytest.fixture
def sample_data_with_issues():
    """Create sample data with various issues for cleanup."""
    np.random.seed(42)
    data = {
        "Barcode": [f"plant{i}" for i in range(20)],
        "geno": ["A"] * 10 + ["B"] * 10,
        "rep": [1, 2] * 10,
        # Good traits
        "good_trait1": np.random.randn(20) * 10 + 50,
        "good_trait2": np.random.randn(20) * 5 + 25,
        # Trait with too many zeros (11/20 = 55%)
        "zero_inflated": [0] * 11 + list(np.random.randn(9) * 2 + 10),
        # Trait with too many NaNs (8/20 = 40%)
        "high_nan_trait": [np.nan] * 8 + list(np.random.randn(12) * 3 + 15),
        # Trait that will have too few samples after sample removal
        "low_sample_trait": [1.0] * 5 + [np.nan] * 15,
    }
    df = pd.DataFrame(data)

    # Add NaNs to some samples to trigger sample removal
    # After bad traits removed, need samples with >50% NaNs in GOOD traits only
    # Make samples 15-17 have NaNs in both good traits (2/2 = 100% NaN)
    df.loc[15, ["good_trait1", "good_trait2"]] = np.nan
    df.loc[16, ["good_trait1", "good_trait2"]] = np.nan
    df.loc[17, ["good_trait1", "good_trait2"]] = np.nan

    return df


@pytest.fixture
def config():
    """Create test configuration."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
        cleanup=CleanupConfig(
            max_zeros_per_trait=0.5,  # Remove if >50% zeros
            max_nans_per_trait=0.3,  # Remove if >30% NaNs
            max_nan_fraction=0.5,  # Remove sample if >50% NaNs
            min_samples_per_trait=10,  # Remove if <10 valid samples
        ),
    )


@pytest.fixture
def prev_result(sample_data_with_issues):
    """Create mock previous step result."""
    trait_cols = [
        "good_trait1",
        "good_trait2",
        "zero_inflated",
        "high_nan_trait",
        "low_sample_trait",
    ]
    return StepResult(
        data=sample_data_with_issues,
        metadata={
            "trait_column_names": trait_cols,
            "metadata_column_names": ["Barcode", "geno", "rep"],
        },
        files_generated=[],
    )


def test_cleanup_traits_step_initialization():
    """Test CleanupTraitsStep initialization."""
    step = CleanupTraitsStep()
    assert step.step_name == "CleanupTraits"
    assert "optimal" in step.description.lower()


def test_cleanup_traits_step_removes_bad_traits(
    sample_data_with_issues, config, prev_result, tmp_path
):
    """Test that CleanupTraitsStep removes problematic traits."""
    step = CleanupTraitsStep()
    result = step.execute(
        data=sample_data_with_issues,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    # Should remove: zero_inflated (55% zeros), high_nan_trait (40% NaNs),
    # low_sample_trait (only 5 valid samples)
    assert result.metadata["traits_removed"] == 3
    assert result.metadata["traits_final"] == 2  # Only good_trait1, good_trait2

    # Check which traits remain
    assert "good_trait1" in result.metadata["valid_trait_names"]
    assert "good_trait2" in result.metadata["valid_trait_names"]
    assert "zero_inflated" not in result.metadata["valid_trait_names"]
    assert "high_nan_trait" not in result.metadata["valid_trait_names"]
    assert "low_sample_trait" not in result.metadata["valid_trait_names"]


def test_cleanup_traits_step_removes_bad_samples(
    sample_data_with_issues, config, prev_result, tmp_path
):
    """Test that CleanupTraitsStep ensures no NaN samples remain."""
    step = CleanupTraitsStep()
    result = step.execute(
        data=sample_data_with_issues,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    # After cleanup, all remaining samples should have valid data in trait columns
    # The cleanup process removes samples with too many NaNs
    trait_cols = result.metadata["valid_trait_names"]
    assert result.data[trait_cols].isna().sum().sum() == 0  # No NaNs in trait columns


def test_cleanup_traits_step_validates_no_nans(
    sample_data_with_issues, config, prev_result, tmp_path
):
    """Test that CleanupTraitsStep validates no NaNs remain."""
    step = CleanupTraitsStep()
    result = step.execute(
        data=sample_data_with_issues,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    # Should pass validation
    assert result.metadata["nan_validation_passed"] is True

    # Verify no NaNs in remaining trait columns
    trait_cols = result.metadata["valid_trait_names"]
    nan_count = result.data[trait_cols].isna().sum().sum()
    assert nan_count == 0


def test_cleanup_traits_step_generates_files(
    sample_data_with_issues, config, prev_result, tmp_path
):
    """Test that CleanupTraitsStep generates all expected files."""
    step = CleanupTraitsStep()
    result = step.execute(
        data=sample_data_with_issues,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    # Should generate 6 files
    assert len(result.files_generated) == 6

    # Check files exist
    assert (tmp_path / "01_data_traits_cleaned.csv").exists()
    assert (tmp_path / "01_removed_traits_detail.csv").exists()
    assert (tmp_path / "01_trait_cleanup_log.json").exists()
    assert (tmp_path / "02_data_samples_cleaned.csv").exists()
    assert (tmp_path / "02_removed_samples_detail.csv").exists()
    assert (tmp_path / "02_cleanup_log.json").exists()


def test_cleanup_traits_step_removed_traits_detail(
    sample_data_with_issues, config, prev_result, tmp_path
):
    """Test that removed traits detail file contains correct information."""
    step = CleanupTraitsStep()
    step.execute(
        data=sample_data_with_issues,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    removed_traits = pd.read_csv(tmp_path / "01_removed_traits_detail.csv")
    assert len(removed_traits) == 3

    # Check that the right traits were removed
    assert "zero_inflated" in removed_traits["trait"].values
    assert "high_nan_trait" in removed_traits["trait"].values
    assert "low_sample_trait" in removed_traits["trait"].values


def test_validate_clean_step_initialization():
    """Test ValidateCleanStep initialization."""
    step = ValidateCleanStep()
    assert step.step_name == "ValidateClean"
    assert "validate" in step.description.lower()


def test_validate_clean_step_passes_with_clean_data(config, tmp_path):
    """Test ValidateCleanStep with clean data (no NaNs)."""
    # Create clean data
    df_clean = pd.DataFrame(
        {
            "Barcode": ["p1", "p2", "p3"],
            "geno": ["A", "B", "A"],
            "rep": [1, 1, 2],
            "trait1": [10.5, 12.3, 11.8],
            "trait2": [5.2, 6.1, 5.8],
        }
    )

    prev_result = StepResult(
        data=df_clean,
        metadata={"valid_trait_names": ["trait1", "trait2"]},
        files_generated=[],
    )

    step = ValidateCleanStep()
    result = step.execute(
        data=df_clean, config=config, run_dir=tmp_path, prev_result=prev_result
    )

    # Should pass validation
    assert result.metadata["validation_passed"] is True
    assert result.metadata["total_nans_in_traits"] == 0


def test_validate_clean_step_fails_with_nans(config, tmp_path):
    """Test ValidateCleanStep fails when NaNs are present."""
    # Create data with NaNs
    df_dirty = pd.DataFrame(
        {
            "Barcode": ["p1", "p2", "p3"],
            "geno": ["A", "B", "A"],
            "rep": [1, 1, 2],
            "trait1": [10.5, np.nan, 11.8],
            "trait2": [5.2, 6.1, 5.8],
        }
    )

    prev_result = StepResult(
        data=df_dirty,
        metadata={"valid_trait_names": ["trait1", "trait2"]},
        files_generated=[],
    )

    step = ValidateCleanStep()
    with pytest.raises(ValueError, match="Validation failed.*NaN values"):
        step.execute(
            data=df_dirty, config=config, run_dir=tmp_path, prev_result=prev_result
        )


def test_validate_clean_step_generates_report(config, tmp_path):
    """Test that ValidateCleanStep generates validation report."""
    df_clean = pd.DataFrame(
        {
            "Barcode": ["p1", "p2", "p3"],
            "geno": ["A", "B", "A"],
            "trait1": [10.5, 12.3, 11.8],
        }
    )

    prev_result = StepResult(
        data=df_clean, metadata={"valid_trait_names": ["trait1"]}, files_generated=[]
    )

    step = ValidateCleanStep()
    result = step.execute(
        data=df_clean, config=config, run_dir=tmp_path, prev_result=prev_result
    )

    # Should generate validation report
    assert len(result.files_generated) == 1
    assert (tmp_path / "03_validation_report.json").exists()


def test_full_cleanup_and_validate_workflow(
    sample_data_with_issues, config, prev_result, tmp_path
):
    """Test the full workflow of cleanup followed by validation."""
    # Step 2: Cleanup
    cleanup_step = CleanupTraitsStep()
    cleanup_result = cleanup_step.execute(
        data=sample_data_with_issues,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    # Step 3: Validate
    validate_step = ValidateCleanStep()
    validate_result = validate_step.execute(
        data=cleanup_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=cleanup_result,
    )

    # Validation should pass
    assert validate_result.metadata["validation_passed"] is True
    assert validate_result.metadata["total_nans_in_traits"] == 0
