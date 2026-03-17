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

    # Check which traits remain (trait names are now sanitized)
    assert "Good Trait1" in result.metadata["valid_trait_names"]
    assert "Good Trait2" in result.metadata["valid_trait_names"]
    assert "Zero Inflated" not in result.metadata["valid_trait_names"]
    assert "High Nan Trait" not in result.metadata["valid_trait_names"]
    assert "Low Sample Trait" not in result.metadata["valid_trait_names"]


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

    # Check that the right traits were removed (trait names are now sanitized)
    assert "Zero Inflated" in removed_traits["trait"].values
    assert "High Nan Trait" in removed_traits["trait"].values
    assert "Low Sample Trait" in removed_traits["trait"].values


def test_removed_samples_csv_contains_correct_rows(
    sample_data_with_issues, config, prev_result, tmp_path
):
    """Regression test: 02_removed_samples_detail.csv must contain rows when samples removed.

    Catches the key mismatch bug that caused the file to always be header-only.
    The fixture has rows 15-17 with 100% NaN in both good traits, so 3 samples
    should be removed and recorded.
    """
    step = CleanupTraitsStep()
    step.execute(
        data=sample_data_with_issues,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    removed_samples = pd.read_csv(tmp_path / "02_removed_samples_detail.csv")

    # Must have exactly 3 removed samples (rows 15, 16, 17)
    assert len(removed_samples) == 3, (
        f"Expected 3 removed sample rows, got {len(removed_samples)} "
        "(empty file indicates key mismatch bug)"
    )

    # Must have all required columns
    required_cols = {
        "sample_index",
        "barcode",
        "genotype",
        "rep",
        "nan_count",
        "nan_fraction",
        "nan_traits",
        "removal_reason",
    }
    assert required_cols.issubset(
        set(removed_samples.columns)
    ), f"Missing columns: {required_cols - set(removed_samples.columns)}"

    # Barcodes of removed samples must match rows 15-17
    assert set(removed_samples["barcode"]) == {"plant15", "plant16", "plant17"}


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


def test_cleanup_traits_step_with_custom_replacements(config, tmp_path):
    """Test that CleanupTraitsStep applies custom_replacements to trait names."""
    # Create test data with wheat-specific terminology (15 samples to pass min_samples_per_trait=10)
    df = pd.DataFrame(
        {
            "Barcode": [f"p{i}" for i in range(15)],
            "geno": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "rep": [1, 2, 3, 4, 5] * 3,
            "crown_length_mm": [10.5 + i * 0.1 for i in range(15)],
            "crown_count": [5 + (i % 3) for i in range(15)],
            "primary_root_angle_deg": [45.0 + i * 0.5 for i in range(15)],
        }
    )

    prev_result = StepResult(
        data=df,
        metadata={
            "trait_column_names": [
                "crown_length_mm",
                "crown_count",
                "primary_root_angle_deg",
            ],
            "metadata_column_names": ["Barcode", "geno", "rep"],
        },
        files_generated=[],
    )

    # Add custom replacements to config (wheat terminology)
    config.cleanup.custom_replacements = {"crown": "seminal", "primary": "main"}

    step = CleanupTraitsStep()
    result = step.execute(
        data=df, config=config, run_dir=tmp_path, prev_result=prev_result
    )

    # Check that trait names were transformed with custom replacements
    assert "Seminal Length (mm)" in result.metadata["trait_names"]
    assert "Seminal Count" in result.metadata["trait_names"]
    assert "Main Root Angle (°)" in result.metadata["trait_names"]

    # Check that mapping is tracked in metadata
    assert result.metadata["trait_name_mapping"] is not None
    mapping = result.metadata["trait_name_mapping"]
    assert "crown_length_mm" in mapping
    assert mapping["crown_length_mm"] == "Seminal Length (mm)"
    assert mapping["crown_count"] == "Seminal Count"
    assert mapping["primary_root_angle_deg"] == "Main Root Angle (°)"

    # Check that DataFrame has sanitized column names
    assert "Seminal Length (mm)" in result.data.columns
    assert "Seminal Count" in result.data.columns
    assert "Main Root Angle (°)" in result.data.columns
