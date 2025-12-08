"""Tests for LoadCrossPlatformDataStep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig


@pytest.fixture
def exp1_csv(tmp_path):
    """Create experiment 1 CSV file for testing."""
    data = {
        "Geno": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "Rep": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "trait1": [10.5, 11.2, 10.8, 12.3, 11.9, 12.5, 9.8, 10.2, 10.0],
        "trait2": [5.2, 5.5, 5.3, 6.1, 5.9, 6.3, 4.8, 5.0, 4.9],
        "trait3": [15.0, 15.5, 15.2, 16.0, 15.8, 16.2, 14.5, 14.8, 14.6],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "exp1_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def exp2_csv(tmp_path):
    """Create experiment 2 CSV file for testing."""
    data = {
        "geno": ["A", "A", "B", "B", "B", "C", "C"],
        "rep": [1, 2, 1, 2, 3, 1, 2],
        "trait_a": [20.1, 20.5, 22.0, 21.8, 22.2, 19.5, 19.8],
        "trait_b": [10.2, 10.5, 11.0, 10.8, 11.2, 9.8, 10.0],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "exp2_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def cross_platform_config(exp1_csv, exp2_csv):
    """Create a test CrossPlatformConfig."""
    return CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Cylinder",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Turface",
        exp2_genotype_col="geno",
        correlation_method="spearman",
        min_samples_per_genotype=2,
    )


def test_load_cross_platform_data_step_initialization():
    """Test LoadCrossPlatformDataStep initialization."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    assert step.step_name == "LoadCrossPlatformData"
    assert "Load and align" in step.description


def test_load_cross_platform_data_step_execute(cross_platform_config, tmp_path):
    """Test LoadCrossPlatformDataStep execution."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check that data was loaded
    assert result.data is not None
    assert "exp1_df" in result.data
    assert "exp2_df" in result.data
    assert "common_genotypes" in result.data

    # Check data types
    assert isinstance(result.data["exp1_df"], pd.DataFrame)
    assert isinstance(result.data["exp2_df"], pd.DataFrame)
    assert isinstance(result.data["common_genotypes"], list)

    # Check common genotypes
    assert len(result.data["common_genotypes"]) == 3  # A, B, C
    assert set(result.data["common_genotypes"]) == {"A", "B", "C"}


def test_load_cross_platform_data_step_metadata(cross_platform_config, tmp_path):
    """Test that metadata is correctly populated."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check metadata
    assert "exp1_samples" in result.metadata
    assert "exp2_samples" in result.metadata
    assert "exp1_traits" in result.metadata
    assert "exp2_traits" in result.metadata
    assert "common_genotypes" in result.metadata

    assert result.metadata["exp1_samples"] == 9
    assert result.metadata["exp2_samples"] == 7
    assert result.metadata["exp1_traits"] == 3  # trait1, trait2, trait3
    assert result.metadata["exp2_traits"] == 2  # trait_a, trait_b
    assert result.metadata["common_genotypes"] == 3


def test_load_cross_platform_data_step_trait_identification(
    cross_platform_config, tmp_path
):
    """Test that trait columns are correctly identified."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check that trait column names are stored in metadata
    assert "exp1_trait_names" in result.metadata
    assert "exp2_trait_names" in result.metadata

    exp1_traits = result.metadata["exp1_trait_names"]
    exp2_traits = result.metadata["exp2_trait_names"]

    assert "trait1" in exp1_traits
    assert "trait2" in exp1_traits
    assert "trait3" in exp1_traits
    assert "trait_a" in exp2_traits
    assert "trait_b" in exp2_traits

    # Genotype and rep columns should NOT be in traits
    assert "Geno" not in exp1_traits
    assert "Rep" not in exp1_traits
    assert "geno" not in exp2_traits
    assert "rep" not in exp2_traits


def test_load_cross_platform_data_step_min_samples_filter(tmp_path):
    """Test that genotypes are filtered by min_samples_per_genotype."""
    # Create data where genotype C has only 1 sample in exp1
    exp1_data = {
        "Geno": ["A", "A", "A", "B", "B", "B", "C"],
        "Rep": [1, 2, 3, 1, 2, 3, 1],
        "trait1": [10.5, 11.2, 10.8, 12.3, 11.9, 12.5, 9.8],
    }
    exp1_df = pd.DataFrame(exp1_data)
    exp1_csv = tmp_path / "exp1_min_samples.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    exp2_data = {
        "geno": ["A", "A", "B", "B", "C", "C"],
        "rep": [1, 2, 1, 2, 1, 2],
        "trait_a": [20.1, 20.5, 22.0, 21.8, 19.5, 19.8],
    }
    exp2_df = pd.DataFrame(exp2_data)
    exp2_csv = tmp_path / "exp2_min_samples.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    config = CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        min_samples_per_genotype=2,  # Require at least 2 samples
    )

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    # C should be excluded because it has only 1 sample in exp1
    # Only A and B should remain
    assert len(result.data["common_genotypes"]) == 2
    assert set(result.data["common_genotypes"]) == {"A", "B"}
    assert "C" not in result.data["common_genotypes"]


def test_load_cross_platform_data_step_no_common_genotypes(tmp_path):
    """Test behavior when there are no common genotypes."""
    exp1_data = {"Geno": ["A", "B"], "Rep": [1, 1], "trait1": [10.5, 12.3]}
    exp1_df = pd.DataFrame(exp1_data)
    exp1_csv = tmp_path / "exp1_no_common.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    exp2_data = {"geno": ["C", "D"], "rep": [1, 1], "trait_a": [20.1, 22.0]}
    exp2_df = pd.DataFrame(exp2_data)
    exp2_csv = tmp_path / "exp2_no_common.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    config = CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Exp2",
        exp2_genotype_col="geno",
    )

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()

    with pytest.raises(ValueError, match="No common genotypes found"):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)


def test_load_cross_platform_data_step_files_generated(cross_platform_config, tmp_path):
    """Test that appropriate files are generated."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check that files were generated
    assert len(result.files_generated) > 0

    # Check specific expected files
    expected_files = [
        "cross_platform_exp1_loaded.csv",
        "cross_platform_exp2_loaded.csv",
        "cross_platform_alignment_summary.csv",
    ]

    for expected_file in expected_files:
        file_path = tmp_path / expected_file
        assert file_path.exists(), f"Expected file {expected_file} not found"
        assert str(file_path) in result.files_generated


def test_load_cross_platform_data_step_alignment_summary(
    cross_platform_config, tmp_path
):
    """Test that alignment summary file contains correct information."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    step.execute(data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None)

    # Read alignment summary
    summary_path = tmp_path / "cross_platform_alignment_summary.csv"
    summary = pd.read_csv(summary_path)

    # Check columns
    assert "genotype" in summary.columns
    assert "exp1_samples" in summary.columns
    assert "exp2_samples" in summary.columns

    # Check content - should have all 3 common genotypes
    assert len(summary) == 3
    assert set(summary["genotype"]) == {"A", "B", "C"}

    # Check sample counts
    geno_a = summary[summary["genotype"] == "A"].iloc[0]
    assert geno_a["exp1_samples"] == 3
    assert geno_a["exp2_samples"] == 2
