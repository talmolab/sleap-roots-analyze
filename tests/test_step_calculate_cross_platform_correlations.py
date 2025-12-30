"""Tests for CalculateCrossPlatformCorrelationsStep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig
from sleap_roots_analyze.pipeline.core import StepResult


@pytest.fixture
def cross_platform_config():
    """Create a test CrossPlatformConfig."""
    return CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        correlation_method="spearman",
        min_samples_per_genotype=2,
    )


@pytest.fixture
def loaded_data_result(cross_platform_exp1_df, cross_platform_exp2_df, tmp_path):
    """Create a mock result from LoadCrossPlatformDataStep."""
    from sleap_roots_analyze.cross_experiment_analysis import load_and_align_experiments
    from sleap_roots_analyze.data_cleanup import get_trait_columns

    # Save DataFrames to CSV files (required by load_and_align_experiments)
    exp1_path = tmp_path / "exp1_temp.csv"
    exp2_path = tmp_path / "exp2_temp.csv"
    cross_platform_exp1_df.to_csv(exp1_path, index=False)
    cross_platform_exp2_df.to_csv(exp2_path, index=False)

    # Use the actual pipeline function to load and standardize column names
    exp1_df, exp2_df, common_genotypes = load_and_align_experiments(
        exp1_path=str(exp1_path),
        exp2_path=str(exp2_path),
        genotype_col1="Geno",
        genotype_col2="geno",
    )

    # Get trait columns using the same function as LoadCrossPlatformDataStep
    exp1_traits = get_trait_columns(
        exp1_df,
        barcode_col=None,
        genotype_col="genotype",
        replicate_col="replicate",
    )
    exp2_traits = get_trait_columns(
        exp2_df,
        barcode_col=None,
        genotype_col="genotype",
        replicate_col="replicate",
    )

    common_genotypes = sorted(common_genotypes)

    return StepResult(
        data={
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
            "common_genotypes": common_genotypes,
        },
        metadata={
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
            "common_genotypes": len(common_genotypes),
        },
        files_generated=[],
    )


def test_calculate_cross_platform_correlations_step_initialization():
    """Test CalculateCrossPlatformCorrelationsStep initialization."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    assert step.step_name == "CalculateCrossPlatformCorrelations"
    assert "Calculate correlations" in step.description


def test_calculate_cross_platform_correlations_step_execute(
    cross_platform_config, loaded_data_result, tmp_path
):
    """Test CalculateCrossPlatformCorrelationsStep execution."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    # Check that correlation data was created
    assert result.data is not None
    assert "correlation_df" in result.data
    assert isinstance(result.data["correlation_df"], pd.DataFrame)

    # Check correlation DataFrame structure
    corr_df = result.data["correlation_df"]
    assert "exp1_trait" in corr_df.columns
    assert "exp2_trait" in corr_df.columns
    assert "spearman_r" in corr_df.columns
    assert "spearman_p" in corr_df.columns
    assert "pearson_r" in corr_df.columns
    assert "pearson_p" in corr_df.columns
    assert "n_genotypes" in corr_df.columns

    # Check that we have correlations (should be many trait pairs)
    assert len(corr_df) > 0


def test_calculate_cross_platform_correlations_step_spearman(
    loaded_data_result, tmp_path
):
    """Test correlation calculation with Spearman method."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        correlation_method="spearman",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    # Verify correlations are in valid range [-1, 1]
    corr_df = result.data["correlation_df"]
    assert (corr_df["spearman_r"].abs() <= 1.0).all()
    assert (corr_df["pearson_r"].abs() <= 1.0).all()


def test_calculate_cross_platform_correlations_step_pearson(
    loaded_data_result, tmp_path
):
    """Test correlation calculation with Pearson method."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        correlation_method="pearson",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    # Should have correlation data
    assert "correlation_df" in result.data
    corr_df = result.data["correlation_df"]
    assert len(corr_df) > 0


def test_calculate_cross_platform_correlations_step_sorted(
    cross_platform_config, loaded_data_result, tmp_path
):
    """Test that correlations are sorted by absolute value."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    corr_df = result.data["correlation_df"]

    # Check sorted by absolute primary correlation (spearman by default, descending)
    abs_corr = corr_df["spearman_r"].abs().values
    assert all(abs_corr[i] >= abs_corr[i + 1] for i in range(len(abs_corr) - 1))


def test_calculate_cross_platform_correlations_step_metadata(
    cross_platform_config, loaded_data_result, tmp_path
):
    """Test that metadata is correctly populated."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    # Check metadata
    assert "total_correlations" in result.metadata
    assert "correlation_method" in result.metadata
    assert result.metadata["correlation_method"] == "spearman"
    assert result.metadata["total_correlations"] > 0


def test_calculate_cross_platform_correlations_step_files_generated(
    cross_platform_config, loaded_data_result, tmp_path
):
    """Test that correlation CSV is generated."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    # Check that correlation CSV was created
    corr_csv = tmp_path / "cross_platform_correlations.csv"
    assert corr_csv.exists()
    assert str(corr_csv) in result.files_generated

    # Verify CSV contents
    saved_df = pd.read_csv(corr_csv)
    assert len(saved_df) == len(result.data["correlation_df"])
    assert "exp1_trait" in saved_df.columns
    assert "spearman_r" in saved_df.columns
    assert "pearson_r" in saved_df.columns


def test_calculate_cross_platform_correlations_step_nan_handling(tmp_path):
    """Test handling of NaN values in trait data."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )
    from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig

    # Create data with some NaN values
    exp1_df = pd.DataFrame(
        {
            "genotype": ["A", "A", "B", "B", "C", "C"],
            "replicate": [1, 2, 1, 2, 1, 2],
            "trait1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "trait2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    exp2_df = pd.DataFrame(
        {
            "genotype": ["A", "A", "B", "B", "C", "C"],
            "replicate": [1, 2, 1, 2, 1, 2],
            "trait_a": [100.0, 200.0, 300.0, np.nan, 500.0, 600.0],
        }
    )

    prev_result = StepResult(
        data={
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
            "common_genotypes": ["A", "B", "C"],
        },
        metadata={
            "exp1_trait_names": ["trait1", "trait2"],
            "exp2_trait_names": ["trait_a"],
        },
        files_generated=[],
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=prev_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    # Should still compute correlations
    corr_df = result.data["correlation_df"]
    assert len(corr_df) > 0

    # Check n_genotypes reflects actual pairs used
    # For trait1 vs trait_a: All genotypes have valid means
    # (calculate_genotype_means handles NaN by averaging available replicates)
    # So A, B, C all contribute to the correlation = 3 genotypes
    trait1_traita = corr_df[
        (corr_df["exp1_trait"] == "trait1") & (corr_df["exp2_trait"] == "trait_a")
    ]
    if len(trait1_traita) > 0:
        assert trait1_traita.iloc[0]["n_genotypes"] == 3


# === TDD Tests for Dual Correlation Metrics (store-dual-correlation-metrics) ===


def test_csv_contains_both_pearson_and_spearman_columns(
    cross_platform_config, loaded_data_result, tmp_path
):
    """Test that CSV output contains both Pearson and Spearman correlation columns.

    TDD Test: This test will FAIL until the fix is implemented.
    Expected new columns: spearman_r, spearman_p, pearson_r, pearson_p
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    corr_df = result.data["correlation_df"]

    # New schema: both Pearson and Spearman columns
    assert "spearman_r" in corr_df.columns, "Missing spearman_r column"
    assert "spearman_p" in corr_df.columns, "Missing spearman_p column"
    assert "pearson_r" in corr_df.columns, "Missing pearson_r column"
    assert "pearson_p" in corr_df.columns, "Missing pearson_p column"

    # Old generic columns should not exist
    assert (
        "correlation" not in corr_df.columns
    ), "Old 'correlation' column should be removed"
    assert "p_value" not in corr_df.columns, "Old 'p_value' column should be removed"


def test_csv_sorted_by_primary_correlation_method(tmp_path):
    """Test that CSV is sorted by the primary correlation method from config.

    When correlation_method='spearman', sort by |spearman_r|.
    When correlation_method='pearson', sort by |pearson_r|.
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    # Create test data with known correlation patterns
    exp1_df = pd.DataFrame(
        {
            "genotype": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "replicate": [1, 2, 1, 2, 1, 2, 1, 2],
            "trait1": [1.0, 1.2, 2.0, 2.1, 3.0, 3.2, 4.0, 4.1],
            "trait2": [10.0, 10.5, 20.0, 20.2, 30.0, 30.3, 40.0, 40.1],
        }
    )

    exp2_df = pd.DataFrame(
        {
            "genotype": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "replicate": [1, 2, 1, 2, 1, 2, 1, 2],
            "trait_a": [100.0, 102.0, 200.0, 205.0, 300.0, 310.0, 400.0, 410.0],
        }
    )

    prev_result = StepResult(
        data={
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
            "common_genotypes": ["A", "B", "C", "D"],
        },
        metadata={
            "exp1_trait_names": ["trait1", "trait2"],
            "exp2_trait_names": ["trait_a"],
        },
        files_generated=[],
    )

    # Test with Spearman as primary
    config_spearman = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        correlation_method="spearman",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=prev_result.data,
        config=config_spearman,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    corr_df = result.data["correlation_df"]

    # Should be sorted by |spearman_r| descending
    abs_spearman = corr_df["spearman_r"].abs().values
    assert all(
        abs_spearman[i] >= abs_spearman[i + 1] for i in range(len(abs_spearman) - 1)
    ), "Results should be sorted by |spearman_r| when correlation_method='spearman'"


def test_visualization_uses_precomputed_pearson_values(tmp_path):
    """Test that create_joint_plot accepts and uses pre-computed Pearson values.

    TDD Test: This will verify that both Pearson and Spearman come from CSV.
    """
    import matplotlib.pyplot as plt

    from sleap_roots_analyze.cross_experiment_analysis import create_joint_plot

    # Create genotype means data
    exp1_means = pd.DataFrame(
        {"trait1": [1.0, 2.0, 3.0, 4.0]}, index=["A", "B", "C", "D"]
    )
    exp2_means = pd.DataFrame(
        {"trait_a": [10.0, 20.0, 30.0, 40.0]}, index=["A", "B", "C", "D"]
    )

    # Pre-computed values (as if from CSV)
    precomputed_spearman_r = 0.95
    precomputed_spearman_p = 0.001
    precomputed_pearson_r = 0.98
    precomputed_pearson_p = 0.0005
    precomputed_n = 4

    # Call create_joint_plot with both pre-computed values
    fig = create_joint_plot(
        exp1_means,
        exp2_means,
        "trait1",
        "trait_a",
        exp1_name="Exp1",
        exp2_name="Exp2",
        correlation=precomputed_spearman_r,
        p_value=precomputed_spearman_p,
        n_genotypes=precomputed_n,
        pearson_r=precomputed_pearson_r,
        pearson_p=precomputed_pearson_p,
    )
    plt.close(fig)

    # If we get here without error, the function accepts the new parameters


# === TDD Tests for FDR Correction (add-fdr-correction) ===


def test_fdr_correction_bh_method(loaded_data_result, tmp_path):
    """Test that fdr_bh correction method produces adjusted p-values.

    TDD Test: Verifies adjusted columns exist and p_adj >= p_raw.
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_bh",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    corr_df = result.data["correlation_df"]

    # Check new columns exist
    assert "spearman_p_adjusted" in corr_df.columns
    assert "pearson_p_adjusted" in corr_df.columns
    assert "significant_fdr" in corr_df.columns

    # Adjusted p-values should be >= raw p-values
    assert (corr_df["spearman_p_adjusted"] >= corr_df["spearman_p"] - 1e-10).all()
    assert (corr_df["pearson_p_adjusted"] >= corr_df["pearson_p"] - 1e-10).all()

    # significant_fdr should be boolean
    assert corr_df["significant_fdr"].dtype == bool


def test_fdr_correction_by_method(loaded_data_result, tmp_path):
    """Test that fdr_by correction method (default) produces more conservative results.

    BY correction should produce larger (more conservative) adjusted p-values than BH.
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    # Run with fdr_bh
    config_bh = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_bh",
    )

    # Run with fdr_by (default, more conservative)
    config_by = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_by",
    )

    step = CalculateCrossPlatformCorrelationsStep()

    # Create subdirectories for each run
    bh_dir = tmp_path / "bh"
    by_dir = tmp_path / "by"
    bh_dir.mkdir(parents=True, exist_ok=True)
    by_dir.mkdir(parents=True, exist_ok=True)

    result_bh = step.execute(
        data=loaded_data_result.data,
        config=config_bh,
        run_dir=bh_dir,
        prev_result=loaded_data_result,
    )

    result_by = step.execute(
        data=loaded_data_result.data,
        config=config_by,
        run_dir=by_dir,
        prev_result=loaded_data_result,
    )

    # BY should be more conservative (fewer significant, or same)
    sig_bh = result_bh.data["correlation_df"]["significant_fdr"].sum()
    sig_by = result_by.data["correlation_df"]["significant_fdr"].sum()
    assert sig_by <= sig_bh


def test_fdr_correction_none_method(loaded_data_result, tmp_path):
    """Test that 'none' correction method uses raw p-values."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="none",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    corr_df = result.data["correlation_df"]

    # With no correction, adjusted should equal raw
    assert np.allclose(
        corr_df["spearman_p_adjusted"], corr_df["spearman_p"], rtol=1e-10
    )
    assert np.allclose(corr_df["pearson_p_adjusted"], corr_df["pearson_p"], rtol=1e-10)


def test_fdr_correction_invalid_method():
    """Test that invalid FDR method raises ValueError."""
    with pytest.raises(ValueError, match="fdr_correction_method must be one of"):
        CrossPlatformConfig(
            exp1_data_path="dummy1.csv",
            exp1_name="Exp1",
            exp1_genotype_col="Geno",
            exp2_data_path="dummy2.csv",
            exp2_name="Exp2",
            exp2_genotype_col="geno",
            fdr_correction_method="invalid_method",
        )


def test_csv_output_contains_fdr_columns(
    cross_platform_config, loaded_data_result, tmp_path
):
    """Test that saved CSV contains FDR correction columns."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    step.execute(
        data=loaded_data_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    # Verify CSV was saved with new columns
    saved_df = pd.read_csv(tmp_path / "cross_platform_correlations.csv")
    assert "spearman_p_adjusted" in saved_df.columns
    assert "pearson_p_adjusted" in saved_df.columns
    assert "significant_fdr" in saved_df.columns


def test_metadata_includes_fdr_info(loaded_data_result, tmp_path):
    """Test that metadata includes FDR correction information."""
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_by",
        significance_level=0.05,
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=loaded_data_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=loaded_data_result,
    )

    assert result.metadata["fdr_correction_method"] == "fdr_by"
    assert result.metadata["significance_level"] == 0.05
    assert "significant_correlations" in result.metadata


# === TDD Tests for FDR Edge Case Handling (add-fdr-edge-case-handling) ===


@pytest.fixture
def data_with_constant_trait(tmp_path):
    """Create test data where one trait has constant values (zero variance).

    This will produce NaN p-values for correlations involving that trait.
    """
    from sleap_roots_analyze.cross_experiment_analysis import load_and_align_experiments
    from sleap_roots_analyze.data_cleanup import get_trait_columns

    # Experiment 1: normal data + one constant trait
    exp1_df = pd.DataFrame(
        {
            "Geno": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "rep": [1, 2, 1, 2, 1, 2, 1, 2],
            "trait1": [1.0, 1.2, 2.0, 2.1, 3.0, 3.2, 4.0, 4.1],
            "trait2": [10.0, 10.5, 20.0, 20.5, 30.0, 30.5, 40.0, 40.5],
            "constant_trait": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # Zero variance
        }
    )

    # Experiment 2: normal data
    exp2_df = pd.DataFrame(
        {
            "geno": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "rep": [1, 2, 1, 2, 1, 2, 1, 2],
            "traitA": [2.0, 2.2, 4.0, 4.1, 6.0, 6.2, 8.0, 8.1],
            "traitB": [5.0, 5.5, 10.0, 10.5, 15.0, 15.5, 20.0, 20.5],
        }
    )

    exp1_path = tmp_path / "exp1_constant.csv"
    exp2_path = tmp_path / "exp2_constant.csv"
    exp1_df.to_csv(exp1_path, index=False)
    exp2_df.to_csv(exp2_path, index=False)

    exp1_loaded, exp2_loaded, common_genotypes = load_and_align_experiments(
        exp1_path=str(exp1_path),
        exp2_path=str(exp2_path),
        genotype_col1="Geno",
        genotype_col2="geno",
    )

    exp1_traits = get_trait_columns(
        exp1_loaded,
        barcode_col=None,
        genotype_col="genotype",
        replicate_col="replicate",
    )
    exp2_traits = get_trait_columns(
        exp2_loaded,
        barcode_col=None,
        genotype_col="genotype",
        replicate_col="replicate",
    )

    return StepResult(
        data={
            "exp1_df": exp1_loaded,
            "exp2_df": exp2_loaded,
            "common_genotypes": common_genotypes,
        },
        metadata={
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
            "exp1_name": "Exp1",
            "exp2_name": "Exp2",
        },
        files_generated=[],
    )


@pytest.fixture
def data_with_single_trait_pair(tmp_path):
    """Create test data with only one trait in each experiment (m=1 test)."""
    from sleap_roots_analyze.cross_experiment_analysis import load_and_align_experiments
    from sleap_roots_analyze.data_cleanup import get_trait_columns

    exp1_df = pd.DataFrame(
        {
            "Geno": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "rep": [1, 2, 1, 2, 1, 2, 1, 2],
            "only_trait": [1.0, 1.2, 2.0, 2.1, 3.0, 3.2, 4.0, 4.1],
        }
    )

    exp2_df = pd.DataFrame(
        {
            "geno": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "rep": [1, 2, 1, 2, 1, 2, 1, 2],
            "single_trait": [2.0, 2.2, 4.0, 4.1, 6.0, 6.2, 8.0, 8.1],
        }
    )

    exp1_path = tmp_path / "exp1_single.csv"
    exp2_path = tmp_path / "exp2_single.csv"
    exp1_df.to_csv(exp1_path, index=False)
    exp2_df.to_csv(exp2_path, index=False)

    exp1_loaded, exp2_loaded, common_genotypes = load_and_align_experiments(
        exp1_path=str(exp1_path),
        exp2_path=str(exp2_path),
        genotype_col1="Geno",
        genotype_col2="geno",
    )

    exp1_traits = get_trait_columns(
        exp1_loaded,
        barcode_col=None,
        genotype_col="genotype",
        replicate_col="replicate",
    )
    exp2_traits = get_trait_columns(
        exp2_loaded,
        barcode_col=None,
        genotype_col="genotype",
        replicate_col="replicate",
    )

    return StepResult(
        data={
            "exp1_df": exp1_loaded,
            "exp2_df": exp2_loaded,
            "common_genotypes": common_genotypes,
        },
        metadata={
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
            "exp1_name": "Exp1",
            "exp2_name": "Exp2",
        },
        files_generated=[],
    )


def test_fdr_correction_with_nan_pvalues(data_with_constant_trait, tmp_path):
    """Test that NaN p-values don't corrupt FDR correction of valid p-values.

    TDD Test: When one trait produces NaN p-values (constant values),
    other valid correlations should still receive correct FDR adjustment.
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_bh",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=data_with_constant_trait.data,
        config=config,
        run_dir=tmp_path,
        prev_result=data_with_constant_trait,
    )

    corr_df = result.data["correlation_df"]

    # Correlations involving constant_trait should have NaN p-values
    constant_rows = corr_df[corr_df["exp1_trait"] == "constant_trait"]
    assert (
        constant_rows["spearman_p"].isna().all()
    ), "Constant trait should produce NaN p-values"
    assert (
        constant_rows["spearman_p_adjusted"].isna().all()
    ), "NaN p-values should remain NaN after adjustment"

    # Valid correlations should NOT have NaN adjusted p-values
    valid_rows = corr_df[corr_df["exp1_trait"] != "constant_trait"]
    assert (
        not valid_rows["spearman_p"].isna().any()
    ), "Valid traits should have p-values"
    assert (
        not valid_rows["spearman_p_adjusted"].isna().any()
    ), "Valid traits should have adjusted p-values"

    # Valid adjusted p-values should be >= raw p-values (FDR property)
    assert (valid_rows["spearman_p_adjusted"] >= valid_rows["spearman_p"] - 1e-10).all()


def test_fdr_correction_single_correlation(data_with_single_trait_pair, tmp_path):
    """Test FDR correction with only one trait pair (m=1).

    TDD Test: With a single test, adjusted p-values should equal raw p-values
    (no multiple testing correction needed).
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_bh",
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=data_with_single_trait_pair.data,
        config=config,
        run_dir=tmp_path,
        prev_result=data_with_single_trait_pair,
    )

    corr_df = result.data["correlation_df"]

    # With m=1, there should be exactly one correlation
    assert len(corr_df) == 1, "Should have exactly one trait pair"

    # Adjusted should equal raw (no correction for single test)
    assert np.isclose(
        corr_df["spearman_p_adjusted"].iloc[0],
        corr_df["spearman_p"].iloc[0],
        rtol=1e-10,
    ), "Single test should not be corrected"


def test_fdr_correction_with_constant_trait(data_with_constant_trait, tmp_path):
    """Test that constant-valued traits produce NaN and don't break the pipeline.

    TDD Test: A trait with zero variance should produce NaN correlations
    but not crash the FDR correction or corrupt other results.
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_by",  # Use BY (more conservative)
    )

    step = CalculateCrossPlatformCorrelationsStep()

    # Should not raise an exception
    result = step.execute(
        data=data_with_constant_trait.data,
        config=config,
        run_dir=tmp_path,
        prev_result=data_with_constant_trait,
    )

    corr_df = result.data["correlation_df"]

    # Should have some rows (3 exp1 traits x 2 exp2 traits = 6 pairs)
    assert len(corr_df) == 6, "Should have 6 trait pairs"

    # constant_trait correlations should have NaN r values
    constant_rows = corr_df[corr_df["exp1_trait"] == "constant_trait"]
    assert len(constant_rows) == 2, "constant_trait should correlate with 2 exp2 traits"
    assert (
        constant_rows["spearman_r"].isna().all()
    ), "Constant trait should produce NaN r"


def test_significant_fdr_false_for_nan(data_with_constant_trait, tmp_path):
    """Test that significant_fdr is False for rows with NaN adjusted p-values.

    TDD Test: Rows with NaN adjusted p-values should never be marked significant.
    """
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )

    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        fdr_correction_method="fdr_bh",
        significance_level=0.99,  # Very high threshold to ensure valid correlations are significant
    )

    step = CalculateCrossPlatformCorrelationsStep()
    result = step.execute(
        data=data_with_constant_trait.data,
        config=config,
        run_dir=tmp_path,
        prev_result=data_with_constant_trait,
    )

    corr_df = result.data["correlation_df"]

    # Rows with NaN adjusted p-values should have significant_fdr = False
    nan_rows = corr_df[corr_df["spearman_p_adjusted"].isna()]
    assert len(nan_rows) > 0, "Should have some NaN p-values from constant trait"
    assert not nan_rows[
        "significant_fdr"
    ].any(), "NaN adjusted p-values should not be significant"

    # Valid rows with high significance threshold should be significant
    valid_rows = corr_df[~corr_df["spearman_p_adjusted"].isna()]
    # With significance_level=0.99, most valid correlations should be significant
    assert valid_rows[
        "significant_fdr"
    ].any(), "Some valid correlations should be significant"
