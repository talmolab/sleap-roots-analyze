"""Tests for VisualizeCrossPlatformStep."""

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
        exp1_name="Cylinder",
        exp1_genotype_col="Geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Turface",
        exp2_genotype_col="geno",
        correlation_method="spearman",
        top_n_correlations=20,
        top_n_joint_plots=6,
        top_n_boxplots=6,
    )


@pytest.fixture
def correlation_result(cross_platform_exp1_df, cross_platform_exp2_df, tmp_path):
    """Create a mock result from CalculateCrossPlatformCorrelationsStep."""
    from sleap_roots_analyze.cross_experiment_analysis import load_and_align_experiments
    from sleap_roots_analyze.data_cleanup import get_trait_columns

    # Save DataFrames to CSV files
    exp1_path = tmp_path / "exp1_temp.csv"
    exp2_path = tmp_path / "exp2_temp.csv"
    cross_platform_exp1_df.to_csv(exp1_path, index=False)
    cross_platform_exp2_df.to_csv(exp2_path, index=False)

    # Load and standardize
    exp1_df, exp2_df, common_genotypes = load_and_align_experiments(
        exp1_path=str(exp1_path),
        exp2_path=str(exp2_path),
        genotype_col1="Geno",
        genotype_col2="geno",
    )

    # Get trait columns
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

    # Create mock correlation data
    np.random.seed(42)
    correlation_data = []
    for i, trait1 in enumerate(exp1_traits[:10]):  # Use first 10 traits
        for j, trait2 in enumerate(exp2_traits[:5]):  # Use first 5 traits
            correlation_data.append(
                {
                    "exp1_trait": trait1,
                    "exp2_trait": trait2,
                    "correlation": np.random.randn() * 0.5,
                    "p_value": np.random.rand(),
                    "n_genotypes": len(common_genotypes),
                }
            )

    correlation_df = pd.DataFrame(correlation_data)
    # Sort by absolute correlation
    correlation_df = correlation_df.assign(
        abs_correlation=correlation_df["correlation"].abs()
    )
    correlation_df = correlation_df.sort_values(
        "abs_correlation", ascending=False
    ).drop(columns=["abs_correlation"])
    correlation_df = correlation_df.reset_index(drop=True)

    return StepResult(
        data={
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
            "common_genotypes": sorted(common_genotypes),
            "correlation_df": correlation_df,
        },
        metadata={
            "exp1_name": "Cylinder",
            "exp2_name": "Turface",
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
            "total_correlations": len(correlation_df),
            "correlation_method": "spearman",
        },
        files_generated=[],
    )


def test_visualize_cross_platform_step_initialization():
    """Test VisualizeCrossPlatformStep initialization."""
    from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
        VisualizeCrossPlatformStep,
    )

    step = VisualizeCrossPlatformStep()
    assert step.step_name == "VisualizeCrossPlatform"
    assert "Visualize" in step.description or "visualize" in step.description


def test_visualize_cross_platform_step_execute(
    cross_platform_config, correlation_result, tmp_path
):
    """Test VisualizeCrossPlatformStep execution."""
    from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
        VisualizeCrossPlatformStep,
    )

    step = VisualizeCrossPlatformStep()
    result = step.execute(
        data=correlation_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=correlation_result,
    )

    # Check that visualization was successful
    assert result.data is not None
    assert len(result.files_generated) > 0


def test_visualize_cross_platform_step_summary_plot(
    cross_platform_config, correlation_result, tmp_path
):
    """Test that correlation summary plot is created."""
    from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
        VisualizeCrossPlatformStep,
    )

    step = VisualizeCrossPlatformStep()
    result = step.execute(
        data=correlation_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=correlation_result,
    )

    # Check for summary plot
    summary_plot = tmp_path / "cross_platform_correlation_summary.png"
    assert summary_plot.exists()
    assert str(summary_plot) in result.files_generated


def test_visualize_cross_platform_step_joint_plots(
    cross_platform_config, correlation_result, tmp_path
):
    """Test that joint plots are created for top correlations."""
    from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
        VisualizeCrossPlatformStep,
    )

    step = VisualizeCrossPlatformStep()
    result = step.execute(
        data=correlation_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=correlation_result,
    )

    # Check for joint plots (should create up to top_n_joint_plots)
    joint_plot_files = list(tmp_path.glob("cross_platform_joint_*.png"))
    assert len(joint_plot_files) > 0
    assert len(joint_plot_files) <= cross_platform_config.top_n_joint_plots


def test_visualize_cross_platform_step_boxplots(
    cross_platform_config, correlation_result, tmp_path
):
    """Test that boxplots are created for top correlations."""
    from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
        VisualizeCrossPlatformStep,
    )

    step = VisualizeCrossPlatformStep()
    result = step.execute(
        data=correlation_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=correlation_result,
    )

    # Check for boxplots (should create up to top_n_boxplots)
    boxplot_files = list(tmp_path.glob("cross_platform_boxplot_*.png"))
    assert len(boxplot_files) > 0
    assert len(boxplot_files) <= cross_platform_config.top_n_boxplots


def test_visualize_cross_platform_step_metadata(
    cross_platform_config, correlation_result, tmp_path
):
    """Test that metadata is correctly populated."""
    from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
        VisualizeCrossPlatformStep,
    )

    step = VisualizeCrossPlatformStep()
    result = step.execute(
        data=correlation_result.data,
        config=cross_platform_config,
        run_dir=tmp_path,
        prev_result=correlation_result,
    )

    # Check metadata
    assert "plots_generated" in result.metadata
    assert result.metadata["plots_generated"] > 0


def test_visualize_cross_platform_step_minimal_correlations(tmp_path):
    """Test behavior with very few correlations."""
    from sleap_roots_analyze.pipeline.steps.visualize_cross_platform import (
        VisualizeCrossPlatformStep,
    )

    # Create minimal correlation data
    correlation_df = pd.DataFrame(
        {
            "exp1_trait": ["trait1", "trait2"],
            "exp2_trait": ["trait_a", "trait_b"],
            "correlation": [0.8, -0.7],
            "p_value": [0.01, 0.02],
            "n_genotypes": [10, 10],
        }
    )

    exp1_df = pd.DataFrame(
        {
            "genotype": ["A", "B", "C"] * 3,
            "replicate": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "trait1": np.random.randn(9),
            "trait2": np.random.randn(9),
        }
    )

    exp2_df = pd.DataFrame(
        {
            "genotype": ["A", "B", "C"] * 3,
            "replicate": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "trait_a": np.random.randn(9),
            "trait_b": np.random.randn(9),
        }
    )

    prev_result = StepResult(
        data={
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
            "common_genotypes": ["A", "B", "C"],
            "correlation_df": correlation_df,
        },
        metadata={
            "exp1_name": "Exp1",
            "exp2_name": "Exp2",
            "total_correlations": 2,
            "exp1_trait_names": ["trait1", "trait2"],
            "exp2_trait_names": ["trait_a", "trait_b"],
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
        top_n_joint_plots=6,  # Request more than available
        top_n_boxplots=6,
    )

    step = VisualizeCrossPlatformStep()
    result = step.execute(
        data=prev_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=prev_result,
    )

    # Should create plots even with minimal data
    assert len(result.files_generated) > 0
    # Should not create more joint plots than correlations available
    joint_plot_files = list(tmp_path.glob("cross_platform_joint_*.png"))
    assert len(joint_plot_files) <= 2
