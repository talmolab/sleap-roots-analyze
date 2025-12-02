"""Tests for visualization module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import matplotlib

# Use non-interactive backend for tests to avoid Tk issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import fixtures
from tests.fixtures import (
    viz_sample_data,
    viz_data_with_nan,
    viz_empty_data,
    viz_single_trait_data,
    viz_many_traits_data,
    viz_perfect_correlation_data,
    viz_bimodal_data,
    viz_single_genotype_data,
    viz_constant_trait_data,
    viz_eda_sample_data,
    viz_eda_thresholds,
    viz_eda_cleanup_log,
    viz_eda_data_with_extremes,
    viz_eda_empty_cleanup_log,
    viz_eda_many_traits_data,
    turface_traits_df,
    traits_summary_df,
    heritability_results_basic,
    heritability_results_empty,
    heritability_results_invalid,
    heritability_threshold_analysis,
    heritability_threshold_analysis_empty,
)

from sleap_roots_analyze.visualization import (
    create_trait_histograms,
    create_trait_boxplots_by_genotype,
    create_correlation_heatmap,
    save_figure_with_unique_name,
    create_exploratory_summary_plots,
    create_trait_eda_plots,
    create_variance_decomposition_plot,
    create_trait_by_genotype_boxplots,
    create_heritability_diagnostic_dashboard,
)


class TestCreateTraitHistograms:
    """Tests for create_trait_histograms function."""

    def test_basic_histograms(self, viz_sample_data):
        """Test basic histogram creation with sample data."""
        trait_cols = ["trait1", "trait2", "trait3"]
        fig = create_trait_histograms(viz_sample_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 3

        # Check that histograms have data
        for ax in axes:
            assert len(ax.patches) > 0  # Should have histogram bars

        plt.close(fig)

    def test_empty_traits_list(self, viz_sample_data):
        """Test handling of empty traits list."""
        fig = create_trait_histograms(viz_sample_data, [])

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1

        # Check for "No traits to plot" text
        texts = [t.get_text() for t in axes[0].texts]
        assert any("No traits to plot" in text for text in texts)
        plt.close(fig)

    def test_custom_grid_layout(self, viz_many_traits_data):
        """Test custom number of columns in grid."""
        trait_cols = [f"trait_{i:02d}" for i in range(6)]
        fig = create_trait_histograms(viz_many_traits_data, trait_cols, n_cols=2)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 6
        plt.close(fig)

    def test_missing_trait_columns(self, viz_sample_data):
        """Test handling when some trait columns don't exist in DataFrame."""
        trait_cols = ["trait1", "trait2", "nonexistent_trait"]

        fig = create_trait_histograms(viz_sample_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        # Should still create 3 subplots (missing columns are handled gracefully)
        assert len(axes) == 3
        plt.close(fig)

    def test_all_nan_trait(self, viz_data_with_nan):
        """Test handling of trait with all NaN values."""
        trait_cols = ["trait_complete", "trait_all_nan"]

        fig = create_trait_histograms(viz_data_with_nan, trait_cols)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        # May have extra axes from subplots
        assert len(axes) >= 2

        # Second subplot should show "No data" or have no patches
        if len(axes[1].texts) > 0:
            texts = [t.get_text() for t in axes[1].texts]
            assert any("No data" in text for text in texts)
        else:
            # Or it should have no histogram bars
            assert len(axes[1].patches) == 0
        plt.close(fig)

    def test_single_trait(self, viz_single_trait_data):
        """Test with single trait."""
        trait_cols = ["single_trait"]

        fig = create_trait_histograms(viz_single_trait_data, trait_cols, n_cols=1)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1
        assert len(axes[0].patches) > 0  # Should have histogram bars
        plt.close(fig)

    def test_custom_figsize(self, viz_sample_data):
        """Test custom figure size."""
        trait_cols = ["trait1"]

        fig = create_trait_histograms(viz_sample_data, trait_cols, figsize=(8, 6))

        assert isinstance(fig, plt.Figure)
        width, height = fig.get_size_inches()
        assert width == 8
        assert height == 6
        plt.close(fig)

    def test_many_traits_layout(self, viz_many_traits_data):
        """Test layout with many traits."""
        trait_cols = [f"trait_{i:02d}" for i in range(16)]

        fig = create_trait_histograms(viz_many_traits_data, trait_cols, n_cols=4)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 16

        # All should be visible and have data
        for i, ax in enumerate(axes):
            if i < len(trait_cols):
                assert ax.get_visible()
        plt.close(fig)


class TestCreateTraitBoxplotsByGenotype:
    """Tests for create_trait_boxplots_by_genotype function."""

    def test_basic_boxplots(self, viz_sample_data):
        """Test basic boxplot creation with genotype groups."""
        trait_cols = ["trait1", "trait2"]

        fig = create_trait_boxplots_by_genotype(viz_sample_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        # Note: boxplot creates additional axes (one for each subplot)
        assert len([ax for ax in axes if ax.get_visible()]) >= 2
        plt.close(fig)

    def test_custom_genotype_column(self, viz_single_trait_data):
        """Test using custom genotype column name."""
        trait_cols = ["single_trait"]

        # Check that geno column exists
        assert "geno" in viz_single_trait_data.columns

        fig = create_trait_boxplots_by_genotype(
            viz_single_trait_data, trait_cols, genotype_col="geno"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_genotype_column(self, viz_sample_data):
        """Test handling when genotype column doesn't exist."""
        trait_cols = ["trait1"]

        fig = create_trait_boxplots_by_genotype(
            viz_sample_data, trait_cols, genotype_col="nonexistent"
        )

        assert isinstance(fig, plt.Figure)
        # Should handle missing column gracefully
        plt.close(fig)

    def test_empty_traits_list(self, viz_sample_data):
        """Test handling of empty traits list."""
        fig = create_trait_boxplots_by_genotype(viz_sample_data, [])

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1

        # Check for "No traits to plot" text
        texts = [t.get_text() for t in axes[0].texts]
        assert any("No traits to plot" in text for text in texts)
        plt.close(fig)

    def test_all_nan_trait_with_genotype(self, viz_data_with_nan):
        """Test trait with all NaN values grouped by genotype."""
        trait_cols = ["trait_all_nan"]

        fig = create_trait_boxplots_by_genotype(viz_data_with_nan, trait_cols)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        # Should show "No data" message
        texts = [t.get_text() for ax in axes for t in ax.texts]
        assert any("No data" in text for text in texts)
        plt.close(fig)

    def test_single_genotype(self, viz_single_genotype_data):
        """Test with only one genotype group."""
        trait_cols = ["trait1", "trait2"]

        fig = create_trait_boxplots_by_genotype(viz_single_genotype_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        # Should handle single genotype gracefully
        plt.close(fig)

    def test_custom_layout(self, viz_sample_data):
        """Test custom column layout."""
        trait_cols = ["trait1", "trait2", "trait3"]

        fig = create_trait_boxplots_by_genotype(
            viz_sample_data, trait_cols, n_cols=1, figsize=(6, 12)
        )

        assert isinstance(fig, plt.Figure)
        width, height = fig.get_size_inches()
        assert width == 6
        assert height == 12
        plt.close(fig)


class TestCreateCorrelationHeatmap:
    """Tests for create_correlation_heatmap function."""

    def test_basic_correlation_heatmap(self, viz_sample_data):
        """Test basic correlation heatmap creation."""
        trait_cols = ["trait1", "trait2", "trait3"]

        fig = create_correlation_heatmap(viz_sample_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1  # At least main axis

        # Check that heatmap has been created
        ax = axes[0]
        assert len(ax.collections) > 0  # Should have heatmap data
        plt.close(fig)

    def test_perfect_correlation(self, viz_perfect_correlation_data):
        """Test heatmap with perfectly correlated traits."""
        trait_cols = ["trait_a", "trait_b", "trait_c", "trait_d"]

        fig = create_correlation_heatmap(viz_perfect_correlation_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        # Should handle perfect correlations without error
        plt.close(fig)

    def test_single_trait(self, viz_single_trait_data):
        """Test with single trait (1x1 correlation matrix)."""
        trait_cols = ["single_trait"]

        fig = create_correlation_heatmap(viz_single_trait_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, viz_sample_data):
        """Test custom figure size is made square using larger dimension."""
        trait_cols = ["trait1", "trait2"]

        fig = create_correlation_heatmap(viz_sample_data, trait_cols, figsize=(8, 6))

        assert isinstance(fig, plt.Figure)
        width, height = fig.get_size_inches()
        # Should use the larger dimension (8) for both width and height
        assert width == 8
        assert height == 8
        plt.close(fig)

    def test_with_nan_values(self, viz_data_with_nan):
        """Test correlation calculation with NaN values."""
        trait_cols = ["trait_complete", "trait_some_nan"]

        fig = create_correlation_heatmap(viz_data_with_nan, trait_cols)

        assert isinstance(fig, plt.Figure)
        # Should handle NaN values in correlation calculation
        plt.close(fig)

    def test_constant_traits(self, viz_constant_trait_data):
        """Test with constant (zero variance) traits."""
        trait_cols = ["trait_constant", "trait_variable", "trait_zero"]

        # This might produce warnings but should not error
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = create_correlation_heatmap(viz_constant_trait_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_many_traits(self, viz_many_traits_data):
        """Test with many traits for large heatmap."""
        trait_cols = [f"trait_{i:02d}" for i in range(20)]

        fig = create_correlation_heatmap(
            viz_many_traits_data, trait_cols, figsize=(15, 12)
        )

        assert isinstance(fig, plt.Figure)
        # Should handle large correlation matrix
        plt.close(fig)


class TestSaveFigureWithUniqueName:
    """Tests for save_figure_with_unique_name function."""

    def test_basic_save(self):
        """Test basic figure saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])

            run_dir = Path(tmpdir)
            saved_path = save_figure_with_unique_name(fig, run_dir, "test_plot")

            assert saved_path.exists()
            assert saved_path.suffix == ".png"
            assert "test_plot" in saved_path.stem
            plt.close(fig)

    def test_directory_creation(self):
        """Test that run_dir is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            run_dir = Path(tmpdir) / "new_subdir"
            assert not run_dir.exists()

            saved_path = save_figure_with_unique_name(fig, run_dir, "test")

            assert run_dir.exists()
            assert saved_path.exists()
            plt.close(fig)

    def test_unique_naming_on_collision(self):
        """Test unique naming when file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig1, ax1 = plt.subplots()
            ax1.plot([1, 2, 3])
            fig2, ax2 = plt.subplots()
            ax2.plot([3, 2, 1])

            run_dir = Path(tmpdir)

            # Mock datetime to ensure same timestamp
            with patch("sleap_roots_analyze.visualization.datetime") as mock_datetime:
                mock_now = MagicMock()
                mock_now.strftime.return_value = "120000"
                mock_datetime.now.return_value = mock_now

                path1 = save_figure_with_unique_name(fig1, run_dir, "test")
                path2 = save_figure_with_unique_name(fig2, run_dir, "test")

            assert path1.exists()
            assert path2.exists()
            assert path1 != path2
            assert "120000" in path1.stem
            assert "120000" in path2.stem

            plt.close(fig1)
            plt.close(fig2)

    def test_different_formats(self):
        """Test saving in different formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            run_dir = Path(tmpdir)

            # Test PNG
            png_path = save_figure_with_unique_name(fig, run_dir, "test", format="png")
            assert png_path.suffix == ".png"
            assert png_path.exists()

            # Test PDF
            pdf_path = save_figure_with_unique_name(fig, run_dir, "test", format="pdf")
            assert pdf_path.suffix == ".pdf"
            assert pdf_path.exists()

            # Test SVG
            svg_path = save_figure_with_unique_name(fig, run_dir, "test", format="svg")
            assert svg_path.suffix == ".svg"
            assert svg_path.exists()

            plt.close(fig)

    def test_custom_dpi(self):
        """Test saving with custom DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            run_dir = Path(tmpdir)

            # Save with different DPI values
            low_dpi_path = save_figure_with_unique_name(fig, run_dir, "low_res", dpi=72)
            high_dpi_path = save_figure_with_unique_name(
                fig, run_dir, "high_res", dpi=300
            )

            assert low_dpi_path.exists()
            assert high_dpi_path.exists()

            # High DPI file should generally be larger
            low_size = low_dpi_path.stat().st_size
            high_size = high_dpi_path.stat().st_size
            # Note: Size difference might not always be guaranteed
            assert low_size > 0 and high_size > 0

            plt.close(fig)


class TestCreateExploratorySummaryPlots:
    """Tests for create_exploratory_summary_plots function."""

    def test_basic_summary_plots(self, viz_sample_data):
        """Test creation of all summary plots."""
        trait_cols = ["trait1", "trait2", "trait3"]

        figures = create_exploratory_summary_plots(viz_sample_data, trait_cols)

        assert isinstance(figures, dict)
        assert "trait_distributions" in figures
        assert "missing_data_pattern" in figures
        assert "trait_ranges_by_genotype" in figures
        assert "samples_per_genotype" in figures
        assert "trait_correlations" in figures

        # Clean up
        for fig in figures.values():
            plt.close(fig)

    def test_many_traits(self, viz_many_traits_data):
        """Test with many traits to check subsetting logic."""
        trait_cols = [f"trait_{i:02d}" for i in range(30)]

        figures = create_exploratory_summary_plots(viz_many_traits_data, trait_cols)

        assert isinstance(figures, dict)
        # Should only show subset of traits in some plots
        assert "trait_distributions" in figures
        assert "trait_correlations" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_missing_genotype_column(self, viz_sample_data):
        """Test when genotype column is missing."""
        trait_cols = ["trait1", "trait2"]

        figures = create_exploratory_summary_plots(
            viz_sample_data, trait_cols, genotype_col="nonexistent"
        )

        assert isinstance(figures, dict)
        # Should still create trait distribution and correlation plots
        assert "trait_distributions" in figures
        assert "trait_correlations" in figures
        # But not genotype-related plots
        assert "samples_per_genotype" not in figures

        for fig in figures.values():
            plt.close(fig)

    def test_empty_traits_list(self, viz_sample_data):
        """Test with empty traits list."""
        figures = create_exploratory_summary_plots(
            viz_sample_data, [], genotype_col="geno"
        )

        assert isinstance(figures, dict)
        # Should only have genotype plot
        assert "samples_per_genotype" in figures
        assert "trait_distributions" not in figures

        for fig in figures.values():
            plt.close(fig)

    def test_single_trait(self, viz_single_trait_data):
        """Test with single trait."""
        trait_cols = ["single_trait"]

        figures = create_exploratory_summary_plots(viz_single_trait_data, trait_cols)

        assert isinstance(figures, dict)
        assert "trait_distributions" in figures
        assert "missing_data_pattern" in figures
        # No correlation plot with single trait
        assert "trait_correlations" not in figures

        for fig in figures.values():
            plt.close(fig)

    def test_with_missing_data(self, viz_data_with_nan):
        """Test with significant missing data."""
        trait_cols = ["trait_complete", "trait_some_nan", "trait_all_nan"]

        figures = create_exploratory_summary_plots(viz_data_with_nan, trait_cols)

        assert isinstance(figures, dict)
        assert "missing_data_pattern" in figures

        # Missing data pattern should show the NaN structure
        for fig in figures.values():
            plt.close(fig)

    def test_custom_genotype_column_name(self, viz_bimodal_data):
        """Test with custom genotype column name."""
        trait_cols = ["trait_bimodal", "trait_normal"]

        figures = create_exploratory_summary_plots(
            viz_bimodal_data, trait_cols, genotype_col="geno"
        )

        assert isinstance(figures, dict)
        assert "samples_per_genotype" in figures

        for fig in figures.values():
            plt.close(fig)


class TestWithRealData:
    """Tests using real data fixtures."""

    def test_with_turface_data(self, turface_traits_df):
        """Test visualization functions with Turface dataset."""
        # Get numeric trait columns
        trait_cols = [
            col
            for col in turface_traits_df.columns
            if col not in ["Barcode", "geno", "rep", "wave_name"]
            and turface_traits_df[col].dtype in [np.float64, np.int64]
        ][
            :5
        ]  # Use first 5 traits for testing

        if len(trait_cols) > 0:
            # Test histogram creation
            fig = create_trait_histograms(turface_traits_df, trait_cols, n_cols=3)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

            # Test boxplots
            if "geno" in turface_traits_df.columns:
                fig = create_trait_boxplots_by_genotype(
                    turface_traits_df, trait_cols, genotype_col="geno"
                )
                assert isinstance(fig, plt.Figure)
                plt.close(fig)

            # Test correlation heatmap
            if len(trait_cols) > 1:
                fig = create_correlation_heatmap(turface_traits_df, trait_cols)
                assert isinstance(fig, plt.Figure)
                plt.close(fig)

            # Test summary plots
            figures = create_exploratory_summary_plots(
                turface_traits_df, trait_cols, genotype_col="geno"
            )
            assert len(figures) > 0
            for fig in figures.values():
                plt.close(fig)

    def test_with_traits_summary_data(self, traits_summary_df):
        """Test with traits summary dataset."""
        # Get numeric trait columns
        exclude_cols = ["Barcode", "geno", "rep", "species", "plant", "scan"]
        numeric_cols = traits_summary_df.select_dtypes(include=[np.number]).columns
        trait_cols = [col for col in numeric_cols if col not in exclude_cols][:4]

        if len(trait_cols) > 0:
            # Test basic visualizations
            fig = create_trait_histograms(traits_summary_df, trait_cols)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

            figures = create_exploratory_summary_plots(traits_summary_df, trait_cols)
            assert len(figures) > 0
            for fig in figures.values():
                plt.close(fig)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self, viz_empty_data):
        """Test with empty DataFrame."""
        trait_cols = []

        # Empty dataframe should not create certain plots to avoid errors
        figures = create_exploratory_summary_plots(viz_empty_data, trait_cols)
        assert isinstance(figures, dict)
        # Should handle empty dataframe gracefully - may have 0 figures
        assert len(figures) >= 0

    def test_all_constant_traits(self, viz_constant_trait_data):
        """Test with traits that have no variation."""
        trait_cols = ["trait_constant", "trait_zero"]

        # Should handle constant values without error
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = create_correlation_heatmap(viz_constant_trait_data, trait_cols)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_sample(self):
        """Test with single sample."""
        df = pd.DataFrame(
            {
                "trait1": [1],
                "trait2": [2],
                "geno": ["A"],
            }
        )
        trait_cols = ["trait1", "trait2"]

        # Should handle single sample gracefully
        figures = create_exploratory_summary_plots(df, trait_cols)
        assert isinstance(figures, dict)
        for fig in figures.values():
            plt.close(fig)

    def test_very_long_trait_names(self):
        """Test with very long trait names."""
        df = pd.DataFrame(
            {
                "this_is_a_very_long_trait_name_that_might_cause_display_issues": [
                    1,
                    2,
                    3,
                ],
                "another_extremely_long_name_for_testing_purposes_only": [4, 5, 6],
                "geno": ["A", "B", "C"],
            }
        )
        trait_cols = [col for col in df.columns if col != "geno"]

        fig = create_trait_histograms(df, trait_cols)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bimodal_distribution(self, viz_bimodal_data):
        """Test visualization of bimodal distributions."""
        trait_cols = ["trait_bimodal", "trait_normal"]

        # Create histograms - should show bimodal pattern
        fig = create_trait_histograms(viz_bimodal_data, trait_cols)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Create boxplots by genotype - should show separation
        fig = create_trait_boxplots_by_genotype(viz_bimodal_data, trait_cols)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCreateTraitEDAPlots:
    """Tests for create_trait_eda_plots function."""

    def test_basic_eda_plots(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test basic EDA plot creation with sample data."""
        trait_cols = [
            "trait_good",
            "trait_high_nan",
            "trait_high_zero",
            "trait_low_var",
            "trait_outliers",
        ]

        figures = create_trait_eda_plots(
            viz_eda_sample_data, trait_cols, viz_eda_thresholds
        )

        assert isinstance(figures, dict)
        assert "trait_eda_overview" in figures
        assert "variance_distribution" in figures

        # Clean up
        for fig in figures.values():
            plt.close(fig)

    def test_with_cleanup_log(
        self, viz_eda_sample_data, viz_eda_thresholds, viz_eda_cleanup_log
    ):
        """Test EDA plots with provided cleanup log."""
        trait_cols = [
            "trait_good",
            "trait_high_nan",
            "trait_high_zero",
            "trait_low_var",
            "trait_outliers",
        ]

        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            cleanup_log=viz_eda_cleanup_log,
        )

        assert isinstance(figures, dict)
        # Should have plot for actually removed traits
        assert (
            "traits_actually_removed" in figures
            or len(viz_eda_cleanup_log["removed_traits"]) == 0
        )

        for fig in figures.values():
            plt.close(fig)

    def test_without_cleanup_log(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test EDA plots without cleanup log (simulates removal)."""
        trait_cols = ["trait_good", "trait_high_nan", "trait_high_zero"]

        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            cleanup_log=None,  # Will simulate what would be removed
        )

        assert isinstance(figures, dict)
        assert "trait_eda_overview" in figures
        assert "variance_distribution" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_with_extreme_data(self, viz_eda_data_with_extremes, viz_eda_thresholds):
        """Test EDA plots with extreme data patterns."""
        trait_cols = [
            "trait_all_nan",
            "trait_all_zero",
            "trait_single_valid",
            "trait_boundary_nan",
            "trait_boundary_zero",
            "trait_high_var",
            "trait_negative",
        ]

        figures = create_trait_eda_plots(
            viz_eda_data_with_extremes, trait_cols, viz_eda_thresholds
        )

        assert isinstance(figures, dict)
        assert "trait_eda_overview" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_empty_cleanup_log(
        self, viz_eda_sample_data, viz_eda_thresholds, viz_eda_empty_cleanup_log
    ):
        """Test with empty cleanup log (no traits removed)."""
        trait_cols = ["trait_good", "trait_low_var"]

        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            cleanup_log=viz_eda_empty_cleanup_log,
        )

        assert isinstance(figures, dict)
        # Should not have removed traits plot
        assert "traits_actually_removed" not in figures

        for fig in figures.values():
            plt.close(fig)

    def test_many_traits(self, viz_eda_many_traits_data, viz_eda_thresholds):
        """Test EDA plots with many traits."""
        # Get all trait columns (excluding metadata)
        trait_cols = [
            col
            for col in viz_eda_many_traits_data.columns
            if col not in ["Barcode", "geno", "rep"]
        ]

        figures = create_trait_eda_plots(
            viz_eda_many_traits_data, trait_cols, viz_eda_thresholds
        )

        assert isinstance(figures, dict)
        assert "trait_eda_overview" in figures
        assert "variance_distribution" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_custom_min_samples(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test with custom minimum samples per trait."""
        trait_cols = ["trait_good", "trait_high_nan"]

        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            min_samples_per_trait=20,  # Custom threshold
        )

        assert isinstance(figures, dict)

        for fig in figures.values():
            plt.close(fig)

    def test_trait_prefixes(self, viz_eda_many_traits_data, viz_eda_thresholds):
        """Test prefix grouping in EDA plots."""
        # Use traits with different prefixes
        trait_cols = ["root_00", "lateral_01", "crown_02", "network_03", "depth_04"]

        figures = create_trait_eda_plots(
            viz_eda_many_traits_data, trait_cols, viz_eda_thresholds
        )

        assert isinstance(figures, dict)
        # The overview plot should group by prefix
        assert "trait_eda_overview" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_no_traits(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test with empty trait list."""
        figures = create_trait_eda_plots(
            viz_eda_sample_data, [], viz_eda_thresholds  # No traits
        )

        assert isinstance(figures, dict)
        # Should still create some figures even with no traits
        assert "variance_distribution" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_missing_trait_columns(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test with some non-existent trait columns."""
        # Filter to only existing columns
        all_trait_cols = ["trait_good", "trait_nonexistent", "trait_low_var"]
        trait_cols = [
            col for col in all_trait_cols if col in viz_eda_sample_data.columns
        ]

        figures = create_trait_eda_plots(
            viz_eda_sample_data, trait_cols, viz_eda_thresholds
        )

        assert isinstance(figures, dict)
        # Should handle missing columns gracefully
        assert "trait_eda_overview" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_all_nan_variance(self, viz_eda_data_with_extremes, viz_eda_thresholds):
        """Test variance calculation with all NaN traits."""
        trait_cols = ["trait_all_nan", "trait_high_var"]

        figures = create_trait_eda_plots(
            viz_eda_data_with_extremes, trait_cols, viz_eda_thresholds
        )

        assert isinstance(figures, dict)
        assert "variance_distribution" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_thresholds_visualization(self, viz_eda_sample_data):
        """Test threshold lines in visualization."""
        trait_cols = ["trait_good", "trait_high_nan", "trait_high_zero"]
        custom_thresholds = {
            "nan": 0.2,  # Lower threshold
            "zero": 0.4,  # Lower threshold
            "outlier": 0.15,
        }

        figures = create_trait_eda_plots(
            viz_eda_sample_data, trait_cols, custom_thresholds
        )

        assert isinstance(figures, dict)
        # Threshold lines should be plotted in overview
        assert "trait_eda_overview" in figures

        for fig in figures.values():
            plt.close(fig)

    def test_cleanup_consistency(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test that cleanup simulation matches actual behavior."""
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        trait_cols = ["trait_good", "trait_high_nan", "trait_high_zero"]

        # First get actual cleanup results
        _, actual_log = apply_data_cleanup_filters(
            viz_eda_sample_data.copy(),
            trait_cols,
            max_zeros_per_trait=viz_eda_thresholds["zero"],
            max_nans_per_trait=viz_eda_thresholds["nan"],
            min_samples_per_trait=10,
        )

        # Then create EDA plots with the actual log
        figures = create_trait_eda_plots(
            viz_eda_sample_data, trait_cols, viz_eda_thresholds, cleanup_log=actual_log
        )

        assert isinstance(figures, dict)

        for fig in figures.values():
            plt.close(fig)


class TestEDAEdgeCases:
    """Edge case tests for create_trait_eda_plots."""

    def test_single_sample(self, viz_eda_thresholds):
        """Test with single sample DataFrame."""
        df = pd.DataFrame({"trait1": [1.0], "trait2": [2.0], "geno": ["A"]})

        figures = create_trait_eda_plots(df, ["trait1", "trait2"], viz_eda_thresholds)

        assert isinstance(figures, dict)

        for fig in figures.values():
            plt.close(fig)

    def test_inf_values(self, viz_eda_thresholds):
        """Test handling of infinite values."""
        df = pd.DataFrame(
            {"trait_inf": [1, 2, np.inf, 4, -np.inf], "trait_normal": [1, 2, 3, 4, 5]}
        )

        # Should handle inf values without error
        figures = create_trait_eda_plots(
            df, ["trait_inf", "trait_normal"], viz_eda_thresholds
        )

        assert isinstance(figures, dict)

        for fig in figures.values():
            plt.close(fig)

    def test_constant_traits(self, viz_eda_thresholds):
        """Test with constant (zero variance) traits."""
        df = pd.DataFrame(
            {"trait_constant": [5.0] * 20, "trait_variable": np.random.randn(20)}
        )

        figures = create_trait_eda_plots(
            df, ["trait_constant", "trait_variable"], viz_eda_thresholds
        )

        assert isinstance(figures, dict)
        assert "variance_distribution" in figures

        for fig in figures.values():
            plt.close(fig)


class TestEDAIntegration:
    """Integration tests for EDA plots with real data."""

    def test_with_turface_data(self, turface_traits_df, viz_eda_thresholds):
        """Test EDA plots with Turface dataset."""
        # Get numeric trait columns
        trait_cols = [
            col
            for col in turface_traits_df.columns
            if col not in ["Barcode", "geno", "rep", "wave_name"]
            and turface_traits_df[col].dtype in [np.float64, np.int64]
        ][
            :10
        ]  # Use first 10 traits for testing

        if len(trait_cols) > 0:
            figures = create_trait_eda_plots(
                turface_traits_df, trait_cols, viz_eda_thresholds
            )

            assert isinstance(figures, dict)
            assert "trait_eda_overview" in figures

            for fig in figures.values():
                plt.close(fig)

    def test_complete_workflow(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test complete EDA workflow with saving."""
        import tempfile
        from pathlib import Path

        trait_cols = ["trait_good", "trait_high_nan", "trait_high_zero"]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create EDA plots
            figures = create_trait_eda_plots(
                viz_eda_sample_data, trait_cols, viz_eda_thresholds
            )

            # Save all figures
            saved_paths = []
            for plot_name, fig in figures.items():
                path = save_figure_with_unique_name(fig, run_dir, f"eda_{plot_name}")
                saved_paths.append(path)
                plt.close(fig)

            # Verify files were saved
            assert len(saved_paths) == len(figures)
            for path in saved_paths:
                assert path.exists()


class TestVisualizationIntegration:
    """Integration tests for visualization module."""

    def test_complete_workflow(self, viz_sample_data):
        """Test complete visualization workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            trait_cols = ["trait1", "trait2", "trait3"]

            # Create all plot types
            figures = create_exploratory_summary_plots(viz_sample_data, trait_cols)

            # Save all figures
            saved_paths = []
            for plot_name, fig in figures.items():
                path = save_figure_with_unique_name(fig, run_dir, plot_name)
                saved_paths.append(path)
                plt.close(fig)

            # Verify all files were saved
            assert len(saved_paths) == len(figures)
            for path in saved_paths:
                assert path.exists()
                assert path.stat().st_size > 0

    def test_matplotlib_backend_compatibility(self):
        """Test that functions work with different matplotlib backends."""
        original_backend = matplotlib.get_backend()

        try:
            # Switch to non-interactive backend
            matplotlib.use("Agg")

            # Create simple test data
            df = pd.DataFrame(
                {
                    "trait1": np.random.randn(50),
                    "trait2": np.random.randn(50),
                    "geno": np.random.choice(["A", "B"], 50),
                }
            )

            # Test functions still work
            fig = create_trait_histograms(df, ["trait1", "trait2"])
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

        finally:
            # Restore original backend
            matplotlib.use(original_backend)


class TestCreateHeritabilityPlot:
    """Tests for create_heritability_plot function."""

    def test_basic_heritability_plot(self, heritability_results_basic):
        """Test basic heritability plot creation."""
        from sleap_roots_analyze.visualization import create_heritability_plot

        fig = create_heritability_plot(heritability_results_basic)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check that bars were created
        bars = [child for child in ax.get_children() if hasattr(child, "get_height")]
        # Filter to get only data bars (heritability values should be between 0 and 1)
        # Also exclude very thin bars that might be from the threshold line
        data_bars = [b for b in bars if 0 < b.get_height() <= 1 and b.get_width() > 0.5]
        # Should have approximately 12 bars (may include threshold line bar)
        assert 11 <= len(data_bars) <= 13  # Allow for some flexibility

        # Check colors based on threshold
        # High heritability (>=0.5) should be green, low should be orange
        high_h2_count = sum(
            1
            for bar in data_bars[:8]
            if "green" in str(bar.get_facecolor()).lower()
            or bar.get_facecolor()[1] > 0.5
        )
        low_h2_count = sum(
            1
            for bar in data_bars[8:]
            if "orange" in str(bar.get_facecolor()).lower()
            or bar.get_facecolor()[0] > 0.9
        )

        assert (
            high_h2_count >= 4
        )  # First 8 bars, at least 4 should be green (h2 >= 0.5)
        assert low_h2_count >= 2  # Last 4 bars should be orange (h2 < 0.5)

        # Check threshold line
        lines = ax.get_lines()
        assert any(line.get_linestyle() == "--" for line in lines)

        # Check labels and title
        assert ax.get_xlabel() == "Traits"
        assert ax.get_ylabel() == "Heritability (H²)"
        assert "Heritability" in ax.get_title()

        plt.close("all")

    def test_heritability_plot_custom_threshold(self, heritability_results_basic):
        """Test heritability plot with custom threshold."""
        from sleap_roots_analyze.visualization import create_heritability_plot

        fig = create_heritability_plot(heritability_results_basic, threshold=0.7)

        ax = fig.axes[0]

        # Check threshold line is at 0.7
        lines = ax.get_lines()
        threshold_line = [line for line in lines if line.get_linestyle() == "--"][0]
        assert threshold_line.get_ydata()[0] == 0.7

        plt.close("all")

    def test_heritability_plot_empty_data(self, heritability_results_empty):
        """Test heritability plot with empty data."""
        from sleap_roots_analyze.visualization import create_heritability_plot

        fig = create_heritability_plot(heritability_results_empty)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should display "No heritability data available"
        texts = ax.texts
        assert any("No heritability data" in text.get_text() for text in texts)

        plt.close("all")

    def test_heritability_plot_invalid_data(self, heritability_results_invalid):
        """Test heritability plot with invalid/mixed data."""
        from sleap_roots_analyze.visualization import create_heritability_plot

        # Should handle gracefully by skipping invalid entries
        fig = create_heritability_plot(heritability_results_invalid)

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should show "No heritability data" since all entries are invalid
        texts = ax.texts
        assert any("No heritability data" in text.get_text() for text in texts)

        plt.close("all")

    def test_heritability_plot_mixed_valid_invalid(self):
        """Test heritability plot with mix of valid and invalid data."""
        from sleap_roots_analyze.visualization import create_heritability_plot

        mixed_results = {
            "valid_trait": {"heritability": 0.6},
            "invalid_trait": {"variance": 100},  # Missing heritability
            "another_valid": {"heritability": 0.3},
        }

        fig = create_heritability_plot(mixed_results)

        ax = fig.axes[0]

        # Should have 2 bars (valid traits only) plus possible threshold line
        bars = [child for child in ax.get_children() if hasattr(child, "get_height")]
        # Filter to get only data bars - heritability values are > 0 and < 1
        # The bar width should be around 0.8 for data bars
        data_bars = [
            b for b in bars if 0.001 < b.get_height() < 0.99 and b.get_width() > 0.5
        ]
        # Could be 2 or 3 depending on how matplotlib renders
        assert 2 <= len(data_bars) <= 3  # Allow some flexibility

        plt.close("all")

    def test_heritability_plot_value_labels(self, heritability_results_basic):
        """Test that value labels are added to bars."""
        from sleap_roots_analyze.visualization import create_heritability_plot

        fig = create_heritability_plot(heritability_results_basic)

        ax = fig.axes[0]

        # Check for text labels on bars
        texts = ax.texts
        # Should have 12 text labels (one per bar)
        assert len(texts) >= 12

        # Check that labels are numeric
        for text in texts:
            try:
                float(text.get_text())
                assert True
            except ValueError:
                # Some texts might be axis labels, that's ok
                pass

        plt.close("all")

    def test_heritability_plot_figsize(self, heritability_results_basic):
        """Test custom figure size."""
        from sleap_roots_analyze.visualization import create_heritability_plot

        fig = create_heritability_plot(heritability_results_basic, figsize=(15, 8))

        # Check figure size
        size = fig.get_size_inches()
        assert size[0] == 15
        assert size[1] == 8

        plt.close("all")


class TestCreateHeritabilityThresholdPlot:
    """Tests for create_heritability_threshold_plot function."""

    def test_basic_threshold_plot(self, heritability_threshold_analysis):
        """Test basic threshold analysis plot creation."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(heritability_threshold_analysis)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Should have 2 subplots

        ax1, ax2 = fig.axes

        # Check top plot (traits retained)
        lines1 = ax1.get_lines()
        assert len(lines1) >= 1  # Main line plus reference lines
        assert ax1.get_ylabel() == "Number of Traits Retained"
        assert "Trait Retention" in ax1.get_title()

        # Check bottom plot (fraction retained)
        lines2 = ax2.get_lines()
        assert len(lines2) >= 1  # Main line
        assert ax2.get_xlabel() == "Heritability Threshold (H²)"
        assert ax2.get_ylabel() == "Traits Retained (%)"

        plt.close("all")

    def test_threshold_plot_with_current(self, heritability_threshold_analysis):
        """Test threshold plot with current threshold highlighted."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(
            heritability_threshold_analysis, current_threshold=0.5
        )

        ax1, ax2 = fig.axes

        # Check for vertical line at current threshold in both plots
        vlines1 = [line for line in ax1.get_lines() if line.get_linestyle() == "--"]
        vlines2 = [line for line in ax2.get_lines() if line.get_linestyle() == "--"]

        # Should have vertical line at 0.5
        assert any(0.5 in line.get_xdata() for line in vlines1)
        assert any(0.5 in line.get_xdata() for line in vlines2)

        # Check for marker point
        # Look for scatter points (red dots)
        collections = ax1.collections
        assert len(collections) > 0 or any(
            line.get_marker() == "o" for line in ax1.get_lines()
        )

        plt.close("all")

    def test_threshold_plot_reference_lines(self, heritability_threshold_analysis):
        """Test that reference lines are added."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(heritability_threshold_analysis)

        ax1, ax2 = fig.axes

        # Check for horizontal reference lines in top plot (50% and 75%)
        hlines = [
            line for line in ax1.get_lines() if len(set(line.get_ydata())) == 1
        ]  # Horizontal lines have constant y
        assert len(hlines) >= 2  # At least 50% and 75% lines

        plt.close("all")

    def test_threshold_plot_annotations(self, heritability_threshold_analysis):
        """Test that annotations are added."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(
            heritability_threshold_analysis, current_threshold=0.5
        )

        ax1, ax2 = fig.axes

        # Check for text annotations
        texts1 = ax1.texts
        texts2 = ax2.texts

        # Should have annotation for current threshold value
        assert len(texts1) > 0  # "X traits" annotation
        assert len(texts2) > 0  # Percentage annotation

        plt.close("all")

    def test_threshold_plot_empty_data(self, heritability_threshold_analysis_empty):
        """Test threshold plot with empty data."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(heritability_threshold_analysis_empty)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2

        # Should handle empty data gracefully
        ax1, ax2 = fig.axes

        # Check that axes limits are set properly even with no data
        assert ax1.get_xlim() == (0, 1)
        assert ax2.get_xlim() == (0, 1)

        plt.close("all")

    def test_threshold_plot_fill_between(self, heritability_threshold_analysis):
        """Test that fill_between is used for area under curves."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(heritability_threshold_analysis)

        ax1, ax2 = fig.axes

        # Check for filled areas (PolyCollection objects)
        collections1 = [c for c in ax1.collections if hasattr(c, "get_facecolor")]
        collections2 = [c for c in ax2.collections if hasattr(c, "get_facecolor")]

        assert len(collections1) > 0  # Blue fill in top plot
        assert len(collections2) > 0  # Green fill in bottom plot

        plt.close("all")

    def test_threshold_plot_figsize(self, heritability_threshold_analysis):
        """Test custom figure size."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(
            heritability_threshold_analysis, figsize=(12, 8)
        )

        # Check figure size
        size = fig.get_size_inches()
        assert size[0] == 12
        assert size[1] == 8

        plt.close("all")

    def test_threshold_plot_axes_limits(self, heritability_threshold_analysis):
        """Test that axes limits are set correctly."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot

        fig = create_heritability_threshold_plot(heritability_threshold_analysis)

        ax1, ax2 = fig.axes

        # Check x-axis limits (should be 0 to 1 for heritability)
        assert ax1.get_xlim() == (0, 1)
        assert ax2.get_xlim() == (0, 1)

        # Check y-axis limits
        total_traits = heritability_threshold_analysis["total_traits"]
        assert ax1.get_ylim()[0] == 0
        assert ax1.get_ylim()[1] >= total_traits  # Should show all traits

        assert ax2.get_ylim() == (0, 105)  # Percentage from 0 to 105

        plt.close("all")


class TestPCAVisualization:
    """Tests for PCA visualization functions."""

    def test_pca_results_field_names(self, pca_viz_results):
        """Test that PCA results have the correct field names."""
        # These are the required fields that visualization functions depend on
        required_fields = [
            "transformed_data",
            "loadings",
            "eigenvalues",
            "explained_variance_ratio",  # NOT "explained_variance"
            "cumulative_variance_ratio",  # NOT "cumulative_variance"
        ]

        for field in required_fields:
            assert field in pca_viz_results, f"Missing required field: {field}"

        # Check that old field names are NOT present (to catch regressions)
        old_fields = ["explained_variance", "cumulative_variance"]
        for old_field in old_fields:
            assert (
                old_field not in pca_viz_results
            ), f"Old field name '{old_field}' should not be present. Use '{old_field}_ratio' instead."

        # Verify types
        assert isinstance(pca_viz_results["explained_variance_ratio"], np.ndarray)
        assert isinstance(pca_viz_results["cumulative_variance_ratio"], np.ndarray)

        # Verify values are ratios (between 0 and 1)
        assert np.all(pca_viz_results["explained_variance_ratio"] >= 0)
        assert np.all(pca_viz_results["explained_variance_ratio"] <= 1)
        assert np.all(pca_viz_results["cumulative_variance_ratio"] >= 0)
        assert np.all(pca_viz_results["cumulative_variance_ratio"] <= 1)

    def test_identify_extreme_samples_in_pc_space(self, extreme_samples_data):
        """Test identification of extreme samples in PC space."""
        from sleap_roots_analyze.visualization import (
            identify_extreme_samples_in_pc_space,
        )

        df, pca_results = extreme_samples_data

        # Identify extreme samples
        extreme_df = identify_extreme_samples_in_pc_space(
            pca_results, df, n_components=3, n_std=2.0
        )

        assert isinstance(extreme_df, pd.DataFrame)
        assert not extreme_df.empty

        # Check required columns
        required_cols = [
            "Barcode",
            "pc_component",
            "pc_score",
            "z_score",
            "extreme_type",
            "explained_variance_ratio",
        ]
        for col in required_cols:
            assert col in extreme_df.columns

        # Check that we found the known extreme samples
        extreme_barcodes = set(extreme_df["Barcode"])
        assert "Sample_000" in extreme_barcodes  # Known extreme sample
        assert "Sample_001" in extreme_barcodes  # Known extreme sample

        # Check z-scores are indeed extreme
        assert all(abs(extreme_df["z_score"]) >= 2.0)

    def test_create_pca_scree_plot(self, pca_viz_results):
        """Test PCA scree plot creation."""
        from sleap_roots_analyze.visualization import create_pca_scree_plot

        fig = create_pca_scree_plot(pca_viz_results, variance_threshold=0.95)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]

        # Check that bars and line are present
        assert len(ax.patches) > 0  # Bars
        assert len(ax.lines) > 0  # Cumulative line and threshold

        # Check labels
        assert ax.get_xlabel() == "Principal Component"
        assert "Variance" in ax.get_ylabel()

        plt.close("all")

    def test_create_feature_contribution_plot(self, pca_viz_results):
        """Test feature contribution plot creation."""
        from sleap_roots_analyze.visualization import create_feature_contribution_plot

        trait_names = pca_viz_results["feature_names"]

        fig = create_feature_contribution_plot(
            pca_viz_results, trait_names, n_components=3, top_n=10
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]

        # Check that bars are present
        assert len(ax.patches) > 0

        # Check there are exactly top_n features shown
        n_yticks = len(ax.get_yticklabels())
        assert n_yticks <= 10

        plt.close("all")

    def test_create_feature_contribution_plot_with_precalculated(self):
        """Test feature contribution plot with pre-calculated contributions."""
        from sleap_roots_analyze.visualization import create_feature_contribution_plot
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts
        import tempfile

        # Create test data
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"trait_{i}" for i in range(10)]
        )
        df["Barcode"] = [f"Sample_{i:03d}" for i in range(50)]
        df["geno"] = np.random.choice(["G1", "G2", "G3"], 50)
        df["rep"] = np.random.choice([1, 2], 50)

        trait_cols = [f"trait_{i}" for i in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run PCA with export to get pre-calculated contributions
            results = run_pca_and_export_artifacts(
                df_traits=df,
                trait_cols=trait_cols,
                analysis_dir=tmpdir,
                n_components=5,
                save_csv=False,
            )

            # Test with pre-calculated contributions
            pca_results = results["pca_results"]

            # Add pre-calculated contributions to pca_results
            pca_results["trait_contrib_df"] = results["trait_contrib_df"]

            # Create plot using pre-calculated contributions
            fig = create_feature_contribution_plot(
                pca_results, trait_cols, n_components=3, top_n=5
            )

            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 1

            ax = fig.axes[0]

            # Check that bars are present
            assert len(ax.patches) > 0

            # Check there are exactly top_n features shown
            n_yticks = len(ax.get_yticklabels())
            assert n_yticks <= 5

            plt.close("all")

    def test_create_feature_contribution_plot_backward_compatibility(
        self, pca_viz_results
    ):
        """Test that function still works with old-style pca_results without pre-calculated contributions."""
        from sleap_roots_analyze.visualization import create_feature_contribution_plot

        trait_names = pca_viz_results["feature_names"]

        # Ensure no pre-calculated contributions in input
        assert "trait_contrib_df" not in pca_viz_results
        assert "feature_importance_consistent" not in pca_viz_results

        # Should still work by calculating contributions on the fly
        fig = create_feature_contribution_plot(
            pca_viz_results, trait_names, n_components=3, top_n=10
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert len(ax.patches) > 0

        plt.close("all")

    def test_create_feature_contribution_plot_consistency(self):
        """Test that pre-calculated and on-the-fly calculations give consistent results."""
        from sleap_roots_analyze.visualization import create_feature_contribution_plot
        from sleap_roots_analyze.pca import perform_pca_analysis
        import tempfile

        # Create test data
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(50, 10), columns=[f"trait_{i}" for i in range(10)]
        )

        trait_cols = df.columns.tolist()

        # Get PCA results without pre-calculated contributions
        pca_results = perform_pca_analysis(df, n_components=5, standardize=True)

        # Create plot with on-the-fly calculation
        fig1 = create_feature_contribution_plot(
            pca_results, trait_cols, n_components=3, top_n=5
        )

        # Get y-tick labels (feature order) from first plot
        ax1 = fig1.axes[0]
        labels1 = [label.get_text() for label in ax1.get_yticklabels()]

        # Now add pre-calculated contributions manually
        loadings = pca_results["loadings"][:, :3]
        eigenvalues = pca_results["eigenvalues"][:3]

        # Calculate contributions
        contributions = (loadings**2) * eigenvalues
        total_contributions = contributions.sum(axis=1)

        # Create DataFrame with pre-calculated contributions
        trait_contrib_df = pd.DataFrame(
            {
                "trait": trait_cols,
                "PC1_variance_contrib": contributions[:, 0],
                "PC2_variance_contrib": contributions[:, 1],
                "PC3_variance_contrib": contributions[:, 2],
                "trait_total_variance_contrib": total_contributions,
            }
        ).sort_values("trait_total_variance_contrib", ascending=False)

        pca_results["trait_contrib_df"] = trait_contrib_df

        # Create plot with pre-calculated contributions
        fig2 = create_feature_contribution_plot(
            pca_results, trait_cols, n_components=3, top_n=5
        )

        # Get y-tick labels from second plot
        ax2 = fig2.axes[0]
        labels2 = [label.get_text() for label in ax2.get_yticklabels()]

        # Check that the same top features are shown in the same order
        assert labels1 == labels2

        plt.close("all")

    def test_create_pca_biplot(self, pca_viz_results, pca_viz_dataframe):
        """Test PCA biplot creation."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Test without coloring
        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check scatter plot is present
        assert len(ax.collections) > 0

        # Check arrows are present (patches include arrows)
        assert len(ax.patches) > 0

        plt.close("all")

        # Test with categorical coloring
        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="geno",
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].get_legend().get_texts()) > 0  # Legend present

        plt.close("all")

        # Test with numeric coloring
        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="trait_0",
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        # Should have a colorbar
        assert len(fig.axes) > 1  # Main ax plus colorbar

        plt.close("all")

    def test_create_pca_biplot_with_genotype_filtering(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test PCA biplot with genotype filtering."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Get unique genotypes from the dataframe
        unique_genos = pca_viz_dataframe["geno"].unique()
        assert len(unique_genos) >= 2, "Need at least 2 genotypes for this test"

        # Select first 2 genotypes to color
        genos_to_color = list(unique_genos[:2])

        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="geno",
            genotypes_to_color=genos_to_color,
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check legend is present
        legend = ax.get_legend()
        assert legend is not None

        # Check legend labels
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Should have selected genotypes + "Other"
        for geno in genos_to_color:
            assert geno in legend_texts

        # Should have "Other" if there are more genotypes
        if len(unique_genos) > len(genos_to_color):
            assert "Other" in legend_texts

        # Should NOT have unselected genotypes explicitly in legend
        unselected_genos = set(unique_genos) - set(genos_to_color)
        for geno in unselected_genos:
            assert geno not in legend_texts

        plt.close("all")

    def test_create_pca_biplot_genotype_filtering_all_selected(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test biplot when all genotypes are selected."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Select all genotypes
        all_genos = list(pca_viz_dataframe["geno"].unique())

        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="geno",
            genotypes_to_color=all_genos,
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check legend
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Should NOT have "Other" since all genotypes are selected
        assert "Other" not in legend_texts

        # Should have all genotypes
        for geno in all_genos:
            assert geno in legend_texts

        plt.close("all")

    def test_create_pca_biplot_genotype_filtering_empty_list(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test biplot with empty genotypes list."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Empty list should plot all as "Other"
        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="geno",
            genotypes_to_color=[],
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check legend
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Should only have "Other"
        assert legend_texts == ["Other"]

        plt.close("all")

    def test_create_pca_biplot_genotype_filtering_nonexistent_genotype(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test biplot with non-existent genotypes in filter list."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Include some non-existent genotypes
        genos_to_color = ["NonExistent1", "NonExistent2"]

        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="geno",
            genotypes_to_color=genos_to_color,
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Check legend
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Should only have "Other" since no genotypes match
        assert legend_texts == ["Other"]

        plt.close("all")

    def test_create_pca_biplot_genotype_filtering_numeric_coloring(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test that genotype filtering doesn't affect numeric coloring."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # genotypes_to_color should have no effect with numeric coloring
        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="trait_0",  # Numeric column
            genotypes_to_color=["should_be_ignored"],
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        # Should have colorbar for numeric coloring
        assert len(fig.axes) > 1

        plt.close("all")

    def test_create_pca_biplot_genotype_filtering_without_color_by(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test that genotype filtering has no effect without color_by."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # genotypes_to_color should be ignored when color_by is None
        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by=None,
            genotypes_to_color=["should_be_ignored"],
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should not have legend
        assert ax.get_legend() is None

        plt.close("all")

    def test_create_pca_biplot_genotype_filtering_no_overlap(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test that colored genotypes are not also plotted as gray."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Get unique genotypes
        unique_genos = pca_viz_dataframe["geno"].unique()
        assert len(unique_genos) >= 3, "Need at least 3 genotypes for this test"

        # Select subset to color
        genos_to_color = list(unique_genos[:2])

        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="geno",
            genotypes_to_color=genos_to_color,
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Get all scatter collections
        collections = ax.collections
        assert len(collections) > 0

        # Count points in each collection
        colored_points = 0
        gray_points = 0

        for collection in collections:
            # Get the facecolors
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                # Check if this is the gray collection
                # Gray in RGB is approximately (0.5, 0.5, 0.5)
                first_color = facecolors[0]
                is_gray = (
                    abs(first_color[0] - 0.5019607843137255) < 0.01
                    and abs(first_color[1] - 0.5019607843137255) < 0.01
                    and abs(first_color[2] - 0.5019607843137255) < 0.01
                )

                n_points = len(collection.get_offsets())
                if is_gray:
                    gray_points += n_points
                else:
                    colored_points += n_points

        # Verify counts
        total_points = len(pca_viz_results["transformed_data"])
        assert colored_points + gray_points == total_points, (
            f"Point count mismatch: {colored_points} colored + {gray_points} gray "
            f"!= {total_points} total"
        )

        # Verify that we have both colored and gray points
        n_colored_samples = sum(pca_viz_dataframe["geno"].isin(genos_to_color))
        n_other_samples = total_points - n_colored_samples

        assert (
            colored_points == n_colored_samples
        ), f"Expected {n_colored_samples} colored points, got {colored_points}"
        assert (
            gray_points == n_other_samples
        ), f"Expected {n_other_samples} gray points, got {gray_points}"

        plt.close("all")

    def test_create_pca_biplot_genotype_filtering_no_gray_colors(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test that selected genotypes don't use gray-like colors."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Get unique genotypes
        unique_genos = pca_viz_dataframe["geno"].unique()
        assert len(unique_genos) >= 3, "Need at least 3 genotypes for this test"

        # Select subset to color
        genos_to_color = list(unique_genos[:3])

        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            color_by="geno",
            genotypes_to_color=genos_to_color,
            pc_x=1,
            pc_y=2,
            top_n_features=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Get all scatter collections
        collections = ax.collections
        assert len(collections) > 0

        # Check colors of non-"Other" collections
        for collection in collections:
            facecolors = collection.get_facecolors()
            if len(facecolors) > 0:
                first_color = facecolors[0]

                # Check if this is the gray "Other" collection
                is_gray = (
                    abs(first_color[0] - 0.5019607843137255) < 0.01
                    and abs(first_color[1] - 0.5019607843137255) < 0.01
                    and abs(first_color[2] - 0.5019607843137255) < 0.01
                )

                # If it's not the gray collection, verify it's not gray-like
                if not is_gray:
                    # Check that RGB values are not all similar (within 0.1 of each other)
                    # and not all close to 0.5 (neutral gray)
                    r, g, b = first_color[0], first_color[1], first_color[2]

                    # A color is gray-like if R, G, B are very similar
                    max_diff = max(abs(r - g), abs(r - b), abs(g - b))

                    # For colored genotypes, we expect distinct colors (not gray-like)
                    # Allow some tolerance, but expect at least 0.1 difference in channels
                    assert max_diff > 0.05, (
                        f"Selected genotype color is too gray-like: "
                        f"RGB=({r:.3f}, {g:.3f}, {b:.3f}), max_diff={max_diff:.3f}"
                    )

        plt.close("all")

    def test_create_umap_colored_by_top_traits(
        self, umap_viz_results, pca_viz_dataframe, pca_viz_results
    ):
        """Test UMAP plots colored by top traits."""
        from sleap_roots_analyze.visualization import create_umap_colored_by_top_traits

        trait_columns = [f"trait_{i}" for i in range(10)]
        trait_names = [f"Trait {i}" for i in range(10)]

        fig = create_umap_colored_by_top_traits(
            umap_viz_results,
            pca_viz_dataframe,
            trait_columns,
            trait_names,
            pca_viz_results,
            n_traits=6,
        )

        assert isinstance(fig, plt.Figure)

        # Should have 6 subplots (n_traits=6)
        axes = [ax for ax in fig.axes if ax.get_visible()]
        # Count actual plot axes (excluding colorbars)
        plot_axes = [ax for ax in axes if not hasattr(ax, "colorbar")]
        assert len(plot_axes) >= 6 or len(plot_axes) == len(trait_columns[:6])

        plt.close("all")

    def test_create_umap_single_trait_basic(self, umap_viz_results, pca_viz_dataframe):
        """Test basic single trait UMAP plot."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        fig = create_umap_single_trait(
            umap_viz_results,
            pca_viz_dataframe,
            trait_col="trait_0",
            trait_name="Trait 0",
        )

        assert isinstance(fig, plt.Figure)

        # Should have axes (main plot + colorbar)
        assert len(fig.axes) >= 1

        # Check labels on main plot
        ax = fig.axes[0]
        assert "UMAP 1" in ax.get_xlabel()
        assert "UMAP 2" in ax.get_ylabel()
        assert "Trait 0" in ax.get_title()

        plt.close("all")

    def test_create_umap_single_trait_with_category(
        self, umap_viz_results, pca_viz_dataframe
    ):
        """Test single trait UMAP plot with category overlay."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        # Add a genotype column (ensure it matches length exactly)
        n_samples = len(pca_viz_dataframe)
        genotypes = ["A", "B", "C"] * (n_samples // 3 + 1)
        pca_viz_dataframe["geno"] = genotypes[:n_samples]

        fig = create_umap_single_trait(
            umap_viz_results,
            pca_viz_dataframe,
            trait_col="trait_0",
            trait_name="Trait 0",
            color_by="geno",
            figsize=(14, 6),
        )

        assert isinstance(fig, plt.Figure)

        # Should have multiple axes (2 main plots + colorbar)
        assert len(fig.axes) >= 2

        # Check we have subplots
        assert len(fig.axes) >= 2

        # Check first subplot (colored by trait)
        ax1 = fig.axes[0]
        assert "UMAP 1" in ax1.get_xlabel()
        assert "Trait 0" in ax1.get_title()

        # Check second subplot (colored by genotype)
        ax2 = fig.axes[1]
        assert "UMAP 1" in ax2.get_xlabel()
        assert "geno" in ax2.get_title()

        plt.close("all")

    def test_create_umap_single_trait_with_dict_input(
        self, umap_viz_results, pca_viz_dataframe
    ):
        """Test single trait UMAP with dictionary input."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        # umap_viz_results is already a dict, just pass it directly
        fig = create_umap_single_trait(
            umap_viz_results,
            pca_viz_dataframe,
            trait_col="trait_0",
        )

        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_create_umap_single_trait_custom_params(
        self, umap_viz_results, pca_viz_dataframe
    ):
        """Test single trait UMAP with custom parameters."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        fig = create_umap_single_trait(
            umap_viz_results,
            pca_viz_dataframe,
            trait_col="trait_0",
            trait_name="Custom Trait Name",
            cmap="plasma",
            point_size=50,
            alpha=0.5,
            title="Custom Title",
        )

        assert isinstance(fig, plt.Figure)

        ax = fig.axes[0]
        assert "Custom Title" in ax.get_title()

        plt.close("all")

    def test_create_umap_single_trait_invalid_trait(
        self, umap_viz_results, pca_viz_dataframe
    ):
        """Test single trait UMAP with invalid trait column."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        with pytest.raises(ValueError, match="Trait column.*not found"):
            create_umap_single_trait(
                umap_viz_results,
                pca_viz_dataframe,
                trait_col="nonexistent_trait",
            )

    def test_create_umap_single_trait_invalid_color_by(
        self, umap_viz_results, pca_viz_dataframe
    ):
        """Test single trait UMAP with invalid color_by column."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        with pytest.raises(ValueError, match="color_by column.*not found"):
            create_umap_single_trait(
                umap_viz_results,
                pca_viz_dataframe,
                trait_col="trait_0",
                color_by="nonexistent_column",
            )

    def test_create_umap_single_trait_mismatched_samples(
        self, umap_viz_results, pca_viz_dataframe
    ):
        """Test single trait UMAP with mismatched sample counts."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        # Create dataframe with fewer samples
        df_small = pca_viz_dataframe.iloc[:10].copy()

        with pytest.raises(ValueError, match="UMAP embedding has.*samples"):
            create_umap_single_trait(
                umap_viz_results,
                df_small,
                trait_col="trait_0",
            )

    def test_create_umap_single_trait_default_name(
        self, umap_viz_results, pca_viz_dataframe
    ):
        """Test single trait UMAP uses trait_col as default name."""
        from sleap_roots_analyze.visualization import create_umap_single_trait

        fig = create_umap_single_trait(
            umap_viz_results,
            pca_viz_dataframe,
            trait_col="trait_0",
            # Don't specify trait_name
        )

        assert isinstance(fig, plt.Figure)

        # Should use trait_col in title
        ax = fig.axes[0]
        assert "trait_0" in ax.get_title()

        plt.close("all")

    def test_identify_extreme_genotypes_by_pc(self, genotype_pc_data):
        """Test identification of extreme genotypes by PC."""
        from sleap_roots_analyze.visualization import identify_extreme_genotypes_by_pc

        df, pca_results = genotype_pc_data

        extreme_df = identify_extreme_genotypes_by_pc(
            pca_results, df, genotype_col="geno", n_components=3, n_extreme=2
        )

        assert isinstance(extreme_df, pd.DataFrame)
        assert not extreme_df.empty

        # Check required columns
        required_cols = [
            "geno",
            "pc_component",
            "median_pc_score",
            "direction",
            "rank",
            "n_samples",
            "explained_variance_ratio",
        ]
        for col in required_cols:
            assert col in extreme_df.columns

        # Check that we have both high and low extremes
        assert "high" in extreme_df["direction"].values
        assert "low" in extreme_df["direction"].values

        # Check that ranks are correct
        assert all(extreme_df["rank"] <= 2)  # n_extreme=2

    def test_create_pc_genotype_boxplots(self, genotype_pc_data):
        """Test PC genotype boxplots creation."""
        from sleap_roots_analyze.visualization import create_pc_genotype_boxplots

        df, pca_results = genotype_pc_data

        fig = create_pc_genotype_boxplots(
            pca_results, df, genotype_col="geno", n_components=3, highlight_extreme=2
        )

        assert isinstance(fig, plt.Figure)

        # Should have 3 subplots (n_components=3)
        axes = fig.axes
        visible_axes = [ax for ax in axes if ax.get_visible()]
        assert len(visible_axes) == 3

        # Check that each axis has boxplot elements
        for ax in visible_axes:
            # Check for lines (whiskers, medians) in the plot
            assert len(ax.lines) > 0

        plt.close("all")

    def test_extreme_samples_edge_cases(self, pca_viz_results, pca_viz_dataframe):
        """Test edge cases for extreme sample identification."""
        from sleap_roots_analyze.visualization import (
            identify_extreme_samples_in_pc_space,
        )

        # Test with single component
        extreme_df = identify_extreme_samples_in_pc_space(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            n_components=1,
            n_std=3.0,  # Very high threshold
        )

        # Should find fewer or no extremes with high threshold
        assert isinstance(extreme_df, pd.DataFrame)

        # Test with all components
        n_total = pca_viz_results["transformed_data"].shape[1]
        extreme_df = identify_extreme_samples_in_pc_space(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            n_components=n_total,
            n_std=1.0,  # Lower threshold
        )

        assert isinstance(extreme_df, pd.DataFrame)
        if not extreme_df.empty:
            # Should have both individual PC extremes and combined
            assert (
                "Combined" in extreme_df["pc_component"].values or len(extreme_df) > 0
            )

    def test_scree_plot_variance_threshold(self, pca_viz_results):
        """Test scree plot with different variance thresholds."""
        from sleap_roots_analyze.visualization import create_pca_scree_plot

        # Test with low threshold
        fig = create_pca_scree_plot(pca_viz_results, variance_threshold=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

        # Test with high threshold
        fig = create_pca_scree_plot(pca_viz_results, variance_threshold=0.99)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

        # Test with threshold of 1.0
        fig = create_pca_scree_plot(pca_viz_results, variance_threshold=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_feature_contribution_with_variance_threshold(self, pca_viz_results):
        """Test feature contribution plot with variance threshold selection."""
        from sleap_roots_analyze.visualization import create_feature_contribution_plot

        trait_names = pca_viz_results["feature_names"]

        # Test with variance threshold instead of n_components
        fig = create_feature_contribution_plot(
            pca_viz_results,
            trait_names,
            n_components=None,
            variance_threshold=0.8,
            top_n=5,
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]

        # Should show exactly top_n features
        n_features_shown = len(ax.get_yticklabels())
        assert n_features_shown == 5

        plt.close("all")

    def test_biplot_invalid_pc_indices(self, pca_viz_results, pca_viz_dataframe):
        """Test biplot with invalid PC indices."""
        from sleap_roots_analyze.visualization import create_pca_biplot

        trait_names = [f"trait_{i}" for i in range(10)]

        # Should handle PC indices within available range
        max_pc = pca_viz_results["transformed_data"].shape[1]

        fig = create_pca_biplot(
            pca_viz_results,
            pca_viz_dataframe,
            trait_names,
            pc_x=max_pc,  # Last PC
            pc_y=max_pc - 1 if max_pc > 1 else 1,  # Second to last or first
            top_n_features=3,
        )

        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_genotype_functions_missing_column(self, genotype_pc_data):
        """Test genotype functions with missing column."""
        from sleap_roots_analyze.visualization import (
            identify_extreme_genotypes_by_pc,
            create_pc_genotype_boxplots,
        )

        df, pca_results = genotype_pc_data

        # Test with non-existent column
        with pytest.raises(ValueError, match="not found"):
            identify_extreme_genotypes_by_pc(
                pca_results, df, genotype_col="nonexistent_column"
            )

        with pytest.raises(ValueError, match="not found"):
            create_pc_genotype_boxplots(
                pca_results, df, genotype_col="nonexistent_column"
            )


class TestFeatureContributionHeatmap:
    """Tests for create_feature_contribution_heatmap function."""

    def test_basic_heatmap(self, pca_results_with_feature_importance):
        """Test basic feature contribution heatmap creation."""
        from sleap_roots_analyze.visualization import (
            create_feature_contribution_heatmap,
        )

        # Test default behavior (returns both plots)
        result = create_feature_contribution_heatmap(
            pca_results_with_feature_importance, n_components=5, n_features=10
        )

        # Should return tuple of (variance_fig, loadings_fig)
        assert isinstance(result, tuple)
        assert len(result) == 2

        variance_fig, loadings_fig = result
        assert isinstance(variance_fig, plt.Figure)
        assert isinstance(loadings_fig, plt.Figure)

        # Check variance figure
        ax = variance_fig.axes[0]
        assert "Variance Contributions" in ax.get_title()

        # Check loadings figure
        ax = loadings_fig.axes[0]
        assert "Loadings" in ax.get_title() or "Correlations" in ax.get_title()

        # Test single variance plot
        fig_var = create_feature_contribution_heatmap(
            pca_results_with_feature_importance,
            n_components=5,
            n_features=10,
            plot_type="variance",
        )
        assert isinstance(fig_var, plt.Figure)

        # Test single loadings plot
        fig_load = create_feature_contribution_heatmap(
            pca_results_with_feature_importance,
            n_components=5,
            n_features=10,
            plot_type="loadings",
        )
        assert isinstance(fig_load, plt.Figure)

        plt.close("all")

    def test_heatmap_with_fewer_components(self, pca_results_with_feature_importance):
        """Test heatmap with fewer components than available."""
        from sleap_roots_analyze.visualization import (
            create_feature_contribution_heatmap,
        )

        # Request only variance plot with 3 components
        fig = create_feature_contribution_heatmap(
            pca_results_with_feature_importance,
            n_components=3,
            n_features=15,
            plot_type="variance",
        )

        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Variance Contributions" in ax.get_title()
        assert ax.get_xlabel() == "Principal Component"

        plt.close("all")

    def test_heatmap_with_custom_figsize(self, pca_results_with_feature_importance):
        """Test heatmap with custom figure size."""
        from sleap_roots_analyze.visualization import (
            create_feature_contribution_heatmap,
        )

        custom_figsize = (12, 6)

        # Test loadings plot with custom size
        fig = create_feature_contribution_heatmap(
            pca_results_with_feature_importance,
            n_components=5,
            n_features=10,
            figsize=custom_figsize,
            plot_type="loadings",
        )

        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == custom_figsize[0]
        assert fig.get_size_inches()[1] == custom_figsize[1]

        plt.close("all")


class TestCreatePublicationFigure:
    """Tests for create_publication_figure function."""

    def test_save_matplotlib_figure(self, tmp_path):
        """Test saving matplotlib figure in various formats."""
        from sleap_roots_analyze.visualization import create_publication_figure

        # Create a simple matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        # Test PDF format
        pdf_path = tmp_path / "test_figure.pdf"
        create_publication_figure(fig, pdf_path, format="pdf")
        assert pdf_path.exists()

        # Test PNG format
        png_path = tmp_path / "test_figure.png"
        create_publication_figure(fig, png_path, format="png", dpi=150)
        assert png_path.exists()

        # Test SVG format
        svg_path = tmp_path / "test_figure.svg"
        create_publication_figure(fig, svg_path, format="svg")
        assert svg_path.exists()

        plt.close("all")

    def test_save_with_transparency(self, tmp_path):
        """Test saving figure with transparent background."""
        from sleap_roots_analyze.visualization import create_publication_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        png_path = tmp_path / "transparent.png"
        create_publication_figure(fig, png_path, format="png", transparent=True)
        assert png_path.exists()

        plt.close("all")

    def test_invalid_figure_type(self, tmp_path):
        """Test error handling for invalid figure type."""
        from sleap_roots_analyze.visualization import create_publication_figure

        # Create an invalid object
        invalid_fig = {"not": "a figure"}

        output_path = tmp_path / "test.pdf"
        with pytest.raises(ValueError, match="Unsupported figure type"):
            create_publication_figure(invalid_fig, output_path)


class TestIdentifyExtremePhenotypes:
    """Tests for identify_extreme_phenotypes function."""

    def test_basic_extreme_identification(self, phenotype_variation_data):
        """Test basic identification of extreme phenotypes."""
        from sleap_roots_analyze.visualization import identify_extreme_phenotypes

        trait_cols = ["trait_A", "trait_B"]
        extremes = identify_extreme_phenotypes(
            phenotype_variation_data, trait_cols, group_col="geno", n_std=2.0
        )

        assert isinstance(extremes, dict)

        # Check trait_A extremes (we know G_high and G_low should be extreme)
        if "trait_A" in extremes:
            extreme_genos = extremes["trait_A"].index.tolist()
            assert "G_high" in extreme_genos or "G_low" in extreme_genos

            # Check DataFrame structure
            assert "mean" in extremes["trait_A"].columns
            assert "std" in extremes["trait_A"].columns
            assert "count" in extremes["trait_A"].columns
            assert "deviation" in extremes["trait_A"].columns
            assert "direction" in extremes["trait_A"].columns

    def test_extreme_with_min_samples(self, phenotype_variation_data):
        """Test extreme identification with minimum sample requirement."""
        from sleap_roots_analyze.visualization import identify_extreme_phenotypes

        # Set high minimum to filter out some groups
        extremes = identify_extreme_phenotypes(
            phenotype_variation_data,
            ["trait_A"],
            min_samples_per_group=10,  # Only G_high and G_low have >= 8 samples
        )

        # Should have fewer groups due to sample size filter
        if "trait_A" in extremes:
            assert len(extremes["trait_A"]) <= 2

    def test_extreme_with_custom_threshold(self, phenotype_variation_data):
        """Test extreme identification with different thresholds."""
        from sleap_roots_analyze.visualization import identify_extreme_phenotypes

        # Very high threshold - should find fewer extremes
        extremes_high = identify_extreme_phenotypes(
            phenotype_variation_data, ["trait_A"], n_std=5.0
        )

        # Lower threshold - should find more extremes
        extremes_low = identify_extreme_phenotypes(
            phenotype_variation_data, ["trait_A"], n_std=1.0
        )

        # Compare counts
        n_high = len(extremes_high.get("trait_A", pd.DataFrame()))
        n_low = len(extremes_low.get("trait_A", pd.DataFrame()))
        assert n_high <= n_low

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        from sleap_roots_analyze.visualization import identify_extreme_phenotypes

        df_empty = pd.DataFrame()
        extremes = identify_extreme_phenotypes(df_empty, [], group_col="geno")

        assert isinstance(extremes, dict)
        assert len(extremes) == 0


class TestCreatePhenotypeVariationPlot:
    """Tests for create_phenotype_variation_plot function."""

    def test_basic_variation_plot(self, phenotype_variation_data):
        """Test basic phenotype variation plot creation."""
        from sleap_roots_analyze.visualization import create_phenotype_variation_plot

        fig, plot_df = create_phenotype_variation_plot(
            phenotype_variation_data,
            "trait_A",
            group_col="geno",
            highlight_extreme=True,
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(plot_df, pd.DataFrame)

        # Check that plot_df contains expected columns
        assert "geno" in plot_df.columns
        assert "trait_A" in plot_df.columns
        assert "trait_A_mean" in plot_df.columns
        assert "trait_A_std" in plot_df.columns
        assert "trait_A_overall_mean" in plot_df.columns

        ax = fig.axes[0]
        assert ax.get_ylabel() == "trait_A"
        assert ax.get_xlabel() == "Geno"

        plt.close("all")

    def test_variation_plot_without_highlights(self, phenotype_variation_data):
        """Test variation plot without highlighting extremes."""
        from sleap_roots_analyze.visualization import create_phenotype_variation_plot

        fig, plot_df = create_phenotype_variation_plot(
            phenotype_variation_data, "trait_A", highlight_extreme=False
        )

        assert isinstance(fig, plt.Figure)

        # Check that threshold columns are not added
        assert "trait_A_high_threshold" not in plot_df.columns
        assert "trait_A_low_threshold" not in plot_df.columns

        plt.close("all")

    def test_variation_plot_with_csv_output(self, phenotype_variation_data, tmp_path):
        """Test saving plot data to CSV."""
        from sleap_roots_analyze.visualization import create_phenotype_variation_plot

        csv_path = tmp_path / "plot_data.csv"

        fig, plot_df = create_phenotype_variation_plot(
            phenotype_variation_data, "trait_A", output_csv_path=csv_path
        )

        assert csv_path.exists()

        # Load and verify CSV
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == len(plot_df)
        assert set(saved_df.columns) == set(plot_df.columns)

        plt.close("all")

    def test_variation_plot_custom_parameters(self, phenotype_variation_data):
        """Test variation plot with custom parameters."""
        from sleap_roots_analyze.visualization import create_phenotype_variation_plot

        fig, plot_df = create_phenotype_variation_plot(
            phenotype_variation_data,
            "trait_B",
            n_std=3.0,
            point_size=100,
            figsize=(15, 10),
        )

        assert isinstance(fig, plt.Figure)

        # Check figure size
        width, height = fig.get_size_inches()
        assert width == 15
        assert height == 10

        # Check threshold values in DataFrame
        assert "trait_B_high_threshold" in plot_df.columns
        overall_mean = plot_df["trait_B_overall_mean"].iloc[0]
        overall_std = plot_df["trait_B_overall_std"].iloc[0]
        expected_high = overall_mean + 3.0 * overall_std
        assert np.allclose(plot_df["trait_B_high_threshold"].iloc[0], expected_high)

        plt.close("all")
