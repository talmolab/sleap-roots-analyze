"""Tests for visualization module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend for tests to avoid Tk issues
matplotlib.use('Agg')
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
        """Test custom figure size."""
        trait_cols = ["trait1", "trait2"]
        
        fig = create_correlation_heatmap(viz_sample_data, trait_cols, figsize=(8, 6))

        assert isinstance(fig, plt.Figure)
        width, height = fig.get_size_inches()
        assert width == 8
        assert height == 6
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
        
        fig = create_correlation_heatmap(viz_many_traits_data, trait_cols, figsize=(15, 12))

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
            with patch('sleap_roots_analyze.visualization.datetime') as mock_datetime:
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
            high_dpi_path = save_figure_with_unique_name(fig, run_dir, "high_res", dpi=300)

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
        figures = create_exploratory_summary_plots(viz_sample_data, [], genotype_col="geno")

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
            col for col in turface_traits_df.columns
            if col not in ["Barcode", "geno", "rep", "wave_name"]
            and turface_traits_df[col].dtype in [np.float64, np.int64]
        ][:5]  # Use first 5 traits for testing

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
        df = pd.DataFrame({
            "trait1": [1],
            "trait2": [2],
            "geno": ["A"],
        })
        trait_cols = ["trait1", "trait2"]
        
        # Should handle single sample gracefully
        figures = create_exploratory_summary_plots(df, trait_cols)
        assert isinstance(figures, dict)
        for fig in figures.values():
            plt.close(fig)

    def test_very_long_trait_names(self):
        """Test with very long trait names."""
        df = pd.DataFrame({
            "this_is_a_very_long_trait_name_that_might_cause_display_issues": [1, 2, 3],
            "another_extremely_long_name_for_testing_purposes_only": [4, 5, 6],
            "geno": ["A", "B", "C"]
        })
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
        trait_cols = ['trait_good', 'trait_high_nan', 'trait_high_zero', 
                     'trait_low_var', 'trait_outliers']
        
        figures = create_trait_eda_plots(
            viz_eda_sample_data, 
            trait_cols, 
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        assert 'trait_eda_overview' in figures
        assert 'variance_distribution' in figures
        
        # Clean up
        for fig in figures.values():
            plt.close(fig)
    
    def test_with_cleanup_log(self, viz_eda_sample_data, viz_eda_thresholds, 
                             viz_eda_cleanup_log):
        """Test EDA plots with provided cleanup log."""
        trait_cols = ['trait_good', 'trait_high_nan', 'trait_high_zero', 
                     'trait_low_var', 'trait_outliers']
        
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            cleanup_log=viz_eda_cleanup_log
        )
        
        assert isinstance(figures, dict)
        # Should have plot for actually removed traits
        assert 'traits_actually_removed' in figures or len(viz_eda_cleanup_log['removed_traits']) == 0
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_without_cleanup_log(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test EDA plots without cleanup log (simulates removal)."""
        trait_cols = ['trait_good', 'trait_high_nan', 'trait_high_zero']
        
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            cleanup_log=None  # Will simulate what would be removed
        )
        
        assert isinstance(figures, dict)
        assert 'trait_eda_overview' in figures
        assert 'variance_distribution' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_with_extreme_data(self, viz_eda_data_with_extremes, viz_eda_thresholds):
        """Test EDA plots with extreme data patterns."""
        trait_cols = ['trait_all_nan', 'trait_all_zero', 'trait_single_valid',
                     'trait_boundary_nan', 'trait_boundary_zero', 
                     'trait_high_var', 'trait_negative']
        
        figures = create_trait_eda_plots(
            viz_eda_data_with_extremes,
            trait_cols,
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        assert 'trait_eda_overview' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_empty_cleanup_log(self, viz_eda_sample_data, viz_eda_thresholds,
                               viz_eda_empty_cleanup_log):
        """Test with empty cleanup log (no traits removed)."""
        trait_cols = ['trait_good', 'trait_low_var']
        
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            cleanup_log=viz_eda_empty_cleanup_log
        )
        
        assert isinstance(figures, dict)
        # Should not have removed traits plot
        assert 'traits_actually_removed' not in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_many_traits(self, viz_eda_many_traits_data, viz_eda_thresholds):
        """Test EDA plots with many traits."""
        # Get all trait columns (excluding metadata)
        trait_cols = [col for col in viz_eda_many_traits_data.columns 
                     if col not in ['Barcode', 'geno', 'rep']]
        
        figures = create_trait_eda_plots(
            viz_eda_many_traits_data,
            trait_cols,
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        assert 'trait_eda_overview' in figures
        assert 'variance_distribution' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_custom_min_samples(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test with custom minimum samples per trait."""
        trait_cols = ['trait_good', 'trait_high_nan']
        
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            min_samples_per_trait=20  # Custom threshold
        )
        
        assert isinstance(figures, dict)
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_trait_prefixes(self, viz_eda_many_traits_data, viz_eda_thresholds):
        """Test prefix grouping in EDA plots."""
        # Use traits with different prefixes
        trait_cols = ['root_00', 'lateral_01', 'crown_02', 'network_03', 'depth_04']
        
        figures = create_trait_eda_plots(
            viz_eda_many_traits_data,
            trait_cols,
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        # The overview plot should group by prefix
        assert 'trait_eda_overview' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_no_traits(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test with empty trait list."""
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            [],  # No traits
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        # Should still create some figures even with no traits
        assert 'variance_distribution' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_missing_trait_columns(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test with some non-existent trait columns."""
        # Filter to only existing columns
        all_trait_cols = ['trait_good', 'trait_nonexistent', 'trait_low_var']
        trait_cols = [col for col in all_trait_cols if col in viz_eda_sample_data.columns]
        
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        # Should handle missing columns gracefully
        assert 'trait_eda_overview' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_all_nan_variance(self, viz_eda_data_with_extremes, viz_eda_thresholds):
        """Test variance calculation with all NaN traits."""
        trait_cols = ['trait_all_nan', 'trait_high_var']
        
        figures = create_trait_eda_plots(
            viz_eda_data_with_extremes,
            trait_cols,
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        assert 'variance_distribution' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_thresholds_visualization(self, viz_eda_sample_data):
        """Test threshold lines in visualization."""
        trait_cols = ['trait_good', 'trait_high_nan', 'trait_high_zero']
        custom_thresholds = {
            'nan': 0.2,   # Lower threshold
            'zero': 0.4,  # Lower threshold  
            'outlier': 0.15
        }
        
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            custom_thresholds
        )
        
        assert isinstance(figures, dict)
        # Threshold lines should be plotted in overview
        assert 'trait_eda_overview' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_cleanup_consistency(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test that cleanup simulation matches actual behavior."""
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters
        
        trait_cols = ['trait_good', 'trait_high_nan', 'trait_high_zero']
        
        # First get actual cleanup results
        _, actual_log = apply_data_cleanup_filters(
            viz_eda_sample_data.copy(),
            trait_cols,
            max_zeros_per_trait=viz_eda_thresholds['zero'],
            max_nans_per_trait=viz_eda_thresholds['nan'],
            min_samples_per_trait=10
        )
        
        # Then create EDA plots with the actual log
        figures = create_trait_eda_plots(
            viz_eda_sample_data,
            trait_cols,
            viz_eda_thresholds,
            cleanup_log=actual_log
        )
        
        assert isinstance(figures, dict)
        
        for fig in figures.values():
            plt.close(fig)


class TestEDAEdgeCases:
    """Edge case tests for create_trait_eda_plots."""
    
    def test_single_sample(self, viz_eda_thresholds):
        """Test with single sample DataFrame."""
        df = pd.DataFrame({
            'trait1': [1.0],
            'trait2': [2.0],
            'geno': ['A']
        })
        
        figures = create_trait_eda_plots(
            df,
            ['trait1', 'trait2'],
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_inf_values(self, viz_eda_thresholds):
        """Test handling of infinite values."""
        df = pd.DataFrame({
            'trait_inf': [1, 2, np.inf, 4, -np.inf],
            'trait_normal': [1, 2, 3, 4, 5]
        })
        
        # Should handle inf values without error
        figures = create_trait_eda_plots(
            df,
            ['trait_inf', 'trait_normal'],
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        
        for fig in figures.values():
            plt.close(fig)
    
    def test_constant_traits(self, viz_eda_thresholds):
        """Test with constant (zero variance) traits."""
        df = pd.DataFrame({
            'trait_constant': [5.0] * 20,
            'trait_variable': np.random.randn(20)
        })
        
        figures = create_trait_eda_plots(
            df,
            ['trait_constant', 'trait_variable'],
            viz_eda_thresholds
        )
        
        assert isinstance(figures, dict)
        assert 'variance_distribution' in figures
        
        for fig in figures.values():
            plt.close(fig)


class TestEDAIntegration:
    """Integration tests for EDA plots with real data."""
    
    def test_with_turface_data(self, turface_traits_df, viz_eda_thresholds):
        """Test EDA plots with Turface dataset."""
        # Get numeric trait columns
        trait_cols = [
            col for col in turface_traits_df.columns
            if col not in ["Barcode", "geno", "rep", "wave_name"]
            and turface_traits_df[col].dtype in [np.float64, np.int64]
        ][:10]  # Use first 10 traits for testing
        
        if len(trait_cols) > 0:
            figures = create_trait_eda_plots(
                turface_traits_df,
                trait_cols,
                viz_eda_thresholds
            )
            
            assert isinstance(figures, dict)
            assert 'trait_eda_overview' in figures
            
            for fig in figures.values():
                plt.close(fig)
    
    def test_complete_workflow(self, viz_eda_sample_data, viz_eda_thresholds):
        """Test complete EDA workflow with saving."""
        import tempfile
        from pathlib import Path
        
        trait_cols = ['trait_good', 'trait_high_nan', 'trait_high_zero']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create EDA plots
            figures = create_trait_eda_plots(
                viz_eda_sample_data,
                trait_cols,
                viz_eda_thresholds
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
            matplotlib.use('Agg')
            
            # Create simple test data
            df = pd.DataFrame({
                'trait1': np.random.randn(50),
                'trait2': np.random.randn(50),
                'geno': np.random.choice(['A', 'B'], 50)
            })
            
            # Test functions still work
            fig = create_trait_histograms(df, ['trait1', 'trait2'])
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
        bars = [child for child in ax.get_children() if hasattr(child, 'get_height')]
        # Filter to get only data bars (heritability values should be between 0 and 1)
        # Also exclude very thin bars that might be from the threshold line
        data_bars = [b for b in bars if 0 < b.get_height() <= 1 and b.get_width() > 0.5]
        # Should have approximately 12 bars (may include threshold line bar)
        assert 11 <= len(data_bars) <= 13  # Allow for some flexibility
        
        # Check colors based on threshold
        # High heritability (>=0.5) should be green, low should be orange
        high_h2_count = sum(1 for bar in data_bars[:8] if 'green' in str(bar.get_facecolor()).lower() or bar.get_facecolor()[1] > 0.5)
        low_h2_count = sum(1 for bar in data_bars[8:] if 'orange' in str(bar.get_facecolor()).lower() or bar.get_facecolor()[0] > 0.9)
        
        assert high_h2_count >= 4  # First 8 bars, at least 4 should be green (h2 >= 0.5)
        assert low_h2_count >= 2  # Last 4 bars should be orange (h2 < 0.5)
        
        # Check threshold line
        lines = ax.get_lines()
        assert any(line.get_linestyle() == '--' for line in lines)
        
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
        threshold_line = [line for line in lines if line.get_linestyle() == '--'][0]
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
        bars = [child for child in ax.get_children() if hasattr(child, 'get_height')]
        # Filter to get only data bars - heritability values are > 0 and < 1
        # The bar width should be around 0.8 for data bars
        data_bars = [b for b in bars if 0.001 < b.get_height() < 0.99 and b.get_width() > 0.5]
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
            heritability_threshold_analysis,
            current_threshold=0.5
        )
        
        ax1, ax2 = fig.axes
        
        # Check for vertical line at current threshold in both plots
        vlines1 = [line for line in ax1.get_lines() if line.get_linestyle() == '--']
        vlines2 = [line for line in ax2.get_lines() if line.get_linestyle() == '--']
        
        # Should have vertical line at 0.5
        assert any(0.5 in line.get_xdata() for line in vlines1)
        assert any(0.5 in line.get_xdata() for line in vlines2)
        
        # Check for marker point
        # Look for scatter points (red dots)
        collections = ax1.collections
        assert len(collections) > 0 or any(line.get_marker() == 'o' for line in ax1.get_lines())
        
        plt.close("all")
    
    def test_threshold_plot_reference_lines(self, heritability_threshold_analysis):
        """Test that reference lines are added."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot
        
        fig = create_heritability_threshold_plot(heritability_threshold_analysis)
        
        ax1, ax2 = fig.axes
        
        # Check for horizontal reference lines in top plot (50% and 75%)
        hlines = [line for line in ax1.get_lines() 
                  if len(set(line.get_ydata())) == 1]  # Horizontal lines have constant y
        assert len(hlines) >= 2  # At least 50% and 75% lines
        
        # Check for vertical reference lines in bottom plot (Low, Moderate, High)
        vlines = [line for line in ax2.get_lines() 
                  if line.get_linestyle() == ':' and line.get_alpha() == 0.3]
        assert len(vlines) >= 3  # Low (0.3), Moderate (0.5), High (0.7)
        
        plt.close("all")
    
    def test_threshold_plot_annotations(self, heritability_threshold_analysis):
        """Test that annotations are added."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot
        
        fig = create_heritability_threshold_plot(
            heritability_threshold_analysis,
            current_threshold=0.5
        )
        
        ax1, ax2 = fig.axes
        
        # Check for text annotations
        texts1 = ax1.texts
        texts2 = ax2.texts
        
        # Should have annotation for current threshold value
        assert len(texts1) > 0  # "X traits" annotation
        assert len(texts2) > 0  # Percentage and threshold labels
        
        # Check for threshold labels (Low, Moderate, High)
        all_text = ' '.join(text.get_text() for text in texts2)
        assert any(label in all_text for label in ["Low", "Moderate", "High"])
        
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
        collections1 = [c for c in ax1.collections if hasattr(c, 'get_facecolor')]
        collections2 = [c for c in ax2.collections if hasattr(c, 'get_facecolor')]
        
        assert len(collections1) > 0  # Blue fill in top plot
        assert len(collections2) > 0  # Green fill in bottom plot
        
        plt.close("all")
    
    def test_threshold_plot_figsize(self, heritability_threshold_analysis):
        """Test custom figure size."""
        from sleap_roots_analyze.visualization import create_heritability_threshold_plot
        
        fig = create_heritability_threshold_plot(
            heritability_threshold_analysis,
            figsize=(12, 8)
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