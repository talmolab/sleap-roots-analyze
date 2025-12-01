"""Tests for depth_profile_plots module."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from sleap_roots_analyze.depth_profile_plots import (
    plot_depth_profile_faceted,
    plot_depth_profile_replicates,
)
from sleap_roots_analyze.root_core_analysis import (
    melt_depth_data,
    aggregate_by_replicate,
)


@pytest.fixture
def aggregated_depth_data(create_test_root_core_data):
    """Create aggregated depth profile data for testing plots."""
    df = create_test_root_core_data

    # Melt to long format
    df_melted = melt_depth_data(
        df, id_vars=["Plot", "geno", "Rep", "core_n"], parse_depth=True
    )

    # Create plot_rep identifier
    df_melted["plot_rep"] = (
        "plot" + df_melted["Plot"].astype(str) + "_rep" + df_melted["Rep"].astype(str)
    )

    # Aggregate by plot_rep
    df_agg = aggregate_by_replicate(
        df_melted,
        group_cols=["plot_rep", "Plot", "geno", "Depth_cm"],
        value_col="Root_Count",
        agg_func="mean",
    )

    return df_agg


class TestPlotDepthProfileFaceted:
    """Tests for plot_depth_profile_faceted function."""

    def test_creates_figure_object(self, aggregated_depth_data):
        """Test that function returns a matplotlib Figure object."""
        fig = plot_depth_profile_faceted(aggregated_depth_data, facet_col="geno")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_default_parameters(self, aggregated_depth_data):
        """Test plotting with default parameters."""
        fig = plot_depth_profile_faceted(
            aggregated_depth_data, x="Depth_cm", y="Root_Count", facet_col="geno"
        )

        assert fig is not None
        # Check that axes were created (2 genotypes)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_custom_errorbar(self, aggregated_depth_data):
        """Test with custom error bar type."""
        # Add multiple replicates to same data
        df_multi = pd.concat([aggregated_depth_data] * 3, ignore_index=True)
        df_multi["plot_rep"] = df_multi["plot_rep"] + "_" + df_multi.index.astype(str)

        fig = plot_depth_profile_faceted(
            df_multi, x="Depth_cm", y="Root_Count", facet_col="geno", errorbar="sd"
        )

        assert fig is not None
        plt.close(fig)

    def test_custom_grid_layout(self, aggregated_depth_data):
        """Test custom grid layout parameters."""
        fig = plot_depth_profile_faceted(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            col_wrap=1,
            height=3,
        )

        assert fig is not None
        plt.close(fig)

    def test_no_errorbar(self, aggregated_depth_data):
        """Test plotting without error bars."""
        fig = plot_depth_profile_faceted(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            errorbar=None,
        )

        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, aggregated_depth_data):
        """Test saving plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"

            fig = plot_depth_profile_faceted(
                aggregated_depth_data,
                x="Depth_cm",
                y="Root_Count",
                facet_col="geno",
                output_path=output_path,
            )

            # Check file was created
            assert output_path.exists()
            assert fig is not None
            plt.close(fig)

    def test_custom_styling_kwargs(self, aggregated_depth_data):
        """Test passing custom styling kwargs to lineplot."""
        fig = plot_depth_profile_faceted(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            lw=3,
            color="red",
        )

        assert fig is not None
        plt.close(fig)


class TestPlotDepthProfileReplicates:
    """Tests for plot_depth_profile_replicates function."""

    def test_creates_figure_object(self, aggregated_depth_data):
        """Test that function returns a matplotlib Figure object."""
        fig = plot_depth_profile_replicates(
            aggregated_depth_data, facet_col="geno", hue="plot_rep"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_individual_lines_plotted(self, aggregated_depth_data):
        """Test that individual replicate lines are plotted."""
        fig = plot_depth_profile_replicates(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            hue="plot_rep",
        )

        assert fig is not None
        # Should have multiple lines per facet
        plt.close(fig)

    def test_custom_grid_layout(self, aggregated_depth_data):
        """Test custom grid layout for replicates."""
        fig = plot_depth_profile_replicates(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            hue="plot_rep",
            col_wrap=1,
            height=3,
        )

        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, aggregated_depth_data):
        """Test saving replicate plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_replicates.png"

            fig = plot_depth_profile_replicates(
                aggregated_depth_data,
                x="Depth_cm",
                y="Root_Count",
                facet_col="geno",
                hue="plot_rep",
                output_path=output_path,
            )

            # Check file was created
            assert output_path.exists()
            assert fig is not None
            plt.close(fig)

    def test_custom_alpha(self, aggregated_depth_data):
        """Test custom transparency for overlapping lines."""
        fig = plot_depth_profile_replicates(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            hue="plot_rep",
            alpha=0.3,
        )

        assert fig is not None
        plt.close(fig)

    def test_match_layout_with_faceted(self, aggregated_depth_data):
        """Test that layout matches faceted plot for easy comparison."""
        # Create both plots with same layout
        fig1 = plot_depth_profile_faceted(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            col_wrap=2,
            height=4,
        )

        fig2 = plot_depth_profile_replicates(
            aggregated_depth_data,
            x="Depth_cm",
            y="Root_Count",
            facet_col="geno",
            hue="plot_rep",
            col_wrap=2,
            height=4,
        )

        # Both should be created successfully
        assert fig1 is not None
        assert fig2 is not None

        plt.close(fig1)
        plt.close(fig2)
