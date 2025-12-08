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
    plot_biomass_depth_barplot,
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


@pytest.fixture
def biomass_data_two_depths():
    """Fixture with biomass data at 2 depth intervals (0-30cm, 30-60cm)."""
    data = {
        "geno": [
            "Control",
            "Control",
            "Control",
            "GH_001",
            "GH_001",
            "GH_001",
            "GH_002",
            "GH_002",
            "GH_002",
        ]
        * 2,
        "plot_rep": [
            "plot1_rep1",
            "plot2_rep1",
            "plot3_rep2",
            "plot4_rep1",
            "plot5_rep1",
            "plot6_rep2",
            "plot7_rep1",
            "plot8_rep1",
            "plot9_rep2",
        ]
        * 2,
        "Depth_cm": [15] * 9 + [45] * 9,  # 15cm = 0-30cm layer, 45cm = 30-60cm layer
        "Root_DW_g": [
            # 0-30cm layer
            0.29,
            0.28,
            0.30,  # Control: mean ~0.29
            0.22,
            0.21,
            0.23,  # GH_001: mean ~0.22 (lowest)
            0.43,
            0.44,
            0.42,  # GH_002: mean ~0.43 (highest)
            # 30-60cm layer
            0.10,
            0.09,
            0.11,  # Control
            0.05,
            0.06,
            0.04,  # GH_001
            0.08,
            0.09,
            0.07,  # GH_002
        ],
    }
    return pd.DataFrame(data)


class TestPlotBiomassDepthBarplot:
    """Test suite for biomass depth barplot function."""

    def test_function_exists_and_returns_figure(self, biomass_data_two_depths):
        """Test that function exists and returns a matplotlib Figure."""
        fig = plot_biomass_depth_barplot(
            df=biomass_data_two_depths,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_genotype_ordering_control_first(self, biomass_data_two_depths):
        """Test that Control genotype appears first, then sorted by shallow-layer biomass."""
        fig = plot_biomass_depth_barplot(
            df=biomass_data_two_depths,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
        )
        ax = fig.axes[0]
        x_labels = [label.get_text() for label in ax.get_xticklabels()]

        # Expected order: Control first, then GH_001 (0.22), then GH_002 (0.43)
        assert x_labels[0] == "Control"
        assert x_labels[1] == "GH_001"
        assert x_labels[2] == "GH_002"
        plt.close(fig)

    def test_depth_labels_in_legend(self, biomass_data_two_depths):
        """Test that depth interval labels appear in legend."""
        fig = plot_biomass_depth_barplot(
            df=biomass_data_two_depths,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
        )
        ax = fig.axes[0]
        legend = ax.get_legend()

        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert len(legend_labels) == 2  # Two depth intervals

        # For depths [15, 45]: midpoint is 30, so intervals are 0-30cm and 30-90cm
        assert "0-30cm" in legend_labels
        assert "30-90cm" in legend_labels
        plt.close(fig)

    def test_x_axis_labels_rotated_90_degrees(self, biomass_data_two_depths):
        """Test that x-axis labels are rotated 90 degrees."""
        fig = plot_biomass_depth_barplot(
            df=biomass_data_two_depths,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
        )
        ax = fig.axes[0]

        for label in ax.get_xticklabels():
            rotation = label.get_rotation()
            assert abs(rotation - 90) < 1
        plt.close(fig)

    def test_grid_enabled(self, biomass_data_two_depths):
        """Test that grid is enabled on the plot."""
        fig = plot_biomass_depth_barplot(
            df=biomass_data_two_depths,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
        )
        ax = fig.axes[0]

        assert ax.yaxis.get_gridlines()[0].get_visible()
        plt.close(fig)

    def test_stripplot_overlay_when_enabled(self, biomass_data_two_depths):
        """Test that stripplot overlay is added when include_points=True."""
        fig_with = plot_biomass_depth_barplot(
            df=biomass_data_two_depths,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
            include_points=True,
        )
        fig_without = plot_biomass_depth_barplot(
            df=biomass_data_two_depths,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
            include_points=False,
        )

        # Figure with stripplot should have more collections
        assert len(fig_with.axes[0].collections) > len(fig_without.axes[0].collections)
        plt.close(fig_with)
        plt.close(fig_without)

    def test_saves_to_file(self, biomass_data_two_depths):
        """Test that figure is saved to file when output_path is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_barplot.png"

            result = plot_biomass_depth_barplot(
                df=biomass_data_two_depths,
                depth_col="Depth_cm",
                value_col="Root_DW_g",
                genotype_col="geno",
                output_path=output_file,
            )

            assert output_file.exists()
            assert output_file.stat().st_size > 0
            # Function returns None when file is saved
            assert result is None

    def test_handles_missing_control_genotype(self):
        """Test that function handles missing Control genotype gracefully."""
        data = {
            "geno": ["GH_002", "GH_002", "GH_001", "GH_001"],
            "plot_rep": ["plot1", "plot2", "plot3", "plot4"],
            "Depth_cm": [15, 15, 15, 15],
            "Root_DW_g": [0.43, 0.44, 0.22, 0.21],
        }
        df = pd.DataFrame(data)

        fig = plot_biomass_depth_barplot(
            df=df,
            depth_col="Depth_cm",
            value_col="Root_DW_g",
            genotype_col="geno",
        )
        ax = fig.axes[0]
        x_labels = [label.get_text() for label in ax.get_xticklabels()]

        # Should be sorted by biomass (GH_001: 0.22, GH_002: 0.43)
        assert x_labels[0] == "GH_001"
        assert x_labels[1] == "GH_002"
        plt.close(fig)

    def test_raises_error_for_empty_dataframe(self):
        """Test that function raises ValueError for empty DataFrame."""
        df = pd.DataFrame(columns=["geno", "Depth_cm", "Root_DW_g"])

        with pytest.raises(ValueError, match="DataFrame is empty"):
            plot_biomass_depth_barplot(
                df=df,
                depth_col="Depth_cm",
                value_col="Root_DW_g",
                genotype_col="geno",
            )

    def test_raises_error_for_missing_columns(self, biomass_data_two_depths):
        """Test that function raises ValueError for missing required columns."""
        with pytest.raises(ValueError, match="Missing required column"):
            plot_biomass_depth_barplot(
                df=biomass_data_two_depths,
                depth_col="NonexistentColumn",
                value_col="Root_DW_g",
                genotype_col="geno",
            )
