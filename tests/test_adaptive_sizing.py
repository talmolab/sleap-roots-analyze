"""Tests for adaptive figure sizing utilities."""

from __future__ import annotations

import pytest

from sleap_roots_analyze.viz_utils import (
    calculate_barplot_size,
    calculate_correlation_matrix_size,
    calculate_figure_size,
    calculate_grid_dimensions,
    calculate_subplot_grid_size,
    suggest_layout_params,
)


class TestCalculateGridDimensions:
    """Tests for calculate_grid_dimensions function."""

    def test_single_item(self):
        """Test with single item."""
        n_rows, n_cols = calculate_grid_dimensions(1, max_cols=4)
        assert n_rows == 1
        assert n_cols == 1

    def test_items_fit_single_row(self):
        """Test items that fit in single row."""
        n_rows, n_cols = calculate_grid_dimensions(3, max_cols=4)
        assert n_rows == 1
        assert n_cols == 3

    def test_items_require_multiple_rows(self):
        """Test items requiring multiple rows."""
        n_rows, n_cols = calculate_grid_dimensions(10, max_cols=4)
        assert n_rows == 3  # 3 rows x 4 cols = 12 >= 10
        assert n_cols == 4

    def test_exact_grid_fit(self):
        """Test when items exactly fill grid."""
        n_rows, n_cols = calculate_grid_dimensions(12, max_cols=4)
        assert n_rows == 3
        assert n_cols == 4
        assert n_rows * n_cols >= 12

    def test_zero_items(self):
        """Test with zero items."""
        n_rows, n_cols = calculate_grid_dimensions(0, max_cols=4)
        assert n_rows == 0
        assert n_cols == 0

    def test_large_item_count(self):
        """Test with many items."""
        n_rows, n_cols = calculate_grid_dimensions(100, max_cols=5)
        assert n_cols == 5
        assert n_rows == 20  # 100 / 5
        assert n_rows * n_cols >= 100

    def test_different_max_cols(self):
        """Test with different max_cols values."""
        # 3 columns
        n_rows_3, n_cols_3 = calculate_grid_dimensions(10, max_cols=3)
        assert n_cols_3 == 3
        assert n_rows_3 == 4  # Need 4 rows for 10 items

        # 5 columns
        n_rows_5, n_cols_5 = calculate_grid_dimensions(10, max_cols=5)
        assert n_cols_5 == 5
        assert n_rows_5 == 2  # Need 2 rows for 10 items


class TestCalculateFigureSize:
    """Tests for calculate_figure_size function."""

    def test_single_layout(self, adaptive_sizing_config):
        """Test single item layout."""
        width, height = calculate_figure_size(
            1, adaptive_sizing_config, layout="single"
        )
        assert width == adaptive_sizing_config.base_width
        assert height == adaptive_sizing_config.base_height

    def test_horizontal_layout(self, adaptive_sizing_config):
        """Test horizontal layout."""
        width, height = calculate_figure_size(
            5, adaptive_sizing_config, layout="horizontal"
        )
        # Width should scale with items
        expected_width = (
            adaptive_sizing_config.base_width
            + 4 * adaptive_sizing_config.width_per_item
        )
        assert width == expected_width
        assert height == adaptive_sizing_config.base_height

    def test_vertical_layout(self, adaptive_sizing_config):
        """Test vertical layout."""
        width, height = calculate_figure_size(
            5, adaptive_sizing_config, layout="vertical"
        )
        # Height should scale with items
        expected_height = (
            adaptive_sizing_config.base_height
            + 4 * adaptive_sizing_config.height_per_item
        )
        assert width == adaptive_sizing_config.base_width
        assert height == expected_height

    def test_grid_layout(self, adaptive_sizing_config):
        """Test grid layout."""
        width, height = calculate_figure_size(
            12, adaptive_sizing_config, layout="grid", max_cols=4
        )
        # 12 items in 4-column grid = 3 rows x 4 cols
        expected_width = (
            adaptive_sizing_config.base_width
            + 3 * adaptive_sizing_config.width_per_item
        )
        expected_height = (
            adaptive_sizing_config.base_height
            + 2 * adaptive_sizing_config.height_per_item
        )
        assert width == expected_width
        assert height == expected_height

    def test_min_bounds_enforcement(self, adaptive_sizing_config):
        """Test that min bounds are enforced."""
        # Set very small base size
        adaptive_sizing_config.base_width = 2.0
        adaptive_sizing_config.base_height = 2.0

        width, height = calculate_figure_size(
            1, adaptive_sizing_config, layout="single"
        )
        # Should be clamped to min
        assert width == adaptive_sizing_config.min_width
        assert height == adaptive_sizing_config.min_height

    def test_max_bounds_enforcement(self, adaptive_sizing_config):
        """Test that max bounds are enforced."""
        width, height = calculate_figure_size(
            1000, adaptive_sizing_config, layout="horizontal"
        )
        # Should be clamped to max
        assert width == adaptive_sizing_config.max_width
        assert height <= adaptive_sizing_config.max_height

    def test_disabled_adaptive_sizing(self, adaptive_sizing_config_disabled):
        """Test when adaptive sizing is disabled."""
        width, height = calculate_figure_size(
            100, adaptive_sizing_config_disabled, layout="grid"
        )
        # Should return base size regardless of item count
        assert width == adaptive_sizing_config_disabled.base_width
        assert height == adaptive_sizing_config_disabled.base_height

    def test_invalid_layout_raises_error(self, adaptive_sizing_config):
        """Test that invalid layout raises ValueError."""
        with pytest.raises(ValueError, match="Invalid layout type"):
            calculate_figure_size(5, adaptive_sizing_config, layout="invalid")


class TestCalculateSubplotGridSize:
    """Tests for calculate_subplot_grid_size function."""

    def test_single_trait(self, adaptive_sizing_config):
        """Test with single trait."""
        width, height = calculate_subplot_grid_size(
            1, adaptive_sizing_config, max_cols=4
        )
        # 1 trait = 1x1 grid, 1*4.0 = 4.0, but clamped to min_width=6.0
        assert width == 6.0  # Clamped to min_width
        assert height == 4.0  # Clamped to min_height (3.0 -> 4.0)

    def test_multiple_traits_grid(self, adaptive_sizing_config):
        """Test with multiple traits in grid."""
        width, height = calculate_subplot_grid_size(
            12, adaptive_sizing_config, max_cols=4
        )
        # 12 traits = 3 rows x 4 cols
        assert width == 16.0  # 4 * 4.0
        assert height == 9.0  # 3 * 3.0

    def test_custom_subplot_dimensions(self, adaptive_sizing_config):
        """Test with custom subplot dimensions."""
        width, height = calculate_subplot_grid_size(
            6,
            adaptive_sizing_config,
            max_cols=3,
            width_per_subplot=5.0,
            height_per_subplot=4.0,
        )
        # 6 traits = 2 rows x 3 cols
        assert width == 15.0  # 3 * 5.0
        assert height == 8.0  # 2 * 4.0

    def test_bounds_clamping(self, adaptive_sizing_config):
        """Test that bounds are respected."""
        # Very large trait count
        width, height = calculate_subplot_grid_size(
            200, adaptive_sizing_config, max_cols=10
        )
        # Should be clamped to max
        assert width <= adaptive_sizing_config.max_width
        assert height <= adaptive_sizing_config.max_height

    def test_disabled_adaptive_sizing(self, adaptive_sizing_config_disabled):
        """Test with adaptive sizing disabled."""
        width, height = calculate_subplot_grid_size(
            100, adaptive_sizing_config_disabled, max_cols=4
        )
        # Should still calculate based on grid but may differ from enabled
        n_rows, n_cols = calculate_grid_dimensions(100, max_cols=4)
        expected_width = n_cols * 4.0
        expected_height = n_rows * 3.0
        assert width == expected_width
        assert height == expected_height


class TestCalculateCorrelationMatrixSize:
    """Tests for calculate_correlation_matrix_size function."""

    def test_small_matrix(self, adaptive_sizing_config):
        """Test with small number of traits."""
        width, height = calculate_correlation_matrix_size(
            5, adaptive_sizing_config, min_size_per_trait=0.5
        )
        # Expected: 5 * 0.5 + 2.0 = 4.5, but clamped to min_width=6.0
        assert width == 6.0  # Clamped to min
        assert height == 6.0  # Clamped to min

    def test_large_matrix(self, adaptive_sizing_config):
        """Test with many traits."""
        width, height = calculate_correlation_matrix_size(
            50, adaptive_sizing_config, min_size_per_trait=0.5
        )
        # Expected: 50 * 0.5 + 2.0 = 27.0, clamped to max_width=20.0, max_height=16.0
        # But width and height are the same (square), so both use max_width
        assert width == 20.0  # Clamped to max_width
        assert height == 20.0  # Also clamped to max_width (square matrix)

    def test_square_output(self, adaptive_sizing_config):
        """Test that output is square."""
        width, height = calculate_correlation_matrix_size(20, adaptive_sizing_config)
        assert width == height

    def test_disabled_adaptive_sizing(self, adaptive_sizing_config_disabled):
        """Test with adaptive sizing disabled."""
        width, height = calculate_correlation_matrix_size(
            100, adaptive_sizing_config_disabled
        )
        assert width == adaptive_sizing_config_disabled.base_width
        assert height == adaptive_sizing_config_disabled.base_height


class TestCalculateBarplotSize:
    """Tests for calculate_barplot_size function."""

    def test_vertical_barplot(self, adaptive_sizing_config):
        """Test vertical bar plot sizing."""
        width, height = calculate_barplot_size(
            20, adaptive_sizing_config, orientation="vertical"
        )
        # Width scales with number of bars
        expected_width = 20 * 0.8  # = 16.0
        assert width == expected_width
        assert height == adaptive_sizing_config.base_height

    def test_horizontal_barplot(self, adaptive_sizing_config):
        """Test horizontal bar plot sizing."""
        width, height = calculate_barplot_size(
            30, adaptive_sizing_config, orientation="horizontal"
        )
        # Height scales with number of bars
        expected_height = 30 * 0.6  # = 18.0
        assert width == adaptive_sizing_config.base_width
        # Should be clamped to max height
        assert height == min(expected_height, adaptive_sizing_config.max_height)

    def test_custom_bar_dimensions(self, adaptive_sizing_config):
        """Test with custom bar dimensions."""
        width, height = calculate_barplot_size(
            10,
            adaptive_sizing_config,
            orientation="vertical",
            width_per_bar=1.0,
            height_per_bar=0.8,
        )
        expected_width = 10 * 1.0  # = 10.0
        assert width == expected_width

    def test_bounds_enforcement(self, adaptive_sizing_config):
        """Test that bounds are enforced."""
        # Very many bars
        width, height = calculate_barplot_size(
            1000, adaptive_sizing_config, orientation="horizontal"
        )
        assert width <= adaptive_sizing_config.max_width
        assert height <= adaptive_sizing_config.max_height

    def test_invalid_orientation_raises_error(self, adaptive_sizing_config):
        """Test that invalid orientation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid orientation"):
            calculate_barplot_size(10, adaptive_sizing_config, orientation="diagonal")


class TestSuggestLayoutParams:
    """Tests for suggest_layout_params function."""

    def test_small_item_count(self):
        """Test with few items."""
        params = suggest_layout_params(6, (12.0, 9.0), max_cols=3)
        assert params["nrows"] == 2
        assert params["ncols"] == 3
        assert params["figsize"] == (12.0, 9.0)
        assert "hspace" in params
        assert "wspace" in params
        assert 0 < params["hspace"] < 1.0
        assert 0 < params["wspace"] < 1.0

    def test_large_item_count(self):
        """Test with many items."""
        params = suggest_layout_params(50, (20.0, 16.0), max_cols=5)
        assert params["nrows"] == 10  # 50 / 5
        assert params["ncols"] == 5
        # Spacing should increase for many items
        assert params["hspace"] > 0.2

    def test_spacing_scales_with_items(self):
        """Test that spacing increases with more items."""
        params_few = suggest_layout_params(4, (10.0, 8.0), max_cols=2)
        params_many = suggest_layout_params(40, (10.0, 8.0), max_cols=2)

        # More items should have more spacing
        assert params_many["hspace"] > params_few["hspace"]
        assert params_many["wspace"] > params_few["wspace"]

    def test_spacing_capped_at_half(self):
        """Test that spacing doesn't exceed 0.5."""
        params = suggest_layout_params(200, (20.0, 16.0), max_cols=10)
        assert params["hspace"] <= 0.5
        assert params["wspace"] <= 0.5

    def test_single_item(self):
        """Test with single item."""
        params = suggest_layout_params(1, (8.0, 6.0), max_cols=4)
        assert params["nrows"] == 1
        assert params["ncols"] == 1
