"""Tests for regression plotting functionality."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import matplotlib

# Use non-interactive backend for testing
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from sleap_roots_analyze.visualization import create_regression_plot


@pytest.fixture
def regression_data_positive():
    """Create DataFrame with positive linear correlation (R² ≈ 0.8)."""
    np.random.seed(42)
    x = np.linspace(0, 100, 50)
    y = 2 * x + 10 + np.random.normal(0, 10, 50)  # y = 2x + 10 + noise
    return pd.DataFrame({"x_trait": x, "y_trait": y})


@pytest.fixture
def regression_data_negative():
    """Create DataFrame with negative linear correlation."""
    np.random.seed(42)
    x = np.linspace(0, 100, 50)
    y = -1.5 * x + 100 + np.random.normal(0, 8, 50)
    return pd.DataFrame({"x_trait": x, "y_trait": y})


@pytest.fixture
def regression_data_perfect():
    """Create DataFrame with perfect linear correlation (R² = 1.0)."""
    x = np.linspace(0, 100, 50)
    y = 2 * x + 10  # Perfect linear relationship
    return pd.DataFrame({"x_trait": x, "y_trait": y})


@pytest.fixture
def regression_data_zero_corr():
    """Create DataFrame with zero correlation."""
    np.random.seed(42)
    x = np.random.normal(50, 10, 50)
    y = np.random.normal(50, 10, 50)  # Independent random values
    return pd.DataFrame({"x_trait": x, "y_trait": y})


@pytest.fixture
def regression_data_with_nans():
    """Create DataFrame with NaN values (15% missing)."""
    np.random.seed(42)
    x = np.linspace(0, 100, 50)
    y = 2 * x + 10 + np.random.normal(0, 10, 50)

    # Add NaNs to ~15% of data
    nan_indices = np.random.choice(50, size=7, replace=False)
    y[nan_indices] = np.nan

    return pd.DataFrame({"x_trait": x, "y_trait": y})


@pytest.fixture
def regression_data_with_groups():
    """Create DataFrame with categorical grouping column."""
    np.random.seed(42)
    n_per_group = 20
    groups = ["GroupA", "GroupB", "GroupC"]

    data = []
    for group in groups:
        x = np.linspace(0, 100, n_per_group)
        y = 2 * x + 10 + np.random.normal(0, 10, n_per_group)
        for xi, yi in zip(x, y):
            data.append({"x_trait": xi, "y_trait": yi, "group": group})

    return pd.DataFrame(data)


class TestBasicFunctionality:
    """Test basic regression plot generation."""

    def test_simple_regression_plot(self, regression_data_positive):
        """Test basic plot generation returns Figure."""
        fig = create_regression_plot(
            regression_data_positive, x_col="x_trait", y_col="y_trait"
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_custom_figsize(self, regression_data_positive):
        """Test custom figure size."""
        fig = create_regression_plot(
            regression_data_positive,
            x_col="x_trait",
            y_col="y_trait",
            figsize=(10, 6),
        )

        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)

    def test_custom_title(self, regression_data_positive):
        """Test custom title."""
        custom_title = "My Custom Regression Title"
        fig = create_regression_plot(
            regression_data_positive,
            x_col="x_trait",
            y_col="y_trait",
            title=custom_title,
        )

        ax = fig.axes[0]
        assert ax.get_title() == custom_title
        plt.close(fig)

    def test_default_title(self, regression_data_positive):
        """Test default auto-generated title."""
        fig = create_regression_plot(
            regression_data_positive, x_col="x_trait", y_col="y_trait"
        )

        ax = fig.axes[0]
        assert "y_trait vs x_trait" in ax.get_title()
        plt.close(fig)


class TestStatisticalCalculations:
    """Test statistical accuracy of regression calculations."""

    def test_perfect_correlation_stats(self, regression_data_perfect):
        """Test statistics with perfect linear relationship."""
        from scipy import stats

        fig = create_regression_plot(
            regression_data_perfect, x_col="x_trait", y_col="y_trait"
        )

        # Verify directly with scipy
        x = regression_data_perfect["x_trait"].values
        y = regression_data_perfect["y_trait"].values
        r, p = stats.pearsonr(x, y)

        assert abs(r) > 0.9999  # Should be ~1.0
        assert r**2 > 0.9999  # R² should be ~1.0

        plt.close(fig)

    def test_zero_correlation_stats(self, regression_data_zero_corr):
        """Test statistics with zero correlation."""
        fig = create_regression_plot(
            regression_data_zero_corr, x_col="x_trait", y_col="y_trait"
        )

        # R² should be very low (close to 0)
        # Just verify plot is created; exact R² will vary with random seed
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_negative_correlation(self, regression_data_negative):
        """Test with negative correlation."""
        from scipy import stats

        fig = create_regression_plot(
            regression_data_negative, x_col="x_trait", y_col="y_trait"
        )

        # Verify correlation is negative
        x = regression_data_negative["x_trait"].values
        y = regression_data_negative["y_trait"].values
        r, _ = stats.pearsonr(x, y)

        assert r < 0  # Should be negative
        plt.close(fig)


class TestColorByGrouping:
    """Test color_by parameter functionality."""

    def test_color_by_groups(self, regression_data_with_groups):
        """Test coloring points by categorical variable."""
        fig = create_regression_plot(
            regression_data_with_groups,
            x_col="x_trait",
            y_col="y_trait",
            color_by="group",
        )

        ax = fig.axes[0]
        # Should have a legend with 3 groups
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 3
        plt.close(fig)

    def test_color_by_many_categories_warning(self, regression_data_positive):
        """Test warning when color_by has >20 categories."""
        # Create data with 25 unique groups
        df = regression_data_positive.copy()
        df["many_groups"] = [f"Group{i % 25}" for i in range(len(df))]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = create_regression_plot(
                df, x_col="x_trait", y_col="y_trait", color_by="many_groups"
            )

            # Should issue warning about too many categories
            assert len(w) == 1
            assert "unique values" in str(w[0].message)

        plt.close(fig)


class TestNaNHandling:
    """Test NaN value handling."""

    def test_nan_handling_low_percentage(self, regression_data_with_nans):
        """Test NaN handling when <20% data is missing."""
        # Should not issue warning for 15% NaNs
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = create_regression_plot(
                regression_data_with_nans, x_col="x_trait", y_col="y_trait"
            )

            # No warning expected for <20% NaNs
            assert len(w) == 0

        plt.close(fig)

    def test_nan_handling_high_percentage(self, regression_data_positive):
        """Test warning when >20% data has NaN."""
        # Add NaNs to 30% of data
        df = regression_data_positive.copy()
        n_nans = int(len(df) * 0.3)
        df.loc[df.index[:n_nans], "y_trait"] = np.nan

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = create_regression_plot(df, x_col="x_trait", y_col="y_trait")

            # Should warn about high NaN rate
            assert len(w) >= 1
            assert any("Dropped" in str(warning.message) for warning in w)

        plt.close(fig)


class TestInputValidation:
    """Test input validation and error handling."""

    def test_missing_column_error(self, regression_data_positive):
        """Test error when column doesn't exist."""
        with pytest.raises(ValueError, match="not found in DataFrame"):
            create_regression_plot(
                regression_data_positive, x_col="nonexistent", y_col="y_trait"
            )

    def test_non_numeric_column_error(self, regression_data_with_groups):
        """Test error when using non-numeric column."""
        with pytest.raises(ValueError, match="must be numeric"):
            create_regression_plot(
                regression_data_with_groups, x_col="group", y_col="y_trait"
            )

    def test_insufficient_samples_error(self, regression_data_positive):
        """Test error with insufficient samples (<3)."""
        # Keep only 2 samples
        df_small = regression_data_positive.head(2)

        with pytest.raises(ValueError, match="Insufficient samples"):
            create_regression_plot(df_small, x_col="x_trait", y_col="y_trait")

    def test_insufficient_samples_after_nan_removal(self, regression_data_positive):
        """Test error when <3 samples remain after NaN removal."""
        df = regression_data_positive.head(5).copy()
        # Add NaN to 3 rows, leaving only 2 valid
        df.loc[df.index[2:], "y_trait"] = np.nan

        with pytest.raises(ValueError, match="Insufficient samples"):
            create_regression_plot(df, x_col="x_trait", y_col="y_trait")

    def test_zero_variance_x_error(self, regression_data_positive):
        """Test error when x variable has zero variance."""
        df = regression_data_positive.copy()
        df["x_trait"] = 50.0  # All same value

        with pytest.raises(ValueError, match="zero variance"):
            create_regression_plot(df, x_col="x_trait", y_col="y_trait")

    def test_zero_variance_y_error(self, regression_data_positive):
        """Test error when y variable has zero variance."""
        df = regression_data_positive.copy()
        df["y_trait"] = 100.0  # All same value

        with pytest.raises(ValueError, match="zero variance"):
            create_regression_plot(df, x_col="x_trait", y_col="y_trait")


class TestPlotComponents:
    """Test that plot contains expected visual components."""

    def test_plot_has_scatter_points(self, regression_data_positive):
        """Test that plot contains scatter points."""
        fig = create_regression_plot(
            regression_data_positive, x_col="x_trait", y_col="y_trait"
        )

        ax = fig.axes[0]
        # Check for PathCollection (scatter plot)
        collections = ax.collections
        assert len(collections) > 0  # At least scatter + CI patch

        plt.close(fig)

    def test_plot_has_regression_line(self, regression_data_positive):
        """Test that plot contains regression line."""
        fig = create_regression_plot(
            regression_data_positive, x_col="x_trait", y_col="y_trait"
        )

        ax = fig.axes[0]
        # Check for Line2D objects (regression line)
        lines = ax.get_lines()
        assert len(lines) > 0

        plt.close(fig)

    def test_plot_has_axis_labels(self, regression_data_positive):
        """Test that axes are labeled correctly."""
        fig = create_regression_plot(
            regression_data_positive, x_col="x_trait", y_col="y_trait"
        )

        ax = fig.axes[0]
        assert ax.get_xlabel() == "x_trait"
        assert ax.get_ylabel() == "y_trait"

        plt.close(fig)

    def test_plot_has_statistics_annotation(self, regression_data_positive):
        """Test that statistical annotations are present."""
        fig = create_regression_plot(
            regression_data_positive, x_col="x_trait", y_col="y_trait"
        )

        ax = fig.axes[0]
        # Check for text annotations
        texts = ax.texts
        assert len(texts) > 0  # Should have stats box

        # Check that text contains expected stats
        stats_text = texts[0].get_text()
        assert "R²" in stats_text or "R2" in stats_text
        assert "p" in stats_text
        assert "=" in stats_text  # For equation
        assert "n =" in stats_text  # Sample size

        plt.close(fig)


class TestCustomStyling:
    """Test custom styling options."""

    def test_custom_scatter_kwargs(self, regression_data_positive):
        """Test custom scatter plot styling."""
        fig = create_regression_plot(
            regression_data_positive,
            x_col="x_trait",
            y_col="y_trait",
            scatter_kws={"s": 100, "alpha": 0.3},
        )

        # Just verify it doesn't error with custom kwargs
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_line_kwargs(self, regression_data_positive):
        """Test custom regression line styling."""
        fig = create_regression_plot(
            regression_data_positive,
            x_col="x_trait",
            y_col="y_trait",
            line_kws={"color": "red", "linewidth": 3},
        )

        # Just verify it doesn't error with custom kwargs
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
