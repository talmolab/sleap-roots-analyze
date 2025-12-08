"""Tests for cross-platform analysis helper functions."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from scipy import stats


def test_calculate_correlations_spearman():
    """Test Spearman correlation captures non-linear monotonic relationships."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Create non-linear monotonic relationship: y = x^2
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = x**2

    r, p = calculate_correlations(x, y, method="spearman")

    # Spearman should detect perfect monotonic relationship
    assert r == pytest.approx(1.0, abs=1e-10)
    assert p < 0.001


def test_calculate_correlations_pearson():
    """Test Pearson correlation captures linear relationships."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Create linear relationship: y = 2x + 3
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = 2 * x + 3

    r, p = calculate_correlations(x, y, method="pearson")

    # Pearson should detect perfect linear relationship
    assert r == pytest.approx(1.0, abs=1e-10)
    assert p < 0.001


def test_calculate_correlations_kendall():
    """Test Kendall correlation for robust small-sample correlation."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Small sample with monotonic relationship
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    r, p = calculate_correlations(x, y, method="kendall")

    # Kendall should detect perfect concordance
    assert r == pytest.approx(1.0, abs=1e-10)
    assert p < 0.05


def test_calculate_correlations_spearman_vs_pearson():
    """Test that Spearman handles non-linear better than Pearson."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Exponential relationship
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.exp(x / 2)

    r_spearman, _ = calculate_correlations(x, y, method="spearman")
    r_pearson, _ = calculate_correlations(x, y, method="pearson")

    # Spearman should be perfect (monotonic), Pearson should be lower (non-linear)
    assert r_spearman == pytest.approx(1.0, abs=1e-10)
    assert r_pearson < r_spearman


def test_calculate_correlations_negative_correlation():
    """Test detection of negative correlations."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = -2 * x + 50

    r_pearson, p_pearson = calculate_correlations(x, y, method="pearson")
    r_spearman, p_spearman = calculate_correlations(x, y, method="spearman")

    # Both should detect perfect negative correlation
    assert r_pearson == pytest.approx(-1.0, abs=1e-10)
    assert r_spearman == pytest.approx(-1.0, abs=1e-10)
    assert p_pearson < 0.001
    assert p_spearman < 0.001


def test_calculate_correlations_no_correlation():
    """Test detection of no correlation."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Random uncorrelated data
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)

    r_pearson, p_pearson = calculate_correlations(x, y, method="pearson")
    r_spearman, p_spearman = calculate_correlations(x, y, method="spearman")

    # Correlation should be close to zero, p-value should be high
    assert abs(r_pearson) < 0.2
    assert abs(r_spearman) < 0.2
    assert p_pearson > 0.05
    assert p_spearman > 0.05


def test_calculate_correlations_with_nans_dropped():
    """Test that NaN values are properly handled (dropped)."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Data with NaN values
    x = np.array([1, 2, np.nan, 4, 5, 6, 7, np.nan, 9, 10], dtype=float)
    y = np.array([2, 4, 6, np.nan, 10, 12, np.nan, 16, 18, 20], dtype=float)

    r, p = calculate_correlations(x, y, method="pearson")

    # After dropping NaNs, should have 6 pairs: (1,2), (2,4), (4,10), (5,10), (6,12), (10,20)
    # Wait, need to align properly - pairs where BOTH are non-NaN
    # (1,2), (2,4), (5,10), (6,12), (10,20) = 5 pairs
    # These should still show strong positive correlation
    assert r > 0.9
    assert p < 0.05


def test_calculate_correlations_insufficient_data():
    """Test that insufficient data (< 3 samples) returns NaN."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Only 2 samples
    x = np.array([1, 2], dtype=float)
    y = np.array([2, 4], dtype=float)

    r, p = calculate_correlations(x, y, method="pearson")

    # Should return NaN for insufficient data
    assert np.isnan(r)
    assert np.isnan(p)


def test_calculate_correlations_all_nans():
    """Test that all NaN values returns NaN."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    x = np.array([np.nan, np.nan, np.nan], dtype=float)
    y = np.array([np.nan, np.nan, np.nan], dtype=float)

    r, p = calculate_correlations(x, y, method="spearman")

    assert np.isnan(r)
    assert np.isnan(p)


def test_calculate_correlations_constant_values():
    """Test that constant values (zero variance) returns NaN."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    # Constant x values - undefined correlation
    x = np.array([5, 5, 5, 5, 5], dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)

    r, p = calculate_correlations(x, y, method="pearson")

    # Correlation with constant variable is undefined
    assert np.isnan(r)
    assert np.isnan(p)


def test_calculate_correlations_invalid_method():
    """Test that invalid method raises ValueError."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    with pytest.raises(ValueError, match="method must be one of"):
        calculate_correlations(x, y, method="invalid_method")


def test_calculate_correlations_different_lengths():
    """Test that different length arrays raises ValueError."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6], dtype=float)

    with pytest.raises(ValueError, match="x and y must have the same length"):
        calculate_correlations(x, y, method="pearson")


def test_calculate_correlations_matches_scipy_spearman():
    """Test that our implementation matches scipy for Spearman."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    np.random.seed(123)
    x = np.random.randn(50)
    y = x**2 + np.random.randn(50) * 0.1

    r_ours, p_ours = calculate_correlations(x, y, method="spearman")
    r_scipy, p_scipy = stats.spearmanr(x, y)

    assert r_ours == pytest.approx(r_scipy, abs=1e-10)
    assert p_ours == pytest.approx(p_scipy, abs=1e-10)


def test_calculate_correlations_matches_scipy_pearson():
    """Test that our implementation matches scipy for Pearson."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    np.random.seed(456)
    x = np.random.randn(50)
    y = 2 * x + np.random.randn(50) * 0.5

    r_ours, p_ours = calculate_correlations(x, y, method="pearson")
    r_scipy, p_scipy = stats.pearsonr(x, y)

    assert r_ours == pytest.approx(r_scipy, abs=1e-10)
    assert p_ours == pytest.approx(p_scipy, abs=1e-10)


def test_calculate_correlations_matches_scipy_kendall():
    """Test that our implementation matches scipy for Kendall."""
    from sleap_roots_analyze.cross_experiment_analysis import calculate_correlations

    np.random.seed(789)
    x = np.random.randn(30)
    y = x + np.random.randn(30) * 0.3

    r_ours, p_ours = calculate_correlations(x, y, method="kendall")
    r_scipy, p_scipy = stats.kendalltau(x, y)

    assert r_ours == pytest.approx(r_scipy, abs=1e-10)
    assert p_ours == pytest.approx(p_scipy, abs=1e-10)


def test_create_correlation_summary_plot_basic():
    """Test basic creation of correlation summary plot."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )
    import matplotlib.pyplot as plt

    # Create sample correlation dataframe
    np.random.seed(42)
    n_correlations = 100
    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(n_correlations)],
            "exp2_trait": [f"trait2_{i}" for i in range(n_correlations)],
            "correlation": np.random.randn(n_correlations) * 0.5,
            "p_value": np.random.rand(n_correlations),
        }
    )

    fig = create_correlation_summary_plot(
        correlation_df,
        correlation_col="correlation",
        pvalue_col="p_value",
        exp1_trait_col="exp1_trait",
        exp2_trait_col="exp2_trait",
    )

    # Check that a figure is returned
    assert fig is not None
    assert isinstance(fig, plt.Figure)

    # Check that it has 4 subplots (2x2 grid)
    assert len(fig.axes) == 4

    plt.close(fig)


def test_create_correlation_summary_plot_custom_figsize():
    """Test correlation summary plot with custom figsize."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )
    import matplotlib.pyplot as plt

    np.random.seed(42)
    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(50)],
            "exp2_trait": [f"trait2_{i}" for i in range(50)],
            "correlation": np.random.randn(50) * 0.4,
            "p_value": np.random.rand(50),
        }
    )

    figsize = (16, 14)
    fig = create_correlation_summary_plot(
        correlation_df,
        correlation_col="correlation",
        pvalue_col="p_value",
        exp1_trait_col="exp1_trait",
        exp2_trait_col="exp2_trait",
        figsize=figsize,
    )

    assert fig.get_figwidth() == figsize[0]
    assert fig.get_figheight() == figsize[1]

    plt.close(fig)


def test_create_correlation_summary_plot_with_significant():
    """Test that plot correctly identifies significant correlations."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )
    import matplotlib.pyplot as plt

    # Create data with some significant correlations
    np.random.seed(123)
    n = 100
    p_values = np.random.rand(n)
    # Make some clearly significant
    p_values[:10] = 0.001
    # Make some borderline
    p_values[10:20] = 0.04

    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(n)],
            "exp2_trait": [f"trait2_{i}" for i in range(n)],
            "correlation": np.random.randn(n) * 0.5,
            "p_value": p_values,
        }
    )

    fig = create_correlation_summary_plot(
        correlation_df,
        correlation_col="correlation",
        pvalue_col="p_value",
        exp1_trait_col="exp1_trait",
        exp2_trait_col="exp2_trait",
        significance_threshold=0.05,
    )

    # Should create figure without errors
    assert fig is not None
    assert len(fig.axes) == 4

    plt.close(fig)


def test_create_correlation_summary_plot_all_positive():
    """Test plot with all positive correlations."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )
    import matplotlib.pyplot as plt

    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(30)],
            "exp2_trait": [f"trait2_{i}" for i in range(30)],
            "correlation": np.random.rand(30) * 0.8,  # All positive
            "p_value": np.random.rand(30),
        }
    )

    fig = create_correlation_summary_plot(
        correlation_df,
        correlation_col="correlation",
        pvalue_col="p_value",
        exp1_trait_col="exp1_trait",
        exp2_trait_col="exp2_trait",
    )

    assert fig is not None
    plt.close(fig)


def test_create_correlation_summary_plot_all_negative():
    """Test plot with all negative correlations."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )
    import matplotlib.pyplot as plt

    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(30)],
            "exp2_trait": [f"trait2_{i}" for i in range(30)],
            "correlation": -np.random.rand(30) * 0.8,  # All negative
            "p_value": np.random.rand(30),
        }
    )

    fig = create_correlation_summary_plot(
        correlation_df,
        correlation_col="correlation",
        pvalue_col="p_value",
        exp1_trait_col="exp1_trait",
        exp2_trait_col="exp2_trait",
    )

    assert fig is not None
    plt.close(fig)


def test_create_correlation_summary_plot_minimal_data():
    """Test plot with minimal data (< 15 correlations)."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )
    import matplotlib.pyplot as plt

    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(5)],
            "exp2_trait": [f"trait2_{i}" for i in range(5)],
            "correlation": [0.8, -0.7, 0.6, -0.5, 0.4],
            "p_value": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )

    fig = create_correlation_summary_plot(
        correlation_df,
        correlation_col="correlation",
        pvalue_col="p_value",
        exp1_trait_col="exp1_trait",
        exp2_trait_col="exp2_trait",
        top_n=15,  # Request more than available
    )

    assert fig is not None
    plt.close(fig)


def test_create_correlation_summary_plot_custom_top_n():
    """Test plot with custom top_n parameter."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )
    import matplotlib.pyplot as plt

    np.random.seed(456)
    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(100)],
            "exp2_trait": [f"trait2_{i}" for i in range(100)],
            "correlation": np.random.randn(100) * 0.5,
            "p_value": np.random.rand(100),
        }
    )

    fig = create_correlation_summary_plot(
        correlation_df,
        correlation_col="correlation",
        pvalue_col="p_value",
        exp1_trait_col="exp1_trait",
        exp2_trait_col="exp2_trait",
        top_n=10,
    )

    assert fig is not None
    plt.close(fig)


def test_create_correlation_summary_plot_empty_dataframe():
    """Test that empty dataframe raises appropriate error."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )

    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [],
            "exp2_trait": [],
            "correlation": [],
            "p_value": [],
        }
    )

    with pytest.raises(ValueError, match="correlation_df cannot be empty"):
        create_correlation_summary_plot(
            correlation_df,
            correlation_col="correlation",
            pvalue_col="p_value",
            exp1_trait_col="exp1_trait",
            exp2_trait_col="exp2_trait",
        )


def test_create_correlation_summary_plot_missing_column():
    """Test that missing column raises appropriate error."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        create_correlation_summary_plot,
    )

    correlation_df = pd.DataFrame(
        {
            "exp1_trait": [f"trait1_{i}" for i in range(10)],
            "exp2_trait": [f"trait2_{i}" for i in range(10)],
            "correlation": np.random.rand(10),
            # Missing p_value column
        }
    )

    with pytest.raises(ValueError, match="Column\\(s\\) .* not found"):
        create_correlation_summary_plot(
            correlation_df,
            correlation_col="correlation",
            pvalue_col="p_value",
            exp1_trait_col="exp1_trait",
            exp2_trait_col="exp2_trait",
        )
