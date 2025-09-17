"""Tests for PCA reconstruction error outlier detection."""

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.outlier_detection import (
    detect_outliers_pca,
    detect_outliers_mahalanobis,
)


class TestDetectOutliersPCA:
    """Test suite for PCA reconstruction error outlier detection."""

    def test_basic_pca_outlier_detection(self, pca_reconstruction_data_low_rank):
        """Test basic PCA reconstruction error outlier detection."""
        df, expected_outliers, metadata = pca_reconstruction_data_low_rank

        result = detect_outliers_pca(
            df, explained_variance_threshold=0.95, outlier_threshold=2.5
        )

        # Check basic structure
        assert result["method"] == "PCA"
        assert "outlier_indices" in result
        assert "reconstruction_errors" in result
        assert "n_components" in result

        # Should use fewer components than features due to low rank
        assert result["n_components"] < metadata["n_features"]

        # Check that we found some outliers
        assert result["n_outliers"] > 0

        # Check dimensions
        assert len(result["reconstruction_errors"]) == len(df)

        # Outliers should have higher reconstruction errors
        errors = np.array(result["reconstruction_errors"])
        outlier_mask = np.zeros(len(df), dtype=bool)
        outlier_mask[result["outlier_indices"]] = True

        if result["n_outliers"] > 0:
            mean_outlier_error = np.mean(errors[outlier_mask])
            mean_normal_error = np.mean(errors[~outlier_mask])
            assert mean_outlier_error > mean_normal_error

    def test_pca_with_perfect_low_rank(self, pca_reconstruction_perfect_low_rank):
        """Test PCA on perfect low-rank data (no outliers expected)."""
        df, metadata = pca_reconstruction_perfect_low_rank

        result = detect_outliers_pca(
            df,
            n_components=metadata["true_rank"],  # Use exact rank
            outlier_threshold=3.0,
        )

        # Reconstruction should be near-perfect
        errors = np.array(result["reconstruction_errors"])
        # After standardization, there may be small numerical errors
        assert np.max(errors) < 1e-6  # Relaxed numerical tolerance

        # Should find no or very few outliers in perfect low-rank data
        assert result["n_outliers"] <= 2  # Allow for numerical noise

    def test_pca_threshold_sensitivity(self, pca_reconstruction_varying_errors):
        """Test effect of different outlier thresholds."""
        df, metadata = pca_reconstruction_varying_errors

        # Lower threshold - more outliers
        result_low = detect_outliers_pca(
            df, explained_variance_threshold=0.90, outlier_threshold=1.5
        )

        # Higher threshold - fewer outliers
        result_high = detect_outliers_pca(
            df, explained_variance_threshold=0.90, outlier_threshold=3.0
        )

        # Higher threshold should find fewer outliers
        assert result_high["n_outliers"] <= result_low["n_outliers"]

        # Check threshold values
        assert result_high["threshold_value"] > result_low["threshold_value"]

    def test_pca_component_selection(self, pca_reconstruction_data_low_rank):
        """Test effect of component selection on outlier detection."""
        df, _, metadata = pca_reconstruction_data_low_rank

        # Fewer components - higher reconstruction error
        result_few = detect_outliers_pca(df, n_components=2, outlier_threshold=2.5)

        # More components - lower reconstruction error
        result_many = detect_outliers_pca(df, n_components=4, outlier_threshold=2.5)

        # More components should have lower average reconstruction error
        mean_error_few = np.mean(result_few["reconstruction_errors"])
        mean_error_many = np.mean(result_many["reconstruction_errors"])
        assert mean_error_many <= mean_error_few

        # Both should have correct number of components
        assert result_few["n_components"] == 2
        assert result_many["n_components"] == 4

    def test_pca_variance_threshold_selection(self, pca_reconstruction_data_low_rank):
        """Test automatic component selection via variance threshold."""
        df, _, _ = pca_reconstruction_data_low_rank

        # Lower threshold - fewer components
        result_low = detect_outliers_pca(
            df, explained_variance_threshold=0.80, outlier_threshold=2.5
        )

        # Higher threshold - more components
        result_high = detect_outliers_pca(
            df, explained_variance_threshold=0.99, outlier_threshold=2.5
        )

        # Higher threshold should use more components
        assert result_high["n_components"] >= result_low["n_components"]

        # Check cumulative variance
        assert result_low["total_variance_explained"] >= 0.80
        assert result_high["total_variance_explained"] >= 0.99

    def test_pca_with_standardization(self, pca_reconstruction_data_low_rank):
        """Test that PCA properly standardizes data."""
        df, _, _ = pca_reconstruction_data_low_rank

        # Scale features differently
        df_scaled = df.copy()
        df_scaled.iloc[:, 0] *= 1000  # Scale first feature
        df_scaled.iloc[:, 1] *= 0.001  # Scale second feature

        result_original = detect_outliers_pca(df, outlier_threshold=2.5)
        result_scaled = detect_outliers_pca(df_scaled, outlier_threshold=2.5)

        # Results should be similar despite scaling (due to standardization)
        # Number of outliers should be close
        assert abs(result_original["n_outliers"] - result_scaled["n_outliers"]) <= 2

    def test_pca_edge_cases(self, outlier_data_edge_cases):
        """Test PCA outlier detection with edge cases."""
        edge_cases = outlier_data_edge_cases

        # Empty data
        result_empty = detect_outliers_pca(edge_cases["empty"])
        assert "error" in result_empty
        assert "Empty" in result_empty["error"]

        # Single sample
        single = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        result_single = detect_outliers_pca(single)
        assert "error" in result_single
        assert "at least 2 samples" in result_single["error"]

        # Data with NaN
        with_nan = pd.DataFrame({"col1": [1, 2, np.nan, 4], "col2": [5, 6, 7, 8]})
        result_nan = detect_outliers_pca(with_nan)
        assert "error" in result_nan
        assert "NaN" in result_nan["error"]

    def test_pca_constant_features(self):
        """Test PCA with constant features."""
        # Create data with a constant feature
        df = pd.DataFrame(
            {
                "var1": np.random.randn(50),
                "const": np.ones(50),  # Constant
                "var2": np.random.randn(50),
            }
        )

        result = detect_outliers_pca(df)

        # Should handle constant features (removed during standardization)
        assert "error" not in result or result.get("error") is None
        # Should use fewer components than original features
        assert result["n_components"] <= 2  # At most 2 non-constant features

    def test_pca_reconstruction_error_calculation(
        self, pca_reconstruction_data_low_rank
    ):
        """Test that reconstruction errors are calculated correctly."""
        df, _, _ = pca_reconstruction_data_low_rank

        result = detect_outliers_pca(df, n_components=3, outlier_threshold=2.5)

        # All reconstruction errors should be non-negative
        errors = np.array(result["reconstruction_errors"])
        assert np.all(errors >= 0)

        # Check threshold calculation
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        expected_threshold = error_mean + 2.5 * error_std
        np.testing.assert_almost_equal(
            result["threshold_value"], expected_threshold, decimal=10
        )

    def test_pca_output_completeness(self, pca_reconstruction_data_low_rank):
        """Test that all expected outputs are present."""
        df, _, _ = pca_reconstruction_data_low_rank

        result = detect_outliers_pca(df)

        # Check all expected keys
        expected_keys = [
            "method",
            "n_components",
            "explained_variance_ratio",
            "cumulative_variance",
            "total_variance_explained",
            "outlier_threshold",
            "threshold_value",
            "reconstruction_errors",
            "outlier_indices",
            "n_outliers",
            "pca_components",
            "loadings",
            "eigenvalues",
            "feature_names",
            "data_indices",
            "explained_variance_per_feature",
            "explained_variance_ratio_per_feature",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_pca_vs_mahalanobis_comparison(self, pca_reconstruction_data_low_rank):
        """Compare PCA reconstruction with Mahalanobis method."""
        df, expected_outliers, _ = pca_reconstruction_data_low_rank

        # Run both methods
        pca_result = detect_outliers_pca(
            df, explained_variance_threshold=0.95, outlier_threshold=2.5
        )

        mahal_result = detect_outliers_mahalanobis(
            df, standardize=True, variance_threshold=0.95, chi2_percentile=97.5
        )

        # Both should find outliers
        assert pca_result["n_outliers"] > 0
        assert mahal_result["n_outliers"] > 0

        # Check overlap - they should find some common outliers
        pca_outliers = set(pca_result["outlier_indices"])
        mahal_outliers = set(mahal_result["outlier_indices"])

        # Some overlap expected for clear outliers
        if len(pca_outliers) > 0 and len(mahal_outliers) > 0:
            overlap = pca_outliers.intersection(mahal_outliers)
            # At least some overlap for obvious outliers
            assert len(overlap) > 0
