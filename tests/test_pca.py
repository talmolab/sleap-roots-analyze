"""Tests for PCA analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from sleap_roots_analyze.pca import (
    calculate_mahalanobis_distances,
    calculate_pca_metrics,
    calculate_pca_reconstruction_error,
    fit_pca,
    perform_pca_analysis,
    perform_pca_with_variance_threshold,
    select_n_components,
    standardize_data,
)


class TestStandardizeData:
    """Test suite for standardize_data function."""

    def test_standardize_basic(self, pca_simple_data):
        """Test basic standardization."""
        data, _ = pca_simple_data
        df = pd.DataFrame(data, columns=["x", "y"])

        X_scaled, scaler, df_clean = standardize_data(df)

        # Check outputs
        assert X_scaled.shape == data.shape
        assert isinstance(scaler, StandardScaler)
        assert df_clean.shape == df.shape

        # Check standardization
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1, atol=1e-10)

    def test_standardize_with_non_numeric(self):
        """Test standardization with non-numeric columns."""
        df = pd.DataFrame(
            {
                "numeric1": np.random.randn(50),
                "numeric2": np.random.randn(50),
                "text": ["A"] * 50,
                "category": pd.Categorical(["cat1", "cat2"] * 25),
            }
        )

        X_scaled, scaler, df_clean = standardize_data(df)

        # Should only keep numeric columns
        assert df_clean.shape[1] == 2
        assert "numeric1" in df_clean.columns
        assert "numeric2" in df_clean.columns

    def test_standardize_zero_variance(self, pca_constant_feature_data):
        """Test standardization with zero variance columns."""
        X_scaled, scaler, df_clean = standardize_data(pca_constant_feature_data)

        # Should remove constant columns
        assert "constant1" not in df_clean.columns
        assert "constant2" not in df_clean.columns
        assert "constant3" not in df_clean.columns
        assert "variable1" in df_clean.columns
        assert "variable2" in df_clean.columns

        # Check only non-constant features remain
        assert df_clean.shape[1] == 2
        assert X_scaled.shape[1] == 2

    def test_standardize_empty_after_cleaning(self):
        """Test error when no valid columns remain."""
        df = pd.DataFrame({"constant": [1] * 10, "text": ["A"] * 10})

        with pytest.raises(
            ValueError, match="No numeric columns with non-zero variance"
        ):
            standardize_data(df)

    def test_standardize_nan_handling(self, pca_nan_data):
        """Test standardization with NaN values."""
        # StandardScaler should handle NaN appropriately
        X_scaled, scaler, df_clean = standardize_data(pca_nan_data)

        # NaN should be preserved
        assert np.isnan(X_scaled).sum() > 0
        assert df_clean.shape == pca_nan_data.shape


class TestSelectNComponents:
    """Test suite for select_n_components function."""

    def test_select_with_specified_n(self, pca_simple_data):
        """Test component selection with specified n_components."""
        data, _ = pca_simple_data

        n = select_n_components(data, n_components=1)
        assert n == 1

        n = select_n_components(data, n_components=2)
        assert n == 2

        # Should cap at max valid
        n = select_n_components(data, n_components=100)
        assert n == min(data.shape)

    def test_select_with_variance_threshold(self, pca_variance_threshold_data):
        """Test automatic selection based on variance threshold."""
        datasets = pca_variance_threshold_data

        # One component dataset
        data = datasets["one_component"].values
        n = select_n_components(data, explained_variance_threshold=0.95)
        assert n == 1

        # Two components dataset
        data = datasets["two_components"].values
        n = select_n_components(data, explained_variance_threshold=0.90)
        assert n >= 2

        # All components dataset
        data = datasets["all_components"].values
        n = select_n_components(data, explained_variance_threshold=0.99)
        assert n == 3

    def test_select_edge_cases(self):
        """Test edge cases in component selection."""
        # Single feature
        data = np.random.randn(100, 1)
        n = select_n_components(data)
        assert n == 1

        # More features than samples
        data = np.random.randn(10, 20)
        n = select_n_components(data)
        assert n <= 9  # max is n_samples - 1

        # Single sample
        data = np.random.randn(1, 5)
        n = select_n_components(data)
        assert n == 0  # Can't do PCA with single sample

    def test_select_low_variance_threshold(self, pca_3d_data):
        """Test with low variance threshold."""
        df, _ = pca_3d_data
        data = df.values

        # Low threshold should select fewer components
        n_low = select_n_components(data, explained_variance_threshold=0.50)
        n_high = select_n_components(data, explained_variance_threshold=0.99)

        assert n_low <= n_high


class TestFitPCA:
    """Test suite for fit_pca function."""

    def test_fit_basic(self, pca_simple_data):
        """Test basic PCA fitting."""
        data, _ = pca_simple_data
        n_components = 2

        pca, X_transformed = fit_pca(data, n_components)

        assert isinstance(pca, SklearnPCA)
        assert X_transformed.shape == (data.shape[0], n_components)
        assert pca.n_components_ == n_components

    def test_fit_single_component(self, pca_3d_data):
        """Test fitting with single component."""
        df, _ = pca_3d_data
        data = df.values

        pca, X_transformed = fit_pca(data, n_components=1)

        assert X_transformed.shape == (data.shape[0], 1)
        assert pca.explained_variance_ratio_.shape == (1,)

    def test_fit_reproducibility(self, pca_simple_data):
        """Test reproducibility with random_state."""
        data, _ = pca_simple_data

        pca1, X1 = fit_pca(data, 2, random_state=42)
        pca2, X2 = fit_pca(data, 2, random_state=42)

        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_almost_equal(pca1.components_, pca2.components_)

    def test_fit_high_dimensional(self, pca_high_dim_data):
        """Test fitting with high-dimensional data."""
        df, expected = pca_high_dim_data
        data = df.values

        n_components = 5
        pca, X_transformed = fit_pca(data, n_components)

        assert X_transformed.shape == (data.shape[0], n_components)
        # First few components should explain most variance
        assert pca.explained_variance_ratio_[0] > 0.3


class TestCalculatePCAMetrics:
    """Test suite for calculate_pca_metrics function."""

    def test_metrics_basic(self, pca_simple_data):
        """Test basic metrics calculation."""
        data, _ = pca_simple_data

        pca, X_transformed = fit_pca(data, 2)
        metrics = calculate_pca_metrics(pca, X_transformed)

        # Check all expected keys
        expected_keys = [
            "pca",
            "n_components_selected",
            "transformed_data",
            "loadings",
            "eigenvalues",
            "explained_variance_ratio",
            "cumulative_variance_ratio",
            "total_variance_explained",
            "explained_variance_per_feature",
            "explained_variance_ratio_per_feature",
        ]
        for key in expected_keys:
            assert key in metrics

        # Check dimensions
        assert metrics["loadings"].shape == (data.shape[1], 2)
        assert metrics["eigenvalues"].shape == (2,)
        assert len(metrics["explained_variance_per_feature"]) == data.shape[1]

    def test_metrics_variance_explained(self, pca_3d_data):
        """Test variance explained calculations."""
        df, _ = pca_3d_data
        data = StandardScaler().fit_transform(df.values)

        pca, X_transformed = fit_pca(data, 2)
        metrics = calculate_pca_metrics(pca, X_transformed)

        # Cumulative variance should be increasing
        cumulative = metrics["cumulative_variance_ratio"]
        assert all(
            cumulative[i] <= cumulative[i + 1] for i in range(len(cumulative) - 1)
        )

        # Total variance explained should match last cumulative
        assert np.isclose(metrics["total_variance_explained"], cumulative[-1])

        # Per-feature variance should be reasonable
        per_feature = metrics["explained_variance_per_feature"]
        assert all(0 <= v <= 3 for v in per_feature)  # For standardized data

    def test_metrics_loadings(self, pca_perfect_correlation_data):
        """Test loadings calculation with correlated features."""
        data = StandardScaler().fit_transform(pca_perfect_correlation_data.values)

        pca, X_transformed = fit_pca(data, 2)
        metrics = calculate_pca_metrics(pca, X_transformed)

        loadings = metrics["loadings"]

        # Loadings should be orthogonal (for different PCs)
        for i in range(loadings.shape[1]):
            for j in range(i + 1, loadings.shape[1]):
                dot_product = np.dot(loadings[:, i], loadings[:, j])
                assert np.abs(dot_product) < 0.1  # Nearly orthogonal


class TestPerformPCAWithVarianceThreshold:
    """Test suite for legacy perform_pca_with_variance_threshold function."""

    def test_legacy_function(self, pca_simple_data):
        """Test legacy function still works."""
        data, _ = pca_simple_data
        X_scaled = StandardScaler().fit_transform(data)

        result = perform_pca_with_variance_threshold(X_scaled)

        assert "pca" in result
        assert "n_components_selected" in result
        assert "transformed_data" in result
        assert result["n_components_selected"] <= 2

    def test_legacy_with_threshold(self, pca_variance_threshold_data):
        """Test legacy function with different thresholds."""
        data = pca_variance_threshold_data["two_components"].values
        X_scaled = StandardScaler().fit_transform(data)

        # Low threshold
        result_low = perform_pca_with_variance_threshold(
            X_scaled, explained_variance_threshold=0.5
        )

        # High threshold
        result_high = perform_pca_with_variance_threshold(
            X_scaled, explained_variance_threshold=0.99
        )

        assert (
            result_low["n_components_selected"] <= result_high["n_components_selected"]
        )

    def test_legacy_specified_components(self, pca_3d_data):
        """Test legacy function with specified n_components."""
        df, _ = pca_3d_data
        X_scaled = StandardScaler().fit_transform(df.values)

        result = perform_pca_with_variance_threshold(X_scaled, n_components=2)

        assert result["n_components_selected"] == 2
        assert result["transformed_data"].shape[1] == 2


class TestPerformPCAAnalysis:
    """Test suite for main perform_pca_analysis function."""

    def test_analysis_with_standardization(self, pca_3d_data):
        """Test PCA analysis with standardization (default)."""
        df, _ = pca_3d_data

        result = perform_pca_analysis(df, standardize=True)

        # Check all expected keys
        assert "scaler" in result
        assert result["scaler"] is not None
        assert "data_processed" in result
        assert "feature_names" in result
        assert result["feature_names"] == df.columns.tolist()

        # Data should be standardized
        processed = result["data_processed"]
        np.testing.assert_allclose(processed.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(processed.std(axis=0), 1, atol=1e-10)

    def test_analysis_without_standardization(self, pca_3d_data):
        """Test PCA analysis without standardization."""
        df, _ = pca_3d_data

        result = perform_pca_analysis(df, standardize=False)

        assert result["scaler"] is None

        # Data should not be standardized
        processed = result["data_processed"]
        original_mean = df.values.mean(axis=0)
        processed_mean = processed.mean(axis=0)
        np.testing.assert_allclose(processed_mean, original_mean)

    def test_analysis_with_array_input(self, pca_simple_data):
        """Test with numpy array input instead of DataFrame."""
        data, _ = pca_simple_data

        result = perform_pca_analysis(data)

        assert "feature_names" in result
        # Should generate feature names
        assert result["feature_names"] == ["Feature_0", "Feature_1"]
        assert result["transformed_data"].shape[0] == data.shape[0]

    def test_analysis_with_non_numeric_columns(self):
        """Test handling of non-numeric columns."""
        df = pd.DataFrame(
            {
                "num1": np.random.randn(50),
                "num2": np.random.randn(50),
                "text": ["A", "B"] * 25,
                "num3": np.random.randn(50),
            }
        )

        result = perform_pca_analysis(df)

        # Should only use numeric columns
        assert set(result["feature_names"]) == {"num1", "num2", "num3"}

    def test_analysis_with_zero_variance(self, pca_constant_feature_data):
        """Test handling of zero variance columns."""
        result = perform_pca_analysis(pca_constant_feature_data)

        # Should exclude constant columns
        assert "constant1" not in result["feature_names"]
        assert "variable1" in result["feature_names"]
        assert "variable2" in result["feature_names"]

    def test_analysis_variance_threshold(self, pca_variance_threshold_data):
        """Test variance threshold in full pipeline."""
        datasets = pca_variance_threshold_data

        # Test with dataset needing 1 component
        result = perform_pca_analysis(
            datasets["one_component"], explained_variance_threshold=0.95
        )
        assert result["n_components_selected"] == 1

        # Test with dataset needing multiple components
        result = perform_pca_analysis(
            datasets["two_components"], explained_variance_threshold=0.95
        )
        assert result["n_components_selected"] >= 2

    def test_analysis_with_real_data(self, pca_real_traits_data):
        """Test with real trait data."""
        df, trait_cols = pca_real_traits_data

        # Should work with real data
        result = perform_pca_analysis(df)

        assert result["n_components_selected"] > 0
        assert result["total_variance_explained"] > 0.5
        assert len(result["feature_names"]) == len(trait_cols)

    def test_analysis_error_cases(self):
        """Test error handling."""
        # Empty DataFrame
        with pytest.raises(ValueError, match="Empty DataFrame"):
            perform_pca_analysis(pd.DataFrame())

        # All constant values
        df = pd.DataFrame({"const": [1] * 10})
        with pytest.raises(
            ValueError, match="No numeric columns with non-zero variance"
        ):
            perform_pca_analysis(df)

        # Wrong array shape
        with pytest.raises(ValueError, match="must be 2D"):
            perform_pca_analysis(np.array([1, 2, 3]))

    def test_analysis_edge_cases(self):
        """Test edge cases."""
        # Single feature
        df = pd.DataFrame({"single": np.random.randn(50)})
        result = perform_pca_analysis(df)
        assert result["n_components_selected"] == 1

        # Single sample - should handle gracefully
        df = pd.DataFrame({"feat1": [1.0], "feat2": [2.0]})
        # Single sample can't do PCA meaningfully
        result = perform_pca_analysis(df, standardize=False)
        assert result["n_components_selected"] == 0  # No components for single sample


class TestCalculateReconstructionError:
    """Test suite for calculate_pca_reconstruction_error function."""

    def test_reconstruction_basic(self, pca_simple_data):
        """Test basic reconstruction error calculation."""
        data, _ = pca_simple_data
        X_scaled = StandardScaler().fit_transform(data)

        # Full components - should have zero error
        pca_result = perform_pca_with_variance_threshold(X_scaled, n_components=2)
        errors = calculate_pca_reconstruction_error(X_scaled, pca_result)

        assert errors.shape == (X_scaled.shape[0],)
        assert all(e >= 0 for e in errors)
        np.testing.assert_allclose(errors, 0, atol=1e-10)

    def test_reconstruction_partial(self, pca_3d_data):
        """Test reconstruction with partial components."""
        df, _ = pca_3d_data
        X_scaled = StandardScaler().fit_transform(df.values)

        # Use only 2 components for 3D data
        pca_result = perform_pca_with_variance_threshold(X_scaled, n_components=2)
        errors = calculate_pca_reconstruction_error(X_scaled, pca_result)

        # Should have non-zero errors
        assert errors.shape == (X_scaled.shape[0],)
        assert all(e >= 0 for e in errors)
        assert errors.mean() > 0

    def test_reconstruction_outliers(self, pca_outlier_data):
        """Test reconstruction errors identify outliers."""
        df, outlier_indices = pca_outlier_data
        X_scaled = StandardScaler().fit_transform(df.values)

        pca_result = perform_pca_with_variance_threshold(X_scaled, n_components=2)
        errors = calculate_pca_reconstruction_error(X_scaled, pca_result)

        # Outliers should have higher reconstruction errors
        outlier_errors = errors[outlier_indices]
        normal_mask = np.ones(len(errors), dtype=bool)
        normal_mask[outlier_indices] = False
        normal_errors = errors[normal_mask]

        assert outlier_errors.mean() > normal_errors.mean()


class TestCalculateMahalanobisDistances:
    """Test suite for calculate_mahalanobis_distances function."""

    def test_mahalanobis_basic(self, pca_simple_data):
        """Test basic Mahalanobis distance calculation."""
        data, _ = pca_simple_data

        distances, mean, covariance = calculate_mahalanobis_distances(data)

        assert distances.shape == (data.shape[0],)
        assert all(d >= 0 for d in distances)
        assert mean.shape == (data.shape[1],)
        assert covariance.shape == (data.shape[1], data.shape[1])

    def test_mahalanobis_1d(self):
        """Test Mahalanobis distance for 1D data."""
        data = np.random.randn(100, 1)

        distances, mean, covariance = calculate_mahalanobis_distances(data)

        assert distances.shape == (100,)
        assert mean.shape == (1,)
        assert covariance.shape == (1, 1)

        # For 1D, Mahalanobis is just standardized distance
        z_scores = np.abs(data[:, 0] - mean[0]) / np.sqrt(covariance[0, 0])
        np.testing.assert_allclose(distances, z_scores)

    def test_mahalanobis_robust(self, pca_outlier_data):
        """Test robust Mahalanobis distance calculation."""
        df, outlier_indices = pca_outlier_data
        data = df.values

        # Non-robust should be affected by outliers
        distances_normal, _, _ = calculate_mahalanobis_distances(data, robust=False)

        # Robust should be less affected
        distances_robust, _, _ = calculate_mahalanobis_distances(data, robust=True)

        # Check that outliers are better identified with robust method
        # (they should have relatively higher distances)
        outlier_rank_normal = np.argsort(distances_normal)[::-1]
        outlier_rank_robust = np.argsort(distances_robust)[::-1]

        # More outliers should be in top ranks for robust method
        top_k = len(outlier_indices)
        found_normal = sum(
            1 for i in outlier_rank_normal[:top_k] if i in outlier_indices
        )
        found_robust = sum(
            1 for i in outlier_rank_robust[:top_k] if i in outlier_indices
        )

        assert found_robust >= found_normal

    def test_mahalanobis_singular(self):
        """Test handling of singular covariance matrix."""
        # Create perfectly correlated data (singular covariance)
        base = np.random.randn(50)
        data = np.column_stack([base, base * 2, base * 3])

        # Should handle singular matrix gracefully
        distances, mean, covariance = calculate_mahalanobis_distances(data)

        assert distances.shape == (50,)
        assert not np.any(np.isnan(distances))
        assert not np.any(np.isinf(distances))

    def test_mahalanobis_zero_variance(self):
        """Test with zero variance in one dimension."""
        data = np.column_stack(
            [np.random.randn(50), np.ones(50), np.random.randn(50)]  # Constant
        )

        # Should handle zero variance gracefully
        distances, mean, covariance = calculate_mahalanobis_distances(data)

        assert distances.shape == (50,)
        assert not np.any(np.isnan(distances))


class TestIntegration:
    """Integration tests for complete PCA pipeline."""

    def test_full_pipeline_with_real_data(self, traits_summary_df):
        """Test complete pipeline with real trait data."""
        # Check if the data has any valid samples after dropping NaNs
        df_numeric = traits_summary_df.select_dtypes(include=[np.number])
        df_clean = df_numeric.dropna()

        if df_clean.empty:
            # Skip test if no valid data
            pytest.skip("No valid samples in real data after removing NaNs")

        # Full pipeline test
        result = perform_pca_analysis(
            traits_summary_df, standardize=True, explained_variance_threshold=0.95
        )

        # Verify all components work together
        assert result["n_components_selected"] > 0
        assert result["transformed_data"].shape[0] == len(traits_summary_df)

        # Test reconstruction error
        if result["scaler"] is not None:
            X_scaled = result["data_processed"]
            errors = calculate_pca_reconstruction_error(X_scaled, result)
            assert len(errors) == len(traits_summary_df)

        # Test Mahalanobis distances
        distances, _, _ = calculate_mahalanobis_distances(result["transformed_data"])
        assert len(distances) == len(traits_summary_df)

    def test_pipeline_consistency(self, pca_3d_data):
        """Test that modular and legacy functions give same results."""
        df, _ = pca_3d_data
        X_scaled = StandardScaler().fit_transform(df.values)

        # Legacy function
        legacy_result = perform_pca_with_variance_threshold(
            X_scaled, explained_variance_threshold=0.95, random_state=42
        )

        # New modular approach
        n_comp = select_n_components(
            X_scaled, explained_variance_threshold=0.95, random_state=42
        )
        pca, X_trans = fit_pca(X_scaled, n_comp, random_state=42)
        modular_result = calculate_pca_metrics(pca, X_trans)

        # Results should be identical
        assert (
            legacy_result["n_components_selected"]
            == modular_result["n_components_selected"]
        )
        np.testing.assert_array_almost_equal(
            legacy_result["transformed_data"], modular_result["transformed_data"]
        )
        np.testing.assert_array_almost_equal(
            legacy_result["eigenvalues"], modular_result["eigenvalues"]
        )

    def test_standardization_effect(self, pca_high_dim_data):
        """Test effect of standardization on PCA results."""
        df, _ = pca_high_dim_data

        # With standardization
        result_std = perform_pca_analysis(
            df, standardize=True, explained_variance_threshold=0.90
        )

        # Without standardization
        result_no_std = perform_pca_analysis(
            df, standardize=False, explained_variance_threshold=0.90
        )

        # Both should work
        assert result_std["n_components_selected"] > 0
        assert result_no_std["n_components_selected"] > 0

        # They might differ based on the data structure
        # but we can't assume which will need more components
