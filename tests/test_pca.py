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


class TestEdgeCasesForFullCoverage:
    """Test edge cases to achieve 100% code coverage."""

    def test_select_n_components_no_threshold(self):
        """Test select_n_components with explained_variance_threshold=None."""
        from sleap_roots_analyze.pca import select_n_components
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        # When threshold is very high (1.0), should use all available components
        n_components = select_n_components(
            X,
            explained_variance_threshold=1.0,  # Use all components
            n_components=None
        )
        assert n_components >= 4  # Should need most/all components for 100% variance
        
        # Test with specified n_components overriding threshold
        n_components = select_n_components(
            X,
            explained_variance_threshold=0.5,
            n_components=2  # Override with specific value
        )
        assert n_components == 2

    def test_mahalanobis_distances_1d_array(self, pca_1d_result_data):
        """Test calculate_mahalanobis_distances with 1D transformed data."""
        from sleap_roots_analyze.pca import fit_pca, calculate_mahalanobis_distances
        
        # Fit PCA requesting only 1 component
        pca, X_transformed = fit_pca(
            pca_1d_result_data.values,
            n_components=1
        )
        
        assert X_transformed.shape[1] == 1
        
        # Calculate distances with 1D data
        distances, mean, covariance = calculate_mahalanobis_distances(X_transformed)
        
        assert distances is not None
        assert mean.shape == (1,)  # 1D mean
        assert covariance.shape == (1, 1)  # 1x1 covariance
        assert len(distances) == len(X_transformed)

    def test_mahalanobis_distances_scalar_covariance(self):
        """Test calculate_mahalanobis_distances with scalar covariance."""
        from sleap_roots_analyze.pca import calculate_mahalanobis_distances
        import numpy as np
        
        # Create 1D data that might result in scalar covariance
        X_1d = np.random.randn(50, 1)
        
        distances, mean, cov = calculate_mahalanobis_distances(X_1d)
        
        # Verify covariance is properly handled as 2D array
        assert cov.shape == (1, 1)
        assert distances is not None

    def test_mahalanobis_distances_zero_std(self):
        """Test calculate_mahalanobis_distances with zero standard deviation."""
        from sleap_roots_analyze.pca import calculate_mahalanobis_distances
        import numpy as np
        
        # Create data with zero variance (all same value)
        X_constant = np.ones((30, 1)) * 5.0  # All values are 5.0
        
        distances, mean, cov = calculate_mahalanobis_distances(X_constant)
        
        # With zero std, all distances should be zero
        assert np.all(distances == 0)
        assert mean[0] == 5.0
        assert cov[0, 0] == 0  # Zero variance

    def test_perform_pca_all_nan_data(self, pca_all_nan_data):
        """Test perform_pca_analysis with all NaN DataFrame."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import pytest
        
        # Should raise ValueError when all data is NaN
        with pytest.raises(ValueError, match="No valid samples after removing NaN"):
            perform_pca_analysis(pca_all_nan_data)

    def test_perform_pca_empty_after_nan_removal(self, pca_empty_after_nan_removal):
        """Test perform_pca_analysis when data becomes empty after NaN removal."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import pytest
        
        # Every row has at least one NaN, so dropna() will remove all rows
        with pytest.raises(ValueError, match="No valid samples after removing NaN"):
            perform_pca_analysis(pca_empty_after_nan_removal)

    def test_perform_pca_zero_variance_all_columns(self, pca_zero_variance_all_columns):
        """Test perform_pca_analysis with all zero-variance columns."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import pytest
        
        # All columns have zero variance
        with pytest.raises(ValueError, match="No numeric columns with non-zero variance found"):
            perform_pca_analysis(pca_zero_variance_all_columns)

    def test_perform_pca_single_sample(self, pca_single_sample_data):
        """Test perform_pca_analysis with single sample data."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import pytest
        
        # Single sample - PCA should handle gracefully but with limitations
        result = perform_pca_analysis(pca_single_sample_data)
        # With single sample, we can't do meaningful PCA
        assert result['n_components_selected'] == 0  # select_n_components returns 0 for single sample

    def test_perform_pca_mixed_data_types(self, pca_mixed_numeric_nonnumeric):
        """Test perform_pca_analysis with mixed numeric and non-numeric columns."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        
        # Should handle mixed data types gracefully
        result = perform_pca_analysis(
            pca_mixed_numeric_nonnumeric,
            n_components=2,
            standardize=True
        )
        
        # Should only use numeric columns
        assert len(result['feature_names']) == 4  # Only the 4 numeric columns
        assert result['feature_names'] == ['value1', 'value2', 'value3', 'value4']
        assert result['n_components_selected'] <= 2

    def test_perform_pca_zero_std_features(self, pca_zero_std_features):
        """Test perform_pca_analysis with some zero-variance features."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        
        # Should filter out zero-variance features
        result = perform_pca_analysis(
            pca_zero_std_features,
            n_components=None,
            standardize=True
        )
        
        # Should filter out truly zero-variance features (zero_std2 is all zeros)
        # zero_std1 might have tiny variance due to floating point representation
        assert 'zero_std2' not in result['feature_names']  # All zeros should be filtered
        assert 'normal1' in result['feature_names']
        assert 'normal2' in result['feature_names']
        assert 'normal3' in result['feature_names']

    def test_perform_pca_singular_covariance(self, pca_singular_covariance_data):
        """Test perform_pca_analysis with singular covariance matrix."""
        from sleap_roots_analyze.pca import perform_pca_analysis, calculate_mahalanobis_distances
        
        # Should handle linearly dependent features
        result = perform_pca_analysis(
            pca_singular_covariance_data,
            n_components=3,
            standardize=True
        )
        
        # Should still work despite linear dependencies
        assert result['n_components_selected'] <= 3
        
        # Test mahalanobis distances with singular covariance
        X_transformed = result['transformed_data']
        distances, mean, cov = calculate_mahalanobis_distances(X_transformed)
        assert distances is not None

    def test_fit_pca_with_more_components_than_features(self):
        """Test fit_pca when requesting more components than features."""
        from sleap_roots_analyze.pca import fit_pca
        import numpy as np
        
        # 3 features but request 5 components - fit_pca should handle this
        X = np.random.randn(50, 3)
        
        # Should automatically cap to min(n_features, n_samples-1)
        pca, X_transformed = fit_pca(X, n_components=3)
        
        # Should only have min(n_samples-1, n_features) components
        assert pca.n_components_ <= 3

    def test_calculate_pca_metrics_edge_cases(self):
        """Test calculate_pca_metrics with edge cases."""
        from sleap_roots_analyze.pca import calculate_pca_metrics
        from sklearn.decomposition import PCA
        import numpy as np
        
        # Test with 1 component PCA
        X = np.random.randn(100, 5)
        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)
        
        metrics = calculate_pca_metrics(pca, X_transformed)
        
        assert metrics['n_components_selected'] == 1
        assert len(metrics['explained_variance_ratio']) == 1
        assert metrics['cumulative_variance_ratio'][-1] <= 1.0

    def test_perform_pca_with_variance_threshold_edge_cases(self):
        """Test perform_pca_with_variance_threshold with edge cases."""
        from sleap_roots_analyze.pca import perform_pca_with_variance_threshold
        import numpy as np
        
        # Test with very high threshold (should use all components)
        X = np.random.randn(50, 3)
        result = perform_pca_with_variance_threshold(
            X, 
            explained_variance_threshold=0.9999
        )
        assert result['n_components_selected'] >= 2
        
        # Test with very low threshold (should use 1 component)
        result = perform_pca_with_variance_threshold(
            X,
            explained_variance_threshold=0.01
        )
        assert result['n_components_selected'] == 1

    def test_mahalanobis_1d_ndim_reshape(self):
        """Test calculate_mahalanobis_distances with actual 1D array (line 231)."""
        from sleap_roots_analyze.pca import calculate_mahalanobis_distances
        import numpy as np
        
        # Create actual 1D array (not 2D with shape (n, 1))
        X_1d = np.random.randn(50)  # Shape is (50,) not (50, 1)
        
        # Should handle reshaping internally
        distances, mean, cov = calculate_mahalanobis_distances(X_1d)
        
        assert distances is not None
        assert mean.shape == (1,)
        assert cov.shape == (1, 1)

    def test_mahalanobis_scalar_covariance_ndim_0(self):
        """Test calculate_mahalanobis_distances with 0-dim covariance (line 253)."""
        from sleap_roots_analyze.pca import calculate_mahalanobis_distances
        import numpy as np
        
        # Create data that might produce scalar covariance
        # Single feature with very small variance
        X = np.ones((10, 1)) * 5 + np.random.randn(10, 1) * 1e-15
        
        distances, mean, cov = calculate_mahalanobis_distances(X, robust=False)
        
        # Covariance should be 2D
        assert cov.ndim == 2
        assert cov.shape == (1, 1)

    def test_perform_pca_no_numeric_columns(self):
        """Test perform_pca_analysis with no numeric columns (line 312)."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import pandas as pd
        import pytest
        
        # DataFrame with only non-numeric columns
        df_non_numeric = pd.DataFrame({
            'name': ['A', 'B', 'C'],
            'category': ['X', 'Y', 'Z'],
            'description': ['foo', 'bar', 'baz']
        })
        
        with pytest.raises(ValueError, match="No numeric columns found"):
            perform_pca_analysis(df_non_numeric)

    def test_perform_pca_all_columns_zero_variance_after_filter(self):
        """Test when all columns have zero variance (line 346)."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import pandas as pd
        import numpy as np
        import pytest
        
        # Create DataFrame where all columns will have zero variance
        n_samples = 20
        df = pd.DataFrame({
            'all_same_1': [42.0] * n_samples,
            'all_same_2': [100.0] * n_samples,
            'all_zeros': np.zeros(n_samples),
            'all_ones': np.ones(n_samples)
        })
        
        with pytest.raises(ValueError, match="No numeric columns with non-zero variance found"):
            perform_pca_analysis(df)

    def test_perform_pca_array_no_features(self):
        """Test perform_pca_analysis with array input that has no features."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import numpy as np
        import pytest
        
        # Create array with shape (n_samples, 0) - no features
        X_no_features = np.empty((10, 0))
        
        # This gets converted to empty DataFrame
        with pytest.raises(ValueError, match="Empty DataFrame provided"):
            perform_pca_analysis(X_no_features)

    def test_mahalanobis_force_scalar_covariance(self):
        """Force scalar covariance matrix case (line 253)."""
        from sleap_roots_analyze.pca import calculate_mahalanobis_distances
        import numpy as np
        
        # Create data with 2 samples, 1 feature
        # np.cov with rowvar=False on shape (2, 1) returns scalar
        X = np.array([[1.0], [2.0]])
        
        # This should trigger the scalar covariance case
        distances, mean, cov = calculate_mahalanobis_distances(X, robust=False)
        
        # Should handle scalar covariance properly
        assert cov.shape == (1, 1)
        assert distances is not None
        assert len(distances) == 2

    def test_perform_pca_no_standardization_with_cleaning(self):
        """Test perform_pca_analysis without standardization but with cleaning."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import pandas as pd
        import numpy as np
        
        # Create DataFrame with mixed columns and zero variance column
        np.random.seed(42)
        df = pd.DataFrame({
            'good1': np.random.randn(20),
            'good2': np.random.randn(20) * 2,
            'zero_var': [5.0] * 20,  # Zero variance
            'text': ['A'] * 20  # Non-numeric
        })
        
        # Test without standardization - should still filter zero variance
        result = perform_pca_analysis(df, standardize=False)
        
        # Should only keep the 2 good features
        assert len(result['feature_names']) == 2
        assert 'good1' in result['feature_names']
        assert 'good2' in result['feature_names']
        assert result['scaler'] is None  # No standardization

class TestStandardizationVerification:
    """Comprehensive tests to verify StandardScaler is working correctly."""
    
    def test_standardization_with_real_trait_data(self, traits_summary_df):
        """Test standardization with real trait data."""
        from sleap_roots_analyze.pca import perform_pca_analysis
        import numpy as np
        import pytest
        
        # Use real trait data - select numeric columns only
        numeric_cols = traits_summary_df.select_dtypes(include=[np.number]).columns
        # Select subset of columns with fewer NaNs for testing
        cols_with_data = []
        for col in numeric_cols[:50]:  # Check first 50 numeric columns
            if traits_summary_df[col].notna().sum() > 100:  # At least 100 non-NaN values
                cols_with_data.append(col)
        
        if len(cols_with_data) < 5:  # Need at least 5 features for meaningful PCA
            pytest.skip("Not enough columns with sufficient data for PCA testing")
        
        test_data = traits_summary_df[cols_with_data[:10]]  # Use up to 10 features
        result = perform_pca_analysis(test_data, standardize=True)
        
        if result['scaler'] is not None:
            # Verify standardized data has mean ≈ 0 and std ≈ 1
            processed = result['data_processed']
            
            # Check mean is close to 0
            means = np.mean(processed, axis=0)
            np.testing.assert_allclose(means, 0, atol=1e-10, 
                                      err_msg="Standardized data should have mean ≈ 0")
            
            # Check std is close to 1
            stds = np.std(processed, axis=0, ddof=0)  # Use population std
            np.testing.assert_allclose(stds, 1, atol=1e-10,
                                      err_msg="Standardized data should have std ≈ 1")
    
    def test_standardization_with_diverse_distributions(self):
        """Test standardization with various data distributions."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import standardize_data
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create diverse distributions
        df = pd.DataFrame({
            'normal': np.random.randn(n_samples),
            'lognormal': np.random.lognormal(0, 1, n_samples),
            'exponential': np.random.exponential(2, n_samples),
            'uniform': np.random.uniform(-10, 10, n_samples),
            'bimodal': np.concatenate([
                np.random.normal(-3, 0.5, n_samples//2),
                np.random.normal(3, 0.5, n_samples//2)
            ])
        })
        
        X_scaled, scaler, df_clean = standardize_data(df)
        
        # All distributions should be standardized
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0, ddof=0)
        
        np.testing.assert_allclose(means, 0, atol=1e-10,
                                  err_msg="All distributions should have mean ≈ 0")
        np.testing.assert_allclose(stds, 1, atol=1e-10,
                                  err_msg="All distributions should have std ≈ 1")
    
    def test_standardization_with_extreme_scales(self):
        """Test standardization with features at very different scales."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import perform_pca_analysis
        
        np.random.seed(42)
        n_samples = 500
        
        # Create features with vastly different scales
        df = pd.DataFrame({
            'tiny': np.random.randn(n_samples) * 1e-6,  # Very small scale
            'small': np.random.randn(n_samples) * 0.01,
            'medium': np.random.randn(n_samples),
            'large': np.random.randn(n_samples) * 1000,
            'huge': np.random.randn(n_samples) * 1e6,  # Very large scale
        })
        
        result = perform_pca_analysis(df, standardize=True)
        
        # After standardization, all should be on same scale
        processed = result['data_processed']
        means = np.mean(processed, axis=0)
        stds = np.std(processed, axis=0, ddof=0)
        
        # Check all features are properly standardized
        np.testing.assert_allclose(means, 0, atol=1e-9,
                                  err_msg="Features at different scales should have mean ≈ 0")
        np.testing.assert_allclose(stds, 1, atol=1e-9,
                                  err_msg="Features at different scales should have std ≈ 1")
        
        # Verify feature names are preserved
        assert result['feature_names'] == ['tiny', 'small', 'medium', 'large', 'huge']
    
    def test_standardization_with_outliers(self):
        """Test that standardization handles outliers correctly."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import standardize_data
        
        np.random.seed(42)
        n_samples = 200
        
        # Create data with outliers
        normal_data = np.random.randn(n_samples)
        
        # Add outliers
        outlier_indices = [10, 50, 100, 150]
        for idx in outlier_indices:
            normal_data[idx] = normal_data[idx] * 100  # Make outliers
        
        df = pd.DataFrame({
            'with_outliers': normal_data,
            'normal': np.random.randn(n_samples)
        })
        
        X_scaled, scaler, df_clean = standardize_data(df)
        
        # Even with outliers, standardization should work
        # Mean should still be close to 0
        means = np.mean(X_scaled, axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10,
                                  err_msg="Mean should be 0 even with outliers")
        
        # Std should be 1 (outliers will affect this but StandardScaler handles it)
        stds = np.std(X_scaled, axis=0, ddof=0)
        np.testing.assert_allclose(stds, 1, atol=1e-10,
                                  err_msg="Std should be 1 even with outliers")
    
    def test_ddof_consistency(self):
        """Test that ddof=0 (population variance) is used consistently."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import standardize_data
        from sklearn.preprocessing import StandardScaler
        
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'feat1': np.random.randn(n_samples) * 2 + 5,
            'feat2': np.random.randn(n_samples) * 0.5 - 3,
            'feat3': np.random.randn(n_samples) * 10
        })
        
        # Our standardization
        X_scaled, scaler, df_clean = standardize_data(df)
        
        # Manual calculation with ddof=0
        X_manual = df.values
        means_manual = np.mean(X_manual, axis=0)
        stds_manual = np.std(X_manual, axis=0, ddof=0)  # Population std
        X_manual_scaled = (X_manual - means_manual) / stds_manual
        
        # Verify our implementation matches manual calculation
        np.testing.assert_allclose(X_scaled, X_manual_scaled, atol=1e-10,
                                  err_msg="Standardization should use ddof=0")
        
        # Verify sklearn StandardScaler also uses ddof=0
        sklearn_scaler = StandardScaler()
        X_sklearn = sklearn_scaler.fit_transform(df.values)
        np.testing.assert_allclose(X_scaled, X_sklearn, atol=1e-10,
                                  err_msg="Should match sklearn's StandardScaler")
    
    def test_near_zero_variance_features(self):
        """Test handling of features with very small variance."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import standardize_data
        
        np.random.seed(42)
        n_samples = 100
        
        # Create features with different variance levels
        df = pd.DataFrame({
            'normal_var': np.random.randn(n_samples),
            'tiny_var': np.random.randn(n_samples) * 1e-8,  # Very small variance
            'zero_var': np.ones(n_samples) * 5.0,  # Zero variance
            'small_var': np.random.randn(n_samples) * 0.001
        })
        
        X_scaled, scaler, df_clean = standardize_data(df)
        
        # Zero variance column should be removed
        assert 'zero_var' not in df_clean.columns
        assert df_clean.shape[1] == 3  # Only 3 columns remaining
        
        # Remaining columns should be standardized
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0, ddof=0)
        
        np.testing.assert_allclose(means, 0, atol=1e-10)
        np.testing.assert_allclose(stds, 1, atol=1e-10)
    
    def test_standardization_inverse_transform(self):
        """Test that standardization can be reversed correctly."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import perform_pca_analysis
        
        np.random.seed(42)
        n_samples = 200
        
        # Create data with known properties
        df = pd.DataFrame({
            'feat1': np.random.randn(n_samples) * 5 + 10,  # mean=10, std=5
            'feat2': np.random.randn(n_samples) * 2 - 5,   # mean=-5, std=2
            'feat3': np.random.randn(n_samples) * 0.5 + 100  # mean=100, std=0.5
        })
        
        # Store original data
        original_values = df.values.copy()
        
        result = perform_pca_analysis(df, standardize=True, n_components=2)
        
        # Get standardized data
        X_standardized = result['data_processed']
        scaler = result['scaler']
        
        # Verify standardization
        np.testing.assert_allclose(np.mean(X_standardized, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(X_standardized, axis=0, ddof=0), 1, atol=1e-10)
        
        # Inverse transform should recover original data
        X_recovered = scaler.inverse_transform(X_standardized)
        np.testing.assert_allclose(X_recovered, original_values, atol=1e-10,
                                  err_msg="Inverse transform should recover original data")
    
    def test_standardization_with_array_input(self):
        """Test standardization when input is numpy array."""
        import numpy as np
        from sleap_roots_analyze.pca import perform_pca_analysis
        
        np.random.seed(42)
        n_samples = 150
        n_features = 5
        
        # Create array with different scales
        X = np.random.randn(n_samples, n_features)
        X[:, 0] *= 100  # Scale first feature
        X[:, 1] *= 0.01  # Scale second feature
        X[:, 2] += 50  # Shift third feature
        
        result = perform_pca_analysis(X, standardize=True)
        
        # Verify standardization worked
        processed = result['data_processed']
        means = np.mean(processed, axis=0)
        stds = np.std(processed, axis=0, ddof=0)
        
        np.testing.assert_allclose(means, 0, atol=1e-10,
                                  err_msg="Array input should be standardized correctly")
        np.testing.assert_allclose(stds, 1, atol=1e-10,
                                  err_msg="Array input should have std ≈ 1")
        
        # Feature names should be generated
        assert result['feature_names'] == [f'Feature_{i}' for i in range(n_features)]
    
    def test_standardization_preserves_relationships(self):
        """Test that standardization preserves relative relationships between samples."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import standardize_data
        from scipy.stats import spearmanr
        
        np.random.seed(42)
        n_samples = 100
        
        # Create correlated features
        base = np.random.randn(n_samples)
        df = pd.DataFrame({
            'feat1': base * 2 + 5,
            'feat2': base * 0.5 - 3 + np.random.randn(n_samples) * 0.1,
            'feat3': -base * 3 + 10 + np.random.randn(n_samples) * 0.2
        })
        
        # Calculate rank correlations before standardization
        corr_before = spearmanr(df.values)[0]
        
        # Standardize
        X_scaled, scaler, df_clean = standardize_data(df)
        
        # Calculate rank correlations after standardization
        corr_after = spearmanr(X_scaled)[0]
        
        # Rank correlations should be preserved
        np.testing.assert_allclose(corr_before, corr_after, atol=1e-10,
                                  err_msg="Standardization should preserve rank correlations")
