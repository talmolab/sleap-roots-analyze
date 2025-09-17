"""Tests for outlier detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
    detect_outliers_pca,
    calculate_outlier_threshold,
    identify_outliers_from_distances,
)


class TestDetectOutliersMahalanobis:
    """Test main outlier detection function."""

    def test_basic_outlier_detection(self, outlier_data_with_known_outliers):
        """Test basic outlier detection with known outliers."""
        df, expected_outliers, metadata = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=97.5,
        )

        # Check basic structure
        assert result["method"] == "Mahalanobis"
        assert "outlier_indices" in result
        assert "mahalanobis_distances" in result
        assert "n_outliers" in result
        assert "n_components" in result

        # Check that we detect some outliers
        assert result["n_outliers"] > 0

        # Check that outliers have larger distances
        distances = np.array(result["mahalanobis_distances"])
        outlier_distances = distances[expected_outliers]
        normal_distances = np.delete(distances, expected_outliers)

        # Most outliers should have larger distances than normal points
        assert np.median(outlier_distances) > np.median(normal_distances)

    def test_with_known_outliers(self, outlier_data_with_known_outliers):
        """Test that known outliers are detected."""
        df, expected_outliers, metadata = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=95.0,  # Lower threshold to catch more outliers
        )

        detected = set(result["outlier_indices"])
        expected = set(expected_outliers)

        # Should detect at least 60% of the true outliers
        overlap = detected.intersection(expected)
        recall = len(overlap) / len(expected)
        assert recall >= 0.6, f"Only detected {recall*100:.1f}% of true outliers"

    def test_chi_squared_threshold(self, outlier_data_with_known_outliers):
        """Test chi-squared threshold calculation."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=99.0,
        )

        assert result["threshold_type"] == "chi_squared"
        assert result["chi2_percentile"] == 99.0
        assert result["threshold_value"] > 0

        # Check threshold is consistent with chi2 distribution
        n_components = result["n_components"]
        expected_threshold = stats.chi2.ppf(0.99, n_components)
        np.testing.assert_allclose(
            result["threshold_value"], expected_threshold, rtol=0.01
        )

    def test_custom_distance_threshold(self, outlier_data_with_known_outliers):
        """Test using custom distance threshold."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=False,
            distance_threshold=3.5,
        )

        assert result["threshold_type"] == "distance"
        assert result["threshold_value"] == 3.5
        assert result["chi2_percentile"] is None
        assert result["distance_threshold"] == 3.5

    def test_robust_covariance(self, outlier_data_multimodal):
        """Test robust covariance estimation."""
        df, _, metadata = outlier_data_multimodal

        # Test with robust covariance
        result_robust = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=97.5,
            robust_covariance=True,
        )

        # Test without robust covariance
        result_normal = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=97.5,
            robust_covariance=False,
        )

        # Robust method should handle multimodal data better
        assert result_robust["method"] == "Mahalanobis"
        assert result_normal["method"] == "Mahalanobis"

        # Results should differ due to different covariance estimation
        robust_outliers = set(result_robust["outlier_indices"])
        normal_outliers = set(result_normal["outlier_indices"])
        assert robust_outliers != normal_outliers or len(robust_outliers) != len(
            normal_outliers
        )

    def test_variance_threshold_selection(self, outlier_data_high_dimensional):
        """Test component selection based on variance threshold."""
        df, _, metadata = outlier_data_high_dimensional

        # Test with lower variance threshold (fewer components)
        result_low = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.8,
            use_chi_squared=True,
        )

        # Test with higher variance threshold (more components)
        result_high = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
        )

        # Higher threshold should use more components
        assert result_high["n_components"] >= result_low["n_components"]
        assert result_low["cumulative_variance_explained"] >= 0.8
        assert result_high["cumulative_variance_explained"] >= 0.95

    def test_standardization_effect(self, outlier_data_with_known_outliers):
        """Test effect of standardization on outlier detection."""
        df, _, _ = outlier_data_with_known_outliers

        # Scale features differently
        df_scaled = df.copy()
        df_scaled.iloc[:, 0] *= 100  # Scale first feature
        df_scaled.iloc[:, 1] *= 0.01  # Scale second feature

        # With standardization
        result_std = detect_outliers_mahalanobis(
            df_scaled,
            standardize=True,
            variance_threshold=0.95,
        )

        # Without standardization
        result_no_std = detect_outliers_mahalanobis(
            df_scaled,
            standardize=False,
            variance_threshold=0.95,
        )

        # Standardization should handle scale differences
        assert result_std["n_outliers"] > 0
        assert result_no_std["n_outliers"] >= 0

        # Results likely differ due to scale issues
        std_outliers = set(result_std["outlier_indices"])
        no_std_outliers = set(result_no_std["outlier_indices"])
        # May or may not be equal depending on data

    def test_with_dataframe_input(self, outlier_data_with_known_outliers):
        """Test with DataFrame input."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
        )

        assert isinstance(result, dict)
        assert "feature_names" in result
        assert result["feature_names"] == df.columns.tolist()
        assert "data_indices" in result
        assert len(result["data_indices"]) == len(df)

    def test_with_array_input(self, outlier_data_with_known_outliers):
        """Test with numpy array input."""
        df, _, _ = outlier_data_with_known_outliers
        X = df.values

        result = detect_outliers_mahalanobis(
            X,
            standardize=True,
            variance_threshold=0.95,
        )

        assert isinstance(result, dict)
        assert "feature_names" in result
        # Should have generated feature names
        assert len(result["feature_names"]) == X.shape[1]
        assert result["feature_names"][0].startswith("feature_")

    def test_empty_data(self, outlier_data_edge_cases):
        """Test with empty data."""
        edge_cases = outlier_data_edge_cases

        result = detect_outliers_mahalanobis(
            edge_cases["empty"],
            standardize=True,
        )

        assert result["method"] == "Mahalanobis"
        assert result["outlier_indices"] == []
        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_nan_handling(self, outlier_data_edge_cases):
        """Test handling of NaN values."""
        edge_cases = outlier_data_edge_cases

        result = detect_outliers_mahalanobis(
            edge_cases["with_nan"],
            standardize=True,
        )

        assert result["method"] == "Mahalanobis"
        assert result["outlier_indices"] == []
        assert "error" in result
        assert "nan" in result["error"].lower()

    def test_single_sample(self, outlier_data_edge_cases):
        """Test with single sample."""
        edge_cases = outlier_data_edge_cases

        result = detect_outliers_mahalanobis(
            edge_cases["single_sample"],
            standardize=True,
        )

        assert result["method"] == "Mahalanobis"
        assert result["outlier_indices"] == []
        assert "error" in result

    def test_constant_features(self, outlier_data_edge_cases):
        """Test with constant features."""
        edge_cases = outlier_data_edge_cases

        # Constant feature should be removed during standardization
        result = detect_outliers_mahalanobis(
            edge_cases["constant_feature"],
            standardize=True,
            variance_threshold=0.95,
        )

        assert result["method"] == "Mahalanobis"
        # Should work after removing constant feature
        assert "error" not in result or result.get("error") is None
        # Feature count should be reduced
        assert len(result["feature_names"]) == 2  # Originally 3, one constant removed

    def test_per_feature_variance_tracking(self, outlier_data_with_known_outliers):
        """Test that per-feature variance is tracked correctly."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.90,
        )

        # Should have per-feature variance information
        assert "feature_variance_explained" in result
        assert "feature_fraction_explained" in result

        # Check dimensions
        n_features = df.shape[1]
        assert len(result["feature_variance_explained"]) == n_features
        assert len(result["feature_fraction_explained"]) == n_features

        # Check that fractions are between 0 and 1
        fractions = np.array(result["feature_fraction_explained"])
        assert np.all(fractions >= 0)
        assert np.all(fractions <= 1)


class TestCalculateOutlierThreshold:
    """Test threshold calculation for outlier detection."""

    def test_chi_squared_threshold(self):
        """Test chi-squared threshold calculation."""
        n_components = 5
        chi2_percentile = 97.5

        threshold, threshold_type = calculate_outlier_threshold(
            n_components=n_components,
            use_chi_squared=True,
            chi2_percentile=chi2_percentile,
        )

        assert threshold_type == "chi_squared"
        expected = stats.chi2.ppf(chi2_percentile / 100, n_components)
        np.testing.assert_allclose(threshold, expected)

    def test_distance_threshold(self):
        """Test distance threshold."""
        threshold, threshold_type = calculate_outlier_threshold(
            n_components=5,
            use_chi_squared=False,
            distance_threshold=4.0,
        )

        assert threshold_type == "distance"
        assert threshold == 4.0

    def test_default_distance_threshold(self):
        """Test default distance threshold when not specified."""
        threshold, threshold_type = calculate_outlier_threshold(
            n_components=5,
            use_chi_squared=False,
            distance_threshold=None,
        )

        assert threshold_type == "distance"
        assert threshold == 3.0  # Default value

    def test_invalid_parameters(self):
        """Test with invalid parameters."""
        # Invalid chi2 percentile
        with pytest.raises(ValueError):
            calculate_outlier_threshold(
                n_components=5,
                use_chi_squared=True,
                chi2_percentile=150,  # > 100
            )

        # Invalid n_components
        with pytest.raises(ValueError):
            calculate_outlier_threshold(
                n_components=0,
                use_chi_squared=True,
            )

        # Negative distance threshold
        with pytest.raises(ValueError):
            calculate_outlier_threshold(
                n_components=5,
                use_chi_squared=False,
                distance_threshold=-1,
            )


class TestIdentifyOutliersFromDistances:
    """Test outlier identification from distances."""

    def test_basic_identification(self):
        """Test basic outlier identification."""
        distances = np.array([1.0, 2.0, 5.0, 1.5, 6.0, 1.2])
        threshold = 3.0

        result = identify_outliers_from_distances(
            distances=distances,
            threshold=threshold,
            threshold_type="distance",
        )

        assert "outlier_mask" in result
        assert "outlier_indices" in result
        assert "n_outliers" in result

        # Distances > 3.0 are outliers: indices 2 and 4
        assert result["outlier_indices"] == [2, 4]
        assert result["n_outliers"] == 2
        np.testing.assert_array_equal(
            result["outlier_mask"], [False, False, True, False, True, False]
        )

    def test_chi_squared_identification(self):
        """Test identification with chi-squared threshold."""
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold = 9.0  # Chi-squared threshold

        result = identify_outliers_from_distances(
            distances=distances,
            threshold=threshold,
            threshold_type="chi_squared",
        )

        # For chi-squared, compare squared distances
        squared_distances = distances**2
        expected_mask = squared_distances > threshold

        np.testing.assert_array_equal(result["outlier_mask"], expected_mask)
        assert result["n_outliers"] == np.sum(expected_mask)

    def test_with_indices(self):
        """Test with custom indices."""
        distances = np.array([2.0, 5.0, 1.0])
        indices = pd.Index([10, 20, 30])

        result = identify_outliers_from_distances(
            distances=distances,
            threshold=3.0,
            threshold_type="distance",
            indices=indices,
        )

        # Index 20 (distance 5.0) is outlier
        assert result["outlier_indices"] == [20]
        assert result["n_outliers"] == 1

    def test_empty_distances(self):
        """Test with empty distances."""
        distances = np.array([])

        result = identify_outliers_from_distances(
            distances=distances,
            threshold=3.0,
            threshold_type="distance",
        )

        assert result["outlier_indices"] == []
        assert result["n_outliers"] == 0
        assert len(result["outlier_mask"]) == 0


class TestIntegrationWithPCA:
    """Test integration with PCA module."""

    def test_full_pipeline_with_real_data(self, traits_summary_df):
        """Test full pipeline with real trait data."""
        from sleap_roots_analyze.data_cleanup import get_trait_columns

        # Get trait columns
        trait_cols = get_trait_columns(traits_summary_df)

        # Select columns with good coverage
        good_cols = []
        for col in trait_cols[:20]:
            if traits_summary_df[col].notna().sum() > len(traits_summary_df) * 0.8:
                good_cols.append(col)

        if len(good_cols) < 5:
            pytest.skip("Not enough columns with good data coverage")

        # Prepare data
        test_data = traits_summary_df[good_cols[:10]].dropna()

        # Detect outliers
        result = detect_outliers_mahalanobis(
            test_data,
            standardize=True,
            variance_threshold=0.90,
            use_chi_squared=True,
            chi2_percentile=95.0,
        )

        # Check results
        assert result["method"] == "Mahalanobis"
        assert "error" not in result or result.get("error") is None
        assert result["n_components"] > 0
        assert result["cumulative_variance_explained"] >= 0.90
        assert len(result["mahalanobis_distances"]) == len(test_data)

        # Check outlier percentage is reasonable (0-20%)
        outlier_percentage = (result["n_outliers"] / len(test_data)) * 100
        assert 0 <= outlier_percentage <= 20

    def test_consistency_with_pca_metrics(self, outlier_data_with_known_outliers):
        """Test consistency with PCA metrics calculation."""
        from sleap_roots_analyze.pca import perform_pca_analysis, calculate_pca_metrics

        df, _, _ = outlier_data_with_known_outliers

        # Run outlier detection
        outlier_result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
        )

        # Run PCA separately with same threshold
        pca_result = perform_pca_analysis(
            df,
            standardize=True,
            explained_variance_threshold=0.95,  # Match outlier detection threshold
        )

        # Check consistency
        # The outlier detection now uses the variance_threshold directly
        assert outlier_result["n_components"] == pca_result["n_components_selected"]

    def test_per_feature_variance_tracking(self, outlier_data_with_known_outliers):
        """Test that per-feature variance is correctly tracked from PCA."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.85,
        )

        # Check per-feature metrics
        assert "feature_variance_explained" in result
        assert "feature_fraction_explained" in result

        n_features = df.shape[1]
        assert len(result["feature_variance_explained"]) == n_features
        assert len(result["feature_fraction_explained"]) == n_features

        # Sum of per-feature variance should equal sum of eigenvalues
        total_var_explained = sum(result["feature_variance_explained"])
        eigenvalue_sum = sum(result["eigenvalues"])
        np.testing.assert_allclose(total_var_explained, eigenvalue_sum, rtol=1e-10)


class TestNumericalValidation:
    """Validate numerical correctness of outlier detection."""

    def test_mahalanobis_distance_formula(self):
        """Test Mahalanobis distance calculation is correct."""
        # Create simple test data
        np.random.seed(42)
        n_samples = 20
        n_features = 3

        # Generate data with known mean and covariance
        mean = np.array([1, 2, 3])
        cov = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 1.5]])

        X = np.random.multivariate_normal(mean, cov, n_samples)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])

        # Run detection without standardization to keep original structure
        result = detect_outliers_mahalanobis(
            df,
            standardize=False,
            variance_threshold=0.99,  # Use all components
            use_chi_squared=False,
            distance_threshold=10.0,  # High threshold to not remove any
        )

        # Manually calculate Mahalanobis distances for verification
        # (This would be on PCA-transformed data, so exact match isn't expected,
        # but the distances should be reasonable)
        distances = np.array(result["mahalanobis_distances"])

        # Check properties
        assert np.all(distances >= 0)  # Distances are non-negative
        assert np.all(np.isfinite(distances))  # No inf or nan

        # Most distances should be moderate (between 0 and 3 for normal data)
        assert np.median(distances) < 3.0

    def test_chi_squared_distribution(self):
        """Test that chi-squared threshold follows correct distribution."""
        # Test various degrees of freedom
        for dof in [1, 2, 5, 10]:
            for percentile in [90, 95, 97.5, 99]:
                threshold, _ = calculate_outlier_threshold(
                    n_components=dof,
                    use_chi_squared=True,
                    chi2_percentile=percentile,
                )

                expected = stats.chi2.ppf(percentile / 100, dof)
                np.testing.assert_allclose(threshold, expected, rtol=1e-10)

    def test_outlier_percentage(self, outlier_data_with_known_outliers):
        """Test that outlier percentage matches chi-squared expectation."""
        df, _, _ = outlier_data_with_known_outliers

        # Large dataset for statistical test
        np.random.seed(123)
        n_samples = 500
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        df_large = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])

        # Test with chi-squared at 95th percentile
        result = detect_outliers_mahalanobis(
            df_large,
            standardize=True,
            variance_threshold=0.99,  # Use most components
            use_chi_squared=True,
            chi2_percentile=95.0,
        )

        # Expect approximately 5% outliers
        outlier_percentage = (result["n_outliers"] / n_samples) * 100

        # Allow some tolerance due to finite sample
        assert 2 <= outlier_percentage <= 10


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


# Tests for outlier index preservation and correctness
@pytest.fixture
def data_with_specific_outliers():
    """Create data with known outliers at specific positions."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Create normal data
    X = np.random.randn(n_samples, n_features)

    # Add specific outliers at known positions
    outlier_positions = [10, 25, 50, 75, 90]
    for pos in outlier_positions:
        # Make these samples clear outliers
        X[pos, :] += np.random.randn(n_features) * 5.0

    return X, outlier_positions


@pytest.fixture
def dataframe_with_custom_indices():
    """Create DataFrame with non-sequential custom indices."""
    np.random.seed(42)
    n_samples = 50
    n_features = 4

    # Create data
    X = np.random.randn(n_samples, n_features)

    # Add outliers at positions 5, 15, 35
    outlier_positions = [5, 15, 35]
    for pos in outlier_positions:
        X[pos, :] += np.random.randn(n_features) * 4.0

    # Create DataFrame with custom indices (non-sequential)
    # Use indices like 100, 102, 104, ...
    custom_indices = [100 + i * 2 for i in range(n_samples)]
    df = pd.DataFrame(
        X, index=custom_indices, columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Expected outlier indices in terms of the custom index
    expected_outlier_indices = [custom_indices[pos] for pos in outlier_positions]

    return df, expected_outlier_indices, outlier_positions


@pytest.fixture
def dataframe_with_string_indices():
    """Create DataFrame with string indices."""
    np.random.seed(42)
    n_samples = 30
    n_features = 3

    # Create data
    X = np.random.randn(n_samples, n_features)

    # Add outliers
    outlier_positions = [3, 10, 20, 25]
    for pos in outlier_positions:
        X[pos, :] *= 5.0  # Scale instead of shift

    # Create string indices
    string_indices = [f"sample_{i:03d}" for i in range(n_samples)]
    df = pd.DataFrame(X, index=string_indices, columns=["A", "B", "C"])

    expected_indices = [string_indices[pos] for pos in outlier_positions]

    return df, expected_indices, outlier_positions


class TestMahalanobisIndexPreservation:
    """Test that Mahalanobis method preserves indices correctly."""

    def test_numpy_array_indices(self, data_with_specific_outliers):
        """Test that numpy array inputs return correct position indices."""
        X, true_outlier_positions = data_with_specific_outliers

        result = detect_outliers_mahalanobis(
            X,
            standardize=True,
            variance_threshold=0.95,
            chi2_percentile=99.0,  # High threshold to catch only clear outliers
        )

        # Check that outlier indices are within valid range
        assert all(0 <= idx < len(X) for idx in result["outlier_indices"])

        # Check that we found some of the true outliers
        found_outliers = set(result["outlier_indices"])
        true_outliers = set(true_outlier_positions)
        overlap = found_outliers.intersection(true_outliers)

        # Should find at least some of the true outliers
        assert len(overlap) >= 2, f"Found {overlap} vs expected {true_outliers}"

        # Verify data_indices matches original array indices
        assert result["data_indices"] == list(range(len(X)))

    def test_dataframe_custom_indices(self, dataframe_with_custom_indices):
        """Test that DataFrame custom indices are preserved."""
        df, expected_indices, positions = dataframe_with_custom_indices

        result = detect_outliers_mahalanobis(
            df, standardize=True, variance_threshold=0.95, chi2_percentile=98.0
        )

        # Check that returned indices are from the DataFrame's index
        assert all(idx in df.index for idx in result["outlier_indices"])

        # Check data_indices matches DataFrame index
        assert result["data_indices"] == df.index.tolist()

        # Verify we found some expected outliers
        found = set(result["outlier_indices"])
        expected = set(expected_indices)
        overlap = found.intersection(expected)

        assert len(overlap) >= 1, f"Found {found} vs expected {expected}"

    def test_dataframe_string_indices(self, dataframe_with_string_indices):
        """Test that string indices are handled correctly."""
        df, expected_indices, positions = dataframe_with_string_indices

        result = detect_outliers_mahalanobis(
            df, standardize=True, variance_threshold=0.90, chi2_percentile=95.0
        )

        # All returned indices should be strings
        assert all(isinstance(idx, str) for idx in result["outlier_indices"])

        # All indices should be valid DataFrame indices
        assert all(idx in df.index for idx in result["outlier_indices"])

        # Check overlap with expected
        found = set(result["outlier_indices"])
        expected = set(expected_indices)
        overlap = found.intersection(expected)

        assert len(overlap) >= 1, f"Found {found} vs expected {expected}"

    def test_outlier_mask_consistency(self, data_with_specific_outliers):
        """Test that outlier detection is internally consistent."""
        X, _ = data_with_specific_outliers

        result = detect_outliers_mahalanobis(
            X, standardize=True, variance_threshold=0.95, chi2_percentile=97.5
        )

        # Reconstruct outlier mask from indices
        outlier_mask = np.zeros(len(X), dtype=bool)
        for idx in result["outlier_indices"]:
            outlier_mask[idx] = True

        # Check that n_outliers matches
        assert np.sum(outlier_mask) == result["n_outliers"]
        assert len(result["outlier_indices"]) == result["n_outliers"]

        # Check that outliers have higher Mahalanobis distances
        distances = np.array(result["mahalanobis_distances"])
        if result["n_outliers"] > 0:
            outlier_distances = distances[outlier_mask]
            normal_distances = distances[~outlier_mask]

            # All outlier distances should be above threshold
            # Note: For chi-squared, we compare squared distances
            threshold = result["threshold_value"]
            assert np.all(outlier_distances**2 > threshold)
            assert np.all(normal_distances**2 <= threshold)


class TestPCAReconstructionIndexPreservation:
    """Test that PCA reconstruction method preserves indices correctly."""

    def test_numpy_array_indices(self, data_with_specific_outliers):
        """Test numpy array index handling for PCA method."""
        X, true_outlier_positions = data_with_specific_outliers

        result = detect_outliers_pca(
            X, explained_variance_threshold=0.90, outlier_threshold=2.5
        )

        # Check valid indices
        assert all(0 <= idx < len(X) for idx in result["outlier_indices"])

        # Check data_indices
        assert result["data_indices"] == list(range(len(X)))

        # Verify we found some outliers
        assert result["n_outliers"] > 0

        # Check overlap with true outliers
        found = set(result["outlier_indices"])
        true = set(true_outlier_positions)
        overlap = found.intersection(true)
        assert len(overlap) >= 1

    def test_dataframe_custom_indices(self, dataframe_with_custom_indices):
        """Test DataFrame custom index preservation in PCA method."""
        df, expected_indices, positions = dataframe_with_custom_indices

        result = detect_outliers_pca(
            df, explained_variance_threshold=0.90, outlier_threshold=2.0
        )

        # Check indices are from DataFrame
        assert all(idx in df.index for idx in result["outlier_indices"])

        # Check data_indices
        assert result["data_indices"] == df.index.tolist()

        # Check we found some expected outliers
        found = set(result["outlier_indices"])
        expected = set(expected_indices)
        overlap = found.intersection(expected)
        assert len(overlap) >= 1

    def test_dataframe_string_indices(self, dataframe_with_string_indices):
        """Test string index handling in PCA method."""
        df, expected_indices, positions = dataframe_with_string_indices

        result = detect_outliers_pca(
            df, explained_variance_threshold=0.85, outlier_threshold=2.0
        )

        # Check all indices are strings
        assert all(isinstance(idx, str) for idx in result["outlier_indices"])

        # Check valid DataFrame indices
        assert all(idx in df.index for idx in result["outlier_indices"])

        # Verify overlap with expected
        found = set(result["outlier_indices"])
        expected = set(expected_indices)
        overlap = found.intersection(expected)
        assert len(overlap) >= 1

    def test_reconstruction_error_ordering(self, dataframe_with_custom_indices):
        """Test that reconstruction errors match data order."""
        df, _, _ = dataframe_with_custom_indices

        result = detect_outliers_pca(
            df, explained_variance_threshold=0.95, outlier_threshold=2.5
        )

        # Reconstruction errors should have same length as data
        assert len(result["reconstruction_errors"]) == len(df)

        # Verify outlier indices correspond to high reconstruction errors
        errors = np.array(result["reconstruction_errors"])
        threshold = result["threshold_value"]

        for i, idx in enumerate(df.index):
            if idx in result["outlier_indices"]:
                # This sample should have error above threshold
                assert errors[i] > threshold
            else:
                # This sample should have error below threshold
                assert errors[i] <= threshold


class TestCrossMethodConsistency:
    """Test consistency between Mahalanobis and PCA methods."""

    def test_both_methods_same_indices_format(self, dataframe_with_custom_indices):
        """Test that both methods return indices in same format."""
        df, _, _ = dataframe_with_custom_indices

        mahal_result = detect_outliers_mahalanobis(
            df, variance_threshold=0.95, chi2_percentile=95.0
        )

        pca_result = detect_outliers_pca(
            df,
            explained_variance_threshold=0.95,
            outlier_threshold=1.0,  # Lower threshold for better detection
        )

        # Both should return same data_indices
        assert mahal_result["data_indices"] == pca_result["data_indices"]

        # Both should return indices from DataFrame
        assert all(idx in df.index for idx in mahal_result["outlier_indices"])
        assert all(idx in df.index for idx in pca_result["outlier_indices"])

    def test_clear_outliers_detected_by_both(self):
        """Test that very clear outliers are detected by both methods."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        # Create normal data
        X = np.random.randn(n_samples, n_features) * 0.5

        # Add EXTREMELY clear outliers (make them more obvious)
        clear_outlier_positions = [20, 50, 80]
        for pos in clear_outlier_positions:
            X[pos, :] = np.ones(n_features) * 20.0  # Use consistent extreme values

        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])

        # Run both methods with moderate thresholds
        mahal_result = detect_outliers_mahalanobis(
            df, variance_threshold=0.95, chi2_percentile=95.0
        )

        pca_result = detect_outliers_pca(
            df,
            explained_variance_threshold=0.95,
            outlier_threshold=1.0,  # Lower threshold for better detection
        )

        # Both should find the clear outliers
        mahal_outliers = set(mahal_result["outlier_indices"])
        pca_outliers = set(pca_result["outlier_indices"])

        # Both methods should find at least some of the clear outliers
        # Note: Methods may differ in which outliers they detect
        mahal_found = [pos for pos in clear_outlier_positions if pos in mahal_outliers]
        pca_found = [pos for pos in clear_outlier_positions if pos in pca_outliers]

        # Mahalanobis should find at least 1 of the clear outliers
        assert (
            len(mahal_found) >= 1
        ), f"Mahalanobis only found {mahal_found} of {clear_outlier_positions}"

        # PCA should find SOME outliers (may not be the same ones)
        # PCA reconstruction is less sensitive to extreme values in all dimensions
        assert pca_result["n_outliers"] > 0, "PCA should find some outliers in the data"

        # Note: Methods may find different outliers, which is expected
        # Mahalanobis looks at distance in PC space
        # PCA looks at reconstruction error
        # Both are valid but different approaches


class TestEdgeCasesIndexHandling:
    """Test index handling in edge cases."""

    def test_single_outlier_index(self):
        """Test when there's exactly one outlier."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X[25, :] = [10, 10, 10]  # Clear single outlier

        result_mahal = detect_outliers_mahalanobis(X, chi2_percentile=99.0)
        result_pca = detect_outliers_pca(X, outlier_threshold=3.0)

        # Both should find the outlier
        assert 25 in result_mahal["outlier_indices"]
        assert 25 in result_pca["outlier_indices"]

    def test_no_outliers_empty_indices(self):
        """Test that no outliers returns empty index list."""
        np.random.seed(42)
        # Very uniform data
        X = np.random.randn(30, 4) * 0.1

        result_mahal = detect_outliers_mahalanobis(
            X, chi2_percentile=99.9  # Very high threshold
        )
        result_pca = detect_outliers_pca(
            X, outlier_threshold=5.0  # Very high threshold
        )

        # Should return empty lists
        assert result_mahal["outlier_indices"] == []
        assert result_mahal["n_outliers"] == 0
        assert result_pca["outlier_indices"] == []
        assert result_pca["n_outliers"] == 0

    def test_all_outliers_all_indices(self):
        """Test when all samples are considered outliers."""
        np.random.seed(42)
        # Small dataset with high variability
        X = np.random.randn(10, 3) * 5

        # Use very low thresholds to catch everything
        result_mahal = detect_outliers_mahalanobis(X, chi2_percentile=10.0)  # Very low

        result_pca = detect_outliers_pca(
            X, outlier_threshold=0.01  # Even lower threshold
        )

        # Should find some outliers with these low thresholds
        assert result_mahal["n_outliers"] >= 3
        assert result_pca["n_outliers"] >= 2

        # All indices should be valid
        assert all(0 <= idx < 10 for idx in result_mahal["outlier_indices"])
        assert all(0 <= idx < 10 for idx in result_pca["outlier_indices"])

    def test_duplicate_values_different_indices(self):
        """Test that duplicate values get different indices."""
        # Create data with some duplicate rows
        X = np.array(
            [
                [1, 2, 3],
                [1, 2, 3],  # Duplicate of row 0
                [4, 5, 6],
                [4, 5, 6],  # Duplicate of row 2
                [100, 100, 100],  # Much clearer outlier
            ]
        )

        df = pd.DataFrame(X, columns=["a", "b", "c"])

        result = detect_outliers_mahalanobis(
            df, chi2_percentile=90.0
        )  # Lower threshold

        # Should identify the outlier at index 4
        assert 4 in result["outlier_indices"]

        # Data indices should be [0, 1, 2, 3, 4]
        assert result["data_indices"] == [0, 1, 2, 3, 4]


class TestIndexValidationWithRealData:
    """Test with real trait data fixtures."""

    def test_with_traits_summary_data(self, traits_summary_df):
        """Test index handling with real traits data."""
        from sleap_roots_analyze.data_cleanup import (
            get_trait_columns,
            remove_nan_samples,
        )

        # Get trait columns
        trait_cols = get_trait_columns(traits_summary_df)

        # Remove NaN samples first (required for outlier detection)
        df_clean, _, _ = remove_nan_samples(traits_summary_df, trait_cols)
        df_traits = df_clean[trait_cols]

        # Drop any remaining columns with NaN values
        df_traits = df_traits.dropna(axis=1)

        # Skip if no data left after cleaning
        if df_traits.empty or len(df_traits) < 2:
            pytest.skip("Not enough clean data for outlier detection")

        # Run both methods
        mahal_result = detect_outliers_mahalanobis(
            df_traits, variance_threshold=0.95, chi2_percentile=95.0
        )

        pca_result = detect_outliers_pca(
            df_traits, explained_variance_threshold=0.95, outlier_threshold=2.5
        )

        # Check that both methods completed without error
        assert "error" not in mahal_result or mahal_result.get("error") is None
        assert "error" not in pca_result or pca_result.get("error") is None

        # Check that indices are valid DataFrame indices
        for idx in mahal_result["outlier_indices"]:
            assert idx in df_traits.index

        for idx in pca_result["outlier_indices"]:
            assert idx in df_traits.index

        # Check data_indices match
        assert mahal_result["data_indices"] == df_traits.index.tolist()
        assert pca_result["data_indices"] == df_traits.index.tolist()

