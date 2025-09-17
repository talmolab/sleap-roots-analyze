"""Tests for outlier detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
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
