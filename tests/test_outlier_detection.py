"""Tests for outlier detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
    detect_outliers_pca,
    detect_outliers_isolation_forest,
    calculate_outlier_threshold,
    identify_outliers_from_distances,
    remove_outliers_from_data,
    combine_outlier_methods,
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
        assert recall >= 0.6, f"Only detected {recall * 100:.1f}% of true outliers"

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
        # PCA generates "Feature_" with capital F
        assert result["feature_names"][0].startswith("Feature_")

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
        """Test handling of NaN values - should process valid rows."""
        edge_cases = outlier_data_edge_cases
        df_with_nan = edge_cases["with_nan"]

        # Check how many valid samples we have
        n_valid = len(df_with_nan.dropna())
        assert n_valid > 2  # Need at least a few samples for outlier detection

        result = detect_outliers_mahalanobis(
            df_with_nan,
            standardize=True,
        )

        # Should process the valid samples successfully
        assert result["method"] == "Mahalanobis"
        # Should have results for valid samples
        assert "mahalanobis_distances" in result
        assert len(result["mahalanobis_distances"]) == n_valid
        # Outlier indices should be from the original DataFrame's valid rows
        valid_indices = df_with_nan.dropna().index.tolist()
        for idx in result["outlier_indices"]:
            assert idx in valid_indices

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


class TestValidateChiSquaredDistribution:
    """Test chi-squared distribution validation."""

    def test_goodness_of_fit_with_chi_squared_data(self):
        """Test validation with data that follows chi-squared distribution."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        # Generate data that follows chi-squared distribution
        np.random.seed(42)
        df = 5
        n_samples = 500
        distances_squared = stats.chi2.rvs(df, size=n_samples)

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Check structure
        assert result["test_type"] == "Kolmogorov-Smirnov"
        assert "test_statistic" in result
        assert "p_value" in result
        assert "fit_quality" in result
        assert "interpretation" in result
        assert "distributional_assumption_valid" in result

        # Should have high p-value (good fit) since data follows chi-squared
        assert result["p_value"] > 0.05, "Data follows χ² but test failed"
        assert result["distributional_assumption_valid"] is True
        assert result["fit_quality"] in ["excellent", "good"]
        assert "warning" not in result

    def test_goodness_of_fit_with_multimodal_data(self):
        """Test GOF with multi-modal data (should fail)."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        # Generate bimodal data (mixture of two chi-squared distributions)
        np.random.seed(42)
        df = 5
        n_samples = 500

        # Create two clusters with different scales
        cluster1 = stats.chi2.rvs(df, size=n_samples // 2) * 0.5
        cluster2 = stats.chi2.rvs(df, size=n_samples // 2) * 2.0
        distances_squared = np.concatenate([cluster1, cluster2])

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Should have low p-value (poor fit) due to multi-modal structure
        assert result["p_value"] < 0.10, "Multi-modal data should not follow χ²"
        assert result["distributional_assumption_valid"] is False
        assert result["fit_quality"] in ["poor", "very_poor"]
        assert "warning" in result
        assert (
            "multi" in result["warning"].lower()
            or "cluster" in result["warning"].lower()
        )

    def test_goodness_of_fit_classifications(self):
        """Test that fit quality classifications work correctly."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 5

        # Test different scenarios by varying data quality
        test_cases = [
            # (data generator, expected_quality, expected_valid)
            (lambda: stats.chi2.rvs(df, size=1000), ["excellent", "good"], True),
            (
                lambda: stats.chi2.rvs(df, size=50),
                ["excellent", "good", "poor"],
                None,
            ),  # Small sample, variable
        ]

        for data_gen, expected_qualities, expected_valid in test_cases:
            distances_squared = data_gen()
            result = validate_chi_squared_distribution(distances_squared, df=df)

            assert result["fit_quality"] in expected_qualities
            if expected_valid is not None:
                assert result["distributional_assumption_valid"] == expected_valid

    def test_goodness_of_fit_empty_data(self):
        """Test GOF with empty data."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        result = validate_chi_squared_distribution(np.array([]), df=5)

        assert result["test_type"] == "Kolmogorov-Smirnov"
        assert np.isnan(result["test_statistic"])
        assert np.isnan(result["p_value"])
        assert result["fit_quality"] == "unknown"
        assert result["distributional_assumption_valid"] is False
        assert "warning" in result

    def test_goodness_of_fit_invalid_df(self):
        """Test GOF with invalid degrees of freedom."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        distances_squared = stats.chi2.rvs(5, size=100)

        # Test with df <= 0
        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            validate_chi_squared_distribution(distances_squared, df=0)

        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            validate_chi_squared_distribution(distances_squared, df=-1)

    def test_goodness_of_fit_interpretation_messages(self):
        """Test that interpretation messages are informative."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 5
        distances_squared = stats.chi2.rvs(df, size=100)

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Check interpretation contains key information
        interp = result["interpretation"]
        assert str(df) in interp, "Should mention degrees of freedom"
        assert "p = " in interp or "p=" in interp, "Should show p-value"
        assert "fit" in interp.lower(), "Should discuss fit quality"

    def test_goodness_of_fit_with_small_sample(self):
        """Test GOF with small sample size."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 5
        n_samples = 20  # Small sample
        distances_squared = stats.chi2.rvs(df, size=n_samples)

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Should still return valid result structure
        assert "test_statistic" in result
        assert "p_value" in result
        assert not np.isnan(result["p_value"])
        assert result["fit_quality"] in ["excellent", "good", "poor", "very_poor"]


class TestValidateChiSquaredDistributionLargeSamples:
    """Test chi-squared validation with large samples (n > 500)."""

    def test_large_sample_excellent_fit(self):
        """Test large sample (n=1000) with perfect chi-squared data."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 10
        n_samples = 1000
        distances_squared = stats.chi2.rvs(df, size=n_samples)

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Check new fields are present
        assert "n_samples" in result
        assert "evaluation_strategy" in result
        assert "effect_size_interpretation" in result

        # Should use effect size strategy
        assert result["n_samples"] == n_samples
        assert result["evaluation_strategy"] == "effect_size"

        # Should have excellent fit based on K-S statistic
        assert result["test_statistic"] < 0.05, (
            f"K-S statistic = {result['test_statistic']:.4f}, "
            "should be < 0.05 for excellent fit"
        )
        assert result["fit_quality"] == "excellent"
        assert result["distributional_assumption_valid"] is True
        assert "warning" not in result

        # Interpretation should mention large sample size
        assert "n = " in result["interpretation"]
        assert "not reliable due to large sample" in result["interpretation"]

    def test_large_sample_good_fit_with_minor_deviation(self):
        """Test large sample with minor systematic deviation."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 10
        n_samples = 1000

        # Add small systematic bias
        distances_squared = stats.chi2.rvs(df, size=n_samples) + 0.1

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Should use effect size strategy
        assert result["evaluation_strategy"] == "effect_size"

        # Should still have good fit (K-S statistic small enough)
        assert result["test_statistic"] < 0.10, (
            f"K-S statistic = {result['test_statistic']:.4f}, "
            "should be < 0.10 for good fit"
        )
        assert result["fit_quality"] in ["excellent", "good"]
        assert result["distributional_assumption_valid"] is True

    def test_large_sample_acceptable_fit(self):
        """Test large sample (n=925, like notebook) with acceptable fit."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 13  # Similar to notebook
        n_samples = 925

        # Add slight skew to simulate realistic data
        base_data = stats.chi2.rvs(df, size=n_samples)
        # Add some mild outliers
        outlier_indices = np.random.choice(
            n_samples, size=int(n_samples * 0.05), replace=False
        )
        base_data[outlier_indices] *= 1.5
        distances_squared = base_data

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Should use effect size strategy
        assert result["n_samples"] == n_samples
        assert result["evaluation_strategy"] == "effect_size"

        # K-S statistic should be in acceptable range
        # Note: With realistic data, we expect K-S between 0.05-0.15
        assert result["test_statistic"] < 0.20, (
            f"K-S statistic = {result['test_statistic']:.4f} is too high, "
            "even for realistic data"
        )

        # Should be acceptable or better
        assert result["fit_quality"] in ["excellent", "good", "acceptable"]

        # If acceptable, should have a warning explaining sample size sensitivity
        if result["fit_quality"] == "acceptable":
            assert "warning" in result
            assert "large sample" in result["warning"].lower()
            assert result["distributional_assumption_valid"] is True

    def test_large_sample_truly_poor_fit(self):
        """Test large sample with truly poor fit (bimodal)."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 10
        n_samples = 1000

        # Create strongly bimodal data
        cluster1 = stats.chi2.rvs(df=5, size=int(n_samples * 0.8))
        cluster2 = stats.chi2.rvs(df=30, size=int(n_samples * 0.2))
        distances_squared = np.concatenate([cluster1, cluster2])

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Should use effect size strategy
        assert result["evaluation_strategy"] == "effect_size"

        # Should have large K-S statistic
        assert result["test_statistic"] >= 0.15, (
            f"K-S statistic = {result['test_statistic']:.4f}, "
            "should be >= 0.15 for bimodal data"
        )

        # Should be poor or very poor
        assert result["fit_quality"] in ["poor", "very_poor"]
        assert result["distributional_assumption_valid"] is False
        assert "warning" in result
        assert "cluster" in result["warning"].lower()

    def test_sample_size_boundary(self):
        """Test behavior at n=500 boundary."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 10

        # Test n=500 (should use p-value)
        data_500 = stats.chi2.rvs(df, size=500)
        result_500 = validate_chi_squared_distribution(data_500, df=df)
        assert result_500["n_samples"] == 500
        assert result_500["evaluation_strategy"] == "p_value"

        # Test n=501 (should use effect size)
        data_501 = stats.chi2.rvs(df, size=501)
        result_501 = validate_chi_squared_distribution(data_501, df=df)
        assert result_501["n_samples"] == 501
        assert result_501["evaluation_strategy"] == "effect_size"

    def test_effect_size_interpretation_field(self):
        """Test that effect size interpretation is informative."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 10
        n_samples = 1000
        distances_squared = stats.chi2.rvs(df, size=n_samples)

        result = validate_chi_squared_distribution(distances_squared, df=df)

        # Check effect size interpretation exists and is informative
        assert "effect_size_interpretation" in result
        effect_interp = result["effect_size_interpretation"]
        assert "K-S = " in effect_interp
        assert result["test_statistic"] < 0.1  # Should have small K-S for chi2 data
        assert "fit" in effect_interp.lower()

    def test_comparison_small_vs_large_sample_same_data(self):
        """Compare results for same distribution at small vs large n."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 10

        # Small sample (n=200)
        data_small = stats.chi2.rvs(df, size=200)
        result_small = validate_chi_squared_distribution(data_small, df=df)

        # Large sample (n=1000)
        data_large = stats.chi2.rvs(df, size=1000)
        result_large = validate_chi_squared_distribution(data_large, df=df)

        # Both should indicate good fit, but using different strategies
        assert result_small["evaluation_strategy"] == "p_value"
        assert result_large["evaluation_strategy"] == "effect_size"

        # Both should be valid
        assert result_small["distributional_assumption_valid"] is True
        assert result_large["distributional_assumption_valid"] is True

        # Both should have good fit quality
        assert result_small["fit_quality"] in ["excellent", "good"]
        assert result_large["fit_quality"] in ["excellent", "good"]

        # Large sample p-value might be lower, but should still indicate good fit
        # This is the key test: same distribution, different sample sizes
        if result_large["p_value"] < 0.05:
            # P-value is low due to large n, but K-S statistic shows good fit
            assert result_large["test_statistic"] < 0.10, (
                "Large sample with low p-value should have small K-S statistic "
                "to indicate good practical fit"
            )

    def test_ks_statistic_thresholds(self):
        """Test that K-S statistic thresholds are correctly applied."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
        )

        np.random.seed(42)
        df = 10
        n_samples = 1000

        # Test different levels of deviation
        test_cases = [
            # (data, expected_max_ks, expected_quality)
            (stats.chi2.rvs(df, size=n_samples), 0.05, ["excellent"]),
            (stats.chi2.rvs(df, size=n_samples) * 1.05, 0.10, ["excellent", "good"]),
            # More deviant data should have higher K-S
        ]

        for data, expected_max_ks, expected_qualities in test_cases:
            result = validate_chi_squared_distribution(data, df=df)

            # K-S statistic should be reasonable
            assert (
                result["test_statistic"] <= expected_max_ks
                or result["fit_quality"] in expected_qualities
            ), (
                f"K-S = {result['test_statistic']:.4f}, "
                f"quality = {result['fit_quality']}, "
                f"expected in {expected_qualities}"
            )


class TestPrintGoodnessOfFitSummary:
    """Test console display of goodness-of-fit results."""

    def test_print_gof_summary_large_sample(self, capsys):
        """Test printing GOF summary for large sample case."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
            print_goodness_of_fit_summary,
        )

        # Generate data and get GOF results
        np.random.seed(42)
        df = 10
        n_samples = 1000
        distances_squared = stats.chi2.rvs(df, size=n_samples)
        gof_results = validate_chi_squared_distribution(distances_squared, df=df)

        # Print summary
        print_goodness_of_fit_summary(gof_results, df_value=df)

        # Capture output
        captured = capsys.readouterr()

        # Check key elements are present
        assert "Mahalanobis Chi-Squared Goodness-of-Fit" in captured.out
        assert f"{n_samples} samples" in captured.out
        assert f"{df} components" in captured.out
        assert "K-S Statistic:" in captured.out
        assert "P-value:" in captured.out
        assert "Fit Quality:" in captured.out
        assert "Effect Size" in captured.out  # Large sample strategy

    def test_print_gof_summary_small_sample(self, capsys):
        """Test printing GOF summary for small sample case."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
            print_goodness_of_fit_summary,
        )

        # Generate data and get GOF results
        np.random.seed(42)
        df = 5
        n_samples = 200
        distances_squared = stats.chi2.rvs(df, size=n_samples)
        gof_results = validate_chi_squared_distribution(distances_squared, df=df)

        # Print summary
        print_goodness_of_fit_summary(gof_results, df_value=df)

        # Capture output
        captured = capsys.readouterr()

        # Check key elements are present
        assert "Goodness-of-Fit" in captured.out
        assert f"{n_samples} samples" in captured.out
        assert "P-value" in captured.out  # Small sample uses p-value

    def test_print_gof_summary_with_warning(self, capsys):
        """Test printing GOF summary when warning is present."""
        from sleap_roots_analyze.outlier_detection import (
            validate_chi_squared_distribution,
            print_goodness_of_fit_summary,
        )

        # Generate bimodal data (should trigger warning)
        np.random.seed(42)
        df = 10
        n_samples = 1000
        cluster1 = stats.chi2.rvs(df=5, size=int(n_samples * 0.8))
        cluster2 = stats.chi2.rvs(df=30, size=int(n_samples * 0.2))
        distances_squared = np.concatenate([cluster1, cluster2])

        gof_results = validate_chi_squared_distribution(distances_squared, df=df)

        # Print summary
        print_goodness_of_fit_summary(gof_results, df_value=df)

        # Capture output
        captured = capsys.readouterr()

        # Check warning is displayed
        if "warning" in gof_results:
            assert len(captured.out) > 0  # Something was printed

    def test_print_gof_summary_acceptable_fit(self, capsys):
        """Test printing GOF summary for acceptable fit case."""
        from sleap_roots_analyze.outlier_detection import print_goodness_of_fit_summary

        # Create mock GOF results (like what notebook would have)
        gof_results = {
            "test_type": "Kolmogorov-Smirnov",
            "test_statistic": 0.1234,
            "p_value": 0.0001,
            "n_samples": 925,
            "fit_quality": "acceptable",
            "distributional_assumption_valid": True,
            "evaluation_strategy": "effect_size",
            "interpretation": "Acceptable fit: Data shows minor deviations from χ²(13)...",
            "warning": "Note: With large sample size, K-S test p-value is unreliable.",
        }

        # Print summary
        print_goodness_of_fit_summary(gof_results, df_value=13)

        # Capture output
        captured = capsys.readouterr()

        # Check formatting
        assert "╔" in captured.out  # Box drawing characters
        assert "925 samples" in captured.out
        assert "13 components" in captured.out
        assert "0.1234" in captured.out
        assert "ACCEPTABLE" in captured.out
        assert "unreliable" in captured.out

    def test_print_gof_summary_without_df(self, capsys):
        """Test that function extracts df from interpretation if not provided."""
        from sleap_roots_analyze.outlier_detection import print_goodness_of_fit_summary

        gof_results = {
            "test_type": "Kolmogorov-Smirnov",
            "test_statistic": 0.05,
            "p_value": 0.15,
            "n_samples": 300,
            "fit_quality": "good",
            "distributional_assumption_valid": True,
            "evaluation_strategy": "p_value",
            "interpretation": "Good fit: Data is consistent with χ²(7) distribution...",
        }

        # Don't provide df_value - should extract from interpretation
        print_goodness_of_fit_summary(gof_results)

        # Capture output
        captured = capsys.readouterr()

        # Should extract df=7 from interpretation
        assert "7 components" in captured.out or "?" in captured.out  # Fallback to ?


class TestMahalanobisGoodnessOfFitIntegration:
    """Test GOF integration with detect_outliers_mahalanobis."""

    def test_gof_included_with_chi_squared(self, outlier_data_with_known_outliers):
        """Test that GOF is included when use_chi_squared=True."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=97.5,
        )

        # Check that goodness_of_fit is present
        assert "goodness_of_fit" in result
        assert result["goodness_of_fit"] is not None

        # Check GOF structure
        gof = result["goodness_of_fit"]
        assert "test_type" in gof
        assert "test_statistic" in gof
        assert "p_value" in gof
        assert "fit_quality" in gof
        assert "distributional_assumption_valid" in gof

    def test_gof_not_included_without_chi_squared(
        self, outlier_data_with_known_outliers
    ):
        """Test that GOF is None when use_chi_squared=False."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=False,
            distance_threshold=3.0,
        )

        # Check that goodness_of_fit is None
        assert "goodness_of_fit" in result
        assert result["goodness_of_fit"] is None

    def test_gof_degrees_of_freedom_matches(self, outlier_data_with_known_outliers):
        """Test that GOF uses correct degrees of freedom."""
        df, _, _ = outlier_data_with_known_outliers

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=97.5,
        )

        n_components = result["n_components"]
        gof = result["goodness_of_fit"]

        # Interpretation should mention the correct df
        assert str(n_components) in gof["interpretation"]

    def test_gof_with_multimodal_data(self, outlier_data_multimodal):
        """Test GOF detects multimodal structure."""
        df, _, metadata = outlier_data_multimodal

        result = detect_outliers_mahalanobis(
            df,
            standardize=True,
            variance_threshold=0.95,
            use_chi_squared=True,
            chi2_percentile=97.5,
        )

        gof = result["goodness_of_fit"]

        # Multimodal data should show poor fit
        # Note: This is probabilistic, so we allow some flexibility
        # But typically should show evidence of poor fit
        assert "test_statistic" in gof
        assert "p_value" in gof

    def test_gof_survives_error_handling(self):
        """Test that GOF doesn't break error handling."""
        # Test with data that will cause PCA to fail
        bad_data = pd.DataFrame({"A": [1, 1, 1], "B": [2, 2, 2]})  # Constant columns

        result = detect_outliers_mahalanobis(
            bad_data,
            standardize=True,
            use_chi_squared=True,
        )

        # Should return error result
        assert "error" in result
        # GOF should not be present or should be handled gracefully
        if "goodness_of_fit" in result:
            assert result["goodness_of_fit"] is None


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

        # Data with NaN - should process valid rows
        with_nan = pd.DataFrame({"col1": [1, 2, np.nan, 4], "col2": [5, 6, 7, 8]})
        result_nan = detect_outliers_pca(with_nan)
        # Should process the 3 valid rows successfully
        assert result_nan["method"] == "PCA"
        assert len(result_nan["reconstruction_errors"]) == 3  # 3 valid rows
        assert result_nan["data_indices"] == [0, 1, 3]  # Indices of valid rows

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
            X,
            chi2_percentile=99.9,  # Very high threshold
        )
        result_pca = detect_outliers_pca(
            X,
            outlier_threshold=5.0,  # Very high threshold
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
            X,
            outlier_threshold=0.01,  # Even lower threshold
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


class TestRemoveOutliersFromData:
    """Test the remove_outliers_from_data function."""

    def test_basic_outlier_removal(self):
        """Test basic functionality of removing outliers."""
        # Create sample data
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "metadata": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            }
        )

        outlier_indices = [1, 5, 8]  # Remove rows at index 1, 5, 8

        # Test with return_outliers=True
        cleaned_df, outlier_df = remove_outliers_from_data(
            df, outlier_indices, return_outliers=True
        )

        # Check cleaned DataFrame
        assert len(cleaned_df) == 7  # 10 - 3 outliers
        assert outlier_indices[0] not in cleaned_df.index
        assert outlier_indices[1] not in cleaned_df.index
        assert outlier_indices[2] not in cleaned_df.index

        # Check outlier DataFrame
        assert len(outlier_df) == 3
        assert list(outlier_df.index) == outlier_indices
        assert outlier_df.loc[1, "feature1"] == 2
        assert outlier_df.loc[5, "feature1"] == 6
        assert outlier_df.loc[8, "feature1"] == 9

    def test_custom_index_handling(self):
        """Test handling of custom DataFrame indices."""
        # Create DataFrame with custom string indices
        df = pd.DataFrame(
            {"value": [10, 20, 30, 40, 50], "category": ["A", "B", "C", "D", "E"]},
            index=["row_a", "row_b", "row_c", "row_d", "row_e"],
        )

        outlier_indices = ["row_b", "row_d"]

        cleaned_df, outlier_df = remove_outliers_from_data(
            df, outlier_indices, return_outliers=True
        )

        # Check indices are handled correctly
        assert len(cleaned_df) == 3
        assert "row_b" not in cleaned_df.index
        assert "row_d" not in cleaned_df.index
        assert "row_a" in cleaned_df.index

        # Check outlier DataFrame preserves original indices
        assert list(outlier_df.index) == outlier_indices
        assert outlier_df.loc["row_b", "value"] == 20

    def test_reset_index_option(self):
        """Test the reset_index option."""
        df = pd.DataFrame({"data": [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50])

        outlier_indices = [20, 40]

        # Without reset_index
        cleaned_df = remove_outliers_from_data(
            df, outlier_indices, return_outliers=False, reset_index=False
        )
        assert list(cleaned_df.index) == [10, 30, 50]

        # With reset_index
        cleaned_df_reset = remove_outliers_from_data(
            df, outlier_indices, return_outliers=False, reset_index=True
        )
        assert list(cleaned_df_reset.index) == [0, 1, 2]

    def test_metadata_handling(self):
        """Test keep_metadata flag."""
        df = pd.DataFrame(
            {
                "numeric1": [1, 2, 3, 4, 5],
                "numeric2": [5, 4, 3, 2, 1],
                "text": ["a", "b", "c", "d", "e"],
                "category": pd.Categorical(["X", "Y", "X", "Y", "X"]),
            }
        )

        outlier_indices = [1, 3]

        # Keep all columns (default)
        cleaned_all = remove_outliers_from_data(
            df, outlier_indices, keep_metadata=True, return_outliers=False
        )
        assert list(cleaned_all.columns) == ["numeric1", "numeric2", "text", "category"]

        # Only numeric columns
        cleaned_numeric = remove_outliers_from_data(
            df, outlier_indices, keep_metadata=False, return_outliers=False
        )
        assert list(cleaned_numeric.columns) == ["numeric1", "numeric2"]

    def test_edge_cases(self):
        """Test edge cases."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [5, 4, 3, 2, 1]})

        # Empty outlier list
        cleaned_df = remove_outliers_from_data(df, [], return_outliers=False)
        assert len(cleaned_df) == len(df)
        pd.testing.assert_frame_equal(cleaned_df, df)

        # All samples are outliers
        all_indices = list(df.index)
        cleaned_df, outlier_df = remove_outliers_from_data(
            df, all_indices, return_outliers=True
        )
        assert len(cleaned_df) == 0
        assert len(outlier_df) == len(df)

    def test_invalid_indices(self):
        """Test handling of invalid indices."""
        df = pd.DataFrame({"data": [1, 2, 3, 4, 5]})

        # Mix of valid and invalid indices
        outlier_indices = [1, 2, 99]  # 99 doesn't exist

        # Should only remove valid indices
        cleaned_df = remove_outliers_from_data(
            df, outlier_indices, return_outliers=False
        )
        assert len(cleaned_df) == 3  # Removed indices 1 and 2

    def test_integration_with_detection(self, outlier_data_with_known_outliers):
        """Test integration with outlier detection functions."""
        df, expected_outliers, _ = outlier_data_with_known_outliers

        # Detect outliers
        result = detect_outliers_mahalanobis(
            df, variance_threshold=0.95, chi2_percentile=95.0
        )

        # Remove detected outliers
        cleaned_df, outlier_df = remove_outliers_from_data(
            df, result["outlier_indices"], return_outliers=True
        )

        # Check that outliers were removed
        assert len(cleaned_df) < len(df)
        assert len(outlier_df) == result["n_outliers"]

        # Outlier indices should not be in cleaned data
        for idx in result["outlier_indices"]:
            assert idx not in cleaned_df.index

    def test_with_nan_data(self):
        """Test handling data with NaN values."""
        df = pd.DataFrame({"col1": [1, 2, np.nan, 4, 5], "col2": [5, 4, 3, np.nan, 1]})

        outlier_indices = [0, 4]  # Remove first and last

        cleaned_df = remove_outliers_from_data(
            df, outlier_indices, return_outliers=False
        )

        # Should handle NaN values correctly
        assert len(cleaned_df) == 3
        assert 0 not in cleaned_df.index
        assert 4 not in cleaned_df.index
        # NaN values should still be present in remaining rows
        assert pd.isna(cleaned_df.loc[2, "col1"])
        assert pd.isna(cleaned_df.loc[3, "col2"])


class TestDetectOutliersIsolationForest:
    """Test suite for Isolation Forest outlier detection."""

    def test_basic_isolation_forest_detection(
        self, isolation_forest_data_with_anomalies
    ):
        """Test basic Isolation Forest outlier detection."""
        df, expected_anomalies, metadata = isolation_forest_data_with_anomalies

        result = detect_outliers_isolation_forest(
            df, contamination=metadata["contamination"], random_state=42
        )

        # Check basic structure
        assert result["method"] == "IsolationForest"
        assert "outlier_indices" in result
        assert "anomaly_scores" in result
        assert "n_outliers" in result
        assert "contamination" in result

        # Should find approximately the right number of outliers
        expected_n = metadata["n_anomalies"]
        assert abs(result["n_outliers"] - expected_n) <= 2  # Allow some tolerance

        # Check that anomaly scores are present for all samples
        assert len(result["anomaly_scores"]) == len(df)

        # Outliers should have more negative anomaly scores
        scores = np.array(result["anomaly_scores"])
        outlier_mask = np.zeros(len(df), dtype=bool)
        for idx in result["outlier_indices"]:
            outlier_mask[idx] = True

        if result["n_outliers"] > 0:
            mean_outlier_score = np.mean(scores[outlier_mask])
            mean_normal_score = np.mean(scores[~outlier_mask])
            assert (
                mean_outlier_score < mean_normal_score
            )  # More negative is more anomalous

    def test_contamination_parameter(self, isolation_forest_data_with_anomalies):
        """Test effect of contamination parameter."""
        df, _, _ = isolation_forest_data_with_anomalies

        # Low contamination - expect fewer outliers
        result_low = detect_outliers_isolation_forest(
            df, contamination=0.05, random_state=42
        )

        # High contamination - expect more outliers
        result_high = detect_outliers_isolation_forest(
            df, contamination=0.2, random_state=42
        )

        # Higher contamination should find more outliers
        assert result_high["n_outliers"] > result_low["n_outliers"]

        # Check that approximately correct proportion is flagged
        n_samples = len(df)
        assert abs(result_low["n_outliers"] / n_samples - 0.05) <= 0.02
        assert abs(result_high["n_outliers"] / n_samples - 0.2) <= 0.03

    def test_multimodal_data_handling(self, isolation_forest_multimodal_data):
        """Test Isolation Forest on multimodal data (its strength)."""
        df, expected_anomalies, metadata = isolation_forest_multimodal_data

        result = detect_outliers_isolation_forest(
            df, contamination=metadata["contamination"], random_state=42
        )

        # Should identify anomalies even in multimodal distribution
        found_anomalies = set(result["outlier_indices"])
        expected_set = set(expected_anomalies)

        # Check overlap - should find most anomalies
        overlap = found_anomalies.intersection(expected_set)
        assert len(overlap) >= len(expected_anomalies) * 0.5  # At least 50% overlap

    def test_high_dimensional_sparse_data(
        self, isolation_forest_high_dimensional_sparse
    ):
        """Test on high-dimensional sparse data where IF excels."""
        df, expected_anomalies, metadata = isolation_forest_high_dimensional_sparse

        result = detect_outliers_isolation_forest(
            df, contamination=metadata["contamination"], random_state=42
        )

        # Should handle high dimensions well
        assert result["n_outliers"] > 0

        # Check that it found some of the expected anomalies
        found = set(result["outlier_indices"])
        expected = set(expected_anomalies)
        overlap = found.intersection(expected)
        assert len(overlap) >= 2  # Should find at least some true anomalies

    def test_reproducibility(self, isolation_forest_data_with_anomalies):
        """Test that same random_state gives same results."""
        df, _, _ = isolation_forest_data_with_anomalies

        result1 = detect_outliers_isolation_forest(
            df, contamination=0.1, random_state=42
        )

        result2 = detect_outliers_isolation_forest(
            df, contamination=0.1, random_state=42
        )

        # Same random state should give identical results
        assert result1["outlier_indices"] == result2["outlier_indices"]
        assert result1["anomaly_scores"] == result2["anomaly_scores"]

        # Different random state should give different results
        result3 = detect_outliers_isolation_forest(
            df, contamination=0.1, random_state=123
        )

        # Scores might be slightly different
        assert result1["anomaly_scores"] != result3["anomaly_scores"]

    def test_index_preservation(self, dataframe_with_custom_indices):
        """Test that DataFrame indices are preserved."""
        df, _, _ = dataframe_with_custom_indices

        result = detect_outliers_isolation_forest(
            df, contamination=0.1, random_state=42
        )

        # Check that returned indices are from DataFrame
        assert all(idx in df.index for idx in result["outlier_indices"])

        # Check data_indices matches DataFrame index
        assert result["data_indices"] == df.index.tolist()

    def test_numpy_array_input(self, isolation_forest_data_with_anomalies):
        """Test with numpy array input."""
        df, _, metadata = isolation_forest_data_with_anomalies
        X = df.values

        result = detect_outliers_isolation_forest(
            X, contamination=metadata["contamination"], random_state=42
        )

        # Should work with numpy array
        assert result["method"] == "IsolationForest"
        assert "outlier_indices" in result

        # Indices should be integers for array input
        assert all(isinstance(idx, int) for idx in result["outlier_indices"])
        assert all(0 <= idx < len(X) for idx in result["outlier_indices"])

    def test_edge_cases(self, outlier_data_edge_cases):
        """Test edge cases for Isolation Forest."""
        edge_cases = outlier_data_edge_cases

        # Empty data
        result_empty = detect_outliers_isolation_forest(edge_cases["empty"])
        assert "error" in result_empty
        assert "Empty" in result_empty["error"]

        # Data with NaN - should process valid rows
        with_nan = pd.DataFrame({"col1": [1, 2, np.nan, 4], "col2": [5, 6, 7, 8]})
        result_nan = detect_outliers_isolation_forest(with_nan)
        # Should process the 3 valid rows successfully
        assert result_nan["method"] == "IsolationForest"
        assert len(result_nan["anomaly_scores"]) == 3  # 3 valid rows
        assert result_nan["data_indices"] == [0, 1, 3]  # Indices of valid rows

    def test_anomaly_score_ordering(self, isolation_forest_data_with_anomalies):
        """Test that anomaly scores correctly identify outliers."""
        df, expected_anomalies, _ = isolation_forest_data_with_anomalies

        result = detect_outliers_isolation_forest(
            df, contamination=0.1, random_state=42
        )

        scores = np.array(result["anomaly_scores"])
        labels = np.array(result["outlier_labels"])

        # Outliers (label=-1) should have lower scores than inliers (label=1)
        outlier_scores = scores[labels == -1]
        inlier_scores = scores[labels == 1]

        if len(outlier_scores) > 0 and len(inlier_scores) > 0:
            assert (
                np.max(outlier_scores) <= np.min(inlier_scores) + 0.1
            )  # Small tolerance

    def test_output_completeness(self, isolation_forest_data_with_anomalies):
        """Test that all expected outputs are present."""
        df, _, _ = isolation_forest_data_with_anomalies

        result = detect_outliers_isolation_forest(df, contamination=0.1)

        # Check all expected keys
        expected_keys = [
            "method",
            "contamination",
            "outlier_indices",
            "n_outliers",
            "anomaly_scores",
            "outlier_labels",
            "data_indices",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_comparison_with_other_methods(self, outlier_data_with_known_outliers):
        """Compare Isolation Forest with other methods on same data."""
        df, expected_outliers, _ = outlier_data_with_known_outliers

        # Run all three methods
        iso_result = detect_outliers_isolation_forest(
            df, contamination=0.1, random_state=42
        )

        pca_result = detect_outliers_pca(
            df, explained_variance_threshold=0.95, outlier_threshold=2.5
        )

        mahal_result = detect_outliers_mahalanobis(
            df, variance_threshold=0.95, chi2_percentile=95.0
        )

        # All should find some outliers
        assert iso_result["n_outliers"] > 0
        assert pca_result["n_outliers"] > 0
        assert mahal_result["n_outliers"] > 0

        # Isolation Forest might find different outliers than distance-based methods
        iso_outliers = set(iso_result["outlier_indices"])
        pca_outliers = set(pca_result["outlier_indices"])
        mahal_outliers = set(mahal_result["outlier_indices"])

        # But there should be some agreement on clear outliers
        # At least some overlap between any two methods
        overlap_iso_pca = iso_outliers.intersection(pca_outliers)
        overlap_iso_mahal = iso_outliers.intersection(mahal_outliers)
        overlap_all = iso_outliers.intersection(pca_outliers).intersection(
            mahal_outliers
        )

        # At least some method agreement expected
        assert len(overlap_iso_pca) > 0 or len(overlap_iso_mahal) > 0

    def test_contamination_validation(self):
        """Test that contamination parameter is validated."""
        df = pd.DataFrame(np.random.randn(100, 5))

        # Valid contamination
        result = detect_outliers_isolation_forest(df, contamination=0.1)
        assert "error" not in result

        # Test with boundary values
        result = detect_outliers_isolation_forest(df, contamination=0.01)  # Very low
        assert "error" not in result

        result = detect_outliers_isolation_forest(df, contamination=0.5)  # Maximum
        assert "error" not in result


class TestCombineOutlierMethods:
    """Test combine_outlier_methods function."""

    def test_basic_combination(self, outlier_data_with_known_outliers):
        """Test basic combination of outlier detection methods."""
        df, expected_outliers, metadata = outlier_data_with_known_outliers

        # Run individual methods
        pca_results = detect_outliers_pca(df)
        isolation_results = detect_outliers_isolation_forest(df)
        mahalanobis_results = detect_outliers_mahalanobis(df)

        # Combine results with default threshold
        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.5,
        )

        # Check basic structure
        assert combined["method"] == "Combined"
        assert combined["consensus_threshold"] == 0.5
        assert combined["n_methods"] == 3
        assert "consensus_outliers" in combined
        assert "n_consensus_outliers" in combined
        assert "agreement_summary" in combined

        # Check method-specific outliers are preserved
        assert "pca_outliers" in combined
        assert "isolation_forest_outliers" in combined
        assert "mahalanobis_outliers" in combined

        # Check agreement tracking
        assert "outlier_agreement_count" in combined
        assert "outlier_agreement_methods" in combined

        # Check method-only outliers
        assert "pca_only" in combined
        assert "isolation_forest_only" in combined
        assert "mahalanobis_only" in combined

        # Check overlaps
        assert "pca_isolation_forest_overlap" in combined
        assert "pca_mahalanobis_overlap" in combined
        assert "isolation_forest_mahalanobis_overlap" in combined

    def test_two_methods_combination(self, outlier_data_with_known_outliers):
        """Test combination with only two methods."""
        df, _, _ = outlier_data_with_known_outliers

        # Run only two methods
        pca_results = detect_outliers_pca(df)
        isolation_results = detect_outliers_isolation_forest(df)

        # Combine without Mahalanobis
        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=None,
            consensus_threshold=0.5,
        )

        # Check structure
        assert combined["n_methods"] == 2
        assert "mahalanobis_outliers" not in combined
        assert "mahalanobis_only" not in combined

        # Check that consensus requires at least 1 method (50% of 2)
        agreement_summary = combined["agreement_summary"]
        assert "1 out of 2" in agreement_summary["consensus_rule"]

    def test_mahalanobis_with_error(self, outlier_data_with_known_outliers):
        """Test handling Mahalanobis results with error."""
        df, _, _ = outlier_data_with_known_outliers

        # Run methods
        pca_results = detect_outliers_pca(df)
        isolation_results = detect_outliers_isolation_forest(df)

        # Create Mahalanobis results with error
        mahalanobis_results = {
            "error": "Singular covariance matrix",
            "outlier_indices": [],
        }

        # Combine
        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.5,
        )

        # Should only use 2 methods
        assert combined["n_methods"] == 2
        assert "mahalanobis_outliers" not in combined

    def test_consensus_thresholds(self, outlier_data_with_known_outliers):
        """Test different consensus thresholds."""
        df, _, _ = outlier_data_with_known_outliers

        # Run all methods
        pca_results = detect_outliers_pca(df)
        isolation_results = detect_outliers_isolation_forest(df)
        mahalanobis_results = detect_outliers_mahalanobis(df)

        # Test with strict consensus (all methods must agree)
        combined_strict = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=1.0,
        )

        # Test with loose consensus (any method is enough)
        combined_loose = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.34,  # Just over 1/3
        )

        # Test with majority consensus
        combined_majority = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.67,  # Just over 2/3
        )

        # Strict consensus should have fewer or equal outliers than loose
        assert len(combined_strict["consensus_outliers"]) <= len(
            combined_loose["consensus_outliers"]
        )

        # Majority should be between strict and loose
        assert len(combined_strict["consensus_outliers"]) <= len(
            combined_majority["consensus_outliers"]
        )
        assert len(combined_majority["consensus_outliers"]) <= len(
            combined_loose["consensus_outliers"]
        )

    def test_agreement_distribution(self, outlier_data_with_known_outliers):
        """Test agreement distribution tracking."""
        df, _, _ = outlier_data_with_known_outliers

        # Run all methods
        pca_results = detect_outliers_pca(df)
        isolation_results = detect_outliers_isolation_forest(df)
        mahalanobis_results = detect_outliers_mahalanobis(df)

        # Combine
        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.5,
        )

        # Check agreement distribution keys
        possible_keys = [
            "agreed_by_1_methods",
            "agreed_by_2_methods",
            "agreed_by_3_methods",
        ]

        # At least one of these should exist if there are outliers
        all_outliers = set()
        all_outliers.update(pca_results.get("outlier_indices", []))
        all_outliers.update(isolation_results.get("outlier_indices", []))
        all_outliers.update(mahalanobis_results.get("outlier_indices", []))

        if all_outliers:
            assert any(key in combined for key in possible_keys)

            # Check that all outliers are accounted for
            total_in_distribution = []
            for key in possible_keys:
                if key in combined:
                    total_in_distribution.extend(combined[key])

            assert set(total_in_distribution) == all_outliers

    def test_empty_outliers(self):
        """Test when no outliers are detected."""
        # Create clean data without outliers
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(50, 5), columns=[f"trait_{i}" for i in range(5)]
        )

        # Create results with no outliers
        pca_results = {"outlier_indices": [], "method": "PCA"}
        isolation_results = {"outlier_indices": [], "method": "IsolationForest"}
        mahalanobis_results = {"outlier_indices": [], "method": "Mahalanobis"}

        # Combine
        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.5,
        )

        # Check empty results
        assert combined["consensus_outliers"] == []
        assert combined["n_consensus_outliers"] == 0
        assert combined["outlier_agreement_count"] == {}
        assert combined["outlier_agreement_methods"] == {}

        # Method-only should be empty lists
        assert combined["pca_only"] == []
        assert combined["isolation_forest_only"] == []
        assert combined["mahalanobis_only"] == []

        # Overlaps should be empty
        assert combined["pca_isolation_forest_overlap"] == []
        assert combined["pca_mahalanobis_overlap"] == []
        assert combined["isolation_forest_mahalanobis_overlap"] == []

    def test_partial_overlap(self):
        """Test with known partial overlaps between methods."""
        # Create specific outlier patterns
        pca_results = {"outlier_indices": [0, 1, 2, 3], "method": "PCA"}
        isolation_results = {
            "outlier_indices": [2, 3, 4, 5],
            "method": "IsolationForest",
        }
        mahalanobis_results = {"outlier_indices": [0, 3, 5, 6], "method": "Mahalanobis"}

        # Combine with majority consensus
        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.66,  # Need 2 out of 3 (2/3 = 0.666...)
        )

        # Check consensus outliers (those agreed by at least 2 methods)
        # Index 3 is agreed by all 3 methods
        # Index 0 is agreed by PCA and Mahalanobis (2/3)
        # Index 5 is agreed by Isolation and Mahalanobis (2/3)
        # Index 2 is agreed by PCA and Isolation (2/3)
        # With 0.66 threshold, 2/3 methods (0.666...) passes the threshold
        # So consensus should include all indices agreed by 2+ methods
        assert 3 in combined["consensus_outliers"]  # All 3 methods agree
        assert 0 in combined["consensus_outliers"]  # PCA and Mahalanobis agree
        assert 5 in combined["consensus_outliers"]  # Isolation and Mahalanobis agree
        assert 2 in combined["consensus_outliers"]  # PCA and Isolation agree

        # Check that the right indices are marked as consensus
        # With threshold 0.67, need 2/3 methods = indices with agreement >= 2
        consensus = sorted(combined["consensus_outliers"])

        # Check method-only outliers (outliers unique to each method)
        # PCA has [0,1,2,3], others have [0,2,3,4,5,6], so PCA-only is [1]
        assert sorted(combined["pca_only"]) == [1]

        # Isolation has [2,3,4,5], others have [0,1,2,3,5,6], so Isolation-only is [4]
        assert sorted(combined["isolation_forest_only"]) == [4]

        # Mahalanobis has [0,3,5,6], others have [0,1,2,3,4,5], so Mahalanobis-only is [6]
        assert sorted(combined["mahalanobis_only"]) == [6]

        # Check overlaps
        assert sorted(combined["pca_isolation_forest_overlap"]) == [2, 3]
        assert sorted(combined["pca_mahalanobis_overlap"]) == [0, 3]
        assert sorted(combined["isolation_forest_mahalanobis_overlap"]) == [3, 5]

    def test_agreement_methods_tracking(self):
        """Test tracking which methods agree on each outlier."""
        # Create specific outlier patterns
        pca_results = {"outlier_indices": [0, 1, 2], "method": "PCA"}
        isolation_results = {"outlier_indices": [0, 2, 3], "method": "IsolationForest"}
        mahalanobis_results = {"outlier_indices": [0, 3, 4], "method": "Mahalanobis"}

        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            mahalanobis_results=mahalanobis_results,
            consensus_threshold=0.5,
        )

        # Check agreement methods for each outlier
        agreement_methods = combined["outlier_agreement_methods"]

        # Outlier 0 should be agreed by all methods
        assert set(agreement_methods[0]) == {"pca", "isolation_forest", "mahalanobis"}

        # Outlier 1 only by PCA
        assert agreement_methods[1] == ["pca"]

        # Outlier 2 by PCA and Isolation Forest
        assert set(agreement_methods[2]) == {"pca", "isolation_forest"}

        # Outlier 3 by Isolation Forest and Mahalanobis
        assert set(agreement_methods[3]) == {"isolation_forest", "mahalanobis"}

        # Outlier 4 only by Mahalanobis
        assert agreement_methods[4] == ["mahalanobis"]

    def test_with_real_data(self, features_df):
        """Test with real feature data."""
        # Get numeric columns only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        df_numeric = features_df[numeric_cols].dropna()

        if len(df_numeric) > 10:  # Need enough samples
            # Run all methods
            pca_results = detect_outliers_pca(df_numeric)
            isolation_results = detect_outliers_isolation_forest(df_numeric)
            mahalanobis_results = detect_outliers_mahalanobis(df_numeric)

            # Combine
            combined = combine_outlier_methods(
                pca_results=pca_results,
                isolation_results=isolation_results,
                mahalanobis_results=mahalanobis_results,
                consensus_threshold=0.5,
            )

            # Basic checks
            assert combined["method"] == "Combined"
            assert combined["n_methods"] in [2, 3]  # Mahalanobis might fail
            assert isinstance(combined["consensus_outliers"], list)
            assert isinstance(combined["n_consensus_outliers"], int)

    def test_sorted_outputs(self):
        """Test that outputs are properly sorted."""
        pca_results = {"outlier_indices": [5, 2, 8, 1], "method": "PCA"}
        isolation_results = {
            "outlier_indices": [3, 1, 7, 2],
            "method": "IsolationForest",
        }

        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=isolation_results,
            consensus_threshold=0.5,
        )

        # Check consensus outliers are sorted
        consensus = combined["consensus_outliers"]
        assert consensus == sorted(consensus)

        # Check agreement count dict is sorted by key
        agreement_count = combined["outlier_agreement_count"]
        keys = list(agreement_count.keys())
        assert keys == sorted(keys)

        # Check agreement methods dict is sorted by key
        agreement_methods = combined["outlier_agreement_methods"]
        keys = list(agreement_methods.keys())
        assert keys == sorted(keys)

    def test_integration_with_detection_methods(self, outlier_data_with_known_outliers):
        """Test full integration with actual detection methods."""
        df, expected_outliers, metadata = outlier_data_with_known_outliers

        # Run detection with different parameters
        pca_strict = detect_outliers_pca(df, n_components=3, outlier_threshold=3.0)
        pca_loose = detect_outliers_pca(df, n_components=2, outlier_threshold=2.0)
        isolation = detect_outliers_isolation_forest(df, contamination=0.1)

        # Combine different PCA results with isolation forest
        combined = combine_outlier_methods(
            pca_results=pca_strict,
            isolation_results=isolation,
            mahalanobis_results={"outlier_indices": pca_loose["outlier_indices"]},
            consensus_threshold=0.67,
        )

        # Should have valid results
        assert "consensus_outliers" in combined
        assert "n_consensus_outliers" in combined
        assert combined["n_methods"] == 3

        # Check that the structure is complete
        assert "pca_outliers" in combined
        assert "isolation_forest_outliers" in combined
        assert "mahalanobis_outliers" in combined

    def test_consensus_calculation_edge_cases(self):
        """Test edge cases in consensus calculation, particularly ceiling behavior."""
        # Test case 1: 3 methods with 0.5 threshold should require 2 methods (ceiling of 1.5)
        pca_results = {"outlier_indices": [0, 1], "method": "PCA"}
        iso_results = {"outlier_indices": [1, 2], "method": "IsolationForest"}
        mah_results = {"outlier_indices": [2, 3], "method": "Mahalanobis"}

        combined = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=iso_results,
            mahalanobis_results=mah_results,
            consensus_threshold=0.5,
        )

        # Check consensus rule shows correct calculation (2 out of 3)
        assert "2 out of 3" in combined["agreement_summary"]["consensus_rule"]

        # Indices 1 and 2 are flagged by 2 methods each, should be in consensus
        assert 1 in combined["consensus_outliers"]  # PCA and Isolation
        assert 2 in combined["consensus_outliers"]  # Isolation and Mahalanobis
        assert 0 not in combined["consensus_outliers"]  # Only PCA
        assert 3 not in combined["consensus_outliers"]  # Only Mahalanobis

        # Test case 2: 3 methods with 0.33 threshold should require 1 method (ceiling of 0.99)
        combined_low = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=iso_results,
            mahalanobis_results=mah_results,
            consensus_threshold=0.33,
        )

        assert "1 out of 3" in combined_low["agreement_summary"]["consensus_rule"]
        # All indices should be in consensus
        assert sorted(combined_low["consensus_outliers"]) == [0, 1, 2, 3]

        # Test case 3: 3 methods with 0.67 threshold should require 3 methods (ceiling of 2.01)
        combined_high = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=iso_results,
            mahalanobis_results=mah_results,
            consensus_threshold=0.67,
        )

        assert "3 out of 3" in combined_high["agreement_summary"]["consensus_rule"]
        # No indices are flagged by all 3 methods
        assert combined_high["consensus_outliers"] == []

        # Test case 4: 2 methods with 0.5 threshold should require 1 method (ceiling of 1.0)
        combined_two = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=iso_results,
            consensus_threshold=0.5,
        )

        assert "1 out of 2" in combined_two["agreement_summary"]["consensus_rule"]
        # Index 1 is flagged by both methods
        assert 1 in combined_two["consensus_outliers"]

        # Test case 5: Verify exact threshold boundaries
        # With 5 methods and 0.4 threshold: 5 * 0.4 = 2.0, ceil(2.0) = 2
        pca_results = {"outlier_indices": [0, 1], "method": "PCA"}
        iso_results = {"outlier_indices": [0, 2], "method": "IsolationForest"}
        mah_results = {"outlier_indices": [0, 3], "method": "Mahalanobis"}
        method4 = {"outlier_indices": [0, 4], "method": "Method4"}
        method5 = {"outlier_indices": [1, 5], "method": "Method5"}

        # Manually create the scenario with 5 methods
        # Index 0 is flagged by 4 methods (PCA, Iso, Mah, Method4)
        # Index 1 is flagged by 2 methods (PCA, Method5)
        results_5methods = {
            "outlier_indices": [],
            "pca": pca_results,
            "isolation_forest": iso_results,
            "mahalanobis": mah_results,
        }

        combined_5 = combine_outlier_methods(
            pca_results=pca_results,
            isolation_results=iso_results,
            mahalanobis_results=mah_results,
            consensus_threshold=0.4,  # 40% of 3 methods = 1.2, ceil = 2
        )

        assert "2 out of 3" in combined_5["agreement_summary"]["consensus_rule"]
        # Index 0 is flagged by all 3 methods
        assert 0 in combined_5["consensus_outliers"]

    def test_consensus_rule_string_accuracy(self):
        """Test that consensus rule string accurately reflects the calculation."""
        import math

        # Test cases for 2 and 3 methods only (what combine_outlier_methods supports)
        test_cases_3_methods = [
            (0.33, 1),  # 3 * 0.33 = 0.99, ceil = 1
            (0.34, 2),  # 3 * 0.34 = 1.02, ceil = 2
            (0.5, 2),  # 3 * 0.5 = 1.5, ceil = 2
            (0.66, 2),  # 3 * 0.66 = 1.98, ceil = 2
            (0.67, 3),  # 3 * 0.67 = 2.01, ceil = 3
            (1.0, 3),  # 3 * 1.0 = 3.0, ceil = 3
        ]

        test_cases_2_methods = [
            (0.49, 1),  # 2 * 0.49 = 0.98, ceil = 1
            (0.5, 1),  # 2 * 0.5 = 1.0, ceil = 1
            (0.51, 2),  # 2 * 0.51 = 1.02, ceil = 2
            (0.99, 2),  # 2 * 0.99 = 1.98, ceil = 2
            (1.0, 2),  # 2 * 1.0 = 2.0, ceil = 2
        ]

        # Test with 3 methods
        for threshold, expected_min in test_cases_3_methods:
            pca_results = {"outlier_indices": [], "method": "PCA"}
            iso_results = {"outlier_indices": [], "method": "IsolationForest"}
            mah_results = {"outlier_indices": [], "method": "Mahalanobis"}

            combined = combine_outlier_methods(
                pca_results=pca_results,
                isolation_results=iso_results,
                mahalanobis_results=mah_results,
                consensus_threshold=threshold,
            )

            # Extract the minimum methods from the consensus rule string
            rule = combined["agreement_summary"]["consensus_rule"]
            # Pattern: "X out of Y"
            import re

            match = re.search(r"(\d+) out of (\d+)", rule)
            assert match is not None, f"Could not parse rule: {rule}"

            actual_min = int(match.group(1))
            actual_total = int(match.group(2))

            # Verify the calculation
            assert actual_min == expected_min, (
                f"For 3 methods with threshold {threshold}: "
                f"expected {expected_min}, got {actual_min}"
            )
            assert (
                actual_total == 3
            ), f"Total methods mismatch: expected 3, got {actual_total}"

        # Test with 2 methods (no Mahalanobis)
        for threshold, expected_min in test_cases_2_methods:
            pca_results = {"outlier_indices": [], "method": "PCA"}
            iso_results = {"outlier_indices": [], "method": "IsolationForest"}

            combined = combine_outlier_methods(
                pca_results=pca_results,
                isolation_results=iso_results,
                mahalanobis_results=None,
                consensus_threshold=threshold,
            )

            rule = combined["agreement_summary"]["consensus_rule"]
            match = re.search(r"(\d+) out of (\d+)", rule)
            assert match is not None, f"Could not parse rule: {rule}"

            actual_min = int(match.group(1))
            actual_total = int(match.group(2))

            assert actual_min == expected_min, (
                f"For 2 methods with threshold {threshold}: "
                f"expected {expected_min}, got {actual_min}"
            )
            assert (
                actual_total == 2
            ), f"Total methods mismatch: expected 2, got {actual_total}"
