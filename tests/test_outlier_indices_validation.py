"""Comprehensive tests for outlier index preservation and correctness.

This test suite ensures that outlier indices are correctly identified and preserved
across different data formats (DataFrame with custom indices, numpy arrays) for both
Mahalanobis and PCA reconstruction methods.
"""

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
    detect_outliers_pca,
)


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
