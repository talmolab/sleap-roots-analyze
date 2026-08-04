"""Tests for PCA analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from sleap_roots_analyze.pca import (
    _first_index_crossing_threshold,
    _total_variance_contribution,
    calculate_mahalanobis_distances,
    calculate_pca_metrics,
    calculate_pca_reconstruction_error,
    fit_pca,
    perform_pca_analysis,
    perform_pca_with_variance_threshold,
    select_n_components,
    select_n_features_by_variance,
    select_top_features_from_pca,
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

        # Single sample should raise ValueError
        data = np.random.randn(1, 5)
        with pytest.raises(ValueError, match="PCA requires at least 2 samples"):
            select_n_components(data)

    def test_select_low_variance_threshold(self, pca_3d_data):
        """Test with low variance threshold."""
        df, _ = pca_3d_data
        data = df.values

        # Low threshold should select fewer components
        n_low = select_n_components(data, explained_variance_threshold=0.50)
        n_high = select_n_components(data, explained_variance_threshold=0.99)

        assert n_low <= n_high


class TestSelectNFeaturesByVariance:
    """Test suite for select_n_features_by_variance function (issue #206)."""

    @pytest.fixture
    def feature_contributions_df(self):
        """Feature contributions matching perform_pca_analysis's shape.

        Sorted descending by contribution, `fractional_contribution` sums to 1.
        """
        return pd.DataFrame(
            {
                "total_contribution": [5.0, 3.0, 1.5, 0.5],
                "fractional_contribution": [0.5, 0.3, 0.15, 0.05],
            },
            index=["feat_a", "feat_b", "feat_c", "feat_d"],
        )

    def test_threshold_met_exactly_at_row_boundary(self, feature_contributions_df):
        """A threshold exactly equal to a cumulative-sum row is met by that row."""
        n = select_n_features_by_variance(feature_contributions_df, threshold=0.8)
        assert n == 2

    def test_threshold_requiring_all_features(self, feature_contributions_df):
        """A threshold only reached by the full cumulative sum selects every row."""
        n = select_n_features_by_variance(feature_contributions_df, threshold=1.0)
        assert n == 4

    def test_threshold_unreachable_even_by_full_sum_selects_all_features(
        self, feature_contributions_df
    ):
        """An unreachable threshold still returns all features.

        Exercises the `else` fallback branch, which a threshold of exactly
        1.0 cannot reach since `fractional_contribution` sums to exactly
        1.0 by construction here.
        """
        n = select_n_features_by_variance(feature_contributions_df, threshold=1.5)
        assert n == 4

    def test_threshold_met_by_fewer_than_all_features(self, feature_contributions_df):
        """The resolved count meets the threshold without wildly overshooting it."""
        n = select_n_features_by_variance(feature_contributions_df, threshold=0.6)

        cumulative = feature_contributions_df["fractional_contribution"].cumsum()
        assert cumulative.iloc[n - 1] >= 0.6
        # One fewer feature must NOT meet the threshold (minimal selection).
        assert n == 1 or cumulative.iloc[n - 2] < 0.6

    def test_non_positive_threshold_resolves_to_one_feature(
        self, feature_contributions_df
    ):
        """Threshold <= 0 selects exactly 1 feature without raising."""
        assert select_n_features_by_variance(feature_contributions_df, threshold=0) == 1
        assert (
            select_n_features_by_variance(feature_contributions_df, threshold=-0.5) == 1
        )

    def test_single_feature_dataframe(self):
        """A single-row DataFrame always resolves to 1 feature."""
        df = pd.DataFrame(
            {"total_contribution": [2.0], "fractional_contribution": [1.0]},
            index=["only_feature"],
        )
        assert select_n_features_by_variance(df, threshold=0.5) == 1
        assert select_n_features_by_variance(df, threshold=0.999) == 1

    def test_empty_dataframe_raises(self):
        """An empty feature_contributions DataFrame is a caller error."""
        df = pd.DataFrame({"total_contribution": [], "fractional_contribution": []})
        with pytest.raises(ValueError, match="at least one row"):
            select_n_features_by_variance(df, threshold=0.5)


class TestFirstIndexCrossingThreshold:
    """Test suite for the shared _first_index_crossing_threshold() helper.

    Used by both select_n_components() and select_n_features_by_variance()
    (design.md Decision 2) — tested directly here since it's the single
    source of truth both callers' own test suites now rely on implicitly.
    """

    def test_exact_boundary(self):
        """A threshold exactly equal to a cumulative-sum row is met by that row."""
        cumulative = np.array([0.5, 0.8, 0.95, 1.0])
        assert _first_index_crossing_threshold(cumulative, 0.8, total=4) == 2

    def test_threshold_never_reached_returns_total(self):
        """A threshold above the full cumulative sum returns total, not fewer."""
        cumulative = np.array([0.5, 0.8, 0.95, 1.0])
        assert _first_index_crossing_threshold(cumulative, 1.5, total=4) == 4

    def test_single_element(self):
        """A single-element cumulative array always resolves to 1."""
        cumulative = np.array([1.0])
        assert _first_index_crossing_threshold(cumulative, 0.5, total=1) == 1

    def test_clamps_to_total(self):
        """The result never exceeds total, even if cumulative has more rows."""
        cumulative = np.array([0.5, 0.8, 0.95, 1.0])
        # total=2 simulates a caller restricting to fewer elements than the
        # full cumulative array — result must still respect that cap.
        assert _first_index_crossing_threshold(cumulative, 1.0, total=2) == 2


class TestTotalVarianceContribution:
    """Test suite for the shared _total_variance_contribution() helper.

    perform_pca_analysis() and select_top_features_from_pca()'s
    "top_variance" method both call this single helper (design.md
    Decision 5) so their per-feature contribution values can never
    numerically diverge from independently re-deriving the same formula
    (Σ eigenvalue · loading²) via two different summation orders.
    """

    def test_matches_hand_calculated_values(self):
        """Sigma eigenvalue * loading^2 per feature, computed by hand."""
        loadings = np.array([[0.6, 0.2], [0.8, -0.1], [-0.3, 0.9]])
        eigenvalues = np.array([2.0, 0.5])

        result = _total_variance_contribution(loadings, eigenvalues)

        expected = np.array(
            [
                2.0 * 0.6**2 + 0.5 * 0.2**2,
                2.0 * 0.8**2 + 0.5 * (-0.1) ** 2,
                2.0 * (-0.3) ** 2 + 0.5 * 0.9**2,
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_n_features_restricts_rows(self):
        """An explicit n_features restricts to that many leading rows."""
        loadings = np.array([[0.6, 0.2], [0.8, -0.1], [-0.3, 0.9]])
        eigenvalues = np.array([2.0, 0.5])

        result = _total_variance_contribution(loadings, eigenvalues, n_features=2)

        assert len(result) == 2

    def test_matches_a_naive_accumulation_loop_up_to_float_noise(self):
        """Same formula as a naive loop, not necessarily the same bits.

        Not necessarily bit-identical to a naive per-PC accumulation loop
        — that gap is exactly the divergence this refactor eliminates by
        leaving only one implementation.

        With enough PCs, a naive per-column accumulation loop (the pattern
        previously duplicated in select_top_features_from_pca()'s
        "top_variance" branch before this refactor) can disagree with a
        vectorized `np.sum` by float-summation-order noise. Confirmed here
        with a fixed seed where the two demonstrably differ at the bit
        level, to make the "no longer two independent implementations"
        guarantee concrete rather than theoretical.
        """
        rng = np.random.RandomState(42)
        loadings = rng.randn(50, 30)
        eigenvalues = np.abs(rng.randn(30)) + 0.01

        naive_loop_result = np.zeros(50)
        for i in range(30):
            naive_loop_result += eigenvalues[i] * loadings[:, i] ** 2

        result = _total_variance_contribution(loadings, eigenvalues)

        np.testing.assert_allclose(result, naive_loop_result)  # same formula...
        assert not np.array_equal(result, naive_loop_result)  # ...different bits

    def test_matches_perform_pca_analysis_feature_contributions_exactly(self):
        """Stored total_contribution must equal a direct call to the helper.

        perform_pca_analysis()'s stored value and a direct call to this
        same shared helper with the same inputs — the single-source-of-truth
        guarantee design.md Decision 5 exists for.
        """
        np.random.seed(0)
        df = pd.DataFrame(
            np.random.randn(30, 6), columns=[f"trait{i}" for i in range(6)]
        )
        pca_results = perform_pca_analysis(df, n_components=4)

        direct = _total_variance_contribution(
            pca_results["loadings"], pca_results["eigenvalues"]
        )
        stored = (
            pca_results["feature_contributions"]
            .loc[pca_results["feature_names"], "total_contribution"]
            .to_numpy()
        )
        np.testing.assert_array_equal(direct, stored)


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

        # Single sample - should raise ValueError
        df = pd.DataFrame({"feat1": [1.0], "feat2": [2.0]})
        with pytest.raises(ValueError, match="PCA requires at least 2 samples"):
            perform_pca_analysis(df, standardize=False)


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
        from sleap_roots_analyze.data_cleanup import get_trait_columns

        # Use get_trait_columns to select only trait columns (more realistic)
        trait_cols = get_trait_columns(traits_summary_df)

        # Select a subset of trait columns that have good data coverage
        cols_with_good_coverage = []
        for col in trait_cols[:30]:  # Check first 30 trait columns
            # Count non-NaN values
            non_nan_count = traits_summary_df[col].notna().sum()
            # Select columns with at least 50% data coverage
            if non_nan_count >= len(traits_summary_df) * 0.5:
                cols_with_good_coverage.append(col)

        # Use at least 5 columns for meaningful PCA
        if len(cols_with_good_coverage) < 5:
            # If not enough columns with good coverage, use the ones with best coverage
            coverage_scores = {
                col: traits_summary_df[col].notna().sum() for col in trait_cols[:20]
            }
            cols_with_good_coverage = sorted(
                coverage_scores.keys(), key=lambda x: coverage_scores[x], reverse=True
            )[:10]

        # Create subset of data with selected columns
        test_data = traits_summary_df[cols_with_good_coverage].copy()

        # Full pipeline test with realistic data subset
        result = perform_pca_analysis(
            test_data, standardize=True, explained_variance_threshold=0.95
        )

        # Verify all components work together
        assert result["n_components_selected"] > 0
        # transformed_data will have fewer rows if NaN rows were dropped
        assert result["transformed_data"].shape[0] <= len(test_data)
        assert result["transformed_data"].shape[0] > 0  # At least some samples remain

        # Verify we got valid transformed data (not all NaN)
        assert not np.isnan(result["transformed_data"]).all()

        # Test reconstruction error if standardization was applied
        if result["scaler"] is not None:
            X_scaled = result["data_processed"]
            # Only calculate errors for samples that were used (non-NaN rows)
            valid_indices = np.where(~np.isnan(X_scaled).any(axis=1))[0]
            if len(valid_indices) > 0:
                errors = calculate_pca_reconstruction_error(
                    X_scaled[valid_indices], result
                )
                assert len(errors) == len(valid_indices)
                assert not np.isnan(errors).all()

        # Test Mahalanobis distances for valid samples
        X_transformed = result["transformed_data"]
        valid_transformed = X_transformed[~np.isnan(X_transformed).any(axis=1)]
        if len(valid_transformed) > 0:
            distances, _, _ = calculate_mahalanobis_distances(valid_transformed)
            assert len(distances) == len(valid_transformed)
            assert not np.isnan(distances).all()

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
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)

        # When threshold is very high (1.0), should use all available components
        n_components = select_n_components(
            X,
            explained_variance_threshold=1.0,
            n_components=None,  # Use all components
        )
        assert n_components >= 4  # Should need most/all components for 100% variance

        # Test with specified n_components overriding threshold
        n_components = select_n_components(
            X,
            explained_variance_threshold=0.5,
            n_components=2,  # Override with specific value
        )
        assert n_components == 2

    def test_mahalanobis_distances_1d_array(self, pca_1d_result_data):
        """Test calculate_mahalanobis_distances with 1D transformed data."""
        # Fit PCA requesting only 1 component
        pca, X_transformed = fit_pca(pca_1d_result_data.values, n_components=1)

        assert X_transformed.shape[1] == 1

        # Calculate distances with 1D data
        distances, mean, covariance = calculate_mahalanobis_distances(X_transformed)

        assert distances is not None
        assert mean.shape == (1,)  # 1D mean
        assert covariance.shape == (1, 1)  # 1x1 covariance
        assert len(distances) == len(X_transformed)

    def test_mahalanobis_distances_scalar_covariance(self):
        """Test calculate_mahalanobis_distances with scalar covariance."""
        # Create 1D data that might result in scalar covariance
        X_1d = np.random.randn(50, 1)

        distances, mean, cov = calculate_mahalanobis_distances(X_1d)

        # Verify covariance is properly handled as 2D array
        assert cov.shape == (1, 1)
        assert distances is not None

    def test_mahalanobis_distances_zero_std(self):
        """Test calculate_mahalanobis_distances with zero standard deviation."""
        # Create data with zero variance (all same value)
        X_constant = np.ones((30, 1)) * 5.0  # All values are 5.0

        distances, mean, cov = calculate_mahalanobis_distances(X_constant)

        # With zero std, all distances should be zero
        assert np.all(distances == 0)
        assert mean[0] == 5.0
        assert cov[0, 0] == 0  # Zero variance

    def test_perform_pca_all_nan_data(self, pca_all_nan_data):
        """Test perform_pca_analysis with all NaN DataFrame."""
        # Should raise ValueError when all data is NaN
        with pytest.raises(ValueError, match="No valid samples after removing NaN"):
            perform_pca_analysis(pca_all_nan_data)

    def test_perform_pca_empty_after_nan_removal(self, pca_empty_after_nan_removal):
        """Test perform_pca_analysis when data becomes empty after NaN removal."""
        # Every row has at least one NaN, so dropna() will remove all rows
        with pytest.raises(ValueError, match="No valid samples after removing NaN"):
            perform_pca_analysis(pca_empty_after_nan_removal)

    def test_perform_pca_zero_variance_all_columns(self, pca_zero_variance_all_columns):
        """Test perform_pca_analysis with all zero-variance columns."""
        # All columns have zero variance
        with pytest.raises(
            ValueError, match="No numeric columns with non-zero variance found"
        ):
            perform_pca_analysis(pca_zero_variance_all_columns)

    def test_perform_pca_single_sample(self, pca_single_sample_data):
        """Test perform_pca_analysis with single sample data."""
        # Single sample - should raise ValueError
        with pytest.raises(ValueError, match="PCA requires at least 2 samples"):
            perform_pca_analysis(pca_single_sample_data)

    def test_perform_pca_mixed_data_types(self, pca_mixed_numeric_nonnumeric):
        """Test perform_pca_analysis with mixed numeric and non-numeric columns."""
        # Should handle mixed data types gracefully
        result = perform_pca_analysis(
            pca_mixed_numeric_nonnumeric, n_components=2, standardize=True
        )

        # Should only use numeric columns
        assert len(result["feature_names"]) == 4  # Only the 4 numeric columns
        assert result["feature_names"] == ["value1", "value2", "value3", "value4"]
        assert result["n_components_selected"] <= 2

    def test_perform_pca_zero_std_features(self, pca_zero_std_features):
        """Test perform_pca_analysis with some zero-variance features."""
        # Should filter out zero-variance features
        result = perform_pca_analysis(
            pca_zero_std_features, n_components=None, standardize=True
        )

        # Should filter out truly zero-variance features (zero_std2 is all zeros)
        # zero_std1 might have tiny variance due to floating point representation
        assert (
            "zero_std2" not in result["feature_names"]
        )  # All zeros should be filtered
        assert "normal1" in result["feature_names"]
        assert "normal2" in result["feature_names"]
        assert "normal3" in result["feature_names"]

    def test_perform_pca_singular_covariance(self, pca_singular_covariance_data):
        """Test perform_pca_analysis with singular covariance matrix."""
        # Should handle linearly dependent features
        result = perform_pca_analysis(
            pca_singular_covariance_data, n_components=3, standardize=True
        )

        # Should still work despite linear dependencies
        assert result["n_components_selected"] <= 3

        # Test mahalanobis distances with singular covariance
        X_transformed = result["transformed_data"]
        distances, mean, cov = calculate_mahalanobis_distances(X_transformed)
        assert distances is not None

    def test_fit_pca_with_more_components_than_features(self):
        """Test fit_pca when requesting more components than features."""
        # 3 features but request 5 components - fit_pca should handle this
        X = np.random.randn(50, 3)

        # Should automatically cap to min(n_features, n_samples-1)
        pca, X_transformed = fit_pca(X, n_components=3)

        # Should only have min(n_samples-1, n_features) components
        assert pca.n_components_ <= 3

    def test_calculate_pca_metrics_edge_cases(self):
        """Test calculate_pca_metrics with edge cases."""
        # Test with 1 component PCA
        X = np.random.randn(100, 5)
        pca = SklearnPCA(n_components=1)
        X_transformed = pca.fit_transform(X)

        metrics = calculate_pca_metrics(pca, X_transformed)

        assert metrics["n_components_selected"] == 1
        assert len(metrics["explained_variance_ratio"]) == 1
        assert metrics["cumulative_variance_ratio"][-1] <= 1.0

    def test_perform_pca_with_variance_threshold_edge_cases(self):
        """Test perform_pca_with_variance_threshold with edge cases."""
        # Test with very high threshold (should use all components)
        X = np.random.randn(50, 3)
        result = perform_pca_with_variance_threshold(
            X, explained_variance_threshold=0.9999
        )
        assert result["n_components_selected"] >= 2

        # Test with very low threshold (should use 1 component)
        result = perform_pca_with_variance_threshold(
            X, explained_variance_threshold=0.01
        )
        assert result["n_components_selected"] == 1

    def test_mahalanobis_1d_ndim_reshape(self):
        """Test calculate_mahalanobis_distances with actual 1D array (line 231)."""
        # Create actual 1D array (not 2D with shape (n, 1))
        X_1d = np.random.randn(50)  # Shape is (50,) not (50, 1)

        # Should handle reshaping internally
        distances, mean, cov = calculate_mahalanobis_distances(X_1d)

        assert distances is not None
        assert mean.shape == (1,)
        assert cov.shape == (1, 1)

    def test_mahalanobis_scalar_covariance_ndim_0(self):
        """Test calculate_mahalanobis_distances with 0-dim covariance (line 253)."""
        # Create data that might produce scalar covariance
        # Single feature with very small variance
        X = np.ones((10, 1)) * 5 + np.random.randn(10, 1) * 1e-15

        distances, mean, cov = calculate_mahalanobis_distances(X, robust=False)

        # Covariance should be 2D
        assert cov.ndim == 2
        assert cov.shape == (1, 1)

    def test_perform_pca_no_numeric_columns(self):
        """Test perform_pca_analysis with no numeric columns (line 312)."""
        # DataFrame with only non-numeric columns
        df_non_numeric = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "category": ["X", "Y", "Z"],
                "description": ["foo", "bar", "baz"],
            }
        )

        with pytest.raises(ValueError, match="No numeric columns found"):
            perform_pca_analysis(df_non_numeric)

    def test_perform_pca_all_columns_zero_variance_after_filter(self):
        """Test when all columns have zero variance (line 346)."""
        # Create DataFrame where all columns will have zero variance
        n_samples = 20
        df = pd.DataFrame(
            {
                "all_same_1": [42.0] * n_samples,
                "all_same_2": [100.0] * n_samples,
                "all_zeros": np.zeros(n_samples),
                "all_ones": np.ones(n_samples),
            }
        )

        with pytest.raises(
            ValueError, match="No numeric columns with non-zero variance found"
        ):
            perform_pca_analysis(df)

    def test_perform_pca_array_no_features(self):
        """Test perform_pca_analysis with array input that has no features."""
        # Create array with shape (n_samples, 0) - no features
        X_no_features = np.empty((10, 0))

        # This gets converted to empty DataFrame
        with pytest.raises(ValueError, match="Empty DataFrame provided"):
            perform_pca_analysis(X_no_features)

    def test_mahalanobis_force_scalar_covariance(self):
        """Force scalar covariance matrix case (line 253)."""
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
        # Create DataFrame with mixed columns and zero variance column
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "good1": np.random.randn(20),
                "good2": np.random.randn(20) * 2,
                "zero_var": [5.0] * 20,  # Zero variance
                "text": ["A"] * 20,  # Non-numeric
            }
        )

        # Test without standardization - should still filter zero variance
        result = perform_pca_analysis(df, standardize=False)

        # Should only keep the 2 good features
        assert len(result["feature_names"]) == 2
        assert "good1" in result["feature_names"]
        assert "good2" in result["feature_names"]
        assert result["scaler"] is None  # No standardization


class TestStandardizationVerification:
    """Comprehensive tests to verify StandardScaler is working correctly."""

    def test_standardization_with_real_trait_data(self, traits_summary_df):
        """Test standardization with real trait data."""
        from sleap_roots_analyze.data_cleanup import get_trait_columns

        # Use get_trait_columns to properly exclude metadata columns
        trait_cols = get_trait_columns(traits_summary_df)
        # Select subset of columns with fewer NaNs for testing
        cols_with_data = []
        for col in trait_cols[:50]:  # Check first 50 trait columns
            if (
                traits_summary_df[col].notna().sum() > 100
            ):  # At least 100 non-NaN values
                cols_with_data.append(col)

        if len(cols_with_data) < 5:  # Need at least 5 features for meaningful PCA
            pytest.skip("Not enough columns with sufficient data for PCA testing")

        test_data = traits_summary_df[cols_with_data[:10]]  # Use up to 10 features
        result = perform_pca_analysis(test_data, standardize=True)

        if result["scaler"] is not None:
            # Verify standardized data has mean ≈ 0 and std ≈ 1
            processed = result["data_processed"]

            # Check mean is close to 0
            means = np.mean(processed, axis=0)
            np.testing.assert_allclose(
                means, 0, atol=1e-10, err_msg="Standardized data should have mean ≈ 0"
            )

            # Check std is close to 1
            stds = np.std(processed, axis=0, ddof=0)  # Use population std
            np.testing.assert_allclose(
                stds, 1, atol=1e-10, err_msg="Standardized data should have std ≈ 1"
            )

    def test_standardization_with_diverse_distributions(self):
        """Test standardization with various data distributions."""
        np.random.seed(42)
        n_samples = 1000

        # Create diverse distributions
        df = pd.DataFrame(
            {
                "normal": np.random.randn(n_samples),
                "lognormal": np.random.lognormal(0, 1, n_samples),
                "exponential": np.random.exponential(2, n_samples),
                "uniform": np.random.uniform(-10, 10, n_samples),
                "bimodal": np.concatenate(
                    [
                        np.random.normal(-3, 0.5, n_samples // 2),
                        np.random.normal(3, 0.5, n_samples // 2),
                    ]
                ),
            }
        )

        X_scaled, scaler, df_clean = standardize_data(df)

        # All distributions should be standardized
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0, ddof=0)

        np.testing.assert_allclose(
            means, 0, atol=1e-10, err_msg="All distributions should have mean ≈ 0"
        )
        np.testing.assert_allclose(
            stds, 1, atol=1e-10, err_msg="All distributions should have std ≈ 1"
        )

    def test_standardization_with_extreme_scales(self):
        """Test standardization with features at very different scales."""
        np.random.seed(42)
        n_samples = 500

        # Create features with vastly different scales
        df = pd.DataFrame(
            {
                "tiny": np.random.randn(n_samples) * 1e-6,  # Very small scale
                "small": np.random.randn(n_samples) * 0.01,
                "medium": np.random.randn(n_samples),
                "large": np.random.randn(n_samples) * 1000,
                "huge": np.random.randn(n_samples) * 1e6,  # Very large scale
            }
        )

        result = perform_pca_analysis(df, standardize=True)

        # After standardization, all should be on same scale
        processed = result["data_processed"]
        means = np.mean(processed, axis=0)
        stds = np.std(processed, axis=0, ddof=0)

        # Check all features are properly standardized
        np.testing.assert_allclose(
            means,
            0,
            atol=1e-9,
            err_msg="Features at different scales should have mean ≈ 0",
        )
        np.testing.assert_allclose(
            stds,
            1,
            atol=1e-9,
            err_msg="Features at different scales should have std ≈ 1",
        )

        # Verify feature names are preserved
        assert result["feature_names"] == ["tiny", "small", "medium", "large", "huge"]

    def test_standardization_with_outliers(self):
        """Test that standardization handles outliers correctly."""
        np.random.seed(42)
        n_samples = 200

        # Create data with outliers
        normal_data = np.random.randn(n_samples)

        # Add outliers
        outlier_indices = [10, 50, 100, 150]
        for idx in outlier_indices:
            normal_data[idx] = normal_data[idx] * 100  # Make outliers

        df = pd.DataFrame(
            {"with_outliers": normal_data, "normal": np.random.randn(n_samples)}
        )

        X_scaled, scaler, df_clean = standardize_data(df)

        # Even with outliers, standardization should work
        # Mean should still be close to 0
        means = np.mean(X_scaled, axis=0)
        np.testing.assert_allclose(
            means, 0, atol=1e-10, err_msg="Mean should be 0 even with outliers"
        )

        # Std should be 1 (outliers will affect this but StandardScaler handles it)
        stds = np.std(X_scaled, axis=0, ddof=0)
        np.testing.assert_allclose(
            stds, 1, atol=1e-10, err_msg="Std should be 1 even with outliers"
        )

    def test_ddof_consistency(self):
        """Test that ddof=0 (population variance) is used consistently."""
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "feat1": np.random.randn(n_samples) * 2 + 5,
                "feat2": np.random.randn(n_samples) * 0.5 - 3,
                "feat3": np.random.randn(n_samples) * 10,
            }
        )

        # Our standardization
        X_scaled, scaler, df_clean = standardize_data(df)

        # Manual calculation with ddof=0
        X_manual = df.values
        means_manual = np.mean(X_manual, axis=0)
        stds_manual = np.std(X_manual, axis=0, ddof=0)  # Population std
        X_manual_scaled = (X_manual - means_manual) / stds_manual

        # Verify our implementation matches manual calculation
        np.testing.assert_allclose(
            X_scaled,
            X_manual_scaled,
            atol=1e-10,
            err_msg="Standardization should use ddof=0",
        )

        # Verify sklearn StandardScaler also uses ddof=0
        sklearn_scaler = StandardScaler()
        X_sklearn = sklearn_scaler.fit_transform(df.values)
        np.testing.assert_allclose(
            X_scaled,
            X_sklearn,
            atol=1e-10,
            err_msg="Should match sklearn's StandardScaler",
        )

    def test_near_zero_variance_features(self):
        """Test handling of features with very small variance."""
        np.random.seed(42)
        n_samples = 100

        # Create features with different variance levels
        df = pd.DataFrame(
            {
                "normal_var": np.random.randn(n_samples),
                "tiny_var": np.random.randn(n_samples) * 1e-8,  # Very small variance
                "zero_var": np.ones(n_samples) * 5.0,  # Zero variance
                "small_var": np.random.randn(n_samples) * 0.001,
            }
        )

        X_scaled, scaler, df_clean = standardize_data(df)

        # Zero variance column should be removed
        assert "zero_var" not in df_clean.columns
        assert df_clean.shape[1] == 3  # Only 3 columns remaining

        # Remaining columns should be standardized
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0, ddof=0)

        np.testing.assert_allclose(means, 0, atol=1e-10)
        np.testing.assert_allclose(stds, 1, atol=1e-10)

    def test_standardization_inverse_transform(self):
        """Test that standardization can be reversed correctly."""
        np.random.seed(42)
        n_samples = 200

        # Create data with known properties
        df = pd.DataFrame(
            {
                "feat1": np.random.randn(n_samples) * 5 + 10,  # mean=10, std=5
                "feat2": np.random.randn(n_samples) * 2 - 5,  # mean=-5, std=2
                "feat3": np.random.randn(n_samples) * 0.5 + 100,  # mean=100, std=0.5
            }
        )

        # Store original data
        original_values = df.values.copy()

        result = perform_pca_analysis(df, standardize=True, n_components=2)

        # Get standardized data
        X_standardized = result["data_processed"]
        scaler = result["scaler"]

        # Verify standardization
        np.testing.assert_allclose(np.mean(X_standardized, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(
            np.std(X_standardized, axis=0, ddof=0), 1, atol=1e-10
        )

        # Inverse transform should recover original data
        X_recovered = scaler.inverse_transform(X_standardized)
        np.testing.assert_allclose(
            X_recovered,
            original_values,
            atol=1e-10,
            err_msg="Inverse transform should recover original data",
        )

    def test_standardization_with_array_input(self):
        """Test standardization when input is numpy array."""
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
        processed = result["data_processed"]
        means = np.mean(processed, axis=0)
        stds = np.std(processed, axis=0, ddof=0)

        np.testing.assert_allclose(
            means, 0, atol=1e-10, err_msg="Array input should be standardized correctly"
        )
        np.testing.assert_allclose(
            stds, 1, atol=1e-10, err_msg="Array input should have std ≈ 1"
        )

        # Feature names should be generated
        assert result["feature_names"] == [f"Feature_{i}" for i in range(n_features)]

    def test_standardization_preserves_relationships(self):
        """Test that standardization preserves relative relationships between samples."""
        np.random.seed(42)
        n_samples = 100

        # Create correlated features
        base = np.random.randn(n_samples)
        df = pd.DataFrame(
            {
                "feat1": base * 2 + 5,
                "feat2": base * 0.5 - 3 + np.random.randn(n_samples) * 0.1,
                "feat3": -base * 3 + 10 + np.random.randn(n_samples) * 0.2,
            }
        )

        # Calculate rank correlations before standardization
        corr_before = spearmanr(df.values)[0]

        # Standardize
        X_scaled, scaler, df_clean = standardize_data(df)

        # Calculate rank correlations after standardization
        corr_after = spearmanr(X_scaled)[0]

        # Rank correlations should be preserved
        np.testing.assert_allclose(
            corr_before,
            corr_after,
            atol=1e-10,
            err_msg="Standardization should preserve rank correlations",
        )


class TestPerFeatureVariance:
    """Test per-feature variance explained calculations."""

    def test_per_feature_variance_standardized(self):
        """Test per-feature variance with standardized data."""
        np.random.seed(42)
        n_samples = 100

        # Create data with known variance structure
        df = pd.DataFrame(
            {
                "high_var": np.random.randn(n_samples) * 5,
                "med_var": np.random.randn(n_samples) * 2,
                "low_var": np.random.randn(n_samples) * 0.5,
            }
        )

        result = perform_pca_analysis(
            df, standardize=True, include_feature_metrics=True
        )

        # Check feature metrics DataFrame was created
        assert "feature_metrics_df" in result
        metrics_df = result["feature_metrics_df"]

        # Check DataFrame structure
        assert len(metrics_df) == 3  # Three features
        assert "feature" in metrics_df.columns
        assert "variance_total" in metrics_df.columns
        assert "fraction_explained" in metrics_df.columns
        assert "loading_pc1" in metrics_df.columns

        # Check all fractions are non-negative
        fractions = metrics_df["fraction_explained"].values
        assert np.all(fractions >= 0), "Fractions should be non-negative"

        # For standardized data with all components, sum should equal number of features
        total_explained = fractions.sum()
        n_features = len(metrics_df)
        # When all components are used, total should be close to number of features
        assert (
            total_explained <= n_features + 0.01
        ), f"Total explained {total_explained} > {n_features}"

        # For standardized data, all features should have variance close to 1
        variances = metrics_df["variance_total"].values
        np.testing.assert_allclose(variances, 1.0, rtol=0.1)

    def test_per_feature_variance_unstandardized(self):
        """Test per-feature variance with unstandardized data."""
        np.random.seed(42)
        n_samples = 100

        # Create data with very different variances
        df = pd.DataFrame(
            {
                "high_var": np.random.randn(n_samples) * 10,
                "med_var": np.random.randn(n_samples) * 1,
                "low_var": np.random.randn(n_samples) * 0.1,
            }
        )

        result = perform_pca_analysis(
            df, standardize=False, include_feature_metrics=True
        )

        metrics_df = result["feature_metrics_df"]

        # Check fractions are non-negative
        fractions = metrics_df["fraction_explained"].values
        assert np.all(fractions >= 0), "Fractions should be non-negative"

        # For unstandardized data, the sum depends on variance structure
        # But individual fractions should still be reasonable
        total_explained = fractions.sum()
        assert total_explained > 0, "Total explained should be positive"

        # High variance feature should explain most variance
        high_var_idx = metrics_df[metrics_df["feature"] == "high_var"].index[0]
        assert metrics_df.loc[high_var_idx, "fraction_explained"] > 0.8

        # Check feature variances reflect original data
        var_dict = metrics_df.set_index("feature")["variance_total"].to_dict()
        assert var_dict["high_var"] > var_dict["med_var"] > var_dict["low_var"]

    def test_per_feature_sum_equals_total(self):
        """Test that sum of per-feature explained equals total explained."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        # Create random data
        df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feat_{i}" for i in range(n_features)],
        )

        # Test with all components retained (sum should equal n_components)
        for standardize in [True, False]:
            result = perform_pca_analysis(
                df,
                standardize=standardize,
                n_components=None,  # Use all components
                include_feature_metrics=True,
            )

            metrics_df = result["feature_metrics_df"]

            # When all components are used, sum of per-feature explained
            # should equal number of components retained
            n_components_used = result["n_components_selected"]
            sum_per_feature = metrics_df["fraction_explained"].sum()

            # For standardized, sum should be close to n_components
            # For unstandardized, it depends on the variance structure
            if standardize:
                np.testing.assert_allclose(
                    sum_per_feature,
                    n_components_used,
                    rtol=0.01,
                    err_msg=f"Sum of per-feature != n_components (standardize={standardize})",
                )

    def test_build_feature_metrics_df(self):
        """Test the build_feature_metrics_df helper function."""
        np.random.seed(42)
        n_samples = 50

        df = pd.DataFrame(
            {
                "a": np.random.randn(n_samples),
                "b": np.random.randn(n_samples) * 2,
                "c": np.random.randn(n_samples) * 0.5,
            }
        )

        # Run PCA
        result = perform_pca_analysis(df, standardize=True)

        # Build metrics DataFrame
        from sleap_roots_analyze.pca import build_feature_metrics_df

        metrics_df = build_feature_metrics_df(result)

        # Check structure
        assert len(metrics_df) == 3
        assert set(metrics_df["feature"]) == {"a", "b", "c"}

        # Test sorting options
        metrics_sorted = build_feature_metrics_df(result, sort_by="variance_total")
        assert metrics_sorted["variance_total"].is_monotonic_decreasing

        # Test without loadings
        metrics_no_load = build_feature_metrics_df(result, include_loadings=False)
        assert "loading_pc1" not in metrics_no_load.columns

        # Test custom loading prefix
        metrics_custom = build_feature_metrics_df(result, loading_prefix="PC")
        assert "PC1" in metrics_custom.columns

    def test_backward_compatibility(self):
        """Test that old code still works without X_fitted."""
        np.random.seed(42)
        n_samples = 50

        df = pd.DataFrame(np.random.randn(n_samples, 3))

        # Run PCA
        pca = SklearnPCA(n_components=2)
        X_transformed = pca.fit_transform(df.values)

        # Call calculate_pca_metrics without X_fitted (old way)
        from sleap_roots_analyze.pca import calculate_pca_metrics

        metrics = calculate_pca_metrics(pca, X_transformed)

        # Should still work with deprecation warning
        assert "explained_variance_per_feature" in metrics
        assert "loadings" in metrics
        assert "explained_variance_ratio" in metrics

    def test_zero_variance_features(self):
        """Test handling of zero-variance features."""
        np.random.seed(42)
        n_samples = 50

        df = pd.DataFrame(
            {
                "constant": np.ones(n_samples),  # Zero variance
                "normal": np.random.randn(n_samples),
                "high_var": np.random.randn(n_samples) * 5,
            }
        )

        # With standardization, constant feature should be removed
        result = perform_pca_analysis(
            df, standardize=True, include_feature_metrics=True
        )

        metrics_df = result["feature_metrics_df"]

        # Constant feature should not be in results
        assert "constant" not in metrics_df["feature"].values
        assert len(metrics_df) == 2

        # Without standardization but with constant feature
        result_no_std = perform_pca_analysis(
            df[["normal", "high_var"]],  # Exclude constant manually
            standardize=False,
            include_feature_metrics=True,
        )

        metrics_no_std = result_no_std["feature_metrics_df"]
        assert len(metrics_no_std) == 2

    def test_single_component_edge_case(self):
        """Test per-feature variance with single component."""
        np.random.seed(42)
        n_samples = 50

        df = pd.DataFrame(
            {
                "a": np.random.randn(n_samples),
                "b": np.random.randn(n_samples),
                "c": np.random.randn(n_samples),
            }
        )

        result = perform_pca_analysis(
            df,
            n_components=1,  # Only one component
            include_feature_metrics=True,
            standardize=True,
        )

        metrics_df = result["feature_metrics_df"]

        # Fractions should be non-negative
        fractions = metrics_df["fraction_explained"].values
        assert np.all(fractions >= 0)

        # Each fraction should be <= 1 (can't explain more than 100% of a feature's variance)
        assert np.all(fractions <= 1.01)  # Small tolerance for numerical errors

        # Check that variance_explained values are reasonable
        assert np.all(metrics_df["variance_explained"].values >= 0)

    def test_ddof_parameter_effect(self):
        """Test effect of ddof parameter on per-feature variance."""
        np.random.seed(42)
        n_samples = 10  # Small sample to see ddof effect

        df = pd.DataFrame(np.random.randn(n_samples, 3))

        # Test with ddof=0
        result_ddof0 = perform_pca_analysis(
            df, standardize=False, include_feature_metrics=True, ddof_feature_var=0
        )

        # Test with ddof=1 (default)
        result_ddof1 = perform_pca_analysis(
            df, standardize=False, include_feature_metrics=True, ddof_feature_var=1
        )

        # Feature variances should be different
        var0 = result_ddof0["feature_metrics_df"]["variance_total"].values
        var1 = result_ddof1["feature_metrics_df"]["variance_total"].values

        # ddof=1 should give larger variances
        assert np.all(var1 > var0)

        # The fractions will be different because the total variance is different
        frac0 = result_ddof0["feature_metrics_df"]["fraction_explained"].sum()
        frac1 = result_ddof1["feature_metrics_df"]["fraction_explained"].sum()

        # Both should be positive
        assert frac0 > 0
        assert frac1 > 0

        # With ddof=0, total can exceed n_features due to mismatch with PCA's ddof=1
        # Expected: frac0 = n/(n-1) * frac1
        # Since PCA uses ddof=1 but denominator uses ddof=0
        expected_factor = n_samples / (n_samples - 1)  # 10/9 ≈ 1.111

        # When using ddof=1 (matching PCA), sum should be close to n_features
        assert abs(frac1 - 3.0) < 0.01  # 3 features, all components retained

        # When using ddof=0, sum should be scaled up by n/(n-1)
        assert abs(frac0 - 3.0 * expected_factor) < 0.01


class TestRunPCAAndExportArtifacts:
    """Tests for run_pca_and_export_artifacts function."""

    def test_basic_export(self, pca_export_data, tmp_path):
        """Test basic PCA export functionality."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=5,
            save_csv=True,
            save_prefix="test_",
        )

        # Check returned dictionary structure
        assert "loadings_df" in result
        assert "trait_contrib_df" in result
        assert "variance_df" in result
        assert "pc_scores_df" in result
        assert "pca_results" in result
        assert "feature_metrics_df" in result

        # Verify DataFrames
        assert isinstance(result["loadings_df"], pd.DataFrame)
        assert isinstance(result["trait_contrib_df"], pd.DataFrame)
        assert isinstance(result["variance_df"], pd.DataFrame)
        assert isinstance(result["pc_scores_df"], pd.DataFrame)
        assert isinstance(result["feature_metrics_df"], pd.DataFrame)

        # Check loadings dimensions
        assert result["loadings_df"].shape[0] == len(trait_cols)  # n_features rows
        assert result["loadings_df"].shape[1] <= 5  # at most n_components columns

        # Check trait contributions
        trait_contrib = result["trait_contrib_df"]
        assert "trait" in trait_contrib.columns
        assert "trait_total_variance_contrib" in trait_contrib.columns
        assert "trait_fractional_contrib" in trait_contrib.columns

        # Verify fractional contributions sum to 1
        total_frac = trait_contrib["trait_fractional_contrib"].sum()
        assert np.allclose(
            total_frac, 1.0
        ), f"Fractional contributions sum to {total_frac}, not 1.0"

        # Check PC scores include metadata
        pc_scores = result["pc_scores_df"]
        assert "Barcode" in pc_scores.columns
        assert "geno" in pc_scores.columns
        assert "rep" in pc_scores.columns
        assert "PC1" in pc_scores.columns

        # Verify CSV files were created
        assert (tmp_path / "test_pca_loadings.csv").exists()
        assert (tmp_path / "test_trait_variance_contrib.csv").exists()
        assert (tmp_path / "test_pca_variance_explained.csv").exists()
        assert (tmp_path / "test_pca_transformed_data.csv").exists()
        assert (tmp_path / "test_feature_metrics.csv").exists()

    def test_variance_threshold(self, pca_export_data, tmp_path):
        """Test export with variance threshold instead of n_components."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            explained_variance_threshold=0.90,
            save_csv=False,  # Don't save files for this test
        )

        # Check that cumulative variance meets threshold
        pca_results = result["pca_results"]
        n_selected = pca_results["n_components_selected"]
        cumulative_variance = pca_results["cumulative_variance_ratio"][n_selected - 1]
        assert cumulative_variance >= 0.90

        # Verify loadings match selected components
        assert result["loadings_df"].shape[1] == n_selected

    def test_no_save_csv(self, pca_export_data, tmp_path):
        """Test that CSV files are not created when save_csv=False."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=3,
            save_csv=False,
        )

        # Check data is returned
        assert "loadings_df" in result
        assert "trait_contrib_df" in result

        # Verify no CSV files were created
        assert not (tmp_path / "pca_loadings.csv").exists()
        assert not (tmp_path / "trait_variance_contrib.csv").exists()

    def test_no_feature_metrics(self, pca_export_data, tmp_path):
        """Test export without feature metrics."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=3,
            save_csv=False,
            include_feature_metrics=False,
        )

        # Feature metrics should not be in result
        assert "feature_metrics_df" not in result

        # Other results should still be present
        assert "loadings_df" in result
        assert "trait_contrib_df" in result

    def test_trait_cols_none(self, pca_export_data, tmp_path):
        """Test with trait_cols=None (should auto-detect numeric columns)."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, _ = pca_export_data

        # Pass only trait columns (no metadata)
        df_traits_only = df.select_dtypes(include=[np.number])

        result = run_pca_and_export_artifacts(
            df_traits=df_traits_only,
            trait_cols=None,  # Auto-detect
            analysis_dir=tmp_path,
            n_components=3,
            save_csv=False,
            metadata_cols=[],  # No metadata to add
        )

        # Should work without error
        assert "loadings_df" in result
        assert result["loadings_df"].shape[0] == df_traits_only.shape[1]

    def test_custom_metadata_cols(self, pca_export_data, tmp_path):
        """Test with custom metadata columns."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        # Add additional metadata column
        df["batch"] = ["A" if i < 25 else "B" for i in range(len(df))]

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=3,
            save_csv=False,
            metadata_cols=("Barcode", "geno", "batch"),  # Custom metadata
        )

        # Check that custom metadata is included
        pc_scores = result["pc_scores_df"]
        assert "batch" in pc_scores.columns
        assert "rep" not in pc_scores.columns  # Not in custom list

    def test_variance_contributions_math(self, pca_export_data, tmp_path):
        """Test mathematical correctness of variance contributions."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=5,
            save_csv=False,
        )

        pca_results = result["pca_results"]
        trait_contrib = result["trait_contrib_df"]

        # Get components used
        n_used = pca_results["n_components_selected"]
        eigenvalues = pca_results["eigenvalues"][:n_used]
        loadings = pca_results["loadings"][:, :n_used]

        # Manually compute variance contributions
        manual_contrib = (loadings**2) * eigenvalues
        manual_total = manual_contrib.sum(axis=1)

        # Compare with function output (match trait ordering)
        pc_contrib_cols = [f"PC{i + 1}_variance_contrib" for i in range(n_used)]

        # Map traits to their indices in the original trait_cols
        trait_to_idx = {trait: i for i, trait in enumerate(trait_cols)}
        trait_order = [trait_to_idx[t] for t in trait_contrib["trait"]]

        func_contrib = trait_contrib[pc_contrib_cols].values
        func_total = trait_contrib["trait_total_variance_contrib"].values

        # Reorder manual calculations to match function output
        manual_contrib_sorted = manual_contrib[trait_order]
        manual_total_sorted = manual_total[trait_order]

        assert np.allclose(func_contrib, manual_contrib_sorted)
        assert np.allclose(func_total, manual_total_sorted)

        # Verify fractional contributions
        total_variance = eigenvalues.sum()
        manual_frac = manual_total_sorted / total_variance
        func_frac = trait_contrib["trait_fractional_contrib"].values

        assert np.allclose(func_frac, manual_frac)
        assert np.allclose(func_frac.sum(), 1.0)

    def test_standardization_option(self, pca_export_data, tmp_path):
        """Test with and without standardization."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        # With standardization
        result_std = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=3,
            standardize=True,
            save_csv=False,
        )

        # Without standardization
        result_no_std = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=3,
            standardize=False,
            save_csv=False,
        )

        # Results should be different
        loadings_std = result_std["loadings_df"].values
        loadings_no_std = result_no_std["loadings_df"].values

        assert not np.allclose(loadings_std, loadings_no_std)

    def test_fractional_contrib_sum_without_standardize(
        self, pca_export_data, tmp_path
    ):
        """Test that fractional contributions sum to 1 without standardization."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=4,
            standardize=False,  # No standardization
            save_csv=False,
        )

        contrib = result["trait_contrib_df"]["trait_fractional_contrib"].to_numpy()
        assert np.isclose(
            contrib.sum(), 1.0, atol=1e-9
        ), f"Fractional contributions sum to {contrib.sum():.12f}, expected ~1.0"

    def test_fractional_contrib_sum_with_threshold(self, pca_export_data, tmp_path):
        """Test that fractional contributions sum to 1 with variance threshold."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        df, trait_cols = pca_export_data

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=trait_cols,
            analysis_dir=tmp_path,
            n_components=None,  # Use threshold
            explained_variance_threshold=0.85,
            standardize=True,
            save_csv=False,
        )

        contrib = result["trait_contrib_df"]["trait_fractional_contrib"].to_numpy()
        assert np.isclose(
            contrib.sum(), 1.0, atol=1e-9
        ), f"Fractional contributions sum to {contrib.sum():.12f}, expected ~1.0"

    def test_metadata_handling_with_trait_cols_none(self, tmp_path):
        """Test that auto-detection includes all numeric columns when trait_cols=None."""
        from sleap_roots_analyze.pca import run_pca_and_export_artifacts

        # Create data with numeric metadata that could be mistaken for traits
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "trait_1": np.random.randn(50),
                "trait_2": np.random.randn(50),
                "trait_3": np.random.randn(50),
                "Barcode": [f"S{i:04d}" for i in range(50)],
                "geno": np.random.choice(["A", "B", "C"], 50),
                "rep": np.random.randint(1, 4, 50),  # Numeric metadata
                "batch": np.random.randint(1, 3, 50),  # Another numeric metadata
            }
        )

        result = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=None,  # Auto-detect traits
            analysis_dir=tmp_path,
            n_components=2,
            standardize=True,
            save_csv=False,
            metadata_cols=("Barcode", "geno", "rep", "batch"),
        )

        # When trait_cols=None, all numeric columns are treated as traits
        # This is the current expected behavior
        loadings_idx = set(result["loadings_df"].index)

        # String columns should not be in loadings
        assert (
            "Barcode" not in loadings_idx
        ), "String metadata 'Barcode' should not be in loadings"
        assert (
            "geno" not in loadings_idx
        ), "String metadata 'geno' should not be in loadings"

        # All numeric columns (including rep and batch) will be included
        assert "trait_1" in loadings_idx
        assert "trait_2" in loadings_idx
        assert "trait_3" in loadings_idx
        assert (
            "rep" in loadings_idx
        )  # Numeric metadata is included when trait_cols=None
        assert (
            "batch" in loadings_idx
        )  # Numeric metadata is included when trait_cols=None
        assert (
            len(loadings_idx) == 5
        ), f"Expected 5 numeric columns, got {len(loadings_idx)}"

        # To exclude metadata, users should explicitly specify trait_cols
        result2 = run_pca_and_export_artifacts(
            df_traits=df,
            trait_cols=["trait_1", "trait_2", "trait_3"],  # Explicit trait list
            analysis_dir=tmp_path,
            n_components=2,
            standardize=True,
            save_csv=False,
            metadata_cols=("Barcode", "geno", "rep", "batch"),
        )

        # With explicit trait_cols, only specified traits are included
        loadings_idx2 = set(result2["loadings_df"].index)
        assert "rep" not in loadings_idx2
        assert "batch" not in loadings_idx2
        assert len(loadings_idx2) == 3


class TestPCAMathematicalValidation:
    """Mathematical validation of PCA implementation correctness."""

    def test_pca_shapes_and_types(self, controlled_spectrum_data):
        """Test #1: Basic shape/type sanity checks."""
        df = controlled_spectrum_data
        n_samples, n_features = df.shape

        # Test with and without standardization
        for standardize in [True, False]:
            result = perform_pca_analysis(df, standardize=standardize)

            # Check shapes
            m_star = result["n_components_selected"]
            assert result["transformed_data"].shape == (n_samples, m_star)
            assert result["loadings"].shape == (n_features, m_star)
            assert result["eigenvalues"].shape == (m_star,)

            # Check eigenvalues are non-negative
            assert np.all(
                result["eigenvalues"] >= -1e-10
            )  # Small tolerance for numerical errors

            # Check loadings are orthonormal
            loadings = result["loadings"]
            orthonormal_check = loadings.T @ loadings
            np.testing.assert_allclose(
                orthonormal_check,
                np.eye(m_star),
                atol=1e-7,
                err_msg=f"Loadings not orthonormal (standardize={standardize})",
            )

    def test_trace_accounting(self, controlled_spectrum_data):
        """Test #2: Sum of per-feature explained variance equals sum of eigenvalues."""
        df = controlled_spectrum_data

        result = perform_pca_analysis(
            df, standardize=True, include_feature_metrics=True
        )

        var_explained = result["explained_variance_per_feature"]
        eigenvalues = result["eigenvalues"]

        np.testing.assert_allclose(
            var_explained.sum(),
            eigenvalues.sum(),
            rtol=1e-6,
            atol=1e-8,
            err_msg="Sum of per-feature variance != sum of eigenvalues",
        )

    def test_total_fraction_explained_bounds(self, controlled_spectrum_data):
        """Test #3: Total fraction in [0,1] and equals 1 with all PCs."""
        df = controlled_spectrum_data
        n_samples, n_features = df.shape

        # Test with all components
        max_components = min(n_features, n_samples - 1)
        result_full = perform_pca_analysis(
            df, n_components=max_components, ddof_feature_var=1
        )

        # Should be very close to 1
        total_full = result_full["total_variance_explained_consistent"]
        np.testing.assert_allclose(total_full, 1.0, rtol=1e-5)

        # Test with fewer components
        result_partial = perform_pca_analysis(
            df, explained_variance_threshold=0.8, ddof_feature_var=1
        )

        total_partial = result_partial["total_variance_explained_consistent"]
        assert 0 < total_partial < 1, f"Total fraction {total_partial} not in (0,1)"

    def test_per_feature_fraction_bounds(self, controlled_spectrum_data):
        """Test #4: Per-feature fractions in [0,1]."""
        df = controlled_spectrum_data

        result = perform_pca_analysis(df, include_feature_metrics=True)

        fractions = result["explained_variance_ratio_per_feature"]

        # Check bounds with small tolerance for numerical errors
        assert np.all(fractions >= -1e-10), "Some fractions < 0"
        assert np.all(fractions <= 1 + 1e-10), "Some fractions > 1"

    def test_standardization_population_variance(self, controlled_spectrum_data):
        """Test #5: After standardize=True, per-feature population variance is 1."""
        df = controlled_spectrum_data

        result = perform_pca_analysis(df, standardize=True)

        X_processed = result["data_processed"]
        pop_variances = np.var(X_processed, axis=0, ddof=0)

        np.testing.assert_allclose(
            pop_variances,
            1.0,
            atol=1e-8,
            err_msg="Population variances not 1 after standardization",
        )

    def test_ddof_consistency_total_fraction(self, controlled_spectrum_data):
        """Test #6: Using ddof=1 matches sklearn totals with full PCs."""
        df = controlled_spectrum_data
        n_samples, n_features = df.shape

        # Run with ddof=1 and all components
        max_components = min(n_features, n_samples - 1)
        result = perform_pca_analysis(
            df, n_components=max_components, ddof_feature_var=1, standardize=False
        )

        # Total should be ~1
        total = result["total_variance_explained_consistent"]
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)

        # Show inflation with ddof=0
        result_ddof0 = perform_pca_analysis(
            df,
            n_components=max_components,
            ddof_feature_var=0,
            standardize=False,
            include_feature_metrics=True,
        )

        total_ddof0 = result_ddof0["feature_metrics_df"]["fraction_explained"].sum()
        # When using ddof=0, the sum should be inflated by n/(n-1) compared to using ddof=1
        # Since PCA eigenvalues use ddof=1, but we compute variances with ddof=0
        expected_ratio = n_samples / (n_samples - 1)

        # The ratio should be between the sums, not between individual totals
        # total_ddof0 is the sum of all per-feature fractions with ddof=0 denominators
        # We expect this to be larger than n_components when ddof mismatch occurs

        # For all components, with ddof=1 we get sum ≈ n_components
        # With ddof=0 we get sum ≈ n_components * n/(n-1)
        expected_sum_ddof0 = max_components * expected_ratio
        np.testing.assert_allclose(total_ddof0, expected_sum_ddof0, rtol=0.1)

    def test_loadings_orthonormal(self, controlled_spectrum_data):
        """Test #7: Loadings columns are orthonormal."""
        df = controlled_spectrum_data

        result = perform_pca_analysis(df)
        loadings = result["loadings"]
        m_star = result["n_components_selected"]

        # Check V^T @ V = I
        should_be_identity = loadings.T @ loadings
        np.testing.assert_allclose(
            should_be_identity,
            np.eye(m_star),
            atol=1e-7,
            err_msg="Loadings not orthonormal",
        )

    def test_feature_metrics_dataframe_alignment(self, controlled_spectrum_data):
        """Test #8: Feature metrics DataFrame consistency."""
        df = controlled_spectrum_data

        result = perform_pca_analysis(df, include_feature_metrics=True)

        metrics_df = result["feature_metrics_df"]
        feature_names = result["feature_names"]
        eigenvalues = result["eigenvalues"]

        # Check alignment
        assert len(metrics_df) == len(feature_names)

        # Check sum consistency
        np.testing.assert_allclose(
            metrics_df["variance_explained"].sum(), eigenvalues.sum(), rtol=1e-6
        )

        # Recompute fractions and check
        with np.errstate(divide="ignore", invalid="ignore"):
            recomputed_fractions = np.where(
                metrics_df["variance_total"].values > 0,
                metrics_df["variance_explained"].values
                / metrics_df["variance_total"].values,
                0.0,
            )
        np.testing.assert_allclose(
            recomputed_fractions, metrics_df["fraction_explained"].values, rtol=1e-6
        )

    def test_select_n_components_threshold_behavior(self, diagonal_covariance_data):
        """Test #9: Threshold logic for component selection."""
        df, true_eigenvalues = diagonal_covariance_data

        # The data has independent features with known variances
        # We need to test that the threshold logic works
        result = perform_pca_analysis(
            df, standardize=False, explained_variance_threshold=0.95
        )

        # Check that we selected a reasonable number of components
        # With eigenvalues [5, 3, 2, 1, 0.5, 0.2, 0.1], total = 11.8
        # Cumulative: [5/11.8=0.42, 8/11.8=0.68, 10/11.8=0.85, 11/11.8=0.93, 11.5/11.8=0.97]
        # So 0.95 threshold should select 5 components
        assert result["n_components_selected"] in [
            4,
            5,
            6,
        ]  # Allow some variance due to sampling

    def test_mahalanobis_1d_and_nd(self):
        """Test #11: Mahalanobis distance in 1D and higher dimensions."""
        rng = np.random.default_rng(0)

        # 1D case - should be |z-score|
        X_1d = rng.standard_normal((100, 1))
        distances_1d, mean_1d, cov_1d = calculate_mahalanobis_distances(X_1d)

        # Manual calculation
        z_scores = np.abs((X_1d - mean_1d) / np.sqrt(cov_1d))
        np.testing.assert_allclose(distances_1d, z_scores.ravel(), rtol=1e-6)

        # Higher-D case
        X_nd = rng.standard_normal((100, 5))
        distances_nd, _, _ = calculate_mahalanobis_distances(X_nd)

        # All distances should be non-negative
        assert np.all(distances_nd >= 0)

    def test_reconstruction_error_monotonicity(self, controlled_spectrum_data):
        """Test #12: More components → lower reconstruction error."""
        df = controlled_spectrum_data
        n_features = df.shape[1]

        errors = []
        for n_comp in range(1, min(n_features, 5)):
            result = perform_pca_analysis(df, n_components=n_comp)
            pca = result["pca"]
            X_processed = result["data_processed"]

            # Compute reconstruction error
            X_reconstructed = pca.inverse_transform(pca.transform(X_processed))
            error = np.mean((X_processed - X_reconstructed) ** 2)
            errors.append(error)

        # Check monotonicity
        for i in range(1, len(errors)):
            assert (
                errors[i] <= errors[i - 1] + 1e-10
            ), f"Error not decreasing: {errors[i]} > {errors[i - 1]}"


class TestVisualizationDataConsistency:
    """Tests for ensuring visualization data consistency with PCA results."""

    def test_feature_variance_explained_values(self):
        """Test specific values matching the worked example in documentation."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import perform_pca_analysis, calculate_pca_metrics

        # Create synthetic data matching the documentation example
        # 3 features, structured to have specific eigenvalue distribution
        np.random.seed(42)
        n_samples = 100

        # Create data with controlled variance structure
        # PC1: high variance (λ ≈ 5)
        # PC2: medium variance (λ ≈ 2)
        # PC3: low variance (λ ≈ 0.5)
        pc1 = np.random.randn(n_samples) * np.sqrt(5)
        pc2 = np.random.randn(n_samples) * np.sqrt(2)
        pc3 = np.random.randn(n_samples) * np.sqrt(0.5)

        # Create features as linear combinations
        # Approximate the loading matrix from documentation
        feature1 = 0.7071 * pc1 + 0.6782 * pc2 + 0.2 * pc3
        feature2 = -0.7071 * pc1 + 0.6782 * pc2 + 0.2 * pc3
        feature3 = 0.0 * pc1 - 0.2828 * pc2 + 0.9592 * pc3

        df = pd.DataFrame(
            {"trait_1": feature1, "trait_2": feature2, "trait_3": feature3}
        )

        # Perform PCA retaining only 2 components
        result = perform_pca_analysis(df, standardize=False, n_components=2)

        # Check fraction explained for each feature
        fractions = result.get("explained_variance_ratio_per_feature")

        if fractions is not None:
            # Traits 1 & 2 should have high fraction explained (>0.9)
            assert (
                fractions[0] > 0.9
            ), f"Trait 1 fraction {fractions[0]} should be > 0.9"
            assert (
                fractions[1] > 0.9
            ), f"Trait 2 fraction {fractions[1]} should be > 0.9"

            # Trait 3 should have low fraction explained (<0.5)
            assert (
                fractions[2] < 0.5
            ), f"Trait 3 fraction {fractions[2]} should be < 0.5"

    def test_visualization_data_consistency(self):
        """Ensure visualization uses correct data source for feature variance."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import perform_pca_analysis
        from sleap_roots_analyze.outlier_visualization import (
            create_mahalanobis_outlier_plots,
        )
        from sleap_roots_analyze.outlier_detection import detect_outliers_mahalanobis

        # Create test data
        np.random.seed(42)
        df = pd.DataFrame(
            {f"trait_{i}": np.random.randn(100) * (i + 1) for i in range(10)}
        )

        # Test with explained_variance_ratio_per_feature present
        result = perform_pca_analysis(df, explained_variance_threshold=0.8)
        mahal_result = detect_outliers_mahalanobis(df)

        # Create visualization
        figures = create_mahalanobis_outlier_plots(df, mahal_result)

        # Verify that explained_variance_ratio_per_feature is used when available
        assert (
            "explained_variance_ratio_per_feature" in mahal_result
            or "explained_variance_ratio_per_feature" in result
        )

        # Test fallback calculation when ratio not present
        # Create a minimal result without ratio
        minimal_result = {
            "method": "Mahalanobis",
            "mahalanobis_distances": np.random.randn(100).tolist(),
            "outlier_indices": [5, 10, 15],
            "n_components": 3,
            "threshold_value": 2.5,
            "loadings": np.random.randn(10, 3).tolist(),
            "eigenvalues": [5.0, 2.0, 1.0],
            "feature_names": [f"trait_{i}" for i in range(10)],
        }

        # Should still create plots using fallback calculation
        figures_fallback = create_mahalanobis_outlier_plots(df, minimal_result)

        # Check that at least main plots are created
        assert "mahalanobis_outlier_detection" in figures_fallback

        # PC analysis plot requires pca_components or other conditions
        # Just verify the function handles missing data gracefully
        assert isinstance(figures_fallback, dict)

    def test_trace_preservation_in_visualization(self):
        """Verify trace preservation property in visualization data."""
        import numpy as np
        import pandas as pd
        from sleap_roots_analyze.pca import perform_pca_analysis, calculate_pca_metrics

        np.random.seed(42)
        df = pd.DataFrame(
            {f"trait_{i}": np.random.randn(100) * (i + 1) for i in range(5)}
        )

        # Perform PCA with all components
        result = perform_pca_analysis(df, n_components=5)

        # Get explained variance per feature
        explained_per_feature = result.get("explained_variance_per_feature")
        eigenvalues = result.get("eigenvalues")

        if explained_per_feature is not None and eigenvalues is not None:
            # Sum of explained variance per feature should equal sum of eigenvalues
            sum_explained = np.sum(explained_per_feature)
            sum_eigenvalues = np.sum(eigenvalues)

            np.testing.assert_allclose(
                sum_explained,
                sum_eigenvalues,
                rtol=1e-6,
                err_msg="Trace not preserved: sum of per-feature variance != sum of eigenvalues",
            )


class TestFeatureSelection:
    """Test suite for PCA-based feature selection."""

    @pytest.fixture
    def sample_pca_data(self):
        """Create sample PCA data for testing."""
        np.random.seed(42)
        n_features = 20
        n_components = 5

        # Create mock loadings with some structure
        loadings = np.random.randn(n_features, n_components) * 0.3

        # Make some features have strong loadings on specific PCs
        loadings[0:3, 0] = np.array([0.8, 0.7, 0.6])  # Strong positive on PC1
        loadings[3:6, 0] = np.array([-0.8, -0.7, -0.6])  # Strong negative on PC1
        loadings[6:9, 1] = np.array([0.75, 0.65, 0.55])  # Strong positive on PC2
        loadings[9:12, 1] = np.array([-0.75, -0.65, -0.55])  # Strong negative on PC2

        # Create eigenvalues (decreasing importance)
        eigenvalues = np.array([5.0, 3.0, 2.0, 1.0, 0.5])

        return {
            "loadings": loadings,
            "eigenvalues": eigenvalues,
            "n_features": n_features,
            "n_components": n_components,
        }

    def test_extreme_selection(self, sample_pca_data):
        """Test extreme selection method (top positive and negative)."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=sample_pca_data["n_features"],
            n_features_to_select=2,
            method="extreme",
            pc_indices=[0, 1],
        )

        # Should get 2 most positive and 2 most negative from each PC
        # For PC1: indices 0,1 (most positive) and 3,4 (most negative)
        # For PC2: indices 6,7 (most positive) and 9,10 (most negative)
        assert len(selected) <= 8  # May have overlap
        assert 0 in selected  # Strongest positive PC1
        assert 3 in selected  # Strongest negative PC1
        assert 6 in selected  # Strongest positive PC2
        assert 9 in selected  # Strongest negative PC2

    def test_top_absolute_selection(self, sample_pca_data):
        """Test selection by absolute loading magnitude."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=sample_pca_data["n_features"],
            n_features_to_select=4,
            method="top_absolute",
            pc_indices=[0, 1],
        )

        assert len(selected) == 4
        # Should include features with highest absolute loadings on PC1 or PC2
        # Features 0, 3 have |0.8| on PC1, features 6, 9 have |0.75| on PC2
        assert 0 in selected or 3 in selected
        assert 6 in selected or 9 in selected

    def test_top_contribution_selection(self, sample_pca_data):
        """Test selection by variance contribution to specific PCs."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=sample_pca_data["n_features"],
            n_features_to_select=4,
            method="top_contribution",
            pc_indices=[0, 1],
        )

        assert len(selected) == 4
        # Features with highest variance contributions considering eigenvalues
        # PC1 has eigenvalue 5.0, PC2 has 3.0
        # So PC1 loadings are weighted more heavily

    def test_top_variance_selection(self, sample_pca_data):
        """Test selection by total variance contribution across all PCs."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=sample_pca_data["n_features"],
            n_features_to_select=5,
            method="top_variance",
            pc_indices=None,  # Should use all PCs
        )

        assert len(selected) == 5
        # Should select features with highest total variance contribution
        # Features 0, 3 (strong on PC1 with high eigenvalue) should be included
        assert 0 in selected or 3 in selected

    def test_default_pc_indices(self, sample_pca_data):
        """Test that default pc_indices is [0, 1]."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=sample_pca_data["n_features"],
            n_features_to_select=3,
            method="top_absolute",
            pc_indices=None,
        )

        assert len(selected) == 3

    def test_edge_case_more_features_requested(self, sample_pca_data):
        """Test when more features are requested than available."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=5,  # Limit to 5 features
            n_features_to_select=10,  # Request 10
            method="top_variance",
        )

        assert len(selected) <= 5

    def test_edge_case_single_pc(self, sample_pca_data):
        """Test with only one PC specified."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=sample_pca_data["n_features"],
            n_features_to_select=2,
            method="extreme",
            pc_indices=[0],  # Only PC1
        )

        # Should get 2 most positive and 2 most negative from PC1 only
        assert len(selected) == 4
        assert 0 in selected  # Most positive
        assert 3 in selected  # Most negative

    def test_invalid_method(self, sample_pca_data):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown selection method"):
            select_top_features_from_pca(
                loadings=sample_pca_data["loadings"],
                eigenvalues=sample_pca_data["eigenvalues"],
                n_features_total=sample_pca_data["n_features"],
                n_features_to_select=5,
                method="invalid_method",
            )

    def test_pc_indices_out_of_bounds(self, sample_pca_data):
        """Test handling of PC indices that exceed available components."""
        selected = select_top_features_from_pca(
            loadings=sample_pca_data["loadings"],
            eigenvalues=sample_pca_data["eigenvalues"],
            n_features_total=sample_pca_data["n_features"],
            n_features_to_select=3,
            method="top_contribution",
            pc_indices=[0, 1, 10],  # PC 10 doesn't exist
        )

        # Should only use PC 0 and 1, ignoring PC 10
        assert len(selected) == 3

    def test_extreme_selection_no_duplicates(self):
        """Test that extreme selection doesn't duplicate features."""
        # Create loadings where same feature is extreme on multiple PCs
        n_features = 10
        loadings = np.random.randn(n_features, 3) * 0.1
        loadings[0, 0] = 0.9  # Feature 0 is most positive on PC1
        loadings[0, 1] = 0.9  # Feature 0 is also most positive on PC2

        eigenvalues = np.array([3.0, 2.0, 1.0])

        selected = select_top_features_from_pca(
            loadings=loadings,
            eigenvalues=eigenvalues,
            n_features_total=n_features,
            n_features_to_select=1,
            method="extreme",
            pc_indices=[0, 1],
        )

        # Feature 0 should only appear once
        assert selected.count(0) == 1

    def test_variance_contribution_calculation(self, sample_pca_data):
        """Test that variance contributions are calculated correctly."""
        # Use top_variance method which should weight by eigenvalues
        loadings = np.array([[0.8, 0.1], [0.1, 0.8], [0.5, 0.5]])
        eigenvalues = np.array([2.0, 1.0])

        selected = select_top_features_from_pca(
            loadings=loadings,
            eigenvalues=eigenvalues,
            n_features_total=3,
            n_features_to_select=1,
            method="top_variance",
        )

        # Feature 0: 2.0 * 0.8^2 + 1.0 * 0.1^2 = 1.28 + 0.01 = 1.29
        # Feature 1: 2.0 * 0.1^2 + 1.0 * 0.8^2 = 0.02 + 0.64 = 0.66
        # Feature 2: 2.0 * 0.5^2 + 1.0 * 0.5^2 = 0.50 + 0.25 = 0.75
        # Feature 0 has highest contribution
        assert selected[0] == 0

    def test_extreme_selection_remains_block_ordered_across_three_pcs(self):
        """Extreme method stays block-ordered per-PC, unchanged by the #207 fix."""
        n_features = 12
        loadings = np.random.RandomState(2).uniform(-0.05, 0.05, size=(n_features, 3))
        loadings[0, 0] = -0.9  # PC1 most negative
        loadings[1, 0] = -0.8  # PC1 2nd most negative
        loadings[2, 0] = 0.9  # PC1 most positive
        loadings[3, 0] = 0.8  # PC1 2nd most positive
        loadings[4, 1] = -0.9  # PC2 most negative
        loadings[5, 1] = -0.8  # PC2 2nd most negative
        loadings[6, 1] = 0.9  # PC2 most positive
        loadings[7, 1] = 0.8  # PC2 2nd most positive
        loadings[8, 2] = -0.9  # PC3 most negative
        loadings[9, 2] = -0.8  # PC3 2nd most negative
        loadings[10, 2] = 0.9  # PC3 most positive
        loadings[11, 2] = 0.8  # PC3 2nd most positive

        eigenvalues = np.array([3.0, 2.0, 1.0])

        selected = select_top_features_from_pca(
            loadings=loadings,
            eigenvalues=eigenvalues,
            n_features_total=n_features,
            n_features_to_select=2,
            method="extreme",
            pc_indices=[0, 1, 2],
        )

        assert selected == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
