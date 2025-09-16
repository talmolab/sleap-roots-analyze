"""Mathematical validation tests for PCA implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pca import (
    calculate_mahalanobis_distances,
    perform_pca_analysis,
)


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
            ), f"Error not decreasing: {errors[i]} > {errors[i-1]}"
