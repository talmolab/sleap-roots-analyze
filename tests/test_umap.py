"""Tests for UMAP analysis functions."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import fixtures
from tests.fixtures import traits_summary_sample


class TestPerformUMAPAnalysis:
    """Tests for perform_umap_analysis function."""

    def test_basic_umap_analysis(self, traits_summary_sample):
        """Test basic UMAP analysis functionality."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        # Select numeric columns for analysis
        feature_cols = [
            "network_length_mean",
            "network_length_max",
            "chull_max_width_mean",
            "crown_lengths_mean_mean",
        ]

        # Perform UMAP
        results = perform_umap_analysis(
            traits_summary_sample,
            feature_cols,
            n_neighbors=5,  # Small value for small dataset
            min_dist=0.1,
            n_components=2,
            random_state=42,
        )

        # Check structure
        assert isinstance(results, dict)
        assert "embedding" in results
        assert "reducer" in results
        assert "scaler" in results
        assert "n_neighbors" in results
        assert "min_dist" in results

        # Check embedding shape
        n_samples = len(traits_summary_sample)
        assert results["embedding"].shape == (n_samples, 2)

        # Check parameters are stored
        assert results["n_neighbors"] == 5
        assert results["min_dist"] == 0.1

    def test_umap_with_3d_embedding(self, traits_summary_sample):
        """Test UMAP with 3D embedding."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = [
            "network_length_mean",
            "network_length_max",
            "chull_max_width_mean",
            "crown_lengths_mean_mean",
        ]

        results = perform_umap_analysis(
            traits_summary_sample, feature_cols, n_components=3, random_state=42
        )

        n_samples = len(traits_summary_sample)
        assert results["embedding"].shape == (n_samples, 3)

    def test_umap_with_single_component(self, traits_summary_sample):
        """Test UMAP with single component."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = [
            "network_length_mean",
            "network_length_max",
            "chull_max_width_mean",
            "crown_lengths_mean_mean",
        ]

        results = perform_umap_analysis(
            traits_summary_sample, feature_cols, n_components=1, random_state=42
        )

        n_samples = len(traits_summary_sample)
        assert results["embedding"].shape == (n_samples, 1)

    def test_umap_different_parameters(self, traits_summary_sample):
        """Test UMAP with different parameter settings."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = [
            "network_length_mean",
            "network_length_max",
            "chull_max_width_mean",
            "crown_lengths_mean_mean",
        ]

        # Test with different n_neighbors
        results1 = perform_umap_analysis(
            traits_summary_sample, feature_cols, n_neighbors=3, random_state=42
        )

        results2 = perform_umap_analysis(
            traits_summary_sample, feature_cols, n_neighbors=10, random_state=42
        )

        # Different parameters should give different embeddings
        assert not np.allclose(results1["embedding"], results2["embedding"])

        # Test with different min_dist
        results3 = perform_umap_analysis(
            traits_summary_sample, feature_cols, min_dist=0.01, random_state=42
        )

        results4 = perform_umap_analysis(
            traits_summary_sample, feature_cols, min_dist=0.5, random_state=42
        )

        assert not np.allclose(results3["embedding"], results4["embedding"])

    def test_umap_reproducibility(self, traits_summary_sample):
        """Test UMAP reproducibility with same random state."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = [
            "network_length_mean",
            "network_length_max",
            "chull_max_width_mean",
            "crown_lengths_mean_mean",
        ]

        # Run twice with same random state
        results1 = perform_umap_analysis(
            traits_summary_sample, feature_cols, random_state=42
        )

        results2 = perform_umap_analysis(
            traits_summary_sample, feature_cols, random_state=42
        )

        # Should get same results
        assert np.allclose(results1["embedding"], results2["embedding"])

    def test_umap_with_nan_values(self, traits_summary_sample):
        """Test UMAP handling of NaN values."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        # Create data with NaN
        df_with_nan = traits_summary_sample.copy()
        df_with_nan.loc[0, "network_length_mean"] = np.nan

        feature_cols = [
            "network_length_mean",
            "network_length_max",
            "chull_max_width_mean",
            "crown_lengths_mean_mean",
        ]

        with pytest.raises(ValueError, match="contains NaN"):
            perform_umap_analysis(df_with_nan, feature_cols)

    def test_umap_with_missing_columns(self, traits_summary_sample):
        """Test UMAP with missing feature columns."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = ["nonexistent_column", "another_missing"]

        with pytest.raises(KeyError):
            perform_umap_analysis(traits_summary_sample, feature_cols)

    def test_umap_import_error(self, traits_summary_sample):
        """Test UMAP when umap-learn is not installed."""
        from sleap_roots_analyze import umap as umap_module

        # Mock UMAP not being available
        with patch.object(umap_module, "UMAP_AVAILABLE", False):
            from sleap_roots_analyze.umap import perform_umap_analysis

            feature_cols = ["network_length_mean", "network_length_max"]

            with pytest.raises(ImportError, match="UMAP is not installed"):
                perform_umap_analysis(traits_summary_sample, feature_cols)

    def test_umap_with_single_feature(self, traits_summary_sample):
        """Test UMAP with single feature."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = ["network_length_mean"]

        # UMAP should still work with single feature
        results = perform_umap_analysis(
            traits_summary_sample,
            feature_cols,
            n_neighbors=3,  # Smaller for single feature
            n_components=1,  # Can't have more components than features
            random_state=42,
        )

        assert results["embedding"].shape == (len(traits_summary_sample), 1)

    def test_umap_parameter_validation(self, traits_summary_sample):
        """Test UMAP parameter validation."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = ["primary_length_mm", "lateral_length_mm"]

        # Test invalid n_neighbors
        with pytest.raises(ValueError, match="n_neighbors must be"):
            perform_umap_analysis(traits_summary_sample, feature_cols, n_neighbors=0)

        # Test invalid min_dist
        with pytest.raises(ValueError, match="min_dist must be"):
            perform_umap_analysis(traits_summary_sample, feature_cols, min_dist=-0.1)

        # Test invalid n_components
        with pytest.raises(ValueError, match="n_components must be"):
            perform_umap_analysis(traits_summary_sample, feature_cols, n_components=0)

    def test_umap_scaler_properties(self, traits_summary_sample):
        """Test that scaler is properly fitted."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        feature_cols = [
            "network_length_mean",
            "network_length_max",
            "chull_max_width_mean",
            "crown_lengths_mean_mean",
        ]

        results = perform_umap_analysis(traits_summary_sample, feature_cols)

        scaler = results["scaler"]

        # Check scaler has correct number of features
        assert len(scaler.mean_) == len(feature_cols)
        assert len(scaler.scale_) == len(feature_cols)

        # Verify standardization worked (mean ~0, std ~1)
        X = traits_summary_sample[feature_cols].values
        X_scaled = scaler.transform(X)

        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)

    def test_umap_empty_dataframe(self):
        """Test UMAP with empty DataFrame."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        empty_df = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            perform_umap_analysis(empty_df, ["col1"])

    def test_umap_single_sample(self):
        """Test UMAP with single sample."""
        from sleap_roots_analyze.umap import perform_umap_analysis

        single_df = pd.DataFrame({"feat1": [1.0], "feat2": [2.0], "feat3": [3.0]})

        # UMAP needs multiple samples
        with pytest.raises(ValueError, match="at least"):
            perform_umap_analysis(single_df, ["feat1", "feat2", "feat3"])
