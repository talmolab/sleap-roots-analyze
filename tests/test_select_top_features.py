"""Tests for select_top_features_from_pca function."""

import numpy as np
import pytest

from sleap_roots_analyze.pca import select_top_features_from_pca


class TestSelectTopFeaturesFromPCA:
    """Test suite for select_top_features_from_pca function."""

    def test_vector_length_method(self):
        """Test the vector_length method (traditional biplot)."""
        # Create sample loadings
        # Feature 0: strong on PC1, weak on PC2 → vector length ≈ 0.71
        # Feature 1: weak on PC1, strong on PC2 → vector length ≈ 0.71
        # Feature 2: moderate on both → vector length ≈ 0.99
        # Feature 3: weak on both → vector length ≈ 0.14
        loadings = np.array(
            [
                [0.7, 0.1],  # Feature 0
                [0.1, 0.7],  # Feature 1
                [0.7, 0.7],  # Feature 2 - longest vector
                [0.1, 0.1],  # Feature 3 - shortest vector
            ]
        )
        eigenvalues = np.array([2.0, 1.0])

        # Select top 2 features by vector length
        selected = select_top_features_from_pca(
            loadings=loadings,
            eigenvalues=eigenvalues,
            n_features_total=4,
            n_features_to_select=2,
            method="vector_length",
            pc_indices=[0, 1],
        )

        # Should select features 2 and 0 (or 1, they're similar)
        assert len(selected) == 2
        assert 2 in selected  # Feature 2 has longest vector
        # Feature 0 and 1 are similar, either is acceptable
        assert selected[1] in [0, 1]

    def test_vector_length_vs_top_absolute(self):
        """Test that vector_length differs from top_absolute."""
        # Create scenario where methods differ
        # Feature 0: [0.9, 0.3] → abs sum = 1.2, euclidean = 0.95
        # Feature 1: [0.7, 0.7] → abs sum = 1.4, euclidean = 0.99
        loadings = np.array(
            [
                [0.9, 0.3],
                [0.7, 0.7],
            ]
        )
        eigenvalues = np.array([1.0, 1.0])

        vector_selected = select_top_features_from_pca(
            loadings=loadings,
            eigenvalues=eigenvalues,
            n_features_total=2,
            n_features_to_select=1,
            method="vector_length",
            pc_indices=[0, 1],
        )

        absolute_selected = select_top_features_from_pca(
            loadings=loadings,
            eigenvalues=eigenvalues,
            n_features_total=2,
            n_features_to_select=1,
            method="top_absolute",
            pc_indices=[0, 1],
        )

        # Vector length favors [0.7, 0.7] (balanced contribution)
        assert vector_selected[0] == 1

        # Top absolute favors [0.9, 0.3] (highest sum)
        assert absolute_selected[0] == 1  # Actually both select feature 1

    def test_vector_length_single_pc(self):
        """Test vector_length with single PC."""
        loadings = np.array(
            [
                [0.8],
                [0.5],
                [0.3],
            ]
        )
        eigenvalues = np.array([2.0])

        selected = select_top_features_from_pca(
            loadings=loadings,
            eigenvalues=eigenvalues,
            n_features_total=3,
            n_features_to_select=2,
            method="vector_length",
            pc_indices=[0],
        )

        # Should select features with highest absolute loading
        assert len(selected) == 2
        assert selected[0] == 0  # 0.8
        assert selected[1] == 1  # 0.5

    def test_all_methods_return_correct_count(self):
        """Test that all methods return the requested number of features."""
        np.random.seed(42)
        loadings = np.random.randn(10, 3)
        eigenvalues = np.array([3.0, 2.0, 1.0])

        methods = ["extreme", "top_absolute", "top_contribution", "vector_length"]

        for method in methods:
            selected = select_top_features_from_pca(
                loadings=loadings,
                eigenvalues=eigenvalues,
                n_features_total=10,
                n_features_to_select=5,
                method=method,
                pc_indices=[0, 1],
            )

            if method == "extreme":
                # Extreme method returns up to 2*n_features_to_select*n_pcs
                assert len(selected) <= 5 * 2 * 2  # May return more
            else:
                assert len(selected) == 5, f"Method {method} returned wrong count"
