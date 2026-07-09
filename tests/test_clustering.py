"""Tests for clustering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.clustering import (
    perform_kmeans_clustering,
    perform_gmm_clustering,
    calculate_cluster_quality_metrics,
    perform_hierarchical_clustering,
    calculate_optimal_clusters_hierarchical,
    hierarchical_cluster_labels,
)


def test_perform_kmeans_clustering_basic(simple_cluster_data):
    """Test basic K-Means clustering functionality."""
    result = perform_kmeans_clustering(simple_cluster_data, n_clusters=3)

    assert result["method"] == "KMeans"
    assert result["n_clusters"] == 3
    assert len(result["cluster_labels"]) == 90
    assert len(result["cluster_sizes"]) == 3
    assert result["cluster_centers"].shape == (3, 5)
    assert "silhouette_score" in result
    assert "davies_bouldin_score" in result
    assert "calinski_harabasz_score" in result


def test_perform_kmeans_clustering_with_array():
    """Test K-Means with numpy array input."""
    np.random.seed(42)
    data = np.random.randn(50, 4)

    result = perform_kmeans_clustering(data, n_clusters=2)

    assert result["method"] == "KMeans"
    assert result["n_clusters"] == 2
    assert len(result["feature_names"]) == 4
    assert all("Feature_" in name for name in result["feature_names"])


def test_perform_kmeans_clustering_no_standardization(simple_cluster_data):
    """Test K-Means without standardization."""
    result = perform_kmeans_clustering(
        simple_cluster_data, n_clusters=3, standardize=False
    )

    assert result["method"] == "KMeans"
    assert result["n_clusters"] == 3


def test_perform_kmeans_clustering_auto_adjust_clusters(simple_cluster_data):
    """Test that K-Means auto-adjusts n_clusters if too large."""
    # Request 20 clusters for 90 samples
    result = perform_kmeans_clustering(simple_cluster_data, n_clusters=20)

    # Should be adjusted to at most 9 (90 // 10)
    assert result["n_clusters"] <= 9


def test_perform_kmeans_clustering_empty_data():
    """Test K-Means with empty data."""
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Empty data"):
        perform_kmeans_clustering(df, n_clusters=3)


def test_perform_kmeans_clustering_insufficient_samples():
    """Test K-Means with insufficient samples."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(ValueError, match="Insufficient samples"):
        perform_kmeans_clustering(df, n_clusters=5)


def test_perform_kmeans_clustering_with_nan():
    """Test K-Means handles NaN by dropping rows."""
    df = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            "b": [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10],
        }
    )

    result = perform_kmeans_clustering(df, n_clusters=2)

    # Should have 8 samples (2 rows dropped due to NaN)
    assert len(result["cluster_labels"]) == 8
    assert len(result["data_indices"]) == 8


def test_perform_gmm_clustering_basic(multimodal_data):
    """Test basic GMM clustering functionality."""
    result = perform_gmm_clustering(multimodal_data, n_components=2)

    assert result["method"] == "GMM"
    assert result["n_components"] == 2
    assert len(result["cluster_labels"]) == 100
    assert result["probabilities"].shape == (100, 2)
    assert result["means"].shape == (2, 3)
    assert "bic" in result
    assert "aic" in result
    assert result["converged"] is True


def test_perform_gmm_clustering_auto_select(multimodal_data):
    """Test GMM with automatic component selection."""
    result = perform_gmm_clustering(
        multimodal_data, n_components=None, max_components=5
    )

    assert result["method"] == "GMM"
    assert 1 <= result["n_components"] <= 5
    assert len(result["bic_scores"]) > 0
    assert len(result["aic_scores"]) > 0


def test_perform_gmm_clustering_covariance_types(simple_cluster_data):
    """Test GMM with different covariance types."""
    covariance_types = ["full", "tied", "diag", "spherical"]

    for cov_type in covariance_types:
        result = perform_gmm_clustering(
            simple_cluster_data, n_components=3, covariance_type=cov_type
        )
        assert result["method"] == "GMM"
        assert result["covariance_type"] == cov_type


def test_perform_gmm_clustering_empty_data():
    """Test GMM with empty data."""
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Empty data"):
        perform_gmm_clustering(df, n_components=2)


def test_perform_gmm_clustering_insufficient_samples():
    """Test GMM with insufficient samples."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(ValueError, match="Insufficient samples"):
        perform_gmm_clustering(df, n_components=5, max_components=10)


def test_calculate_cluster_quality_metrics():
    """Test cluster quality metrics calculation."""
    np.random.seed(42)

    # Create simple clustered data
    data = np.vstack([np.random.randn(30, 3), np.random.randn(30, 3) + 5])
    labels = np.array([0] * 30 + [1] * 30)

    metrics = calculate_cluster_quality_metrics(data, labels)

    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert "calinski_harabasz_score" in metrics
    assert -1 <= metrics["silhouette_score"] <= 1
    assert metrics["davies_bouldin_score"] >= 0
    assert metrics["calinski_harabasz_score"] >= 0


def test_calculate_cluster_quality_metrics_invalid_input():
    """Test quality metrics with invalid inputs."""
    data = np.random.randn(10, 3)
    labels = np.array([0] * 10)  # Only one cluster

    with pytest.raises(ValueError, match="at least 2 clusters"):
        calculate_cluster_quality_metrics(data, labels)


def test_kmeans_results_consistency(simple_cluster_data):
    """Test that K-Means results are consistent across runs."""
    result1 = perform_kmeans_clustering(
        simple_cluster_data, n_clusters=3, random_state=42
    )
    result2 = perform_kmeans_clustering(
        simple_cluster_data, n_clusters=3, random_state=42
    )

    np.testing.assert_array_equal(result1["cluster_labels"], result2["cluster_labels"])
    np.testing.assert_array_almost_equal(
        result1["cluster_centers"], result2["cluster_centers"]
    )


def test_gmm_results_consistency(multimodal_data):
    """Test that GMM results are consistent across runs."""
    result1 = perform_gmm_clustering(multimodal_data, n_components=2, random_state=42)
    result2 = perform_gmm_clustering(multimodal_data, n_components=2, random_state=42)

    np.testing.assert_array_equal(result1["cluster_labels"], result2["cluster_labels"])
    np.testing.assert_array_almost_equal(result1["means"], result2["means"])


def test_kmeans_preserves_indices(simple_cluster_data):
    """Test that K-Means preserves original DataFrame indices."""
    # Create DataFrame with custom index
    df = simple_cluster_data.copy()
    df.index = [f"sample_{i}" for i in range(len(df))]

    result = perform_kmeans_clustering(df, n_clusters=3)

    assert len(result["data_indices"]) == len(df)
    assert result["data_indices"] == df.index.tolist()


def test_gmm_soft_assignments(multimodal_data):
    """Test that GMM probabilities sum to 1."""
    result = perform_gmm_clustering(multimodal_data, n_components=2)

    # Each sample's probabilities should sum to ~1.0
    prob_sums = result["probabilities"].sum(axis=1)
    np.testing.assert_array_almost_equal(prob_sums, np.ones(len(multimodal_data)))


def test_kmeans_cluster_sizes_match_labels(simple_cluster_data):
    """Test that cluster sizes match actual label counts."""
    result = perform_kmeans_clustering(simple_cluster_data, n_clusters=3)

    for i, size in enumerate(result["cluster_sizes"]):
        actual_size = np.sum(result["cluster_labels"] == i)
        assert size == actual_size


def test_calculate_optimal_k_kmeans_basic(simple_cluster_data):
    """Test K-Means auto-optimization with silhouette score."""
    from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

    result = calculate_optimal_k_kmeans(
        simple_cluster_data, max_clusters=5, method="silhouette"
    )

    assert "optimal_n_clusters" in result
    assert "scores" in result
    assert "k_values" in result
    assert "method" in result
    assert result["method"] == "silhouette"
    assert 2 <= result["optimal_n_clusters"] <= 5
    assert len(result["scores"]) == len(result["k_values"])
    assert result["k_values"] == list(range(2, 6))


def test_calculate_optimal_k_kmeans_calinski(simple_cluster_data):
    """Test K-Means optimization with Calinski-Harabasz score."""
    from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

    result = calculate_optimal_k_kmeans(
        simple_cluster_data, max_clusters=5, method="calinski"
    )

    assert result["method"] == "calinski"
    assert 2 <= result["optimal_n_clusters"] <= 5


def test_calculate_optimal_k_kmeans_davies_bouldin(simple_cluster_data):
    """Test K-Means optimization with Davies-Bouldin score."""
    from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

    result = calculate_optimal_k_kmeans(
        simple_cluster_data, max_clusters=5, method="davies_bouldin"
    )

    assert result["method"] == "davies_bouldin"
    assert 2 <= result["optimal_n_clusters"] <= 5


def test_calculate_optimal_k_kmeans_invalid_method(simple_cluster_data):
    """Test that invalid method raises RuntimeError."""
    from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

    with pytest.raises(RuntimeError, match="Unknown method"):
        calculate_optimal_k_kmeans(simple_cluster_data, method="invalid")


def test_calculate_optimal_k_kmeans_small_data():
    """Test optimization with small dataset."""
    from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

    # Create small dataset
    small_data = pd.DataFrame(np.random.randn(10, 3))

    result = calculate_optimal_k_kmeans(small_data, max_clusters=10)

    # Should limit max_clusters based on sample size
    assert result["optimal_n_clusters"] <= 5  # 10 samples / 2


def test_perform_kmeans_clustering_with_auto_k(simple_cluster_data):
    """Test K-Means clustering with automatic k selection."""
    result = perform_kmeans_clustering(
        simple_cluster_data, n_clusters=None, max_clusters=5
    )

    assert result["method"] == "KMeans"
    assert "n_clusters" in result
    assert 2 <= result["n_clusters"] <= 5
    assert len(result["cluster_labels"]) == 90
    assert "silhouette_score" in result


def test_perform_kmeans_clustering_auto_k_vs_manual():
    """Test that auto-k produces valid clustering compared to manual k."""
    np.random.seed(42)

    # Create data with 3 clear clusters
    cluster1 = np.random.randn(30, 4) + [0, 0, 0, 0]
    cluster2 = np.random.randn(30, 4) + [5, 5, 5, 5]
    cluster3 = np.random.randn(30, 4) + [-5, -5, -5, -5]
    data = pd.DataFrame(np.vstack([cluster1, cluster2, cluster3]))

    # Test auto-k
    result_auto = perform_kmeans_clustering(data, n_clusters=None, max_clusters=5)

    # Test manual k=3
    result_manual = perform_kmeans_clustering(data, n_clusters=3)

    # Both should produce valid results
    assert result_auto["n_clusters"] >= 2
    assert result_manual["n_clusters"] == 3
    assert result_auto["silhouette_score"] > 0  # Should have decent separation


# ============================================================================
# Edge Case and Error Handling Tests
# ============================================================================


class TestKMeansEdgeCases:
    """Test edge cases and error handling for K-Means clustering."""

    def test_kmeans_with_all_nan_data(self, edge_case_cluster_data):
        """Test K-Means with all NaN data after dropna."""
        with pytest.raises(ValueError, match="All rows contain NaN"):
            perform_kmeans_clustering(edge_case_cluster_data["all_nan"], n_clusters=2)

    def test_kmeans_error_handling(self):
        """Test K-Means error handling for clustering failure."""
        # Create data that might cause issues
        invalid_data = pd.DataFrame({"x": [np.inf, -np.inf, 1, 2, 3]})

        with pytest.raises(RuntimeError, match="K-Means clustering failed"):
            perform_kmeans_clustering(invalid_data, n_clusters=2)


class TestOptimalKEdgeCases:
    """Test edge cases for optimal k calculation."""

    def test_optimal_k_calinski_method(self, optimal_k_test_data):
        """Test optimal k selection using Calinski-Harabasz method."""
        from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

        result = calculate_optimal_k_kmeans(
            optimal_k_test_data["data"],
            max_clusters=optimal_k_test_data["max_k"],
            method="calinski",
        )

        assert result["method"] == "calinski"
        assert "optimal_n_clusters" in result
        assert result["optimal_n_clusters"] >= 2

    def test_optimal_k_davies_bouldin_method(self, optimal_k_test_data):
        """Test optimal k selection using Davies-Bouldin method."""
        from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

        result = calculate_optimal_k_kmeans(
            optimal_k_test_data["data"],
            max_clusters=optimal_k_test_data["max_k"],
            method="davies_bouldin",
        )

        assert result["method"] == "davies_bouldin"
        assert "optimal_n_clusters" in result
        assert result["optimal_n_clusters"] >= 2

    def test_optimal_k_invalid_method(self, simple_cluster_data):
        """Test error handling for invalid optimization method."""
        from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

        with pytest.raises(RuntimeError, match="Unknown method"):
            calculate_optimal_k_kmeans(
                simple_cluster_data, max_clusters=5, method="invalid_method"
            )

    def test_optimal_k_error_handling(self):
        """Test error handling in optimal k calculation."""
        from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

        # Create problematic data with insufficient samples
        bad_data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        with pytest.raises(ValueError, match="Insufficient samples"):
            calculate_optimal_k_kmeans(bad_data, max_clusters=10, method="silhouette")

    def test_optimal_k_max_clusters_adjustment(self):
        """Test that max_clusters is adjusted based on sample size."""
        from sleap_roots_analyze.clustering import calculate_optimal_k_kmeans

        # Small dataset
        small_data = pd.DataFrame(np.random.randn(15, 3))

        result = calculate_optimal_k_kmeans(small_data, max_clusters=20)

        # max_clusters should be limited
        assert len(result["k_values"]) < 20


class TestGMMEdgeCases:
    """Test edge cases and error handling for GMM clustering."""

    def test_gmm_auto_n_components(self, multimodal_data):
        """Test GMM with automatic component selection."""
        result = perform_gmm_clustering(
            multimodal_data, n_components=None, max_components=5
        )

        assert result["method"] == "GMM"
        assert result["n_components"] >= 2

    def test_gmm_insufficient_samples(self, edge_case_cluster_data):
        """Test GMM with insufficient samples."""
        with pytest.raises(ValueError, match="Insufficient samples"):
            perform_gmm_clustering(
                edge_case_cluster_data["insufficient_samples"], n_components=2
            )

    def test_gmm_auto_selection_with_limit(self):
        """Test that auto-selection respects sample size limits."""
        # Create small dataset: 50 samples -> max = 50 // 10 = 5 components
        np.random.seed(42)
        small_data = pd.DataFrame(np.random.randn(50, 3))

        # Auto-select with large max_components - should be limited by sample size
        result = perform_gmm_clustering(
            small_data, n_components=None, max_components=20
        )

        # Should be limited to 5 or less (50 samples / 10)
        assert result["n_components"] <= 5
        assert result["n_components"] >= 1
        assert "bic_scores" in result  # Should have auto-selected

    def test_gmm_single_component_handles_gracefully(self, simple_cluster_data):
        """Test GMM with n_components=1 works (returns silhouette=0)."""
        # Single component should work now (silhouette metrics set to 0)
        result = perform_gmm_clustering(simple_cluster_data, n_components=1)

        assert result["method"] == "GMM"
        assert result["n_components"] == 1
        # Quality metrics should be 0 for single cluster
        assert result["silhouette_score"] == 0.0
        assert result["davies_bouldin_score"] == 0.0
        assert result["calinski_harabasz_score"] == 0.0

    def test_gmm_different_covariance_types(self, multimodal_data):
        """Test GMM with different covariance types."""
        for cov_type in ["full", "tied", "diag", "spherical"]:
            result = perform_gmm_clustering(
                multimodal_data, n_components=2, covariance_type=cov_type
            )

            assert result["method"] == "GMM"
            assert "cluster_labels" in result

    def test_gmm_error_handling(self):
        """Test GMM error handling for clustering failure."""
        # Create problematic data
        bad_data = pd.DataFrame({"x": [np.inf] * 10, "y": [-np.inf] * 10})

        with pytest.raises(RuntimeError, match="GMM clustering failed"):
            perform_gmm_clustering(bad_data, n_components=2)


class TestHierarchicalEdgeCases:
    """Test edge cases and error handling for hierarchical clustering."""

    def test_hierarchical_ward_non_euclidean_error(self, hierarchical_edge_cases):
        """Test that ward linkage with non-euclidean metric raises error."""
        from sleap_roots_analyze.clustering import perform_hierarchical_clustering

        with pytest.raises(ValueError, match="Ward linkage requires euclidean metric"):
            perform_hierarchical_clustering(
                hierarchical_edge_cases["ward_manhattan"],
                method="ward",
                metric="manhattan",
            )

    def test_hierarchical_different_methods(self, hierarchical_edge_cases):
        """Test hierarchical clustering with different linkage methods."""
        from sleap_roots_analyze.clustering import perform_hierarchical_clustering

        methods = ["complete", "average", "single"]

        for method in methods:
            result = perform_hierarchical_clustering(
                hierarchical_edge_cases["complete_manhattan"],
                method=method,
                metric="euclidean",
            )

            assert result["method"] == "Hierarchical"
            assert result["linkage_method"] == method

    def test_hierarchical_different_metrics(self, hierarchical_edge_cases):
        """Test hierarchical clustering with different distance metrics."""
        from sleap_roots_analyze.clustering import perform_hierarchical_clustering

        # Use scipy-compatible metric names
        metrics = [
            "euclidean",
            "cityblock",
            "cosine",
        ]  # scipy uses 'cityblock' not 'manhattan'

        for metric in metrics:
            method = "complete" if metric != "euclidean" else "ward"
            result = perform_hierarchical_clustering(
                hierarchical_edge_cases["complete_manhattan"],
                method=method,
                metric=metric,
            )

            assert result["method"] == "Hierarchical"
            assert result["distance_metric"] == metric

    def test_hierarchical_error_handling(self):
        """Test hierarchical clustering error handling."""
        from sleap_roots_analyze.clustering import perform_hierarchical_clustering

        # Create problematic data
        bad_data = pd.DataFrame({"x": [1, np.inf], "y": [2, -np.inf]})

        with pytest.raises(RuntimeError, match="Hierarchical clustering failed"):
            perform_hierarchical_clustering(bad_data, method="ward", metric="euclidean")


class TestCutDendrogramEdgeCases:
    """Test edge cases for dendrogram cutting."""

    def test_cut_dendrogram_by_height(self, hierarchical_cluster_result):
        """Test cutting dendrogram by height threshold."""
        from sleap_roots_analyze.clustering import cut_dendrogram

        result = cut_dendrogram(hierarchical_cluster_result, height_threshold=5.0)

        assert "cluster_labels" in result
        assert "n_clusters" in result
        assert "cut_height" in result
        assert result["cut_height"] == 5.0

    def test_cut_dendrogram_single_cluster(self, hierarchical_cluster_result):
        """Test cutting dendrogram to create single cluster."""
        from sleap_roots_analyze.clustering import cut_dendrogram

        result = cut_dendrogram(hierarchical_cluster_result, n_clusters=1)

        assert result["n_clusters"] == 1
        assert len(np.unique(result["cluster_labels"])) == 1
        # Quality metrics should be 0 for single cluster
        assert result["silhouette_score"] == 0.0
        assert result["davies_bouldin_score"] == 0.0
        assert result["calinski_harabasz_score"] == 0.0

    def test_cut_dendrogram_neither_parameter_error(self, hierarchical_cluster_result):
        """Test error when neither n_clusters nor height_threshold is provided."""
        from sleap_roots_analyze.clustering import cut_dendrogram

        with pytest.raises(ValueError, match="Must provide either"):
            cut_dendrogram(hierarchical_cluster_result)

    def test_cut_dendrogram_error_handling(self, hierarchical_cluster_result):
        """Test error handling in dendrogram cutting."""
        from sleap_roots_analyze.clustering import cut_dendrogram

        # Try to create more clusters than samples
        with pytest.raises(RuntimeError, match="Failed to cut dendrogram"):
            cut_dendrogram(hierarchical_cluster_result, n_clusters=1000)


class TestOptimalClustersHierarchical:
    """Test optimal cluster calculation for hierarchical clustering."""

    def test_optimal_hierarchical_davies_bouldin(self, hierarchical_cluster_result):
        """Test optimal k with Davies-Bouldin method."""
        from sleap_roots_analyze.clustering import (
            calculate_optimal_clusters_hierarchical,
        )

        result = calculate_optimal_clusters_hierarchical(
            hierarchical_cluster_result, max_clusters=8, method="davies_bouldin"
        )

        assert result["method"] == "davies_bouldin"
        assert result["optimal_n_clusters"] >= 2

    def test_optimal_hierarchical_calinski(self, hierarchical_cluster_result):
        """Test optimal k with Calinski-Harabasz method."""
        from sleap_roots_analyze.clustering import (
            calculate_optimal_clusters_hierarchical,
        )

        result = calculate_optimal_clusters_hierarchical(
            hierarchical_cluster_result, max_clusters=8, method="calinski"
        )

        assert result["method"] == "calinski"
        assert result["optimal_n_clusters"] >= 2

    def test_optimal_hierarchical_invalid_method(self, hierarchical_cluster_result):
        """Test error for invalid optimization method."""
        from sleap_roots_analyze.clustering import (
            calculate_optimal_clusters_hierarchical,
        )

        with pytest.raises(RuntimeError, match="Unknown method"):
            calculate_optimal_clusters_hierarchical(
                hierarchical_cluster_result, max_clusters=5, method="invalid"
            )

    def test_optimal_hierarchical_insufficient_clusters(self):
        """Test error when max_clusters < 2."""
        from sleap_roots_analyze.clustering import (
            perform_hierarchical_clustering,
            calculate_optimal_clusters_hierarchical,
        )

        # Create minimal hierarchical result
        small_data = pd.DataFrame(np.random.randn(3, 2))
        hier_result = perform_hierarchical_clustering(small_data)

        with pytest.raises(ValueError, match="Need at least 2 clusters"):
            calculate_optimal_clusters_hierarchical(hier_result, max_clusters=1)

    def test_optimal_hierarchical_error_handling(self):
        """Test error handling in optimal cluster calculation."""
        from sleap_roots_analyze.clustering import (
            calculate_optimal_clusters_hierarchical,
        )

        # Create invalid hierarchical result with only 2 samples
        # This will cause max_clusters to be limited to 1, raising ValueError
        invalid_result = {
            "linkage_matrix": np.array([[0, 1, 0.5, 2]]),  # Minimal linkage
            "data_processed": np.random.randn(2, 2),
        }

        with pytest.raises(ValueError, match="Need at least 2 clusters"):
            calculate_optimal_clusters_hierarchical(invalid_result, max_clusters=10)


class TestClusteringWithNaN:
    """Test clustering with NaN handling."""

    def test_kmeans_drops_nan_rows(self, cluster_result_with_nan):
        """Test that K-Means correctly drops NaN rows."""
        result = perform_kmeans_clustering(cluster_result_with_nan, n_clusters=2)

        # Should have fewer samples than input (NaN rows dropped)
        assert len(result["cluster_labels"]) < len(cluster_result_with_nan)
        assert len(result["cluster_labels"]) == len(cluster_result_with_nan.dropna())

    def test_gmm_drops_nan_rows(self, cluster_result_with_nan):
        """Test that GMM correctly drops NaN rows."""
        result = perform_gmm_clustering(cluster_result_with_nan, n_components=2)

        # Should have fewer samples than input (NaN rows dropped)
        assert len(result["cluster_labels"]) < len(cluster_result_with_nan)
        assert len(result["cluster_labels"]) == len(cluster_result_with_nan.dropna())

    def test_hierarchical_drops_nan_rows(self, cluster_result_with_nan):
        """Test that hierarchical clustering correctly drops NaN rows."""
        from sleap_roots_analyze.clustering import perform_hierarchical_clustering

        result = perform_hierarchical_clustering(cluster_result_with_nan)

        # Should have fewer samples than input
        assert len(result["data_indices"]) < len(cluster_result_with_nan)


_LABELED_KEYS = {
    "cluster_labels",
    "n_clusters",
    "cluster_sizes",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "feature_names",
    "linkage_method",
    "distance_metric",
    "cophenetic_correlation",
    "cut_height",
}


class TestHierarchicalClusterLabels:
    """Public labeled hierarchical entry point hierarchical_cluster_labels (#179)."""

    def test_perform_hierarchical_shape_unchanged(self, simple_cluster_data):
        """Regression pin: perform_hierarchical_clustering keeps its dendrogram keys."""
        result = perform_hierarchical_clustering(simple_cluster_data)
        assert set(result.keys()) == {
            "method",
            "linkage_matrix",
            "linkage_method",
            "distance_metric",
            "cophenetic_correlation",
            "data_indices",
            "feature_names",
            "data_processed",
        }
        assert "cluster_labels" not in result

    def test_returns_labeled_dict_auto_k(self, simple_cluster_data):
        """Auto-k call returns the labeled key set with structural invariants."""
        result = hierarchical_cluster_labels(simple_cluster_data)
        assert set(result.keys()) == _LABELED_KEYS
        n_samples = len(simple_cluster_data)
        assert len(result["cluster_labels"]) == n_samples
        assert result["n_clusters"] == len(set(result["cluster_labels"].tolist()))
        assert 2 <= result["n_clusters"] <= 10

    def test_auto_k_delegates_to_optimizer(self, simple_cluster_data):
        """Auto-k equals the optimizer's optimal_n_clusters for the same input."""
        labeled = hierarchical_cluster_labels(simple_cluster_data)
        dendrogram = perform_hierarchical_clustering(simple_cluster_data)
        optimal = calculate_optimal_clusters_hierarchical(
            dendrogram, method="silhouette"
        )
        assert labeled["n_clusters"] == optimal["optimal_n_clusters"]

    def test_honors_explicit_cluster_count(self, simple_cluster_data):
        """An explicit n_clusters is honored, sizes sum to the sample count."""
        result = hierarchical_cluster_labels(simple_cluster_data, n_clusters=3)
        assert result["n_clusters"] == 3
        assert len(result["cluster_sizes"]) == 3
        assert sum(result["cluster_sizes"]) == len(result["cluster_labels"])

    def test_deterministic_labels(self, simple_cluster_data):
        """Identical input yields identical labels (same-process, no golden)."""
        r1 = hierarchical_cluster_labels(simple_cluster_data, n_clusters=3)
        r2 = hierarchical_cluster_labels(simple_cluster_data, n_clusters=3)
        assert np.array_equal(r1["cluster_labels"], r2["cluster_labels"])

    def test_ward_non_euclidean_raises_value_error(self, simple_cluster_data):
        """Ward linkage + non-euclidean metric propagates ValueError."""
        with pytest.raises(ValueError, match="Ward linkage requires euclidean metric"):
            hierarchical_cluster_labels(
                simple_cluster_data, method="ward", metric="manhattan"
            )

    def test_too_few_rows_raises_value_error(self):
        """Fewer than 2 valid rows propagates ValueError."""
        one_row = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="at least 2 samples"):
            hierarchical_cluster_labels(one_row, n_clusters=1)

    def test_all_nan_raises_value_error(self):
        """All-NaN input propagates ValueError."""
        all_nan = pd.DataFrame(
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], columns=["a", "b"]
        )
        with pytest.raises(ValueError, match="All rows contain NaN"):
            hierarchical_cluster_labels(all_nan, n_clusters=2)

    @pytest.mark.parametrize("opt_method", ["silhouette", "calinski", "davies_bouldin"])
    def test_optimization_method_accepted_values(self, simple_cluster_data, opt_method):
        """Each accepted optimization_method value returns a valid labeled dict."""
        result = hierarchical_cluster_labels(
            simple_cluster_data, optimization_method=opt_method
        )
        assert 2 <= result["n_clusters"] <= 10

    def test_optimization_method_footgun_raises_runtime_error(
        self, simple_cluster_data
    ):
        """The metric-key-name footgun 'silhouette_score' raises RuntimeError (wrapped)."""
        with pytest.raises(RuntimeError, match="Failed to calculate optimal clusters"):
            hierarchical_cluster_labels(
                simple_cluster_data, optimization_method="silhouette_score"
            )

    def test_single_cluster_finite_zero_metrics(self, simple_cluster_data):
        """n_clusters=1 yields finite 0.0 metrics; the typed view serializes."""
        from sleap_roots_analyze.result_types import ClusterResult

        result = hierarchical_cluster_labels(simple_cluster_data, n_clusters=1)
        assert result["n_clusters"] == 1
        assert len(result["cluster_sizes"]) == 1
        assert result["silhouette_score"] == 0.0
        assert result["davies_bouldin_score"] == 0.0
        assert result["calinski_harabasz_score"] == 0.0
        # finite metrics => strict-JSON serialization succeeds
        ClusterResult.from_hierarchical_dict(result).to_json()

    def test_cluster_per_sample_raises_runtime_error(self, simple_cluster_data):
        """n_clusters == n_samples propagates RuntimeError (silhouette undefined)."""
        n_samples = len(simple_cluster_data)
        with pytest.raises(RuntimeError, match="Failed to cut dendrogram"):
            hierarchical_cluster_labels(simple_cluster_data, n_clusters=n_samples)

    def test_two_row_auto_k_raises_value_error(self):
        """A 2-row input with auto-k propagates ValueError (max_clusters < 2)."""
        two_rows = pd.DataFrame(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], columns=["a", "b", "c"]
        )
        with pytest.raises(ValueError, match="at least 2 clusters"):
            hierarchical_cluster_labels(two_rows)

    def test_end_to_end_typed_view(self, simple_cluster_data):
        """from_hierarchical_dict(producer(df)) yields a populated HierarchicalResult."""
        from sleap_roots_analyze.result_types import (
            ALGORITHM_HIERARCHICAL,
            ClusterResult,
            HierarchicalResult,
        )

        view = ClusterResult.from_hierarchical_dict(
            hierarchical_cluster_labels(simple_cluster_data, n_clusters=3)
        )
        assert isinstance(view, HierarchicalResult)
        assert view.algorithm == ALGORITHM_HIERARCHICAL
        assert view.n_clusters == 3
        assert len(view.cluster_labels) == len(simple_cluster_data)
        assert len(view.cluster_sizes) == 3
        assert all(
            isinstance(s, float)
            for s in (
                view.silhouette_score,
                view.davies_bouldin_score,
                view.calinski_harabasz_score,
            )
        )
        assert not hasattr(view, "cluster_centers")
        view.to_json()
