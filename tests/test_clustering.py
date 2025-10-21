"""Tests for clustering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.clustering import (
    perform_kmeans_clustering,
    perform_gmm_clustering,
    calculate_cluster_quality_metrics,
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
