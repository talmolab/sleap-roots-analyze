"""Tests for the serializable clustering result dataclasses and adapters (#129).

Covers the serializable-result-types contract for clustering: native-type JSON
round-trips for both KMeans and GMM (the reproducibility CI gate), adapter field
mapping (incl. GMM means -> cluster_centers and n_components -> n_clusters),
random_state stamping, determinism (same seed -> identical labels, #118), public
export, and the non-breaking / non-mutating guarantee.
"""

from __future__ import annotations

import dataclasses
import json

from sleap_roots_analyze.clustering import perform_kmeans_clustering
from sleap_roots_analyze.result_types import ClusterResult, GMMResult, KMeansResult


class TestKMeansResultJSON:
    """KMeans clean view serializes to native Python types."""

    def test_json_roundtrip_native_types(self, kmeans_cluster_result):
        """KMeans view round-trips to native types with the right discriminator."""
        result = ClusterResult.from_kmeans_dict(kmeans_cluster_result, random_state=42)
        assert isinstance(result, KMeansResult)
        parsed = json.loads(json.dumps(dataclasses.asdict(result)))

        assert parsed["algorithm"] == "kmeans"
        assert all(type(v) is int for v in parsed["cluster_labels"])
        assert all(type(v) is float for row in parsed["cluster_centers"] for v in row)
        assert type(parsed["inertia"]) is float
        assert type(parsed["silhouette_score"]) is float
        assert type(parsed["random_state"]) is int


class TestGMMResultJSON:
    """GMM clean view serializes to native Python types."""

    def test_json_roundtrip_native_types(self, gmm_cluster_result):
        """GMM view round-trips to native types with the right discriminator."""
        result = ClusterResult.from_gmm_dict(gmm_cluster_result, random_state=42)
        assert isinstance(result, GMMResult)
        parsed = json.loads(json.dumps(dataclasses.asdict(result)))

        assert parsed["algorithm"] == "gmm"
        assert all(type(v) is float for row in parsed["cluster_centers"] for v in row)
        assert all(type(v) is float for v in parsed["weights"])
        assert type(parsed["bic"]) is float
        assert type(parsed["aic"]) is float
        assert type(parsed["converged"]) is bool
        assert type(parsed["n_iter"]) is int


class TestClusterAdapters:
    """Adapters map the legacy dicts faithfully."""

    def test_kmeans_adapter_maps_fields(self, kmeans_cluster_result):
        """KMeans adapter maps counts, centers shape, and random_state."""
        d = kmeans_cluster_result
        result = ClusterResult.from_kmeans_dict(d, random_state=42)

        n_features = len(d["feature_names"])
        assert result.n_clusters == int(d["n_clusters"])
        assert len(result.cluster_centers) == result.n_clusters
        assert all(len(row) == n_features for row in result.cluster_centers)
        assert sum(result.cluster_sizes) == len(result.cluster_labels)
        assert result.random_state == 42

    def test_gmm_adapter_maps_n_components_and_means(self, gmm_cluster_result):
        """GMM adapter maps n_components->n_clusters and means->cluster_centers."""
        d = gmm_cluster_result
        result = ClusterResult.from_gmm_dict(d, random_state=42)

        assert result.n_clusters == int(d["n_components"])
        assert result.cluster_centers == [list(row) for row in d["means"]]
        assert result.covariance_type == d["covariance_type"]
        assert len(result.weights) == result.n_clusters


class TestClusterDeterminism:
    """Same seed -> identical cluster labels via the typed view (#118)."""

    def test_same_seed_identical_labels(self, simple_cluster_data):
        """Re-running KMeans with the same seed yields identical labels."""
        r1 = ClusterResult.from_kmeans_dict(
            perform_kmeans_clustering(
                simple_cluster_data, n_clusters=3, random_state=42
            ),
            random_state=42,
        )
        r2 = ClusterResult.from_kmeans_dict(
            perform_kmeans_clustering(
                simple_cluster_data, n_clusters=3, random_state=42
            ),
            random_state=42,
        )
        assert r1.cluster_labels == r2.cluster_labels


class TestClusterResultExport:
    """Public API surface."""

    def test_importable_from_package_root(self):
        """All three clustering result types are importable and in __all__."""
        import sleap_roots_analyze as sra

        assert sra.ClusterResult is ClusterResult
        assert sra.KMeansResult is KMeansResult
        assert sra.GMMResult is GMMResult
        for name in ("ClusterResult", "KMeansResult", "GMMResult"):
            assert name in sra.__all__
        assert len(sra.__all__) == len(set(sra.__all__))


class TestClusterResultNonBreaking:
    """Legacy clustering dicts are preserved and not mutated."""

    def test_kmeans_dict_not_mutated(self, kmeans_cluster_result):
        """KMeans adapter does not mutate its input dict."""
        d = kmeans_cluster_result
        before = set(d.keys())
        ClusterResult.from_kmeans_dict(d, random_state=42)
        assert set(d.keys()) == before

    def test_gmm_dict_not_mutated(self, gmm_cluster_result):
        """GMM adapter does not mutate its input dict."""
        d = gmm_cluster_result
        before = set(d.keys())
        ClusterResult.from_gmm_dict(d, random_state=42)
        assert set(d.keys()) == before
