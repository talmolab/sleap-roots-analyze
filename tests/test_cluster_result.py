"""Tests for the serializable clustering result dataclasses and adapters (#129).

Covers the serializable-result-types contract for clustering: native-type JSON
round-trips for both KMeans and GMM (the reproducibility CI gate), adapter field
mapping (incl. GMM means -> cluster_centers and n_components -> n_clusters),
random_state stamping, determinism (same seed -> identical labels, #118), public
export, and the non-breaking / non-mutating guarantee.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

from sleap_roots_analyze.clustering import (
    perform_gmm_clustering,
    perform_kmeans_clustering,
)
from sleap_roots_analyze.result_types import (
    ALGORITHM_GMM,
    ALGORITHM_KMEANS,
    ClusterResult,
    GMMResult,
    KMeansResult,
)


def _assert_dict_unchanged(d, before):
    """Assert every value in ``d`` deep-equals ``before`` (numpy-aware)."""
    assert set(d.keys()) == set(before.keys())
    for k, prev in before.items():
        cur = d[k]
        if isinstance(prev, np.ndarray):
            np.testing.assert_array_equal(np.asarray(cur), prev)
        else:
            assert cur == prev, k


class TestKMeansResultJSON:
    """KMeans clean view serializes to native Python types."""

    def test_fields_are_native_types_pre_serialization(self, kmeans_cluster_result):
        """Float fields are native ``float`` on the dataclass, not laundered by JSON.

        ``np.float64`` is a subclass of ``float``, so a JSON round-trip silently
        casts a leak to native float before any assertion — assert on the fields.
        """
        result = ClusterResult.from_kmeans_dict(kmeans_cluster_result, random_state=42)

        assert result.algorithm == ALGORITHM_KMEANS
        assert type(result.inertia) is float
        assert type(result.silhouette_score) is float
        assert type(result.random_state) is int
        assert all(type(v) is int for v in result.cluster_labels)
        assert all(type(v) is float for row in result.cluster_centers for v in row)

    def test_json_roundtrip_native_types(self, kmeans_cluster_result):
        """KMeans view round-trips to native types with the right discriminator."""
        result = ClusterResult.from_kmeans_dict(kmeans_cluster_result, random_state=42)
        assert isinstance(result, KMeansResult)
        parsed = json.loads(json.dumps(dataclasses.asdict(result)))

        assert parsed["algorithm"] == ALGORITHM_KMEANS
        assert all(type(v) is int for v in parsed["cluster_labels"])
        assert all(type(v) is float for row in parsed["cluster_centers"] for v in row)
        assert type(parsed["inertia"]) is float
        assert type(parsed["silhouette_score"]) is float
        assert type(parsed["random_state"]) is int

    def test_to_json_roundtrips_finite_data(self, kmeans_cluster_result):
        """On finite data, to_json emits strict JSON that parses back to to_dict."""
        result = ClusterResult.from_kmeans_dict(kmeans_cluster_result, random_state=42)
        assert json.loads(result.to_json()) == json.loads(json.dumps(result.to_dict()))


class TestGMMResultJSON:
    """GMM clean view serializes to native Python types."""

    def test_fields_are_native_types_pre_serialization(self, gmm_cluster_result):
        """Float fields are native ``float`` on the dataclass, not laundered by JSON."""
        result = ClusterResult.from_gmm_dict(gmm_cluster_result, random_state=42)

        assert result.algorithm == ALGORITHM_GMM
        assert type(result.bic) is float
        assert type(result.aic) is float
        assert type(result.converged) is bool
        assert type(result.n_iter) is int
        assert all(type(v) is float for v in result.weights)
        assert all(type(v) is float for row in result.cluster_centers for v in row)

    def test_json_roundtrip_native_types(self, gmm_cluster_result):
        """GMM view round-trips to native types with the right discriminator."""
        result = ClusterResult.from_gmm_dict(gmm_cluster_result, random_state=42)
        assert isinstance(result, GMMResult)
        parsed = json.loads(json.dumps(dataclasses.asdict(result)))

        assert parsed["algorithm"] == ALGORITHM_GMM
        assert all(type(v) is float for row in parsed["cluster_centers"] for v in row)
        assert all(type(v) is float for v in parsed["weights"])
        assert type(parsed["bic"]) is float
        assert type(parsed["aic"]) is float
        assert type(parsed["converged"]) is bool
        assert type(parsed["n_iter"]) is int

    def test_to_json_roundtrips_finite_data(self, gmm_cluster_result):
        """On finite data, to_json emits strict JSON that parses back to to_dict."""
        result = ClusterResult.from_gmm_dict(gmm_cluster_result, random_state=42)
        assert json.loads(result.to_json()) == json.loads(json.dumps(result.to_dict()))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_to_json_rejects_non_finite_bic(self, gmm_cluster_result, bad):
        """A non-finite bic raises at to_json instead of emitting invalid JSON."""
        result = ClusterResult.from_gmm_dict(gmm_cluster_result, random_state=42)
        tainted = dataclasses.replace(result, bic=bad)
        with pytest.raises(ValueError, match="not JSON compliant"):
            tainted.to_json()


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

    def test_gmm_adapter_value_asserts_all_scalar_fields(self, gmm_cluster_result):
        """bic/aic/converged/n_iter/weights map by value, guarding a bic<->aic swap."""
        d = gmm_cluster_result
        result = ClusterResult.from_gmm_dict(d, random_state=42)

        assert result.bic == pytest.approx(float(d["bic"]))
        assert result.aic == pytest.approx(float(d["aic"]))
        assert result.bic != result.aic  # distinct keys, not swapped/aliased
        assert result.converged is bool(d["converged"])
        assert result.n_iter == int(d["n_iter"])
        assert result.weights == pytest.approx([float(w) for w in d["weights"]])

    def test_gmm_adapter_retains_covariances(self, gmm_cluster_result):
        """The fitted per-component covariances are retained (cluster shapes)."""
        d = gmm_cluster_result
        result = ClusterResult.from_gmm_dict(d, random_state=42)

        np.testing.assert_allclose(
            np.asarray(result.covariances), np.asarray(d["covariances"])
        )
        # covariance_type="full" -> (n_clusters, n_features, n_features)
        if result.covariance_type == "full":
            n_features = len(result.feature_names)
            assert np.asarray(result.covariances).shape == (
                result.n_clusters,
                n_features,
                n_features,
            )


class TestClusterDeterminism:
    """Same seed -> identical result via the typed view (#118)."""

    def test_kmeans_same_seed_identical_labels(self, simple_cluster_data):
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

    def test_gmm_same_seed_identical_serialized(self, multimodal_data):
        """GMM's EM is the most init-sensitive case: same seed -> identical result.

        Asserts the full serialized view (labels + bic/aic/weights/covariances),
        which a labels-only check would miss.
        """
        r1 = ClusterResult.from_gmm_dict(
            perform_gmm_clustering(multimodal_data, n_components=2, random_state=42),
            random_state=42,
        )
        r2 = ClusterResult.from_gmm_dict(
            perform_gmm_clustering(multimodal_data, n_components=2, random_state=42),
            random_state=42,
        )
        assert r1.cluster_labels == r2.cluster_labels
        assert dataclasses.asdict(r1) == dataclasses.asdict(r2)


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
        """KMeans adapter does not mutate its input dict (keys *and* values)."""
        d = kmeans_cluster_result
        before = copy.deepcopy(d)
        ClusterResult.from_kmeans_dict(d, random_state=42)
        _assert_dict_unchanged(d, before)

    def test_gmm_dict_not_mutated(self, gmm_cluster_result):
        """GMM adapter does not mutate its input dict (keys *and* values)."""
        d = gmm_cluster_result
        before = copy.deepcopy(d)
        ClusterResult.from_gmm_dict(d, random_state=42)
        _assert_dict_unchanged(d, before)
