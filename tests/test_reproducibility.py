"""Determinism regression tests for stochastic public functions (issue #118).

bloom-mcp's Phase 2 golden-value tests reproduce past analyses (wheat EDPIE PCA,
clustering, UMAP) and rely on every stochastic function returning identical output
for a fixed ``random_state``. These tests run each stochastic public function twice
with the same seed and assert the reproducibility-bearing outputs match — exactly for
integer labels/indices, within ``rtol`` for floating-point arrays.

See ``docs/reproducibility.md`` for the seeding and tolerance policy.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import sleap_roots_analyze as sra
from sleap_roots_analyze.clustering import perform_hierarchical_clustering

# Tolerance for floating-point arrays. Within a single machine the outputs are
# bit-identical; rtol=1e-6 leaves headroom for cross-platform BLAS differences
# (see docs/reproducibility.md).
RTOL = 1e-6
ATOL = 1e-9

SEED = 42
N_FEATURES = 6


@pytest.fixture(scope="module")
def synthetic_df():
    """Small clustered, NaN-free dataset shared across determinism checks.

    Returns:
        DataFrame of 60 samples x 6 features drawn from 3 well-separated clusters,
        built from a fixed seed so the fixture itself is reproducible.
    """
    rng = np.random.RandomState(0)
    centers = rng.randn(3, N_FEATURES) * 5
    blocks = [centers[i] + rng.randn(20, N_FEATURES) for i in range(3)]
    data = np.vstack(blocks)
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(N_FEATURES)])


@pytest.fixture(scope="module")
def feature_cols():
    """Feature column names matching :func:`synthetic_df`.

    Returns:
        List of the six feature column names.
    """
    return [f"f{i}" for i in range(N_FEATURES)]


def _assert_key_equal(result_a, result_b, key, mode):
    """Assert two result dicts agree on one key.

    Args:
        result_a: First result dictionary.
        result_b: Second result dictionary.
        key: Key to compare.
        mode: "exact" for array-equal (labels/indices) or "close" for ``rtol``.

    Returns:
        None. Raises AssertionError if the values differ.
    """
    a = np.asarray(result_a[key])
    b = np.asarray(result_b[key])
    if mode == "exact":
        assert np.array_equal(a, b), f"{key} differs between runs (exact)"
    else:
        assert np.allclose(
            a.astype(float), b.astype(float), rtol=RTOL, atol=ATOL, equal_nan=True
        ), f"{key} differs between runs (rtol={RTOL})"


# (label, callable, extra-kwargs-builder, [(key, mode), ...])
# extra-kwargs-builder receives (df, feature_cols) and returns the non-seed kwargs.
_CASES = [
    (
        "perform_pca_analysis",
        sra.perform_pca_analysis,
        lambda df, fc: {},
        [
            ("transformed_data", "close"),
            ("loadings", "close"),
            ("eigenvalues", "close"),
        ],
    ),
    (
        "perform_umap_analysis",
        sra.perform_umap_analysis,
        lambda df, fc: {"feature_cols": fc},
        [("embedding", "close")],
    ),
    (
        "perform_kmeans_clustering",
        sra.perform_kmeans_clustering,
        lambda df, fc: {"n_clusters": 3},
        [
            ("cluster_labels", "exact"),
            ("cluster_centers", "close"),
            ("inertia", "close"),
        ],
    ),
    (
        "perform_gmm_clustering",
        sra.perform_gmm_clustering,
        lambda df, fc: {"n_components": 3},
        [("cluster_labels", "exact"), ("means", "close")],
    ),
    (
        "detect_outliers_isolation_forest",
        sra.detect_outliers_isolation_forest,
        lambda df, fc: {},
        [("outlier_indices", "exact"), ("anomaly_scores", "close")],
    ),
    (
        "detect_outliers_kmeans",
        sra.detect_outliers_kmeans,
        lambda df, fc: {},
        [("cluster_labels", "exact"), ("min_distances_to_centers", "close")],
    ),
    (
        "detect_outliers_gmm",
        sra.detect_outliers_gmm,
        lambda df, fc: {},
        [("cluster_labels", "exact"), ("probabilities", "close")],
    ),
    (
        "detect_outliers_mahalanobis",
        sra.detect_outliers_mahalanobis,
        lambda df, fc: {},
        [("outlier_indices", "exact"), ("mahalanobis_distances", "close")],
    ),
]


@pytest.mark.parametrize(
    "name, func, kwargs_for, keys", _CASES, ids=[c[0] for c in _CASES]
)
def test_same_seed_is_reproducible(
    name, func, kwargs_for, keys, synthetic_df, feature_cols
):
    """Each seeded stochastic function returns identical output across two runs."""
    kwargs = kwargs_for(synthetic_df, feature_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_a = func(synthetic_df, random_state=SEED, **kwargs)
        result_b = func(synthetic_df, random_state=SEED, **kwargs)
    for key, mode in keys:
        _assert_key_equal(result_a, result_b, key, mode)


def test_hierarchical_clustering_is_deterministic(synthetic_df):
    """perform_hierarchical_clustering is deterministic and needs no seed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_a = perform_hierarchical_clustering(synthetic_df)
        result_b = perform_hierarchical_clustering(synthetic_df)
    _assert_key_equal(result_a, result_b, "linkage_matrix", "close")


@pytest.mark.parametrize(
    "name, func, kwargs_for, keys", _CASES, ids=[c[0] for c in _CASES]
)
def test_accepts_random_state_none(
    name, func, kwargs_for, keys, synthetic_df, feature_cols
):
    """Each seeded function accepts random_state=None without raising."""
    kwargs = kwargs_for(synthetic_df, feature_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = func(synthetic_df, random_state=None, **kwargs)
    assert isinstance(result, dict)
