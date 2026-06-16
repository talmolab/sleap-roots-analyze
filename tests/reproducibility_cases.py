"""Shared registry of stochastic-function cases for the reproducibility gates.

This is the single source of truth for *which* stochastic functions the
reproducibility gates exercise. Both the determinism sweep
(``tests/test_reproducibility.py``) and its whole-package coverage guard consult
this registry, and the guard fails CI if any in-package function accepting
``random_state`` is absent from it (or from :data:`EXCLUDED`).

See ``docs/reproducibility.md`` for the seeding and tolerance policy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd

from sleap_roots_analyze import clustering, outlier_detection, pca, umap

# Tolerance for floating-point arrays. Within a single machine the outputs are
# bit-identical; rtol=1e-6 leaves headroom for cross-platform BLAS differences
# (see docs/reproducibility.md).
RTOL = 1e-6
ATOL = 1e-9

SEED = 42
N_FEATURES = 6

# Stochastic functions intentionally NOT given their own determinism case, mapped
# to the reason. A dict (not a bare set) so each exclusion documents *why* it is
# safe to skip. Every key must still be a discoverable ``random_state`` function and
# must not overlap the registry (enforced by ``test_excluded_set_is_consistent``).
# Empty today: the approved scope covers every in-package stochastic function
# directly — including the formerly seed-hardcoded ``calculate_mahalanobis_distances``
# and ``detect_outliers_pca``, now caller-seedable and swept (#118).
EXCLUDED: dict[str, str] = {}


@dataclass(frozen=True)
class Case:
    """One stochastic function under the reproducibility gates.

    Attributes:
        func: The stochastic function itself (identity for the coverage guard).
        run: ``run(ctx, seed)`` calls ``func`` with the given ``random_state`` on
            the shared synthetic context and returns its raw result.
        compare: ``compare(a, b)`` asserts two results from the same seed are
            equal (exact for labels/indices, ``rtol`` for float arrays).
    """

    func: Callable
    run: Callable
    compare: Callable

    @property
    def label(self) -> str:
        """Short function name, used as the parametrize id."""
        return self.func.__name__

    @property
    def qualname(self) -> str:
        """Fully-qualified ``module.func`` name, matching the coverage guard."""
        return f"{self.func.__module__}.{self.func.__name__}"


@dataclass
class Context:
    """Shared synthetic inputs reused across every case.

    Attributes:
        df: 60 samples x 6 features from 3 well-separated clusters.
        feature_cols: The six feature column names.
        X: ``df`` as a float ndarray, for the array-input PCA helpers.
    """

    df: pd.DataFrame
    feature_cols: List[str]
    X: np.ndarray


def make_context() -> Context:
    """Build the shared synthetic context from a fixed seed.

    Returns:
        A :class:`Context` whose data is reproducible across runs.
    """
    rng = np.random.RandomState(0)
    centers = rng.randn(3, N_FEATURES) * 5
    blocks = [centers[i] + rng.randn(20, N_FEATURES) for i in range(3)]
    data = np.vstack(blocks)
    cols = [f"f{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(data, columns=cols)
    return Context(df=df, feature_cols=cols, X=df.to_numpy(dtype=float))


# --- comparison helpers -------------------------------------------------------


def _close(a, b, what: str) -> None:
    """Assert two float arrays match within tolerance (NaN-aware)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert np.allclose(
        a, b, rtol=RTOL, atol=ATOL, equal_nan=True
    ), f"{what} differs between runs (rtol={RTOL})"


def _exact(a, b, what: str) -> None:
    """Assert two arrays/values are exactly equal (labels, indices, ints)."""
    assert np.array_equal(
        np.asarray(a), np.asarray(b)
    ), f"{what} differs between runs (exact)"


def _dict_compare(keys):
    """Build a comparator over dict results for the given ``(key, mode)`` pairs."""

    def compare(a, b):
        for key, mode in keys:
            (_exact if mode == "exact" else _close)(a[key], b[key], key)

    return compare


def _compare_fit_pca(a, b):
    """Compare ``fit_pca`` results: (PCA model, transformed array)."""
    pca_a, xt_a = a
    pca_b, xt_b = b
    _close(xt_a, xt_b, "transformed")
    _close(pca_a.components_, pca_b.components_, "components_")
    _close(pca_a.explained_variance_, pca_b.explained_variance_, "explained_variance_")


def _compare_int(a, b):
    """Compare scalar integer results (e.g. selected n_components)."""
    _exact(a, b, "value")


def _compare_mahalanobis(a, b):
    """Compare ``calculate_mahalanobis_distances``: (distances, mean, covariance)."""
    (dist_a, mean_a, cov_a) = a
    (dist_b, mean_b, cov_b) = b
    _close(dist_a, dist_b, "distances")
    _close(mean_a, mean_b, "mean")
    _close(cov_a, cov_b, "covariance")


def _silence(fn):
    """Run ``fn`` with warnings suppressed (sklearn/umap chatter)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


# --- the registry -------------------------------------------------------------

CASES: List[Case] = [
    Case(
        pca.perform_pca_analysis,
        lambda ctx, seed: _silence(
            lambda: pca.perform_pca_analysis(ctx.df, random_state=seed)
        ),
        _dict_compare(
            [
                ("transformed_data", "close"),
                ("loadings", "close"),
                ("eigenvalues", "close"),
            ]
        ),
    ),
    Case(
        pca.fit_pca,
        lambda ctx, seed: _silence(
            lambda: pca.fit_pca(ctx.X, n_components=3, random_state=seed)
        ),
        _compare_fit_pca,
    ),
    Case(
        # Deterministic by construction: select_n_components reads PCA variance
        # ratios (full/exact SVD), so its seed is not load-bearing. Swept anyway so
        # the coverage guard stays complete and a future randomized path is caught.
        pca.select_n_components,
        lambda ctx, seed: _silence(
            lambda: pca.select_n_components(ctx.X, random_state=seed)
        ),
        _compare_int,
    ),
    Case(
        pca.perform_pca_with_variance_threshold,
        lambda ctx, seed: _silence(
            lambda: pca.perform_pca_with_variance_threshold(ctx.X, random_state=seed)
        ),
        _dict_compare(
            [
                ("transformed_data", "close"),
                ("loadings", "close"),
                ("eigenvalues", "close"),
            ]
        ),
    ),
    Case(
        umap.perform_umap_analysis,
        lambda ctx, seed: _silence(
            lambda: umap.perform_umap_analysis(
                ctx.df, feature_cols=ctx.feature_cols, random_state=seed
            )
        ),
        _dict_compare([("embedding", "close")]),
    ),
    Case(
        clustering.perform_kmeans_clustering,
        lambda ctx, seed: _silence(
            lambda: clustering.perform_kmeans_clustering(
                ctx.df, n_clusters=3, random_state=seed
            )
        ),
        _dict_compare(
            [
                ("cluster_labels", "exact"),
                ("cluster_centers", "close"),
                ("inertia", "close"),
            ]
        ),
    ),
    Case(
        clustering.perform_gmm_clustering,
        lambda ctx, seed: _silence(
            lambda: clustering.perform_gmm_clustering(
                ctx.df, n_components=3, random_state=seed
            )
        ),
        _dict_compare([("cluster_labels", "exact"), ("means", "close")]),
    ),
    Case(
        clustering.calculate_optimal_k_kmeans,
        lambda ctx, seed: _silence(
            lambda: clustering.calculate_optimal_k_kmeans(ctx.df, random_state=seed)
        ),
        _dict_compare(
            [
                ("optimal_n_clusters", "exact"),
                ("k_values", "exact"),
                ("scores", "close"),
            ]
        ),
    ),
    Case(
        outlier_detection.detect_outliers_isolation_forest,
        lambda ctx, seed: _silence(
            lambda: outlier_detection.detect_outliers_isolation_forest(
                ctx.df, random_state=seed
            )
        ),
        _dict_compare([("outlier_indices", "exact"), ("anomaly_scores", "close")]),
    ),
    Case(
        outlier_detection.detect_outliers_kmeans,
        lambda ctx, seed: _silence(
            lambda: outlier_detection.detect_outliers_kmeans(ctx.df, random_state=seed)
        ),
        _dict_compare(
            [("cluster_labels", "exact"), ("min_distances_to_centers", "close")]
        ),
    ),
    Case(
        outlier_detection.detect_outliers_gmm,
        lambda ctx, seed: _silence(
            lambda: outlier_detection.detect_outliers_gmm(ctx.df, random_state=seed)
        ),
        _dict_compare([("cluster_labels", "exact"), ("probabilities", "close")]),
    ),
    Case(
        outlier_detection.detect_outliers_mahalanobis,
        lambda ctx, seed: _silence(
            lambda: outlier_detection.detect_outliers_mahalanobis(
                ctx.df, random_state=seed
            )
        ),
        _dict_compare(
            [("outlier_indices", "exact"), ("mahalanobis_distances", "close")]
        ),
    ),
    Case(
        outlier_detection.detect_outliers_pca,
        # Seed is forwarded to PCA; not load-bearing under full/exact SVD, but the
        # function is now caller-seedable (#118) so the sweep exercises it.
        lambda ctx, seed: _silence(
            lambda: outlier_detection.detect_outliers_pca(ctx.df, random_state=seed)
        ),
        _dict_compare(
            [("outlier_indices", "exact"), ("reconstruction_errors", "close")]
        ),
    ),
    Case(
        pca.calculate_mahalanobis_distances,
        # robust=True makes MinCovDet's seed load-bearing — the path the old
        # hardcoded random_state=42 left un-exercised (#118).
        lambda ctx, seed: _silence(
            lambda: pca.calculate_mahalanobis_distances(
                ctx.X, robust=True, random_state=seed
            )
        ),
        _compare_mahalanobis,
    ),
]

# Stable anchor for the count/label pins in tests/test_reproducibility.py.
EXPECTED_LABELS = frozenset(c.label for c in CASES)
