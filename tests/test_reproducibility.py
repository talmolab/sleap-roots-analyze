"""Determinism regression tests for stochastic functions (issues #118, #133).

bloom-mcp's Phase 2 golden-value tests reproduce past analyses (wheat EDPIE PCA,
clustering, UMAP) and rely on every stochastic function returning identical output
for a fixed ``random_state``. These tests run each stochastic function (from the
shared registry in ``tests/reproducibility_cases.py``) twice with the same seed and
assert the reproducibility-bearing outputs match.

A whole-package coverage guard (``test_sweep_covers_all_stochastic_functions``)
walks every module in ``sleap_roots_analyze`` and fails if any function accepting
``random_state`` is missing from the registry, so the sweep stays complete without
hand vigilance. See ``docs/reproducibility.md`` for the seeding/tolerance policy.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings

import numpy as np
import pytest

import sleap_roots_analyze as sra
from tests.reproducibility_cases import CASES, EXCLUDED, make_context
from sleap_roots_analyze.clustering import perform_hierarchical_clustering

# Hardcoded anchor: the exact set of stochastic functions covered today. Pinned
# (not derived from CASES) so silently dropping a case during refactors fails
# loudly rather than passing with fewer parametrized runs.
EXPECTED_QUALNAMES = frozenset(
    {
        "sleap_roots_analyze.pca.perform_pca_analysis",
        "sleap_roots_analyze.pca.fit_pca",
        "sleap_roots_analyze.pca.select_n_components",
        "sleap_roots_analyze.pca.perform_pca_with_variance_threshold",
        "sleap_roots_analyze.umap.perform_umap_analysis",
        "sleap_roots_analyze.clustering.perform_kmeans_clustering",
        "sleap_roots_analyze.clustering.perform_gmm_clustering",
        "sleap_roots_analyze.clustering.calculate_optimal_k_kmeans",
        "sleap_roots_analyze.outlier_detection.detect_outliers_isolation_forest",
        "sleap_roots_analyze.outlier_detection.detect_outliers_kmeans",
        "sleap_roots_analyze.outlier_detection.detect_outliers_gmm",
        "sleap_roots_analyze.outlier_detection.detect_outliers_mahalanobis",
        "sleap_roots_analyze.outlier_detection.detect_outliers_pca",
        "sleap_roots_analyze.pca.calculate_mahalanobis_distances",
    }
)


@pytest.fixture(scope="module")
def ctx():
    """Shared synthetic context (DataFrame, feature columns, array)."""
    return make_context()


def discover_stochastic_qualnames():
    """Walk the whole package for module-level ``random_state`` functions.

    Returns:
        Set of fully-qualified ``module.func`` names for every function defined
        in ``sleap_roots_analyze`` whose signature accepts ``random_state``.
    """
    found = set()
    for _, modname, _ in pkgutil.walk_packages(sra.__path__, sra.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            # A module that fails to import cannot define a usable public
            # function; importing the package already exercises the real ones.
            continue
        for _, fn in inspect.getmembers(mod, inspect.isfunction):
            if not fn.__module__.startswith("sleap_roots_analyze"):
                continue
            try:
                params = inspect.signature(fn).parameters
            except (ValueError, TypeError):
                continue
            if "random_state" in params:
                found.add(f"{fn.__module__}.{fn.__name__}")
    return found


def uncovered_functions(found):
    """Names in ``found`` neither covered by the registry nor excluded.

    Args:
        found: Set of fully-qualified stochastic-function names.

    Returns:
        The subset that has no determinism case and is not in :data:`EXCLUDED`.
    """
    covered = {c.qualname for c in CASES}
    return set(found) - covered - set(EXCLUDED)


# --- coverage guard -----------------------------------------------------------


def test_registry_matches_expected_set():
    """The registry covers exactly the pinned set (guards silent case drops)."""
    assert {c.qualname for c in CASES} == EXPECTED_QUALNAMES
    assert len(CASES) == len(EXPECTED_QUALNAMES)


def test_sweep_covers_all_stochastic_functions():
    """Every in-package ``random_state`` function has a determinism case."""
    missing = uncovered_functions(discover_stochastic_qualnames())
    assert not missing, f"stochastic functions missing a determinism case: {missing}"


def test_coverage_guard_detects_missing_function():
    """The guard reports an uncovered function (proves it can go red)."""
    fake = "sleap_roots_analyze.fake.fake_fn"
    covered = {c.qualname for c in CASES}
    assert uncovered_functions(covered | {fake}) == {fake}


def test_excluded_set_is_consistent():
    """Excluded names must still exist as stochastic functions and not overlap."""
    assert set(EXCLUDED).isdisjoint({c.qualname for c in CASES})
    assert set(EXCLUDED) <= discover_stochastic_qualnames()


# --- determinism sweep --------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_same_seed_is_reproducible(case, ctx):
    """Each stochastic function returns identical output across two runs."""
    result_a = case.run(ctx, 42)
    result_b = case.run(ctx, 42)
    case.compare(result_a, result_b)


def test_hierarchical_clustering_is_deterministic(ctx):
    """perform_hierarchical_clustering is deterministic and needs no seed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_a = perform_hierarchical_clustering(ctx.df)
        result_b = perform_hierarchical_clustering(ctx.df)
    assert np.allclose(
        result_a["linkage_matrix"], result_b["linkage_matrix"], equal_nan=True
    )


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_accepts_random_state_none(case, ctx):
    """Each seeded function accepts random_state=None without raising."""
    result = case.run(ctx, None)
    assert result is not None
