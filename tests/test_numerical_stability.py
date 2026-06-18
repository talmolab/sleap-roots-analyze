"""Numerical-stability golden gate (drift detector).

This gate is **distinct from** the same-machine determinism sweep
(``tests/test_reproducibility.py``). Determinism double-runs each function on one machine
and asserts the two runs agree — but under a ``numba`` / ``numpy`` / ``umap-learn`` /
``pandas`` upgrade *both* runs drift together, so it cannot catch silent numerical drift.
This gate instead compares freshly-recomputed outputs against **committed golden
artifacts**, so a library upgrade that moves the numbers fails here.

Assertions are tolerance-based, not bit-exact, with every threshold grounded in the
reference dataset's measured same-stack spread (see
``tests/numerical_stability_recompute.py`` and ``docs/reproducibility.md``):

- **UMAP embedding** — Procrustes superimposition, then ``np.allclose`` on the *aligned
  coordinate matrices* (not the disparity scalar, which is too insensitive).
- **Cluster labels** — Adjusted Rand Index (permutation-invariant) above a floor.
- **Trait table** — ``assert_frame_equal`` with a tight ``rtol`` (same-stack pure-float).

**Single-OS by design.** UMAP is not bit-reproducible across operating systems / BLAS
backends, and these tolerances are tighter than the cross-OS float floor. The golden is
generated on one OS; this module is skipped elsewhere so the cross-platform ``tests``
matrix stays green, and a dedicated single-OS CI job runs it. See ``docs/reproducibility.md``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import procrustes
from sklearn.metrics import adjusted_rand_score

from tests.numerical_stability_recompute import (
    ARI_FLOOR,
    ATOL_PROCRUSTES,
    CANONICAL_PLATFORM,
    N_CLUSTERS,
    TRAIT_RTOL,
    align_embedding,
    align_labels,
    load_input,
    recompute_embedding,
    recompute_labels,
    recompute_trait_summary,
)

pytestmark = pytest.mark.skipif(
    sys.platform != CANONICAL_PLATFORM,
    reason=(
        f"Numerical-stability gate is single-OS by design (golden generated on "
        f"'{CANONICAL_PLATFORM}'); UMAP is not bit-reproducible across operating systems. "
        "See docs/reproducibility.md."
    ),
)


# --- Comparison helpers (raise AssertionError naming the drifted artifact) ---------


def _assert_embedding_matches(golden_emb: pd.DataFrame, new_emb: pd.DataFrame) -> None:
    """Assert two embeddings match after Procrustes superimposition."""
    g = align_embedding(golden_emb)
    r = align_embedding(new_emb)
    aligned_g, aligned_r, _ = procrustes(g, r)
    if not np.allclose(aligned_g, aligned_r, atol=ATOL_PROCRUSTES):
        max_delta = np.max(np.abs(aligned_g - aligned_r))
        raise AssertionError(
            f"UMAP embedding drifted: Procrustes-aligned max|delta|={max_delta:.3e} "
            f"exceeds atol={ATOL_PROCRUSTES:.0e}"
        )


def _assert_labels_match(golden_labels: pd.DataFrame, new_labels: pd.DataFrame) -> None:
    """Assert two label sets agree by Adjusted Rand Index."""
    ari = adjusted_rand_score(align_labels(golden_labels), align_labels(new_labels))
    if not ari > ARI_FLOOR:
        raise AssertionError(
            f"cluster labels drifted: ARI={ari:.4f} is not above floor {ARI_FLOOR}"
        )


def _assert_trait_summary_matches(golden: pd.DataFrame, new: pd.DataFrame) -> None:
    """Assert two trait-summary tables match within ``TRAIT_RTOL``."""
    try:
        pd.testing.assert_frame_equal(new, golden, rtol=TRAIT_RTOL)
    except AssertionError as exc:  # re-raise naming the artifact
        raise AssertionError(f"trait summary drifted: {exc}") from exc


# --- Fixtures (recompute the stochastic outputs once; UMAP JIT is the slow part) ----


@pytest.fixture(scope="module")
def reference_df() -> pd.DataFrame:
    """Load the reference input table once for the module."""
    return load_input()


@pytest.fixture(scope="module")
def recomputed(reference_df):
    """Recompute the three outputs once on the current stack."""
    labels, effective_k = recompute_labels(reference_df)
    return {
        "embedding": recompute_embedding(reference_df),
        "labels": labels,
        "n_clusters": effective_k,
        "trait_summary": recompute_trait_summary(reference_df),
    }


# --- The gate: recomputed outputs match the committed golden ------------------------


def test_umap_embedding_matches_golden(recomputed, numerical_stability_golden):
    """Recomputed UMAP embedding matches the golden under Procrustes superimposition."""
    _assert_embedding_matches(
        numerical_stability_golden["embedding"], recomputed["embedding"]
    )


def test_cluster_labels_match_golden(recomputed, numerical_stability_golden):
    """Recomputed cluster labels match the golden by ARI."""
    _assert_labels_match(numerical_stability_golden["labels"], recomputed["labels"])


def test_cluster_count_honored(recomputed):
    """The pinned cluster count is honored (guards the internal len//10 clamp)."""
    assert recomputed["n_clusters"] == N_CLUSTERS


def test_trait_summary_matches_golden(recomputed, numerical_stability_golden):
    """Recomputed per-genotype trait summary matches the golden within ``rtol``."""
    _assert_trait_summary_matches(
        numerical_stability_golden["trait_summary"], recomputed["trait_summary"]
    )


# --- Sensitivity: tolerant of rigid transforms, intolerant of structural drift ------


def test_embedding_tolerates_rigid_transform(numerical_stability_golden):
    """A rotated/reflected/translated/scaled copy of the golden still passes."""
    coords = align_embedding(numerical_stability_golden["embedding"])
    theta = 0.7
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    reflection = np.array([[1.0, 0.0], [0.0, -1.0]])
    transformed = 3.5 * coords @ rotation @ reflection + np.array([100.0, -42.0])

    golden_emb = numerical_stability_golden["embedding"].sort_values("Barcode")
    transformed_emb = golden_emb.assign(
        umap_1=transformed[:, 0], umap_2=transformed[:, 1]
    )
    # Should NOT raise — Procrustes removes rigid-transform degrees of freedom.
    _assert_embedding_matches(numerical_stability_golden["embedding"], transformed_emb)


def test_embedding_flags_structural_drift(numerical_stability_golden):
    """A small structural perturbation (above the sensitivity floor) is caught."""
    perturbed = numerical_stability_golden["embedding"].copy()
    # 1e-3 nudge on one coordinate -> aligned max|delta| ~2.6e-5 > atol=1e-6 (measured).
    perturbed.loc[0, "umap_1"] = perturbed.loc[0, "umap_1"] + 1e-3
    with pytest.raises(AssertionError, match="UMAP embedding drifted"):
        _assert_embedding_matches(numerical_stability_golden["embedding"], perturbed)


# --- Negative controls: the comparison bites, and the recompute path is live --------


def test_drift_fails_per_artifact(recomputed, numerical_stability_golden):
    """Mutating each golden beyond tolerance fails its own assertion (comparison bites)."""
    # Embedding: shift every point by a large structural amount.
    bad_emb = recomputed["embedding"].copy()
    bad_emb["umap_1"] = bad_emb["umap_1"] + np.linspace(0, 5, len(bad_emb))
    with pytest.raises(AssertionError, match="UMAP embedding drifted"):
        _assert_embedding_matches(numerical_stability_golden["embedding"], bad_emb)

    # Labels: scramble half the labels so ARI collapses.
    bad_labels = recomputed["labels"].copy()
    half = len(bad_labels) // 2
    bad_labels.loc[: half - 1, "cluster_label"] = (
        bad_labels.loc[: half - 1, "cluster_label"] + 1
    ) % N_CLUSTERS
    with pytest.raises(AssertionError, match="cluster labels drifted"):
        _assert_labels_match(numerical_stability_golden["labels"], bad_labels)

    # Trait summary: bump one cell well past rtol.
    bad_summary = recomputed["trait_summary"].copy()
    col = bad_summary.columns[0]
    bad_summary.iloc[0, 0] = bad_summary.iloc[0, 0] * 1.01 + 1.0
    with pytest.raises(AssertionError, match="trait summary drifted"):
        _assert_trait_summary_matches(
            numerical_stability_golden["trait_summary"], bad_summary
        )


def test_recompute_path_is_live_not_tautology(reference_df, numerical_stability_golden):
    """Perturbing the *recompute* (different seed / k) fails — proves it is not golden-vs-golden.

    If the gate accidentally compared the golden to itself, changing the recompute inputs
    would not change the outcome. It must.
    """
    # A different seed produces a structurally different embedding.
    other_seed_emb = recompute_embedding(reference_df)  # same seed baseline
    # Re-run with an explicitly different seed via the underlying function.
    from sleap_roots_analyze.umap import perform_umap_analysis
    from tests.numerical_stability_recompute import ID_COL, TRAIT_COLS

    drifted = perform_umap_analysis(
        reference_df, feature_cols=TRAIT_COLS, random_state=7, n_components=2
    )["embedding"]
    drifted_emb = pd.DataFrame(
        {
            ID_COL: reference_df[ID_COL].to_numpy(),
            "umap_1": np.asarray(drifted)[:, 0],
            "umap_2": np.asarray(drifted)[:, 1],
        }
    )
    # Baseline (same seed) matches golden; the different-seed recompute should not.
    _assert_embedding_matches(numerical_stability_golden["embedding"], other_seed_emb)
    with pytest.raises(AssertionError, match="UMAP embedding drifted"):
        _assert_embedding_matches(numerical_stability_golden["embedding"], drifted_emb)


# --- Robust failure modes -----------------------------------------------------------


def test_missing_input_fails_loudly(monkeypatch):
    """A missing input fixture fails with a message pointing to the regen script."""
    import tests.numerical_stability_recompute as nsr

    monkeypatch.setattr(nsr, "INPUT_CSV", nsr.INPUT_CSV.with_name("does_not_exist.csv"))
    with pytest.raises(
        FileNotFoundError, match="regenerate_numerical_stability_golden"
    ):
        nsr.load_input()


def test_new_nan_in_output_is_caught(recomputed, numerical_stability_golden):
    """A NaN appearing in a recomputed output that the golden lacks fails the gate."""
    with_nan = recomputed["trait_summary"].copy()
    with_nan.iloc[0, 0] = np.nan
    with pytest.raises(AssertionError, match="trait summary drifted"):
        _assert_trait_summary_matches(
            numerical_stability_golden["trait_summary"], with_nan
        )


def test_provenance_records_stack(numerical_stability_golden):
    """The provenance record captures the stack needed to judge golden staleness."""
    prov = numerical_stability_golden["provenance"]
    assert prov["os"] and prov["python"]
    assert prov["seed"] == 42 and prov["n_clusters"] == N_CLUSTERS
    for pkg in ("numpy", "pandas", "umap-learn", "numba", "scipy", "scikit-learn"):
        assert pkg in prov["packages"]
