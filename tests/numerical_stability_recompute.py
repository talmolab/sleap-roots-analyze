"""Shared recompute logic for the numerical-stability golden gate.

This module is the single source of truth for *how* the numerical-stability golden
artifacts are computed. It is imported by **both**:

- ``tests/test_numerical_stability.py`` — recomputes outputs and compares them to the
  committed golden artifacts (drift detection), and
- ``scripts/regenerate_numerical_stability_golden.py`` — writes the golden artifacts and
  the provenance record.

Keeping the constants and compute here (under ``tests/``, not ``scripts/``) means a
script-only edit cannot silently break test collection, and the test and the golden can
never compute different things. See ``docs/reproducibility.md`` and
``tests/fixtures/README.md`` for the gate's rationale and regeneration policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sleap_roots_analyze.clustering import perform_kmeans_clustering
from sleap_roots_analyze.umap import perform_umap_analysis

# --- Pinned configuration (the only stack-independent inputs) ---------------------

#: Fixed seed for every stochastic step in the gate.
SEED = 42

#: Pinned cluster count — never silhouette auto-selected, so the golden is stable.
N_CLUSTERS = 3

#: The OS the golden artifacts are generated on. The gate runs only here; UMAP is not
#: bit-reproducible across operating systems / BLAS backends, and the tolerances below
#: are tighter than the cross-OS float floor. ``sys.platform`` value: "darwin" = macOS.
CANONICAL_PLATFORM = "darwin"

#: Reference dataset: the turface_19 post-QC final-data slice (153 samples).
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "real"
    / "wheat_edpie"
    / "inputs"
    / "post_qc"
    / "turface_19_final_data.csv"
)

#: Directory holding the committed golden artifacts + provenance record.
GOLDEN_DIR = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "real"
    / "wheat_edpie"
    / "expected"
    / "numerical_stability"
)
GOLDEN_EMBEDDING = GOLDEN_DIR / "golden_umap_embedding.csv"
GOLDEN_LABELS = GOLDEN_DIR / "golden_cluster_labels.csv"
GOLDEN_TRAIT_SUMMARY = GOLDEN_DIR / "golden_trait_summary.csv"
GOLDEN_PROVENANCE = GOLDEN_DIR / "golden_provenance.json"

#: Stable sample identifier used to align golden vs recomputed rows.
ID_COL = "Barcode"

#: Genotype grouping column for the trait summary.
GENOTYPE_COL = "Genotype"

#: Explicit, fixed trait columns the gate computes on. Pinned (not derived from the
#: heritability step) so the gate is self-contained and the golden cannot drift because
#: an upstream filter changed. These are the 12 numeric traits in the post-QC input,
#: excluding the metadata columns (Barcode, Genotype, Replicate).
TRAIT_COLS = [
    "Maximum.Width.mm",
    "Width-to-Depth.Ratio",
    "Solidity",
    "Computation.Time.s",
    "Holes",
    "Network Area (mm²)",
    "Perimeter (mm)",
    "Root Biomass (mg)",
    "Shoot Biomass (mg)",
    "Surface Area (mm²)",
    "Total Root Length (mm)",
    "Volume (mm³)",
]

#: Aggregations for the per-genotype trait summary — the CoW-sensitive groupby path.
TRAIT_AGGS = ["mean", "std"]

# --- Tolerances (grounded in measured same-stack spread; see docs) ----------------
# Measured on turface_19 under the canonical stack:
#   UMAP Procrustes-aligned same-stack max|delta| ~= 2.8e-17 (bit-identical),
#     a real 1e-3 coordinate nudge -> aligned max|delta| ~= 2.6e-5,
#     so atol=1e-6 sits ~11 orders above noise yet ~26x below the drift it must catch.
#   KMeans same-stack ARI = 1.0 (sizes [32, 86, 35]); 0.95 floor leaves headroom.
#   Trait summary same-stack max relative delta = 0.0; 1e-10 is safe for pure-float
#     groupby on a fixed input (no RNG, tiny per-genotype reductions).

#: ``atol`` for ``np.allclose`` on the Procrustes-aligned embedding coordinate matrices.
ATOL_PROCRUSTES = 1e-6

#: Minimum Adjusted Rand Index between recomputed and golden cluster labels.
ARI_FLOOR = 0.95

#: ``rtol`` for ``assert_frame_equal`` on the trait summary. Tighter than the project's
#: standing cross-OS ``rtol=1e-6`` policy because this comparison is same-stack pure-float
#: on a fixed input — see the reconciliation in ``docs/reproducibility.md``.
TRAIT_RTOL = 1e-10

#: Dependencies whose versions are stamped into the provenance record.
PROVENANCE_PACKAGES = [
    "numpy",
    "pandas",
    "umap-learn",
    "numba",
    "scipy",
    "scikit-learn",
]


def load_input() -> pd.DataFrame:
    """Load the reference input table, failing loudly if it is missing.

    Returns:
        The turface_19 post-QC final-data table.

    Raises:
        FileNotFoundError: If the committed input fixture is absent.
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Reference input not found: {INPUT_CSV}. "
            "Restore the fixture or regenerate via "
            "scripts/regenerate_numerical_stability_golden.py."
        )
    return pd.read_csv(INPUT_CSV)


def recompute_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the UMAP embedding for the reference dataset.

    Args:
        df: The reference input table.

    Returns:
        DataFrame with columns ``[ID_COL, "umap_1", "umap_2"]`` in input row order.
    """
    result = perform_umap_analysis(
        df, feature_cols=TRAIT_COLS, random_state=SEED, n_components=2
    )
    embedding = np.asarray(result["embedding"], dtype=float)
    return pd.DataFrame(
        {
            ID_COL: df[ID_COL].to_numpy(),
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
        }
    )


def recompute_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Recompute KMeans cluster labels for the reference dataset.

    Args:
        df: The reference input table.

    Returns:
        Tuple of (DataFrame with columns ``[ID_COL, "cluster_label"]``, the effective
        number of clusters reported by the clusterer).
    """
    result = perform_kmeans_clustering(
        df[TRAIT_COLS], n_clusters=N_CLUSTERS, random_state=SEED
    )
    labels = pd.DataFrame(
        {
            ID_COL: df[ID_COL].to_numpy(),
            "cluster_label": np.asarray(result["cluster_labels"], dtype=int),
        }
    )
    return labels, int(result["n_clusters"])


def recompute_trait_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the per-genotype trait-summary table (the CoW-sensitive path).

    Args:
        df: The reference input table.

    Returns:
        A genotype-indexed table whose columns are flattened ``"trait::agg"`` strings
        (flattened so the golden CSV round-trips without MultiIndex-header fragility).
    """
    summary = df.groupby(GENOTYPE_COL)[TRAIT_COLS].agg(TRAIT_AGGS)
    summary.columns = [f"{trait}::{agg}" for trait, agg in summary.columns]
    return summary


def align_embedding(emb: pd.DataFrame) -> np.ndarray:
    """Return embedding coordinates sorted by ``ID_COL`` for order-stable comparison.

    Args:
        emb: An embedding frame with ``[ID_COL, "umap_1", "umap_2"]``.

    Returns:
        An ``(N, 2)`` float array ordered by ``ID_COL``.
    """
    ordered = emb.sort_values(ID_COL)
    return ordered[["umap_1", "umap_2"]].to_numpy(dtype=float)


def align_labels(labels: pd.DataFrame) -> np.ndarray:
    """Return cluster labels sorted by ``ID_COL`` for order-stable comparison.

    Args:
        labels: A labels frame with ``[ID_COL, "cluster_label"]``.

    Returns:
        An ``(N,)`` int array ordered by ``ID_COL``.
    """
    ordered = labels.sort_values(ID_COL)
    return ordered["cluster_label"].to_numpy(dtype=int)
