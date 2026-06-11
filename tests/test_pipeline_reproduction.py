"""Per-stage reproduction tests for the wheat-EDPIE pipeline fixtures (#120).

These tests assert the committed golden fixtures under ``tests/fixtures/`` for the
``turface_19`` platform across the three pipeline stages (QC -> viz -> cross-platform),
plus harness-config validity. Numeric comparisons follow the tolerance policy in
``tests/fixtures/README.md`` / ``docs/reproducibility.md`` (#118): integers/rosters
exact, floats with ``rtol=1e-6, atol=1e-9``. Analysis-input contract conformance lives
in a follow-up PR (it depends on the unreleased ``sleap-roots-contracts``).

See ``tests/fixtures.py`` for the ``scope="session"`` loaders these tests share.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sleap_roots_analyze.pca import perform_pca_analysis
from sleap_roots_analyze.pipeline.config.utils import (
    load_qc_config,
    load_viz_config,
    validate_qc_config,
    validate_viz_config,
)

# Numeric tolerance per docs/reproducibility.md (#118).
RTOL = 1e-6
ATOL = 1e-9

# Golden trait roster used by the turface_19 viz PCA step (8 high-H2 traits).
PCA_TRAITS = [
    "Holes",
    "Network Area (mm²)",
    "Perimeter (mm)",
    "Root Biomass (mg)",
    "Shoot Biomass (mg)",
    "Surface Area (mm²)",
    "Total Root Length (mm)",
    "Volume (mm³)",
]


# ---------------------------------------------------------------------------
# Fixture tree layout + curation policy
# ---------------------------------------------------------------------------


def test_fixture_tree_present(repro_fixtures_dir):
    """The reproduction fixture tree exists with all required sub-trees."""
    assert (repro_fixtures_dir / "README.md").is_file()
    assert (repro_fixtures_dir / "harness" / "run_manifest.yaml").is_file()
    for sub in ("harness/qc", "harness/viz", "harness/cross_platform"):
        assert (repro_fixtures_dir / sub).is_dir(), sub
    edpie = repro_fixtures_dir / "real" / "wheat_edpie"
    assert (edpie / "inputs").is_dir()
    assert (edpie / "expected").is_dir()


def test_curation_excludes_non_assertable_artifacts(repro_fixtures_dir):
    """No run logs / source tarballs are committed (curation policy)."""
    banned = {"pipeline.log", "viz_pipeline.log", "code_snapshot.tar.gz"}
    offenders = [p for p in repro_fixtures_dir.rglob("*") if p.name in banned]
    assert not offenders, f"non-assertable artifacts committed: {offenders}"


# ---------------------------------------------------------------------------
# Harness config validity
# ---------------------------------------------------------------------------


def test_harness_qc_config_valid(harness_dir):
    """The committed turface_19 QC harness config passes structural validation."""
    cfg = load_qc_config(harness_dir / "qc" / "qc_turface_19genotypes.yaml")
    validate_qc_config(cfg, check_files=False)


def test_harness_viz_config_valid(harness_dir):
    """The committed turface_19 viz harness config passes validation."""
    cfg = load_viz_config(harness_dir / "viz" / "viz_turface_19genotypes.yaml")
    validate_viz_config(cfg)


# ---------------------------------------------------------------------------
# QC stage
# ---------------------------------------------------------------------------


def test_qc_final_data_shape_and_roles(turface19_final_data):
    """Post-QC 10_final_data has the golden shape and required role columns."""
    df = turface19_final_data
    assert df.shape == (153, 15)
    for role in ("Barcode", "Genotype", "Replicate"):
        assert role in df.columns
    numeric = df.select_dtypes(include="number").columns
    assert len(numeric) >= 1


def test_qc_heritability_filter_golden(turface19_qc_heritability_summary):
    """QC heritability filter retains/removes the golden traits at threshold 0.6."""
    s = turface19_qc_heritability_summary
    assert s["filtering_enabled"] is True
    assert s["threshold"] == 0.6
    assert s["traits_original"] == 16
    assert s["traits_retained"] == 8
    assert s["traits_removed"] == 8
    assert len(s["removed_trait_names"]) == 8
    assert np.isclose(
        s["mean_heritability_retained"], 0.7650052743157677, rtol=RTOL, atol=ATOL
    )


def test_qc_removed_outliers_count(turface19_qc_removed_outliers):
    """Exactly 5 samples were removed as outliers (golden)."""
    assert len(turface19_qc_removed_outliers) == 5


def test_qc_no_traits_or_samples_removed_in_cleanup(
    turface19_qc_removed_traits, turface19_qc_removed_samples
):
    """Cleanup removed no traits and no samples for turface_19 (golden)."""
    assert len(turface19_qc_removed_traits) == 0
    assert len(turface19_qc_removed_samples) == 0


# ---------------------------------------------------------------------------
# Viz stage
# ---------------------------------------------------------------------------


def test_viz_summary_golden(turface19_viz_summary, turface19_qc_heritability_summary):
    """Viz summary headline metrics match golden and agree with the QC stage."""
    s = turface19_viz_summary
    assert s["n_samples"] == 153
    assert s["n_traits_final"] == 8
    assert s["results"]["anova"]["n_significant"] == 8
    viz_mean_h2 = s["results"]["heritability"]["mean_h2"]
    assert np.isclose(viz_mean_h2, 0.7650052743157678, rtol=RTOL, atol=ATOL)
    # Cross-stage consistency: viz mean H2 == QC retained mean H2.
    assert np.isclose(
        viz_mean_h2,
        turface19_qc_heritability_summary["mean_heritability_retained"],
        rtol=RTOL,
        atol=ATOL,
    )


def test_viz_pca_reproduction(turface19_final_data, turface19_viz_pca_metadata):
    """Re-running PCA on post-QC data reproduces the golden explained variance.

    This is the genuine per-stage reproduction assertion: feed the stage its input
    (10_final_data restricted to the 8 PCA traits) through ``perform_pca_analysis``
    and compare to the committed golden within ``rtol=1e-6``.
    """
    golden = turface19_viz_pca_metadata
    assert golden["trait_cols"] == PCA_TRAITS
    X = turface19_final_data[PCA_TRAITS]
    res = perform_pca_analysis(
        X, standardize=True, explained_variance_threshold=0.95, random_state=42
    )
    assert res["n_components_selected"] == golden["n_pca_components"] == 3
    reproduced_ev = float(np.sum(res["explained_variance_ratio"][:3]))
    assert np.isclose(
        reproduced_ev, golden["pca_explained_variance"], rtol=RTOL, atol=ATOL
    )


def test_viz_umap_embedding_structural(turface19_viz_umap_embedding):
    """UMAP golden embedding has the expected shape and is finite.

    UMAP coordinates are the most environment-sensitive output (numba/BLAS across
    OSes), so cross-platform CI asserts shape + finiteness, not exact coordinates.
    """
    emb = np.asarray(turface19_viz_umap_embedding, dtype=float)
    assert emb.shape == (153, 2)
    assert np.isfinite(emb).all()


# ---------------------------------------------------------------------------
# Cross-platform stage
# ---------------------------------------------------------------------------

CROSS_PAIRINGS = [
    "turface_150_vs_turface_19",
    "turface_19_vs_cylinder",
    "root_core_vs_turface_19",
]


@pytest.mark.parametrize("pairing", CROSS_PAIRINGS)
def test_crossplatform_correlations_structure(turface19_crossplatform_dir, pairing):
    """Each turface_19 cross-platform correlations table has the golden structure."""
    import pandas as pd

    corr = pd.read_csv(
        turface19_crossplatform_dir / pairing / "cross_platform_correlations.csv"
    )
    required = {
        "exp1_trait",
        "exp2_trait",
        "spearman_r",
        "spearman_p",
        "pearson_r",
        "pearson_p",
        "n_genotypes",
        "significant_fdr",
    }
    assert required.issubset(corr.columns)
    assert len(corr) > 0
    finite_sp = corr["spearman_r"].dropna()
    assert finite_sp.between(-1.0, 1.0).all()
    assert (corr["n_genotypes"] > 0).all()


@pytest.mark.parametrize("pairing", CROSS_PAIRINGS)
def test_crossplatform_alignment_consistency(turface19_crossplatform_dir, pairing):
    """Alignment summary lists shared genotypes with positive per-experiment counts."""
    import pandas as pd

    align = pd.read_csv(
        turface19_crossplatform_dir / pairing / "cross_platform_alignment_summary.csv"
    )
    assert "genotype" in align.columns
    assert len(align) > 0
    assert (align["exp1_samples"] > 0).all()
    assert (align["exp2_samples"] > 0).all()


# Analysis-input contract conformance (post-QC canonicalization + canonical examples)
# is intentionally NOT in this PR: it depends on sleap-roots-contracts (unreleased) and
# is covered by the follow-up "analysis-input contract conformance" PR. This module
# imports no contracts package — it is the reproduction harness, mergeable on its own.
# The post-QC turface_19_final_data.csv fixture stays committed here; the follow-up
# reuses it (canonicalizing a copy, never the frame that feeds QC/viz/cross-platform).
