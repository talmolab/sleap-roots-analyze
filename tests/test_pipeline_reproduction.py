"""Per-stage reproduction tests for the wheat-EDPIE pipeline fixtures (#120).

These tests assert the committed golden fixtures under ``tests/fixtures/`` across the
four EDPIE platforms (``turface_19``, ``turface_150``, ``cylinder``, ``root_core``) for
the three pipeline stages (QC -> viz -> cross-platform), plus harness-config validity.
Numeric comparisons follow the tolerance policy in ``tests/fixtures/README.md`` /
``docs/reproducibility.md`` (#118): integers/rosters exact, floats with
``rtol=1e-6, atol=1e-9``. Analysis-input contract conformance lives in a follow-up PR
(it depends on the unreleased ``sleap-roots-contracts``).

See ``tests/fixtures.py`` for the ``scope="session"`` loaders these tests share.
"""

from __future__ import annotations

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

PLATFORMS = ["turface_19", "turface_150", "cylinder", "root_core"]
# root_core's viz run produced no UMAP embedding, so it is excluded from UMAP tests.
UMAP_PLATFORMS = ["turface_19", "turface_150", "cylinder"]

# Harness config stems per platform (committed under harness/{qc,viz}/).
HARNESS_QC = {
    "turface_19": "qc_turface_19genotypes",
    "turface_150": "qc_turface_150genotypes",
    "cylinder": "qc_cylinder_edpie",
    "root_core": "qc_root_core_edpie",
}
HARNESS_VIZ = {
    "turface_19": "viz_turface_19genotypes",
    "turface_150": "viz_turface_150genotypes",
    "cylinder": "viz_cylinder_edpie",
    "root_core": "viz_root_coring",
}

# Explicit golden table — the headline values for each platform, kept visible here so a
# fixture corruption (or a method-change drift) fails loudly. Values come from the EDPIE
# paper run's committed golden artifacts.
EXPECTED = {
    "turface_19": dict(
        n_samples=153, threshold=0.6, retained=8, removed=8, outliers=5, anova_sig=8
    ),
    "turface_150": dict(
        n_samples=886, threshold=0.4, retained=13, removed=2, outliers=39, anova_sig=13
    ),
    "cylinder": dict(
        n_samples=123,
        threshold=0.6,
        retained=588,
        removed=231,
        outliers=6,
        anova_sig=587,
    ),
    "root_core": dict(
        n_samples=58, threshold=0.5, retained=24, removed=11, outliers=2, anova_sig=24
    ),
}

# Cross-platform golden pairings (the 4 EDPIE manifest pairings).
CROSS_PAIRINGS = [
    "turface_150_vs_turface_19",
    "turface_19_vs_cylinder",
    "root_core_vs_turface_19",
    "root_core_vs_cylinder",
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


@pytest.mark.parametrize("platform", PLATFORMS)
def test_platform_golden_dirs_present(edpie_real_dir, platform):
    """Each platform has its QC + viz golden directories and post-QC input."""
    assert (edpie_real_dir / "expected" / "qc" / platform).is_dir()
    assert (edpie_real_dir / "expected" / "viz" / platform).is_dir()
    assert (
        edpie_real_dir / "inputs" / "post_qc" / f"{platform}_final_data.csv"
    ).is_file()
    assert (edpie_real_dir / "inputs" / "raw" / platform).is_dir()


def test_curation_excludes_non_assertable_artifacts(repro_fixtures_dir):
    """No run logs / source tarballs / oversized stage summaries are committed."""
    banned = {"pipeline.log", "viz_pipeline.log", "code_snapshot.tar.gz"}
    offenders = [p for p in repro_fixtures_dir.rglob("*") if p.name in banned]
    assert not offenders, f"non-assertable artifacts committed: {offenders}"
    # The oversized per-stage QC/viz summaries (52 MB cylinder / 13 MB turface_150)
    # are excluded; compact viz_pca_metadata.json + viz_umap_embedding.csv replace them.
    expected = repro_fixtures_dir / "real" / "wheat_edpie" / "expected"
    for stage in ("qc", "viz"):
        big = list((expected / stage).rglob("pipeline_summary.json"))
        assert not big, f"{stage} pipeline_summary.json should be excluded: {big}"


# ---------------------------------------------------------------------------
# Harness config validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_harness_qc_config_valid(harness_dir, platform):
    """Each committed QC harness config passes structural validation."""
    cfg = load_qc_config(harness_dir / "qc" / f"{HARNESS_QC[platform]}.yaml")
    validate_qc_config(cfg, check_files=False)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_harness_viz_config_valid(harness_dir, platform):
    """Each committed viz harness config passes validation."""
    cfg = load_viz_config(harness_dir / "viz" / f"{HARNESS_VIZ[platform]}.yaml")
    validate_viz_config(cfg)


# ---------------------------------------------------------------------------
# QC stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_qc_final_data_shape_and_roles(final_data_by_platform, platform):
    """Post-QC 10_final_data has the golden sample count, roles, and a numeric trait."""
    df = final_data_by_platform[platform]
    assert df.shape[0] == EXPECTED[platform]["n_samples"]
    for role in ("Barcode", "Genotype", "Replicate"):
        assert role in df.columns
    assert len(df.select_dtypes(include="number").columns) >= 1


@pytest.mark.parametrize("platform", PLATFORMS)
def test_qc_heritability_filter_golden(qc_heritability_by_platform, platform):
    """QC heritability filter retains/removes the golden trait counts at threshold."""
    s = qc_heritability_by_platform[platform]
    exp = EXPECTED[platform]
    assert s["filtering_enabled"] is True
    assert s["threshold"] == exp["threshold"]
    assert s["traits_retained"] == exp["retained"]
    assert s["traits_removed"] == exp["removed"]
    assert len(s["removed_trait_names"]) == exp["removed"]
    assert 0.0 <= s["mean_heritability_retained"] <= 1.0


@pytest.mark.parametrize("platform", PLATFORMS)
def test_qc_removed_outliers_count(qc_removed_counts_by_platform, platform):
    """The golden number of samples were removed as outliers."""
    assert (
        qc_removed_counts_by_platform[platform]["outliers"]
        == EXPECTED[platform]["outliers"]
    )


# ---------------------------------------------------------------------------
# Viz stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_viz_summary_golden(
    viz_summary_by_platform, qc_heritability_by_platform, platform
):
    """Viz summary headline metrics match golden and agree with the QC stage."""
    s = viz_summary_by_platform[platform]
    exp = EXPECTED[platform]
    assert s["n_samples"] == exp["n_samples"]
    assert s["n_traits_final"] == exp["retained"]
    assert s["results"]["anova"]["n_significant"] == exp["anova_sig"]
    # Cross-stage consistency: viz mean H2 == QC retained mean H2.
    assert np.isclose(
        s["results"]["heritability"]["mean_h2"],
        qc_heritability_by_platform[platform]["mean_heritability_retained"],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("platform", PLATFORMS)
def test_viz_pca_reproduction(final_data_by_platform, viz_pca_by_platform, platform):
    """Re-running PCA on post-QC data reproduces the golden explained variance.

    The eigenvalue spectrum is deterministic, so summing the first
    ``n_pca_components`` explained-variance ratios from a fresh ``perform_pca_analysis``
    must equal the committed golden ``pca_explained_variance`` within ``rtol=1e-6``.
    (The pipeline's own component-selection rule is not re-derived here; the golden
    component count is used to index the reproduced spectrum.)
    """
    golden = viz_pca_by_platform[platform]
    n = golden["n_pca_components"]
    res = perform_pca_analysis(
        final_data_by_platform[platform][golden["trait_cols"]],
        standardize=True,
        explained_variance_threshold=0.95,
        random_state=42,
    )
    reproduced = float(np.sum(np.asarray(res["explained_variance_ratio"])[:n]))
    assert np.isclose(
        reproduced, golden["pca_explained_variance"], rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("platform", UMAP_PLATFORMS)
def test_viz_umap_embedding_structural(viz_umap_by_platform, platform):
    """UMAP golden embedding has the expected shape and is finite.

    UMAP coordinates are the most environment-sensitive output (numba/BLAS across
    OSes), so cross-platform CI asserts shape + finiteness, not exact coordinates.
    """
    emb = np.asarray(viz_umap_by_platform[platform], dtype=float)
    assert emb.shape == (EXPECTED[platform]["n_samples"], 2)
    assert np.isfinite(emb).all()


# ---------------------------------------------------------------------------
# Cross-platform stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pairing", CROSS_PAIRINGS)
def test_crossplatform_correlations_structure(crossplatform_dir, pairing):
    """Each cross-platform correlations table has the golden structure."""
    import pandas as pd

    corr = pd.read_csv(crossplatform_dir / pairing / "cross_platform_correlations.csv")
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
    assert corr["spearman_r"].dropna().between(-1.0, 1.0).all()
    assert (corr["n_genotypes"] > 0).all()


@pytest.mark.parametrize("pairing", CROSS_PAIRINGS)
def test_crossplatform_alignment_consistency(crossplatform_dir, pairing):
    """Alignment summary lists shared genotypes with positive per-experiment counts."""
    import pandas as pd

    align = pd.read_csv(
        crossplatform_dir / pairing / "cross_platform_alignment_summary.csv"
    )
    assert "genotype" in align.columns
    assert len(align) > 0
    assert (align["exp1_samples"] > 0).all()
    assert (align["exp2_samples"] > 0).all()


# Analysis-input contract conformance (post-QC canonicalization + canonical examples)
# is intentionally NOT in this PR: it depends on sleap-roots-contracts (unreleased) and
# is covered by the follow-up "analysis-input contract conformance" PR. This module
# imports no contracts package — it is the reproduction harness, mergeable on its own.
# The post-QC *_final_data.csv fixtures stay committed here; the follow-up reuses them
# (canonicalizing a copy, never the frame that feeds QC/viz/cross-platform).
