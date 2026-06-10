"""Tests for the public cross_platform_pc_correlations workflow (issue #119)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze import (
    CrossPlatformPCResult,
    cross_platform_pc_correlations,
)


# ---------------------------------------------------------------------------
# Synthetic 3-platform fixture
# ---------------------------------------------------------------------------
def _make_platform(rng, genotypes, n_reps, n_traits):
    """Build a sample-level trait table: each genotype has n_reps noisy samples."""
    rows = []
    for g in genotypes:
        center = rng.normal(0, 5, n_traits)  # per-genotype trait signal
        for _ in range(n_reps):
            rows.append(
                {
                    "genotype": g,
                    **{f"t{i}": center[i] + rng.normal(0, 1) for i in range(n_traits)},
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def three_platforms():
    """Three platforms sharing 10 genotypes, with 3/4/5 retained PCs (47 tests)."""
    rng = np.random.default_rng(0)
    genotypes = [f"G{i:02d}" for i in range(10)]
    platforms = {
        "A": _make_platform(rng, genotypes, n_reps=4, n_traits=6),
        "B": _make_platform(rng, genotypes, n_reps=4, n_traits=7),
        "C": _make_platform(rng, genotypes, n_reps=4, n_traits=8),
    }
    trait_cols = {
        "A": [f"t{i}" for i in range(6)],
        "B": [f"t{i}" for i in range(7)],
        "C": [f"t{i}" for i in range(8)],
    }
    n_components = {"A": 3, "B": 4, "C": 5}
    return platforms, trait_cols, n_components


# ---------------------------------------------------------------------------
# Shape / API
# ---------------------------------------------------------------------------
def test_returns_result_with_expected_structure(three_platforms):
    platforms, trait_cols, n_components = three_platforms
    res = cross_platform_pc_correlations(
        platforms, trait_cols, n_components, random_state=0
    )

    assert isinstance(res, CrossPlatformPCResult)
    assert set(res.pca) == {"A", "B", "C"}
    assert set(res.pc_scores) == {"A", "B", "C"}
    # genotype-mean PC score matrices have the requested PC counts
    assert list(res.pc_scores["A"].columns) == ["PC1", "PC2", "PC3"]
    assert list(res.pc_scores["C"].columns) == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert res.common_genotypes == [f"G{i:02d}" for i in range(10)]


def test_public_api_exports():
    """Both names are importable from the package root and listed in __all__."""
    import sleap_roots_analyze as sra

    assert sra.cross_platform_pc_correlations is cross_platform_pc_correlations
    assert sra.CrossPlatformPCResult is CrossPlatformPCResult
    assert "cross_platform_pc_correlations" in sra.__all__
    assert "CrossPlatformPCResult" in sra.__all__


# ---------------------------------------------------------------------------
# Test count: 3x4 + 3x5 + 4x5 = 47 for a 3/4/5 PC configuration
# ---------------------------------------------------------------------------
def test_total_test_count_is_47_for_3_4_5(three_platforms):
    platforms, trait_cols, n_components = three_platforms
    res = cross_platform_pc_correlations(
        platforms, trait_cols, n_components, random_state=0
    )

    assert res.summary["n_tests"] == 47
    assert len(res.correlations) == 47
    assert res.summary["tests_per_pair"] == {
        "A_vs_B": 12,
        "A_vs_C": 15,
        "B_vs_C": 20,
    }


# ---------------------------------------------------------------------------
# CI + power present on every test
# ---------------------------------------------------------------------------
def test_every_test_has_ci_and_power(three_platforms):
    platforms, trait_cols, n_components = three_platforms
    res = cross_platform_pc_correlations(
        platforms, trait_cols, n_components, random_state=0
    )

    for col in ["ci_low", "ci_high", "power", "p_value", "p_value_fdr"]:
        assert col in res.correlations.columns
        assert res.correlations[col].notna().all(), f"{col} has NaN"
    assert (res.correlations["ci_low"] <= res.correlations["ci_high"]).all()
    assert ((res.correlations["power"] >= 0) & (res.correlations["power"] <= 1)).all()


# ---------------------------------------------------------------------------
# Pooled FDR is computed across ALL tests, not per pair
# ---------------------------------------------------------------------------
def test_fdr_is_pooled_across_all_tests(three_platforms):
    platforms, trait_cols, n_components = three_platforms
    res = cross_platform_pc_correlations(
        platforms, trait_cols, n_components, correction_method="fdr_bh", random_state=0
    )

    from statsmodels.stats.multitest import multipletests

    # Recompute FDR over the full pooled p-value vector and compare.
    expected_reject, expected_p_adj, _, _ = multipletests(
        res.correlations["p_value"].to_numpy(), alpha=0.05, method="fdr_bh"
    )
    np.testing.assert_allclose(
        res.correlations["p_value_fdr"].to_numpy(), expected_p_adj
    )
    assert res.summary["n_fdr_significant"] == int(expected_reject.sum())
    # A per-pair correction would generally differ from the pooled q-values; the
    # pooled q-value for a test must be >= its raw p-value (BH is monotone up).
    assert (res.correlations["p_value_fdr"] >= res.correlations["p_value"] - 1e-9).all()


# ---------------------------------------------------------------------------
# Ordering pin: genotype-mean PC scores == mean of sample-level PC scores
# (i.e. sample-level PCA THEN aggregate, never average-then-PCA).
# ---------------------------------------------------------------------------
def test_genotype_pc_means_are_aggregated_after_pca(three_platforms):
    platforms, trait_cols, n_components = three_platforms
    res = cross_platform_pc_correlations(
        platforms, trait_cols, n_components, random_state=0
    )

    # The data has no NaN, so the function's sample order == input row order.
    for name in platforms:
        k = n_components[name]
        sample_scores = np.asarray(res.pca[name]["transformed_data"])[:, :k]
        pc_cols = [f"PC{i + 1}" for i in range(k)]
        manual = pd.DataFrame(sample_scores, columns=pc_cols)
        manual["genotype"] = platforms[name]["genotype"].to_numpy()
        manual_means = manual.groupby("genotype")[pc_cols].mean()

        pd.testing.assert_frame_equal(
            res.pc_scores[name].loc[manual_means.index],
            manual_means,
            check_names=False,
        )


def test_pearson_signal_recovered(three_platforms):
    """A platform correlated with itself (renamed) yields a strong PC1-PC1 r."""
    platforms, trait_cols, n_components = three_platforms
    # Use the same underlying data for two platforms -> PC1 should track PC1.
    plats = {"A": platforms["A"], "B": platforms["A"].copy()}
    res = cross_platform_pc_correlations(
        plats,
        {"A": trait_cols["A"], "B": trait_cols["A"]},
        {"A": 2, "B": 2},
        method="pearson",
        random_state=0,
    )
    pc1_pc1 = res.correlations.query("pc1 == 'PC1' and pc2 == 'PC1'")
    assert abs(pc1_pc1["r"].iloc[0]) > 0.99  # identical inputs -> |r| ~ 1


# ---------------------------------------------------------------------------
# Edge case: disjoint genotypes must not raise
# ---------------------------------------------------------------------------
def test_disjoint_genotypes_do_not_raise():
    rng = np.random.default_rng(1)
    a = _make_platform(rng, [f"A{i}" for i in range(6)], n_reps=3, n_traits=5)
    b = _make_platform(rng, [f"B{i}" for i in range(6)], n_reps=3, n_traits=5)
    res = cross_platform_pc_correlations(
        {"A": a, "B": b},
        {"A": [f"t{i}" for i in range(5)], "B": [f"t{i}" for i in range(5)]},
        {"A": 2, "B": 2},
        random_state=0,
    )
    assert res.summary["n_genotypes"] == 0
    assert res.common_genotypes == []
    assert res.summary["n_fdr_significant"] == 0


def test_requires_two_platforms():
    rng = np.random.default_rng(2)
    a = _make_platform(rng, [f"G{i}" for i in range(5)], n_reps=3, n_traits=4)
    with pytest.raises(ValueError, match="At least two platforms"):
        cross_platform_pc_correlations(
            {"A": a}, {"A": [f"t{i}" for i in range(4)]}, {"A": 2}
        )


# ---------------------------------------------------------------------------
# Wheat EDPIE golden regression (skip-guarded on fixture presence, issue #120)
# ---------------------------------------------------------------------------
_WHEAT_DIR = (
    Path(__file__).parent / "fixtures" / "real" / "wheat_edpie" / "inputs" / "post_qc"
)
_WHEAT_FILES = {
    "Turface": _WHEAT_DIR / "turface_19_final_data.csv",
    "Cylinder": _WHEAT_DIR / "cylinder_edpie_final_data.csv",
    "Field": _WHEAT_DIR / "field_root_core_final_data.csv",
}
_WHEAT_N_PCS = {"Turface": 3, "Cylinder": 4, "Field": 5}
_WHEAT_METADATA_COLS = {"Barcode", "Genotype", "Replicate"}


@pytest.mark.skipif(
    not all(p.exists() for p in _WHEAT_FILES.values()),
    reason="wheat EDPIE post-QC fixture not present (issue #120)",
)
def test_wheat_edpie_golden_47_tests_19_genotypes_0_fdr():
    """Reproduce the paper headline numbers: 47 tests, 19 genotypes, 0 FDR-significant."""
    platforms = {name: pd.read_csv(path) for name, path in _WHEAT_FILES.items()}
    # The QC'd trait columns are complete; a handful of numeric metadata columns
    # carry scattered NaN, so restrict to complete numeric non-metadata columns.
    trait_cols = {
        name: [
            c
            for c in df.columns
            if c not in _WHEAT_METADATA_COLS
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().all()
        ]
        for name, df in platforms.items()
    }

    res = cross_platform_pc_correlations(
        platforms,
        trait_cols,
        _WHEAT_N_PCS,
        genotype_col="Genotype",
        method="spearman",
        correction_method="fdr_bh",
        random_state=42,
    )

    assert res.summary["n_tests"] == 47
    assert res.summary["n_genotypes"] == 19
    assert res.summary["n_fdr_significant"] == 0
