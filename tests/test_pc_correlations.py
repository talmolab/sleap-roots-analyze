"""Tests for the PC-correlation and trait-enrichment workflows (issue #119).

Coverage is split into three layers:

- DAG nodes (aggregate / correlate / enrichment) tested directly on small
  in-memory data.
- The two public orchestrators, each driven from a synthetic on-disk fixture
  (a fake pipeline-run directory; fake ``cross_platform_correlations.csv``
  files), asserting the DAG *through its outputs*.
- Workflow independence: the trait-enrichment workflow runs from correlation
  CSVs alone, with no PC-workflow outputs present.

The synthetic-fixture assertions are genuine re-computations (the workflow runs
end to end), not golden pinning; the wheat-EDPIE golden numbers are pinned
separately in the skip-guarded regression below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze import (
    CrossPlatformPCResult,
    EnrichmentResult,
    cross_platform_pc_correlations,
    trait_correlation_enrichment,
)
from sleap_roots_analyze.pc_correlations.aggregate import (
    aggregate_pc_scores_by_genotype,
    align_genotypes_across_platforms,
)
from sleap_roots_analyze.pc_correlations.correlate import (
    calculate_all_platform_correlations,
    summarize_correlations,
)
from sleap_roots_analyze.pc_correlations.enrichment import (
    calculate_all_enrichment_tests,
    calculate_enrichment_test,
)

# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------
_PLATFORM_PCS = {"A": 3, "B": 4, "C": 5}  # 3*4 + 3*5 + 4*5 = 47 tests


def _make_sample_pc_scores(rng, genotypes, n_reps, n_pc_cols):
    """Build sample-level PC scores + aligned genotype labels for one platform."""
    rows, genos = [], []
    for g in genotypes:
        center = rng.normal(0, 5, n_pc_cols)
        for _ in range(n_reps):
            rows.append(center + rng.normal(0, 1, n_pc_cols))
            genos.append(g)
    cols = [f"PC{i + 1}" for i in range(n_pc_cols)]
    pc_scores = pd.DataFrame(rows, columns=cols)
    pc_scores.index = [f"s{i}" for i in range(len(pc_scores))]
    return pc_scores, pd.Series(genos, index=pc_scores.index)


def _write_platform(run_dir, name, pc_scores, genotypes):
    """Write a platform's pca/ dir + QC final_data; return its config entry."""
    pca_dir = run_dir / name / "pca"
    pca_dir.mkdir(parents=True)
    pc_scores.to_csv(pca_dir / "pc_scores.csv")
    # Minimal loadings / explained variance so load_pc_scores succeeds.
    pd.DataFrame(
        np.eye(pc_scores.shape[1]),
        index=[f"trait{i}" for i in range(pc_scores.shape[1])],
        columns=pc_scores.columns,
    ).to_csv(pca_dir / "loadings.csv")
    pd.DataFrame(
        {
            "explained_variance_ratio": np.full(
                pc_scores.shape[1], 1.0 / pc_scores.shape[1]
            )
        },
        index=pc_scores.columns,
    ).to_csv(pca_dir / "explained_variance.csv")

    final_data = run_dir / name / "10_final_data.csv"
    df = pd.DataFrame({"Genotype": genotypes.to_numpy()})
    df["trait0"] = np.arange(len(df), dtype=float)
    df.to_csv(final_data, index=False)

    return {
        "pca_dir": f"{name}/pca",
        "final_data": f"{name}/10_final_data.csv",
        "genotype_col": "Genotype",
        "n_pcs": _PLATFORM_PCS[name],
    }


@pytest.fixture
def synthetic_pipeline_run(tmp_path):
    """A 3-platform synthetic pipeline-run dir sharing 10 genotypes."""
    rng = np.random.default_rng(0)
    genotypes = [f"G{i:02d}" for i in range(10)]
    run_dir = tmp_path / "run"
    config = {}
    for name in ("A", "B", "C"):
        pcs, genos = _make_sample_pc_scores(rng, genotypes, n_reps=4, n_pc_cols=6)
        config[name] = _write_platform(run_dir, name, pcs, genos)
    return run_dir, config


def _write_corr_csv(path, n_tests, n_sig, seed):
    """Write a synthetic cross_platform_correlations.csv with n_sig p<0.05 rows."""
    rng = np.random.default_rng(seed)
    p = np.concatenate(
        [rng.uniform(0, 0.01, n_sig), rng.uniform(0.2, 1.0, n_tests - n_sig)]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trait1": [f"t{i}" for i in range(n_tests)],
            "trait2": [f"u{i}" for i in range(n_tests)],
            "spearman_r": rng.uniform(-1, 1, n_tests),
            "spearman_p": p,
        }
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# DAG node: aggregate
# ---------------------------------------------------------------------------
def test_genotype_mean_equals_mean_of_sample_pc_scores():
    pc_scores = pd.DataFrame(
        {"PC1": [1.0, 3.0, 10.0, 20.0], "PC2": [0.0, 2.0, 5.0, 7.0]},
        index=["a", "b", "c", "d"],
    )
    genotypes = pd.Series(["g1", "g1", "g2", "g2"], index=pc_scores.index)
    means = aggregate_pc_scores_by_genotype(pc_scores, genotypes)
    assert means.loc["g1", "PC1"] == 2.0
    assert means.loc["g2", "PC1"] == 15.0
    assert means.loc["g1", "n_samples"] == 2


def test_align_returns_common_genotypes():
    pd1 = {"genotype_means": pd.DataFrame({"PC1": [1, 2, 3]}, index=["g1", "g2", "g3"])}
    pd2 = {"genotype_means": pd.DataFrame({"PC1": [4, 5]}, index=["g2", "g3"])}
    common, aligned = align_genotypes_across_platforms({"A": pd1, "B": pd2})
    assert common == ["g2", "g3"]
    assert list(aligned["A"].index) == ["g2", "g3"]


# ---------------------------------------------------------------------------
# DAG node: correlate
# ---------------------------------------------------------------------------
@pytest.fixture
def aligned_three_platforms():
    rng = np.random.default_rng(1)
    genos = [f"G{i:02d}" for i in range(10)]
    aligned = {}
    for name in ("A", "B", "C"):
        k = _PLATFORM_PCS[name]
        aligned[name] = pd.DataFrame(
            rng.normal(0, 1, (len(genos), k)),
            index=genos,
            columns=[f"PC{i + 1}" for i in range(k)],
        )
    return aligned


def test_total_test_count_is_47_for_3_4_5(aligned_three_platforms):
    df = calculate_all_platform_correlations(
        aligned_three_platforms, _PLATFORM_PCS, fdr_methods=["fdr_by"]
    )
    assert len(df) == 47
    summary = summarize_correlations(df, primary_method="fdr_by")
    assert summary["tests_per_pair"] == {"A_vs_B": 12, "A_vs_C": 15, "B_vs_C": 20}


def test_combined_and_per_pair_columns_distinct(aligned_three_platforms):
    df = calculate_all_platform_correlations(
        aligned_three_platforms, _PLATFORM_PCS, fdr_methods=["fdr_by"], fdr_scope="both"
    )
    assert "significant_combined_fdr_by" in df.columns
    assert "significant_per_pair_fdr_by" in df.columns
    for col in ("ci_lower", "ci_upper", "power"):
        assert col in df.columns


def test_combined_fdr_is_pooled_across_all_tests(aligned_three_platforms):
    from statsmodels.stats.multitest import multipletests

    df = calculate_all_platform_correlations(
        aligned_three_platforms, _PLATFORM_PCS, fdr_methods=["fdr_by"], fdr_scope="both"
    )
    _, expected_p_adj, _, _ = multipletests(
        df["spearman_p"].to_numpy(), alpha=0.05, method="fdr_by"
    )
    np.testing.assert_allclose(df["p_adj_combined_fdr_by"].to_numpy(), expected_p_adj)


# ---------------------------------------------------------------------------
# DAG node: enrichment
# ---------------------------------------------------------------------------
def test_enrichment_depletion_detected():
    # Far fewer than expected (1331 * 0.05 ~ 66) -> depleted.
    res = calculate_enrichment_test(n_significant=26, n_tests=1331)
    assert res.interpretation == "depleted"
    assert res.p_value_depletion < 0.05
    assert res.fold_enrichment < 1


def test_enrichment_validation_rejects_bad_counts():
    with pytest.raises(ValueError):
        calculate_enrichment_test(n_significant=5, n_tests=3)


def test_all_enrichment_tests_combined_pools_counts(tmp_path):
    files = {}
    for i, (n, sig) in enumerate([(20, 1), (15, 0), (12, 0)]):
        p = tmp_path / f"pair{i}.csv"
        _write_corr_csv(p, n, sig, seed=i)
        files[f"pair{i}"] = p
    results = calculate_all_enrichment_tests(files)
    assert results[0].platform_pair == "Combined"
    assert results[0].n_tests == 47  # 20 + 15 + 12
    assert len(results) == 4  # combined + 3 pairs


# ---------------------------------------------------------------------------
# Workflow 1: PC-level correlations (synthetic pipeline-run dir)
# ---------------------------------------------------------------------------
def test_pc_workflow_produces_artifacts_and_matching_dict(synthetic_pipeline_run):
    run_dir, config = synthetic_pipeline_run
    out = run_dir.parent / "pc_out"
    result = cross_platform_pc_correlations(
        pipeline_run=run_dir,
        platform_config=config,
        output_dir=out,
        make_figures=False,
    )
    # Typed result reflects the DAG.
    assert isinstance(result, CrossPlatformPCResult)
    assert result.summary["n_tests"] == 47
    assert result.summary["n_genotypes"] == 10
    assert result.common_genotypes == [f"G{i:02d}" for i in range(10)]
    # Artifacts on disk match the returned table.
    corr_path = Path(result.output_paths["correlations"])
    assert corr_path.exists()
    assert len(pd.read_csv(corr_path)) == 47
    assert (out / "metadata.json").exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["results"]["n_tests"] == 47


def test_pc_workflow_writes_figures_when_requested(synthetic_pipeline_run):
    run_dir, config = synthetic_pipeline_run
    out = run_dir.parent / "pc_out_fig"
    result = cross_platform_pc_correlations(
        run_dir, config, out, fdr_scope="both", make_figures=True
    )
    figures_dir = Path(result.output_paths["figures_dir"])
    assert (figures_dir / "sensitivity_analysis.png").exists()
    assert (figures_dir / "combined" / "correlation_summary.png").exists()


# ---------------------------------------------------------------------------
# Workflow 2: trait enrichment (synthetic CSVs only) + independence
# ---------------------------------------------------------------------------
def test_enrichment_workflow_runs_from_csv_fixtures_only(tmp_path):
    files = {}
    for i, (n, sig) in enumerate([(20, 0), (15, 0), (12, 0)]):
        p = tmp_path / "cross_platform" / f"pair{i}" / "cross_platform_correlations.csv"
        _write_corr_csv(p, n, sig, seed=10 + i)
        files[f"pair{i}"] = p
    out = tmp_path / "enrich_out"

    result = trait_correlation_enrichment(files, out, make_figure=False)

    assert isinstance(result["results"][0], EnrichmentResult)
    csv_path = Path(result["output_paths"]["enrichment_results"])
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == len(result["results"])
    assert (out / "metadata.json").exists()


def test_enrichment_independent_of_pc_workflow(tmp_path):
    """Enrichment must not require any PC-workflow output to exist."""
    p = tmp_path / "cross_platform_correlations.csv"
    _write_corr_csv(p, n_tests=30, n_sig=0, seed=99)
    out = tmp_path / "enrich_only"
    # No pipeline-run dir, no PC-workflow artifacts anywhere.
    result = trait_correlation_enrichment({"X vs Y": p}, out, make_figure=False)
    assert result["results"][0].platform_pair == "Combined"
    assert not (out / "data" / "correlations.csv").exists()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def test_public_api_exports():
    import sleap_roots_analyze as sra

    for name in (
        "cross_platform_pc_correlations",
        "trait_correlation_enrichment",
        "CrossPlatformPCResult",
        "EnrichmentResult",
    ):
        assert name in sra.__all__
        assert hasattr(sra, name)


def test_pc_workflow_rejects_unknown_fdr_method(synthetic_pipeline_run):
    """PC FDR validation is its own (broader than CrossPlatformConfig)."""
    run_dir, config = synthetic_pipeline_run
    out = run_dir.parent / "bad_fdr"
    with pytest.raises(ValueError, match="Unsupported FDR method"):
        cross_platform_pc_correlations(
            run_dir, config, out, fdr_methods=["fdr_by", "not_a_method"]
        )


def test_pc_workflow_accepts_bonferroni(synthetic_pipeline_run):
    """bonferroni is valid for PC FDR even though CrossPlatformConfig forbids it."""
    run_dir, config = synthetic_pipeline_run
    out = run_dir.parent / "bonf"
    result = cross_platform_pc_correlations(
        run_dir, config, out, fdr_methods=["bonferroni"], make_figures=False
    )
    assert "significant_combined_bonferroni" in result.correlations.columns


# ---------------------------------------------------------------------------
# Edge cases: messy data must NOT crash (NaN/empty + survive, never raise)
# ---------------------------------------------------------------------------
def test_pc_workflow_zero_common_genotypes_does_not_crash(tmp_path):
    """No shared genotypes -> NaN summary, not a crash (min-detectable n<4)."""
    rng = np.random.default_rng(7)
    run_dir = tmp_path / "run"
    config = {}
    # Two platforms with entirely disjoint genotype panels.
    for name, gset in (("A", range(0, 6)), ("B", range(100, 106))):
        pcs, genos = _make_sample_pc_scores(
            rng, [f"G{i:02d}" for i in gset], n_reps=3, n_pc_cols=4
        )
        config[name] = _write_platform(run_dir, name, pcs, genos)
    config["A"]["n_pcs"] = 2
    config["B"]["n_pcs"] = 2

    out = tmp_path / "pc_out"
    result = cross_platform_pc_correlations(run_dir, config, out, make_figures=False)

    assert result.summary["n_genotypes"] == 0
    assert result.common_genotypes == []
    assert np.isnan(result.summary["min_detectable_r_80_power"])
    # Per-PC rows still produced (all-NaN correlations), and FDR is all-NaN.
    assert result.summary["n_significant_combined"] == 0
    assert (tmp_path / "pc_out" / "metadata.json").exists()


def test_summarize_correlations_small_n_returns_nan_mdr():
    from sleap_roots_analyze.pc_correlations.correlate import summarize_correlations

    # Empty table -> n_tests 0, n_genotypes 0, NaN MDR, no crash.
    summary = summarize_correlations(pd.DataFrame())
    assert summary["n_tests"] == 0
    assert summary["n_genotypes"] == 0
    assert np.isnan(summary["min_detectable_r_80_power"])


def test_degenerate_single_genotype_correlation_is_nan():
    from sleap_roots_analyze.pc_correlations.correlate import (
        calculate_all_platform_correlations,
    )

    one = pd.DataFrame({"PC1": [1.0], "PC2": [2.0]}, index=["g1"])
    df = calculate_all_platform_correlations(
        {"A": one, "B": one}, {"A": 2, "B": 2}, fdr_methods=["fdr_by"]
    )
    # n_genotypes < 2 -> correlations are NaN, not a spurious value or a crash.
    assert df["spearman_r"].isna().all()
    assert df["spearman_p"].isna().all()


def test_count_significant_excludes_nan_p_rows():
    from sleap_roots_analyze.pc_correlations.enrichment import count_significant

    df = pd.DataFrame({"spearman_p": [0.01, 0.5, np.nan, np.nan, 0.04]})
    n_sig, n_tests = count_significant(df, alpha=0.05)
    assert n_tests == 3  # NaN rows excluded from the denominator
    assert n_sig == 2


# ---------------------------------------------------------------------------
# Wheat EDPIE PC-correlation golden regression (pending issue #120 fixtures)
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="PENDING issue #120: needs all 3 platforms' sample-level PCA outputs "
    "(pc_scores.csv) + QC final_data; only the turface_19 slice ships today. "
    "Wired to real 47/19/0 assertions in a follow-up — this is a known-pending "
    "placeholder, not live coverage."
)
def test_wheat_edpie_pc_golden_47_19_0_pending_120():
    """Reproduce 47 PC tests / 19 genotypes / 0 combined-fdr_by-significant.

    Decorator-level skip so it is visible as a known-pending test at collection
    rather than reading as live coverage that silently no-ops.
    """
