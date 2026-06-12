"""Tests for the config-gated trait-enrichment DAG step + config validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps.calculate_trait_enrichment import (
    CalculateTraitEnrichmentStep,
)


def _config(**overrides):
    """Build a minimal CrossPlatformConfig with dummy required paths."""
    base = dict(
        exp1_data_path="e1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="Genotype",
        exp2_data_path="e2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="Genotype",
    )
    base.update(overrides)
    return CrossPlatformConfig(**base)


def _corr_df(n_tests, n_sig, seed=0):
    """A representative-only correlation table with n_sig nominal-significant rows."""
    rng = np.random.default_rng(seed)
    p = np.concatenate(
        [rng.uniform(0, 0.01, n_sig), rng.uniform(0.2, 1.0, n_tests - n_sig)]
    )
    return pd.DataFrame(
        {
            "exp1_trait": [f"t{i}" for i in range(n_tests)],
            "exp2_trait": [f"u{i}" for i in range(n_tests)],
            "spearman_r": rng.uniform(-1, 1, n_tests),
            "spearman_p": p,
        }
    )


def _prev_result(corr_df):
    return StepResult(
        data={"correlation_df": corr_df},
        metadata={"exp1_name": "Exp1", "exp2_name": "Exp2"},
        files_generated=[],
    )


# ---------------------------------------------------------------------------
# DAG step: gate
# ---------------------------------------------------------------------------
def test_enrichment_step_skips_when_disabled(tmp_path):
    corr = _corr_df(20, 5)
    prev = _prev_result(corr)
    cfg = _config(enrichment_enabled=False)
    result = CalculateTraitEnrichmentStep().execute(
        data=prev.data, config=cfg, run_dir=tmp_path, prev_result=prev
    )
    assert result.metadata["enrichment_enabled"] is False
    assert result.files_generated == []
    assert not (tmp_path / "trait_enrichment.csv").exists()
    # Data passes through unchanged so visualization still gets the table.
    assert result.data is prev.data


def test_enrichment_step_empty_table_does_not_crash(tmp_path):
    """An empty/all-degenerate correlation table is survivable, not a DAG crash."""
    empty = pd.DataFrame(
        columns=["exp1_trait", "exp2_trait", "spearman_r", "spearman_p"]
    )
    prev = _prev_result(empty)
    cfg = _config(enrichment_enabled=True)
    result = CalculateTraitEnrichmentStep().execute(
        data=prev.data, config=cfg, run_dir=tmp_path, prev_result=prev
    )
    assert result.metadata["enrichment_enabled"] is True
    assert result.metadata["enrichment_n_tests"] == 0
    assert "enrichment_skipped_reason" in result.metadata
    assert (tmp_path / "trait_enrichment.csv").exists()  # empty/skipped CSV written
    assert result.data is prev.data  # data passes through to visualize


def test_enrichment_representative_count_pinned(
    cross_platform_exp1_df, cross_platform_exp2_df, tmp_path
):
    """End-to-end pin: enrichment counts the representative trait pairs.

    Runs clustering -> correlations -> enrichment on real cross-platform data so
    a future change to the upstream representative selection cannot silently
    change the enrichment denominator.
    """
    from sleap_roots_analyze.cross_experiment_analysis import load_and_align_experiments
    from sleap_roots_analyze.data_cleanup import get_trait_columns
    from sleap_roots_analyze.pipeline.core import StepResult
    from sleap_roots_analyze.pipeline.steps.calculate_cross_platform_correlations import (
        CalculateCrossPlatformCorrelationsStep,
    )
    from sleap_roots_analyze.pipeline.steps.reduce_trait_redundancy import (
        ReduceTraitRedundancyStep,
    )

    exp1_path = tmp_path / "e1.csv"
    exp2_path = tmp_path / "e2.csv"
    cross_platform_exp1_df.to_csv(exp1_path, index=False)
    cross_platform_exp2_df.to_csv(exp2_path, index=False)
    exp1_df, exp2_df, common = load_and_align_experiments(
        exp1_path=str(exp1_path),
        exp2_path=str(exp2_path),
        genotype_col1="Geno",
        genotype_col2="geno",
    )
    exp1_traits = get_trait_columns(
        exp1_df, barcode_col=None, genotype_col="genotype", replicate_col="replicate"
    )
    exp2_traits = get_trait_columns(
        exp2_df, barcode_col=None, genotype_col="genotype", replicate_col="replicate"
    )

    cfg = _config(
        exp1_genotype_col="genotype",
        exp2_genotype_col="genotype",
        trait_reduction_method="clustering",
        trait_reduction_target="both",
        enrichment_enabled=True,
        min_genotypes_for_correlation=3,
    )
    loaded = StepResult(
        data={
            "exp1_df": exp1_df,
            "exp2_df": exp2_df,
            "common_genotypes": sorted(common),
        },
        metadata={
            "exp1_name": "Exp1",
            "exp2_name": "Exp2",
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
        },
        files_generated=[],
    )

    reduced = ReduceTraitRedundancyStep().execute(
        data=loaded.data, config=cfg, run_dir=tmp_path, prev_result=loaded
    )
    correlated = CalculateCrossPlatformCorrelationsStep().execute(
        data=reduced.data, config=cfg, run_dir=tmp_path, prev_result=reduced
    )
    enriched = CalculateTraitEnrichmentStep().execute(
        data=correlated.data, config=cfg, run_dir=tmp_path, prev_result=correlated
    )

    corr_df = correlated.data["correlation_df"]
    valid = int(corr_df["spearman_p"].notna().sum())
    # Enrichment counts exactly the representative-only correlation rows that
    # carry a real p-value — not the full pre-clustering N*M trait grid.
    assert enriched.metadata["enrichment_n_tests"] == valid
    assert valid <= len(corr_df)
    # Clustering actually reduced the trait set (representative-only table).
    assert reduced.metadata.get("exp1_reduced_traits", 999) < 50


def test_enrichment_step_runs_and_counts_representative_rows(tmp_path):
    # 20 representative trait pairs, 3 nominally significant.
    corr = _corr_df(20, 3)
    prev = _prev_result(corr)
    cfg = _config(enrichment_enabled=True)  # spearman + spearman_p default
    result = CalculateTraitEnrichmentStep().execute(
        data=prev.data, config=cfg, run_dir=tmp_path, prev_result=prev
    )
    assert result.metadata["enrichment_enabled"] is True
    # Counts the rows of the (representative-only) correlation table directly.
    assert result.metadata["enrichment_n_tests"] == 20
    assert result.metadata["enrichment_n_significant"] == 3
    out = tmp_path / "trait_enrichment.csv"
    assert out.exists()
    row = pd.read_csv(out).iloc[0]
    assert row["n_tests"] == 20
    assert row["n_significant"] == 3
    assert row["platform_pair"] == "Exp1 vs Exp2"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_enrichment_pvalue_must_match_method():
    with pytest.raises(ValueError, match="must match correlation_method"):
        _config(
            enrichment_enabled=True,
            correlation_method="spearman",
            enrichment_p_value_column="pearson_p",
        )


def test_config_enrichment_kendall_unsupported():
    with pytest.raises(ValueError, match="not supported for correlation_method"):
        _config(enrichment_enabled=True, correlation_method="kendall")


def test_config_enrichment_pearson_ok():
    cfg = _config(
        enrichment_enabled=True,
        correlation_method="pearson",
        enrichment_p_value_column="pearson_p",
    )
    assert cfg.enrichment_enabled is True


def test_config_enrichment_disabled_skips_validation():
    # Mismatched column is fine when enrichment is off (default).
    cfg = _config(enrichment_p_value_column="pearson_p")
    assert cfg.enrichment_enabled is False
