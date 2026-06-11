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
