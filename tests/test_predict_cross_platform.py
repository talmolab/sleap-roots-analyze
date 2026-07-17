"""Tests for PredictCrossPlatformStep (Tier 3.5, add-prediction-pipeline-step, #196).

tasks.md Section 4. Step-level tests only -- they call
``PredictCrossPlatformStep.execute()`` directly, constructing ``data``/
``prev_result`` by hand to simulate whatever ``01_load_cross_platform_data``
would have produced (Decision 8/13/15). Two tasks.md tests genuinely need the
full pipeline wiring instead of a bare step call (4.2a's
trait_reduction_method="clustering" interaction, and 4.9b's task-5-is-
ordering-only spy) -- those live in ``test_predict_cross_platform_pipeline.py``
(Section 5) since they exercise ``CrossPlatformPipeline.create_tasks()``, not
the step in isolation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.components import (
    CrossPlatformConfig,
    PredictionConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps.predict_cross_platform import (
    PredictCrossPlatformStep,
)

N_GENOTYPES = 19
GENOTYPES = [f"G{i:02d}" for i in range(1, N_GENOTYPES + 1)]


def _make_blup_tables(genotype_col="Genotype", n=N_GENOTYPES, seed=0):
    """Build a small (source_df, target_df) BLUP-shaped pair with planted signal."""
    rng = np.random.default_rng(seed)
    genotypes = [f"G{i:02d}" for i in range(1, n + 1)]
    source = pd.DataFrame(
        {
            "trait_a": rng.normal(10, 2, n),
            "trait_b": rng.normal(5, 1, n),
        },
        index=genotypes,
    )
    target = pd.DataFrame(
        {
            "trait_x": 2.0 * source["trait_a"].to_numpy() + rng.normal(0, 0.1, n),
            "trait_y": rng.normal(3, 1, n),
        },
        index=genotypes,
    )
    source.index.name = genotype_col
    target.index.name = genotype_col
    return source, target, genotypes


def _write_blup_csv(df, path, genotype_col="Genotype"):
    df.reset_index(names=genotype_col).to_csv(path, index=False)
    return path


def _blup_config(
    tmp_path,
    source_df=None,
    target_df=None,
    genotype_col="Genotype",
    **prediction_overrides,
):
    """Build a CrossPlatformConfig with predictor_source='blup' over given tables."""
    if source_df is None or target_df is None:
        source_df, target_df, _ = _make_blup_tables()
    source_path = tmp_path / "source_blup.csv"
    target_path = tmp_path / "target_blup.csv"
    _write_blup_csv(source_df, source_path, genotype_col)
    _write_blup_csv(target_df, target_path, genotype_col)

    prediction_kwargs = dict(
        enabled=True,
        predictor_source="blup",
        source_blup_path=str(source_path),
        target_blup_path=str(target_path),
        platform_pairs=[{"source": "SourcePlatform", "target": "TargetPlatform"}],
    )
    prediction_kwargs.update(prediction_overrides)
    prediction = PredictionConfig(**prediction_kwargs)
    return CrossPlatformConfig(
        exp1_data_path="unused_exp1.csv",
        exp1_name="SourcePlatform",
        exp1_genotype_col="Genotype",
        exp2_data_path="unused_exp2.csv",
        exp2_name="TargetPlatform",
        exp2_genotype_col="Genotype",
        prediction=prediction,
    )


def _genotype_means_config(tmp_path, exp1_df, exp2_df, **prediction_overrides):
    """Build a CrossPlatformConfig + (data, prev_result) for predictor_source='genotype_means'."""
    prediction_kwargs = dict(
        enabled=True,
        predictor_source="genotype_means",
        platform_pairs=[{"source": "SourcePlatform", "target": "TargetPlatform"}],
    )
    prediction_kwargs.update(prediction_overrides)
    prediction = PredictionConfig(**prediction_kwargs)
    config = CrossPlatformConfig(
        exp1_data_path="unused_exp1.csv",
        exp1_name="SourcePlatform",
        exp1_genotype_col="Genotype",
        exp2_data_path="unused_exp2.csv",
        exp2_name="TargetPlatform",
        exp2_genotype_col="Genotype",
        trait_reduction_method="none",
        prediction=prediction,
    )
    data = {"exp1_df": exp1_df, "exp2_df": exp2_df, "common_genotypes": GENOTYPES}
    exp1_traits = [c for c in exp1_df.columns if c not in ("genotype", "replicate")]
    exp2_traits = [c for c in exp2_df.columns if c not in ("genotype", "replicate")]
    prev_result = StepResult(
        data=data,
        metadata={
            "exp1_name": "SourcePlatform",
            "exp2_name": "TargetPlatform",
            "exp1_trait_names": exp1_traits,
            "exp2_trait_names": exp2_traits,
        },
    )
    return config, data, prev_result


def _raw_per_sample_df(trait_values: dict, genotypes=GENOTYPES, extra_cols=None):
    """Build a raw per-sample df: one row per genotype (n_reps=1 for simplicity)."""
    df = pd.DataFrame({"genotype": genotypes, "replicate": [1] * len(genotypes)})
    for trait, values in trait_values.items():
        df[trait] = values
    if extra_cols:
        for col, values in extra_cols.items():
            df[col] = values
    return df


# =============================================================================
# 4.1 / 4.1a / 4.1b -- BLUP loading, NaN handling, genotype column resolution
# =============================================================================


def test_predict_step_builds_source_matrix_from_blup_when_predictor_source_blup(
    tmp_path,
):
    """source_blup_path is loaded as the predictor matrix X (tasks.md 4.1)."""
    source_df, target_df, genotypes = _make_blup_tables()
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    saved = json.loads((tmp_path / "06_prediction_pls_latent.json").read_text())
    assert set(saved["predictions"][0]["genotype_names"]) == set(genotypes)
    assert result.metadata["source_platform"] == "SourcePlatform"
    assert result.metadata["target_platform"] == "TargetPlatform"


def test_predict_step_drops_trait_columns_containing_any_nan(tmp_path):
    """A trait column with any NaN is dropped before X/target construction (tasks.md 4.1a)."""
    source_df, target_df, genotypes = _make_blup_tables()
    source_df["failed_trait"] = np.nan
    target_df["failed_trait"] = np.nan
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    assert "failed_trait" not in result.metadata["source_trait_columns"]
    assert "failed_trait" not in result.metadata["target_candidate_columns"]


def test_predict_step_raises_clear_error_when_source_matrix_is_empty_after_nan_drop(
    tmp_path,
):
    """A source table with every trait column containing a NaN raises ValueError (tasks.md 4.1a)."""
    source_df, target_df, _ = _make_blup_tables()
    source_df["trait_a"] = np.nan
    source_df["trait_b"] = np.nan
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    with pytest.raises(ValueError, match="[Ss]ource"):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)


def test_predict_step_raises_clear_error_when_target_matrix_is_empty_after_nan_drop(
    tmp_path,
):
    """A target table with every trait column containing a NaN raises a clear ValueError.

    Found during code review: unlike the source side (which already had this
    guard), a zero-column target previously fell through to an opaque sklearn
    error from fitting PCA (for the PC1 target) on a zero-feature array,
    rather than a clear, step-level ValueError naming the target platform.
    """
    source_df, target_df, _ = _make_blup_tables()
    target_df["trait_x"] = np.nan
    target_df["trait_y"] = np.nan
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    with pytest.raises(ValueError, match="[Tt]arget"):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)


def test_predict_step_resolves_blup_genotype_column_name(tmp_path):
    """BLUP genotype column resolves 'Genotype' then 'genotype'; else clear error (tasks.md 4.1b)."""
    source_df, target_df, _ = _make_blup_tables(genotype_col="Genotype")
    config = _blup_config(tmp_path, source_df, target_df, genotype_col="Genotype")
    step = PredictCrossPlatformStep()
    step.execute(
        data=None, config=config, run_dir=tmp_path, prev_result=None
    )  # no raise

    source_df2, target_df2, _ = _make_blup_tables(genotype_col="genotype")
    (tmp_path / "b").mkdir()
    config2 = _blup_config(
        tmp_path / "b", source_df2, target_df2, genotype_col="genotype"
    )
    step2 = PredictCrossPlatformStep()
    step2.execute(
        data=None, config=config2, run_dir=tmp_path / "b", prev_result=None
    )  # no raise

    # Neither "Genotype" nor "genotype" present.
    bad_source = tmp_path / "bad_source.csv"
    bad_target = tmp_path / "bad_target.csv"
    pd.DataFrame({"id": GENOTYPES, "trait_a": range(N_GENOTYPES)}).to_csv(
        bad_source, index=False
    )
    pd.DataFrame({"id": GENOTYPES, "trait_x": range(N_GENOTYPES)}).to_csv(
        bad_target, index=False
    )
    prediction = PredictionConfig(
        enabled=True,
        predictor_source="blup",
        source_blup_path=str(bad_source),
        target_blup_path=str(bad_target),
        platform_pairs=[{"source": "SourcePlatform", "target": "TargetPlatform"}],
    )
    bad_config = CrossPlatformConfig(
        exp1_data_path="unused.csv",
        exp1_name="SourcePlatform",
        exp1_genotype_col="Genotype",
        exp2_data_path="unused.csv",
        exp2_name="TargetPlatform",
        exp2_genotype_col="Genotype",
        prediction=prediction,
    )
    (tmp_path / "c").mkdir()
    step3 = PredictCrossPlatformStep()
    with pytest.raises(ValueError, match="Genotype.*genotype|genotype.*Genotype"):
        step3.execute(
            data=None, config=bad_config, run_dir=tmp_path / "c", prev_result=None
        )


# =============================================================================
# 4.2 / 4.2b -- genotype_means predictor_source
# =============================================================================


def test_predict_step_builds_source_matrix_from_genotype_means_when_predictor_source_genotype_means(
    tmp_path,
):
    """genotype_means aggregates task 1's trait-filtered raw data (tasks.md 4.2)."""
    rng = np.random.default_rng(1)
    exp1_df = _raw_per_sample_df(
        {
            "trait_a": rng.normal(10, 2, N_GENOTYPES),
            "trait_b": rng.normal(5, 1, N_GENOTYPES),
        }
    )
    exp2_df = _raw_per_sample_df(
        {
            "trait_x": rng.normal(20, 3, N_GENOTYPES),
            "trait_y": rng.normal(1, 1, N_GENOTYPES),
        }
    )
    config, data, prev_result = _genotype_means_config(tmp_path, exp1_df, exp2_df)
    step = PredictCrossPlatformStep()

    result = step.execute(
        data=data, config=config, run_dir=tmp_path, prev_result=prev_result
    )

    assert set(result.metadata["source_trait_columns"]) == {"trait_a", "trait_b"}
    assert set(result.metadata["target_candidate_columns"]) == {"trait_x", "trait_y"}


def test_predict_step_genotype_means_averages_multiple_replicates_per_genotype(
    tmp_path,
):
    """genotype_means computes a genuine mean across >1 replicate, not an identity op.

    Found during code review: every other genotype_means fixture in this file
    uses exactly one replicate per genotype, so `.groupby("genotype").mean()`
    is never actually exercised as an aggregation (a single-row mean is a
    no-op). This test uses 3 replicates/genotype with distinct per-replicate
    values and inspects the actual predictor matrix passed to
    logo_cv_predict.
    """
    genotypes = ["G01", "G01", "G01", "G02", "G02", "G02", "G03", "G03", "G03"]
    replicates = [1, 2, 3] * 3
    trait_a_values = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    exp1_df = pd.DataFrame(
        {"genotype": genotypes, "replicate": replicates, "trait_a": trait_a_values}
    )
    exp2_df = pd.DataFrame(
        {
            "genotype": genotypes,
            "replicate": replicates,
            # Varies per genotype (not constant): a constant target trait
            # produces a non-finite spearman_rho/p (legal logo_cv_predict
            # input, but CrossPlatformPredictionResult.to_json's finite-floats
            # contract then raises) -- an unrelated edge case to this test.
            "trait_x": [5.0, 6.0, 7.0, 15.0, 16.0, 17.0, 25.0, 26.0, 27.0],
        }
    )
    prediction = PredictionConfig(
        enabled=True,
        predictor_source="genotype_means",
        platform_pairs=[{"source": "SourcePlatform", "target": "TargetPlatform"}],
    )
    config = CrossPlatformConfig(
        exp1_data_path="unused_exp1.csv",
        exp1_name="SourcePlatform",
        exp1_genotype_col="Genotype",
        exp2_data_path="unused_exp2.csv",
        exp2_name="TargetPlatform",
        exp2_genotype_col="Genotype",
        prediction=prediction,
    )
    data = {
        "exp1_df": exp1_df,
        "exp2_df": exp2_df,
        "common_genotypes": ["G01", "G02", "G03"],
    }
    prev_result = StepResult(
        data=data,
        metadata={
            "exp1_name": "SourcePlatform",
            "exp2_name": "TargetPlatform",
            "exp1_trait_names": ["trait_a"],
            "exp2_trait_names": ["trait_x"],
        },
    )
    step = PredictCrossPlatformStep()

    captured_X = {}

    def _capture(*args, **kwargs):
        captured_X["X"] = kwargs["X"].copy()
        return real_logo_cv_predict(*args, **kwargs)

    from sleap_roots_analyze.cross_platform_prediction import (
        logo_cv_predict as real_logo_cv_predict,
    )

    with patch(
        "sleap_roots_analyze.pipeline.steps.predict_cross_platform.logo_cv_predict",
        side_effect=_capture,
    ):
        step.execute(
            data=data, config=config, run_dir=tmp_path, prev_result=prev_result
        )

    expected_means = {"G01": 2.0, "G02": 20.0, "G03": 200.0}
    for genotype, expected in expected_means.items():
        assert captured_X["X"].loc[genotype, "trait_a"] == pytest.approx(expected)


def test_predict_step_genotype_means_excludes_metadata_columns(tmp_path):
    """Non-trait metadata columns are excluded from the genotype_means predictor matrix (tasks.md 4.2b)."""
    rng = np.random.default_rng(2)
    exp1_df = _raw_per_sample_df(
        {"trait_a": rng.normal(10, 2, N_GENOTYPES)},
        extra_cols={"notes": ["some text"] * N_GENOTYPES},
    )
    exp2_df = _raw_per_sample_df({"trait_x": rng.normal(20, 3, N_GENOTYPES)})
    prediction = PredictionConfig(
        enabled=True,
        predictor_source="genotype_means",
        platform_pairs=[{"source": "SourcePlatform", "target": "TargetPlatform"}],
    )
    config = CrossPlatformConfig(
        exp1_data_path="unused_exp1.csv",
        exp1_name="SourcePlatform",
        exp1_genotype_col="Genotype",
        exp2_data_path="unused_exp2.csv",
        exp2_name="TargetPlatform",
        exp2_genotype_col="Genotype",
        prediction=prediction,
    )
    data = {"exp1_df": exp1_df, "exp2_df": exp2_df, "common_genotypes": GENOTYPES}
    # Simulate task 1's own exclude_cols-filtered trait-name metadata: "notes" is
    # NOT in exp1_trait_names even though it's a (non-numeric) column of exp1_df.
    prev_result = StepResult(
        data=data,
        metadata={
            "exp1_name": "SourcePlatform",
            "exp2_name": "TargetPlatform",
            "exp1_trait_names": ["trait_a"],
            "exp2_trait_names": ["trait_x"],
        },
    )
    step = PredictCrossPlatformStep()

    result = step.execute(
        data=data, config=config, run_dir=tmp_path, prev_result=prev_result
    )

    assert "notes" not in result.metadata["source_trait_columns"]


# =============================================================================
# 4.3 / 4.3a -- target-side representative trait selection
# =============================================================================


def test_predict_step_selects_target_representative_traits(tmp_path):
    """The target platform's cluster-representative traits become prediction targets (tasks.md 4.3)."""
    rng = np.random.default_rng(3)
    genotypes = GENOTYPES
    base = rng.normal(0, 1, N_GENOTYPES)
    # trait_x1/trait_x2 are near-duplicates (one cluster); trait_y is independent.
    target_df = pd.DataFrame(
        {
            "trait_x1": base + rng.normal(0, 0.01, N_GENOTYPES),
            "trait_x2": base + rng.normal(0, 0.01, N_GENOTYPES),
            "trait_y": rng.normal(0, 1, N_GENOTYPES),
        },
        index=genotypes,
    )
    source_df = pd.DataFrame(
        {"trait_a": rng.normal(10, 2, N_GENOTYPES)}, index=genotypes
    )
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    saved = json.loads((tmp_path / "06_prediction_pls_latent.json").read_text())
    target_names = {p["target_name"] for p in saved["predictions"]}
    assert "PC1" in target_names
    # Exactly one of trait_x1/trait_x2 (the cluster representative) plus trait_y.
    assert len(target_names) == 3


def test_predict_step_handles_zero_target_representative_traits(tmp_path):
    """Zero target representatives still runs successfully with only PC1 (tasks.md 4.3a)."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        select_cluster_representatives as real_select_cluster_representatives,
    )

    source_df, target_df, _ = _make_blup_tables()
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    def _target_only_empty(df, clusters):
        # Force zero representatives for the TARGET-side call only (identified
        # by its trait columns); the SOURCE-side call must still return real
        # representatives, or reduction_method="representatives" would itself
        # fail for an unrelated reason (no representative_names available).
        if set(df.columns) == set(target_df.columns):
            return []
        return real_select_cluster_representatives(df, clusters)

    with patch(
        "sleap_roots_analyze.pipeline.steps.predict_cross_platform.select_cluster_representatives",
        side_effect=_target_only_empty,
    ):
        result = step.execute(
            data=None, config=config, run_dir=tmp_path, prev_result=None
        )

    saved = json.loads((tmp_path / "06_prediction_pls_latent.json").read_text())
    assert len(saved["predictions"]) == 1
    assert saved["predictions"][0]["target_name"] == "PC1"


# =============================================================================
# 4.4 / 4.4a -- PC1-as-target
# =============================================================================


def test_predict_step_computes_target_pc1_via_whole_dataset_pca_not_per_fold(tmp_path):
    """PC1 target is computed via pca.fit_pca(), never fit_pca_on_fold (tasks.md 4.4)."""
    source_df, target_df, _ = _make_blup_tables()
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    with patch(
        "sleap_roots_analyze.cross_platform_prediction.fit_pca_on_fold"
    ) as mock_fit_pca_on_fold:
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)
        # pc1 is not in reduction_method/comparison_methods here, so
        # fit_pca_on_fold (the source-side per-fold reducer, internal to
        # logo_cv_predict's "pc1" branch) is never invoked for the PC1
        # *target*'s computation, which uses pca.fit_pca() directly instead.
        mock_fit_pca_on_fold.assert_not_called()


def test_predict_step_target_pc1_values_match_independent_whole_dataset_pca(tmp_path):
    """PC1 target values match an independent whole-dataset PCA computation (tasks.md 4.4a)."""
    from sklearn.preprocessing import StandardScaler

    from sleap_roots_analyze.pca import fit_pca

    source_df, target_df, genotypes = _make_blup_tables()
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    saved = json.loads((tmp_path / "06_prediction_pls_latent.json").read_text())
    pc1_entry = next(p for p in saved["predictions"] if p["target_name"] == "PC1")

    _, expected = fit_pca(
        StandardScaler().fit_transform(target_df.to_numpy()),
        n_components=1,
        random_state=42,
    )
    expected_by_geno = dict(zip(target_df.index.tolist(), expected.ravel().tolist()))
    for geno, y_true in zip(pc1_entry["genotype_names"], pc1_entry["y_true"]):
        assert y_true == pytest.approx(expected_by_geno[geno], rel=1e-6, abs=1e-9)


# =============================================================================
# 4.5 / 4.5a / 4.5b -- common-genotype alignment
# =============================================================================


def test_predict_step_aligns_to_common_genotypes(tmp_path):
    """Genotypes present in only one side are excluded before logo_cv_predict (tasks.md 4.5)."""
    source_df, target_df, genotypes = _make_blup_tables()
    source_only = source_df.copy()
    source_only.loc["EXTRA_SOURCE_ONLY"] = source_only.iloc[0]
    target_only = target_df.copy()
    target_only.loc["EXTRA_TARGET_ONLY"] = target_only.iloc[0]

    config = _blup_config(tmp_path, source_only, target_only)
    step = PredictCrossPlatformStep()

    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    assert "EXTRA_SOURCE_ONLY" not in result.metadata["common_genotypes"]
    assert "EXTRA_TARGET_ONLY" not in result.metadata["common_genotypes"]
    assert set(result.metadata["common_genotypes"]) == set(genotypes)


def test_predict_step_raises_clear_error_when_common_genotypes_below_minimum(tmp_path):
    """Fewer than 3 common genotypes raises a clear, step-level ValueError (tasks.md 4.5a)."""
    source_df, target_df, _ = _make_blup_tables(n=2)
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    with pytest.raises(ValueError, match="SourcePlatform.*TargetPlatform"):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)


def test_predict_step_raises_clear_error_when_zero_genotype_overlap(tmp_path):
    """Zero genotype overlap between source/target raises a clear ValueError (tasks.md 4.5a)."""
    source_df, target_df, _ = _make_blup_tables()
    target_df = target_df.copy()
    target_df.index = [f"OTHER_{g}" for g in target_df.index]
    target_df.index.name = "Genotype"
    config = _blup_config(tmp_path, source_df, target_df)
    step = PredictCrossPlatformStep()

    with pytest.raises(ValueError, match="SourcePlatform.*TargetPlatform"):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)


def test_predict_step_derives_X_y_genotypes_from_one_canonical_index(tmp_path):
    """Reversed row order between source/target still pairs genotypes correctly (tasks.md 4.5b)."""
    source_df, target_df, genotypes = _make_blup_tables()
    target_reversed = target_df.iloc[::-1]

    config = _blup_config(tmp_path, source_df, target_reversed)
    step = PredictCrossPlatformStep()
    step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    saved = json.loads((tmp_path / "06_prediction_pls_latent.json").read_text())
    for prediction in saved["predictions"]:
        for geno, y_true in zip(prediction["genotype_names"], prediction["y_true"]):
            expected = (
                target_df.loc[geno, prediction["target_name"]]
                if (prediction["target_name"] in target_df.columns)
                else None
            )
            if expected is not None:
                assert y_true == pytest.approx(expected, rel=1e-6, abs=1e-9)


# =============================================================================
# 4.6 / 4.7 / 4.8 -- logo_cv_predict call count, result assembly, JSON output
# =============================================================================


def test_predict_step_calls_logo_cv_predict_once_per_target_per_method(tmp_path):
    """logo_cv_predict is called N targets x M methods times (tasks.md 4.6)."""
    source_df, target_df, _ = _make_blup_tables()
    config = _blup_config(
        tmp_path,
        source_df,
        target_df,
        reduction_method="pls_latent",
        comparison_methods=["representatives"],
    )
    step = PredictCrossPlatformStep()

    from sleap_roots_analyze.cross_platform_prediction import logo_cv_predict as real

    with patch(
        "sleap_roots_analyze.pipeline.steps.predict_cross_platform.logo_cv_predict",
        side_effect=real,
    ) as mock_logo:
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    saved_pls = json.loads((tmp_path / "06_prediction_pls_latent.json").read_text())
    n_targets = len(saved_pls["predictions"])
    assert mock_logo.call_count == n_targets * 2  # 2 methods


def test_predict_step_builds_one_result_per_method(tmp_path):
    """One CrossPlatformPredictionResult per method, each holding all N targets (tasks.md 4.7)."""
    source_df, target_df, _ = _make_blup_tables()
    config = _blup_config(
        tmp_path,
        source_df,
        target_df,
        reduction_method="pls_latent",
        comparison_methods=["representatives"],
    )
    step = PredictCrossPlatformStep()
    step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    pls_saved = json.loads((tmp_path / "06_prediction_pls_latent.json").read_text())
    rep_saved = json.loads(
        (tmp_path / "06_prediction_representatives.json").read_text()
    )
    assert len(pls_saved["predictions"]) == len(rep_saved["predictions"])
    assert pls_saved["reduction_method"] == "pls_latent"
    assert rep_saved["reduction_method"] == "representatives"


def test_predict_step_saves_json_output_per_method(tmp_path):
    """One JSON file per method is written to run_dir, no collisions (tasks.md 4.8)."""
    source_df, target_df, _ = _make_blup_tables()
    config = _blup_config(
        tmp_path,
        source_df,
        target_df,
        reduction_method="pls_latent",
        comparison_methods=["representatives"],
    )
    step = PredictCrossPlatformStep()
    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    assert (tmp_path / "06_prediction_pls_latent.json").is_file()
    assert (tmp_path / "06_prediction_representatives.json").is_file()
    assert len(result.files_generated) == 2


# =============================================================================
# 4.9a -- blup_refit_per_fold inertness (post-implementation tripwire)
# =============================================================================


def test_predict_step_blup_refit_per_fold_is_inert(tmp_path):
    """blup_refit_per_fold has no observable effect (tasks.md 4.9a)."""
    source_df, target_df, _ = _make_blup_tables()

    (tmp_path / "true").mkdir(exist_ok=True)
    (tmp_path / "false").mkdir(exist_ok=True)
    config_true = _blup_config(
        tmp_path / "true", source_df, target_df, blup_refit_per_fold=True
    )
    config_false = _blup_config(
        tmp_path / "false", source_df, target_df, blup_refit_per_fold=False
    )

    step_true = PredictCrossPlatformStep()
    step_false = PredictCrossPlatformStep()
    step_true.execute(
        data=None, config=config_true, run_dir=tmp_path / "true", prev_result=None
    )
    step_false.execute(
        data=None, config=config_false, run_dir=tmp_path / "false", prev_result=None
    )

    saved_true = json.loads(
        (tmp_path / "true" / "06_prediction_pls_latent.json").read_text()
    )
    saved_false = json.loads(
        (tmp_path / "false" / "06_prediction_pls_latent.json").read_text()
    )
    assert saved_true == saved_false
