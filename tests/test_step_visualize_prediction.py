"""Tests for VisualizePredictionStep (Tier 4, #200, tasks.md Section 7).

Distinct from ``tests/test_visualize_prediction.py`` (Section 6's
``create_prediction_figure()`` tests) -- see proposal.md's naming-collision
note. Step-level tests call ``PredictCrossPlatformStep.execute()`` for real
first (to get a genuine task-6 ``StepResult``, including
``predictor_matrices``), then feed that directly into
``VisualizePredictionStep.execute()`` -- matching
``test_predict_cross_platform.py``'s own "construct inputs by hand" style.

**Mocking-across-process-boundary note (tasks.md Section 7a, added during
`/review-openspec` round 3):** every test below that mocks/spies on
``permutation_test`` MUST fix ``config.prediction.permutation_n_jobs=1`` in
its fixture. ``joblib.Parallel(n_jobs=1)`` runs sequentially in-process
(never touching the ``loky`` backend), so a ``unittest.mock.patch`` in the
test process stays valid; at ``n_jobs>1``, ``loky`` dispatches to separate
worker processes where a parent-process mock is invisible.
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
from sleap_roots_analyze.pipeline.steps.predict_cross_platform import (
    PredictCrossPlatformStep,
)
from sleap_roots_analyze.pipeline.steps.visualize_prediction import (
    VisualizePredictionStep,
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


def _visualize_config(
    tmp_path,
    source_df=None,
    target_df=None,
    genotype_col="Genotype",
    **prediction_overrides,
):
    """Build a CrossPlatformConfig with visualize=True, CI-fast permutation defaults."""
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
        visualize=True,
        n_permutations=5,
        permutation_random_state=42,
        permutation_n_jobs=1,
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


def _run_predict_step(config, run_dir):
    step = PredictCrossPlatformStep()
    return step.execute(data=None, config=config, run_dir=run_dir, prev_result=None)


# =============================================================================
# 7a. Wiring and predictor-matrix reuse
# =============================================================================


def test_visualize_prediction_step_reuses_task6_predictor_matrices(tmp_path):
    """No BLUP-loading call happens when predictor_matrices is already supplied."""
    config = _visualize_config(tmp_path)
    predict_result = _run_predict_step(config, tmp_path)
    step = VisualizePredictionStep()

    with patch(
        "sleap_roots_analyze.pipeline.steps.predict_cross_platform."
        "PredictCrossPlatformStep._load_blup_table"
    ) as mock_load:
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )
    mock_load.assert_not_called()


def test_visualize_prediction_step_handles_pc1_only_targets(tmp_path):
    """With zero representative-trait targets, exactly N=1 unit per method runs."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        select_cluster_representatives as real_select_cluster_representatives,
    )

    source_df, target_df, _ = _make_blup_tables()
    config = _visualize_config(
        tmp_path,
        source_df,
        target_df,
        reduction_method="pls_latent",
        comparison_methods=["representatives"],
    )

    def _target_only_empty(df, clusters):
        if set(df.columns) == set(target_df.columns):
            return []
        return real_select_cluster_representatives(df, clusters)

    with patch(
        "sleap_roots_analyze.pipeline.steps.predict_cross_platform."
        "select_cluster_representatives",
        side_effect=_target_only_empty,
    ):
        predict_result = _run_predict_step(config, tmp_path)

    step = VisualizePredictionStep()
    from sleap_roots_analyze.cross_platform_prediction import (
        permutation_test as real_permutation_test,
    )

    with patch(
        "sleap_roots_analyze.pipeline.steps.visualize_prediction.permutation_test",
        side_effect=real_permutation_test,
    ) as mock_perm:
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )

    n_methods = 1 + len(config.prediction.comparison_methods)
    assert mock_perm.call_count == n_methods


def test_visualize_prediction_step_calls_permutation_test_once_per_target_per_method(
    tmp_path,
):
    """permutation_test is called exactly N targets x M methods times."""
    config = _visualize_config(
        tmp_path, reduction_method="pls_latent", comparison_methods=["representatives"]
    )
    predict_result = _run_predict_step(config, tmp_path)
    step = VisualizePredictionStep()

    from sleap_roots_analyze.cross_platform_prediction import (
        permutation_test as real_permutation_test,
    )

    with patch(
        "sleap_roots_analyze.pipeline.steps.visualize_prediction.permutation_test",
        side_effect=real_permutation_test,
    ) as mock_perm:
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )

    n_targets = len(predict_result.metadata["target_names"])
    n_methods = 2
    assert mock_perm.call_count == n_targets * n_methods


def test_visualize_prediction_step_calls_permutation_test_n_times_when_comparison_methods_empty(
    tmp_path,
):
    """With comparison_methods=[] (K=0), permutation_test is called exactly N times."""
    config = _visualize_config(
        tmp_path, reduction_method="pls_latent", comparison_methods=[]
    )
    predict_result = _run_predict_step(config, tmp_path)
    step = VisualizePredictionStep()

    from sleap_roots_analyze.cross_platform_prediction import (
        permutation_test as real_permutation_test,
    )

    with patch(
        "sleap_roots_analyze.pipeline.steps.visualize_prediction.permutation_test",
        side_effect=real_permutation_test,
    ) as mock_perm:
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )

    n_targets = len(predict_result.metadata["target_names"])
    assert mock_perm.call_count == n_targets


# =============================================================================
# 7b. joblib parallelization across targets
# =============================================================================


def test_visualize_prediction_step_parallelizes_across_target_method_units_not_within_one(
    tmp_path,
):
    """One dispatched job per (target, method) combination, not per permutation."""
    config = _visualize_config(
        tmp_path, reduction_method="pls_latent", comparison_methods=["representatives"]
    )
    predict_result = _run_predict_step(config, tmp_path)
    step = VisualizePredictionStep()

    from joblib import Parallel as RealParallel

    captured = {}

    class _SpyParallel(RealParallel):
        def __call__(self, iterable):
            jobs = list(iterable)
            captured["n_jobs_dispatched"] = len(jobs)
            return super().__call__(jobs)

    with patch(
        "sleap_roots_analyze.pipeline.steps.visualize_prediction.Parallel",
        _SpyParallel,
    ):
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )

    n_targets = len(predict_result.metadata["target_names"])
    n_methods = 2
    assert captured["n_jobs_dispatched"] == n_targets * n_methods


def test_visualize_prediction_step_joblib_n_jobs_and_backend_match_config(tmp_path):
    """joblib.Parallel is constructed with n_jobs=config's value, backend='loky'."""
    config = _visualize_config(tmp_path, permutation_n_jobs=3)
    predict_result = _run_predict_step(config, tmp_path)
    step = VisualizePredictionStep()

    from joblib import Parallel as RealParallel

    captured = {}

    class _SpyParallel(RealParallel):
        def __init__(self, *args, **kwargs):
            captured["n_jobs"] = kwargs.get("n_jobs")
            captured["backend"] = kwargs.get("backend")
            super().__init__(*args, **kwargs)

    with patch(
        "sleap_roots_analyze.pipeline.steps.visualize_prediction.Parallel",
        _SpyParallel,
    ):
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )

    assert captured["n_jobs"] == 3
    assert captured["backend"] == "loky"


def test_visualize_prediction_step_derives_independent_seed_per_target_method(tmp_path):
    """N (target, method) combinations get N distinct SeedSequence.spawn(N) children."""
    config = _visualize_config(
        tmp_path, reduction_method="pls_latent", comparison_methods=["representatives"]
    )
    predict_result = _run_predict_step(config, tmp_path)
    step = VisualizePredictionStep()

    from sleap_roots_analyze.cross_platform_prediction import (
        permutation_test as real_permutation_test,
    )

    captured_seeds = []

    def spy(*args, **kwargs):
        captured_seeds.append(kwargs["random_state"])
        return real_permutation_test(*args, **kwargs)

    with patch(
        "sleap_roots_analyze.pipeline.steps.visualize_prediction.permutation_test",
        side_effect=spy,
    ):
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )

    n_targets = len(predict_result.metadata["target_names"])
    n_methods = 2
    n_combinations = n_targets * n_methods
    assert len(captured_seeds) == n_combinations

    expected_children = np.random.SeedSequence(
        config.prediction.permutation_random_state
    ).spawn(n_combinations)
    # SeedSequence has no value __eq__ -- compare via generated random state.
    for actual, expected in zip(captured_seeds, expected_children):
        np.testing.assert_array_equal(
            actual.generate_state(4), expected.generate_state(4)
        )
    # Re-running with the same permutation_random_state and combinations
    # reproduces the same derived seeds.
    assert len({id(s) for s in captured_seeds}) == n_combinations  # no two identical


def test_visualize_prediction_step_permutation_test_receives_derived_seed(tmp_path):
    """Each permutation_test call's random_state is that combination's derived child."""
    config = _visualize_config(
        tmp_path, reduction_method="pls_latent", comparison_methods=[]
    )
    predict_result = _run_predict_step(config, tmp_path)
    step = VisualizePredictionStep()

    from sleap_roots_analyze.cross_platform_prediction import (
        permutation_test as real_permutation_test,
    )

    captured_seeds = []

    def spy(*args, **kwargs):
        captured_seeds.append(kwargs["random_state"])
        return real_permutation_test(*args, **kwargs)

    with patch(
        "sleap_roots_analyze.pipeline.steps.visualize_prediction.permutation_test",
        side_effect=spy,
    ):
        step.execute(
            data=predict_result.data,
            config=config,
            run_dir=tmp_path,
            prev_result=predict_result,
        )

    n_targets = len(predict_result.metadata["target_names"])
    expected_children = np.random.SeedSequence(
        config.prediction.permutation_random_state
    ).spawn(n_targets)
    assert len(captured_seeds) == n_targets
    for actual, expected in zip(captured_seeds, expected_children):
        assert isinstance(actual, np.random.SeedSequence)
        np.testing.assert_array_equal(
            actual.generate_state(4), expected.generate_state(4)
        )


def test_visualize_prediction_step_parallel_vs_serial_results_agree_within_tolerance(
    tmp_path,
):
    """n_jobs=1 vs n_jobs=4 agree via assert_allclose(rtol=1e-6, atol=1e-9).

    Uses an explicitly small n_permutations=50 to bound the elementwise-
    comparison surface and keep this test CI-fast. Calls the real
    permutation_test (no mocking) at both n_jobs settings.
    """
    source_df, target_df, _ = _make_blup_tables()
    (tmp_path / "serial").mkdir()
    (tmp_path / "parallel").mkdir()
    config_serial = _visualize_config(
        tmp_path / "serial",
        source_df,
        target_df,
        reduction_method="pls_latent",
        comparison_methods=[],
        n_permutations=50,
        permutation_n_jobs=1,
    )
    config_parallel = _visualize_config(
        tmp_path / "parallel",
        source_df,
        target_df,
        reduction_method="pls_latent",
        comparison_methods=[],
        n_permutations=50,
        permutation_n_jobs=4,
    )

    predict_serial = _run_predict_step(config_serial, tmp_path / "serial")
    predict_parallel = _run_predict_step(config_parallel, tmp_path / "parallel")

    result_serial = VisualizePredictionStep().execute(
        data=predict_serial.data,
        config=config_serial,
        run_dir=tmp_path / "serial",
        prev_result=predict_serial,
    )
    result_parallel = VisualizePredictionStep().execute(
        data=predict_parallel.data,
        config=config_parallel,
        run_dir=tmp_path / "parallel",
        prev_result=predict_parallel,
    )

    for method in result_serial.data:
        for target_name in result_serial.data[method]:
            r_serial = result_serial.data[method][target_name]
            r_parallel = result_parallel.data[method][target_name]
            for field in (
                "observed_r2",
                "observed_rmse",
                "observed_spearman_rho",
                "observed_top_quartile_recovery",
                "p_value_r2",
                "p_value_rmse",
                "p_value_spearman_rho",
            ):
                np.testing.assert_allclose(
                    getattr(r_serial, field),
                    getattr(r_parallel, field),
                    rtol=1e-6,
                    atol=1e-9,
                )
            for field in (
                "null_r2",
                "null_rmse",
                "null_spearman_rho",
                "null_top_quartile_recovery",
            ):
                np.testing.assert_allclose(
                    getattr(r_serial, field),
                    getattr(r_parallel, field),
                    rtol=1e-6,
                    atol=1e-9,
                )


def test_visualize_prediction_step_handles_pc1_only_targets_via_joblib(tmp_path):
    """The PC1-only degenerate case still works dispatched through real joblib.Parallel."""
    from sleap_roots_analyze.cross_experiment_analysis import (
        select_cluster_representatives as real_select_cluster_representatives,
    )

    source_df, target_df, _ = _make_blup_tables()
    config = _visualize_config(
        tmp_path,
        source_df,
        target_df,
        reduction_method="pls_latent",
        comparison_methods=["representatives"],
        permutation_n_jobs=4,
    )

    def _target_only_empty(df, clusters):
        if set(df.columns) == set(target_df.columns):
            return []
        return real_select_cluster_representatives(df, clusters)

    with patch(
        "sleap_roots_analyze.pipeline.steps.predict_cross_platform."
        "select_cluster_representatives",
        side_effect=_target_only_empty,
    ):
        predict_result = _run_predict_step(config, tmp_path)

    step = VisualizePredictionStep()
    result = step.execute(
        data=predict_result.data,
        config=config,
        run_dir=tmp_path,
        prev_result=predict_result,
    )

    assert set(result.data.keys()) == {"pls_latent", "representatives"}
    for method in result.data:
        assert set(result.data[method].keys()) == {"PC1"}
