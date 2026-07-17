"""Tests for CrossPlatformPipeline's task-6 wiring (Tier 3.5, #196, tasks.md Section 5).

Also covers two tasks.md Section 4 tests that genuinely need the full pipeline
wiring rather than a bare ``PredictCrossPlatformStep.execute()`` call: 4.2a
(the ``trait_reduction_method="clustering"`` interaction with
``predictor_source="genotype_means"``) and 4.9b (task 5's dependency is
ordering-only, never data) -- see
``tests/test_predict_cross_platform.py``'s module docstring for why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.utils import load_cross_platform_config
from sleap_roots_analyze.pipeline.pipelines.cross_platform_pipeline import (
    CrossPlatformPipeline,
)

HARNESS_DIR = Path(__file__).parent / "fixtures" / "harness" / "cross_platform"
SNAPSHOT_DIR = (
    Path(__file__).parent
    / "fixtures"
    / "synthetic"
    / "cross_platform_prediction"
    / "backward_compat_snapshot"
)

# Fields whose values are inherently non-deterministic across runs (timestamps,
# elapsed time, absolute output paths) -- normalized out before comparing
# pipeline_summary.json against the committed snapshot (see
# tests/fixtures/synthetic/cross_platform_prediction/README.md).
_NONDETERMINISTIC_KEYS = {
    "start_time",
    "end_time",
    "total_elapsed_time",
    "elapsed_time",
    "output_directory",
}


def _normalize_summary(obj):
    """Recursively zero out known non-deterministic pipeline_summary.json fields.

    Also drops the top-level "config" key: pipeline_summary.json embeds the
    complete resolved config (mirroring config.yaml), so it too gains a
    `prediction: {...}` block the moment the field exists, regardless of
    `enabled` -- the same expected, harmless diff design.md Decision 9
    documents for config.yaml itself.
    """
    if isinstance(obj, dict):
        return {
            k: (None if k in _NONDETERMINISTIC_KEYS else _normalize_summary(v))
            for k, v in obj.items()
            if k != "config"
        }
    if isinstance(obj, list):
        return [_normalize_summary(v) for v in obj]
    return obj


def _prediction_config(enabled: bool):
    yaml_name = (
        "cross_platform_prediction_wiring.yaml"
        if enabled
        else "cross_platform_prediction_wiring_baseline.yaml"
    )
    return load_cross_platform_config(HARNESS_DIR / yaml_name)


# =============================================================================
# 5.1 / 5.2 -- task presence/absence
# =============================================================================


def test_cross_platform_pipeline_appends_predict_task_when_enabled():
    """A 6th task, depending on both task 1 and task 5, is added when enabled (tasks.md 5.1)."""
    config = _prediction_config(enabled=True)
    pipeline = CrossPlatformPipeline(config=config)

    tasks = pipeline.create_tasks()

    assert len(tasks) == 6
    predict_task = tasks[5]
    assert predict_task.name == "06_predict_cross_platform"
    assert set(predict_task.depends_on) == {
        "01_load_cross_platform_data",
        "05_visualize_cross_platform",
    }


def test_cross_platform_pipeline_omits_predict_task_when_disabled():
    """create_tasks() returns exactly the existing 5 tasks when disabled (tasks.md 5.2)."""
    config = _prediction_config(enabled=False)
    pipeline = CrossPlatformPipeline(config=config)

    tasks = pipeline.create_tasks()

    assert len(tasks) == 5
    assert all(not t.name.startswith("06_") for t in tasks)


# =============================================================================
# 5.3 -- backward-compat oracle against the committed pre-Tier-3.5 snapshot
# =============================================================================


# Excluded from both the committed snapshot and this test's live-run comparison:
# config.yaml (Decision 9 -- its content depends on the prediction field's mere
# presence, not on enabled); pipeline.log (timestamps, never asserted anywhere in
# this repo); and the two per-sample "loaded" intermediates, which this repo's
# own pre-existing curation policy (tests/fixtures/README.md, enforced by
# test_pipeline_reproduction.py::test_curation_excludes_non_assertable_artifacts)
# forbids committing anywhere under tests/fixtures/, regardless of this tier's
# own "capture the full file list" intent for Section 1.3's snapshot.
_EXCLUDED_FROM_BACKWARD_COMPAT_COMPARISON = {
    "config.yaml",
    "pipeline.log",
    "cross_platform_exp1_loaded.csv",
    "cross_platform_exp2_loaded.csv",
}


def test_cross_platform_pipeline_backward_compat_disabled_by_default(tmp_path):
    """Disabled-by-default output matches the pre-Tier-3.5 golden snapshot (tasks.md 5.3).

    Compares the run's analysis output (file list + content, excluding
    config.yaml -- design.md Decision 9 -- and the other names in
    _EXCLUDED_FROM_BACKWARD_COMPAT_COMPARISON) against
    tests/fixtures/synthetic/cross_platform_prediction/backward_compat_snapshot/.
    CSV content is compared value-by-value with this repo's documented
    cross-OS/BLAS tolerance (rtol=1e-6, atol=1e-9, docs/reproducibility.md) --
    found via a real Ubuntu/macOS CI failure that exact-text comparison
    doesn't survive, since correlation statistics (Spearman rho, CI bounds,
    p-values) differ in their last significant digit(s) across BLAS
    implementations, even though the underlying computation is otherwise
    identical. pipeline_summary.json is compared with known non-deterministic
    fields normalized out first; PNG figures are compared for presence only,
    matching this repo's existing convention that image outputs are never
    byte-compared across environments (see that snapshot directory's
    README.md).
    """
    config = _prediction_config(enabled=False)
    pipeline = CrossPlatformPipeline(config=config, output_dir=tmp_path)
    pipeline.run()

    run_dir = pipeline.run_dir
    actual_files = {
        p.name
        for p in run_dir.iterdir()
        if p.name not in _EXCLUDED_FROM_BACKWARD_COMPAT_COMPARISON
    }
    expected_files = {p.name for p in SNAPSHOT_DIR.iterdir()}
    assert actual_files == expected_files

    for name in expected_files:
        actual_path = run_dir / name
        expected_path = SNAPSHOT_DIR / name
        if name.endswith(".png"):
            assert actual_path.is_file()
        elif name == "pipeline_summary.json":
            actual = _normalize_summary(json.loads(actual_path.read_text()))
            expected = _normalize_summary(json.loads(expected_path.read_text()))
            assert actual == expected
        else:
            # Parsed value comparison, not text/bytes: pandas.to_csv() writes
            # platform-native line endings (CRLF on Windows) against an
            # LF-normalized committed snapshot (.gitattributes: *.csv text
            # eol=lf), so even a text comparison isn't enough on its own --
            # and correlation statistics (Spearman rho, CI bounds, p-values)
            # differ in their last significant digit(s) across BLAS
            # implementations (confirmed via a real Ubuntu/macOS CI failure),
            # so exact string equality doesn't survive cross-OS either.
            # Matches this repo's documented rtol=1e-6/atol=1e-9 convention
            # for cross-OS/BLAS numerical comparisons (docs/reproducibility.md).
            actual_df = pd.read_csv(actual_path)
            expected_df = pd.read_csv(expected_path)
            pd.testing.assert_frame_equal(
                actual_df, expected_df, check_exact=False, rtol=1e-6, atol=1e-9
            )


# =============================================================================
# Section 4 tests requiring full pipeline wiring
# =============================================================================


def test_predict_step_genotype_means_uses_full_raw_trait_set_even_when_trait_reduction_clustering_enabled(
    tmp_path,
):
    """genotype_means uses task 1's full trait set, unaffected by clustering reduction (tasks.md 4.2a).

    Regression test for the bug found during round 1 (Decision 8) and
    corrected during round 2 (Decision 13): ReduceTraitRedundancyStep (task 2)
    replaces exp1_df/exp2_df with only the cluster-representative columns
    before forwarding them onward, which would silently defeat
    predictor_source="genotype_means"'s "full raw-data ablation" contract if
    task 6 read from task 5 instead of task 1 directly.
    """
    from sleap_roots_analyze.pipeline.config.components import (
        CrossPlatformConfig,
        PredictionConfig,
    )

    base_config = _prediction_config(enabled=True)
    config = CrossPlatformConfig(
        exp1_data_path=base_config.exp1_data_path,
        exp1_name=base_config.exp1_name,
        exp1_genotype_col=base_config.exp1_genotype_col,
        exp2_data_path=base_config.exp2_data_path,
        exp2_name=base_config.exp2_name,
        exp2_genotype_col=base_config.exp2_genotype_col,
        min_samples_per_genotype=base_config.min_samples_per_genotype,
        trait_reduction_method="clustering",
        trait_reduction_target="both",
        trait_clustering_threshold=0.01,  # aggressive: collapses to 1 representative/side
        prediction=PredictionConfig(
            enabled=True,
            predictor_source="genotype_means",
            reduction_method="pls_latent",
            comparison_methods=["representatives"],
            platform_pairs=[{"source": "SourcePlatform", "target": "TargetPlatform"}],
        ),
    )
    pipeline = CrossPlatformPipeline(config=config, output_dir=tmp_path)
    results = pipeline.run()

    task1_result = results["01_load_cross_platform_data"].data
    predict_result = results["06_predict_cross_platform"].data

    assert set(predict_result.metadata["source_trait_columns"]) == set(
        task1_result.metadata["exp1_trait_names"]
    )
    # Sanity: clustering really did reduce task 2's own trait set, so this
    # assertion is not vacuously true.
    task2_result = results["02_reduce_trait_redundancy"].data
    assert len(task2_result.metadata["exp1_trait_names"]) < len(
        task1_result.metadata["exp1_trait_names"]
    )


def test_predict_step_never_reads_task5_data(tmp_path):
    """Task 5's dependency is for DAG ordering only, never data (tasks.md 4.9b, Decision 15)."""
    from sleap_roots_analyze.pipeline.task import TaskResult

    config = _prediction_config(enabled=True)
    pipeline = CrossPlatformPipeline(config=config, output_dir=tmp_path)

    class _PoisonedData:
        """Raises if ever accessed -- proves task 6 never reads task 5's data."""

        def __getattr__(self, item):
            raise AssertionError(
                "PredictCrossPlatformStep must never read "
                "kwargs['05_visualize_cross_platform'].data"
            )

    tasks = pipeline.create_tasks()
    task1_result = None
    dependency_results = {}
    logger = pipeline.logger
    for task in tasks[:5]:
        result = task.execute(
            config=config,
            run_dir=pipeline.run_dir,
            logger=logger,
            dependency_results=dependency_results,
        )
        dependency_results[task.name] = result
        if task.name == "01_load_cross_platform_data":
            task1_result = result

    # Poison task 5's TaskResult before task 6 runs.
    dependency_results["05_visualize_cross_platform"] = TaskResult(data=_PoisonedData())

    predict_task = tasks[5]
    result = predict_task.execute(
        config=config,
        run_dir=pipeline.run_dir,
        logger=logger,
        dependency_results=dependency_results,
    )

    assert result.data.data is not None
    assert task1_result is not None


# =============================================================================
# Tier 4 (add-prediction-permutation-and-figure, #200), tasks.md 8.1/8.2 --
# task presence/absence for VisualizePredictionStep (7th task).
# =============================================================================


def test_cross_platform_pipeline_appends_visualize_prediction_task_when_visualize_enabled():
    """A 7th task, depending on task 6, is added when prediction.visualize=True (tasks.md 8.1)."""
    config = load_cross_platform_config(
        HARNESS_DIR / "cross_platform_prediction_wiring_visualize.yaml"
    )
    pipeline = CrossPlatformPipeline(config=config)

    tasks = pipeline.create_tasks()

    assert len(tasks) == 7
    visualize_task = tasks[6]
    assert visualize_task.name == "07_visualize_prediction"
    assert set(visualize_task.depends_on) == {"06_predict_cross_platform"}


def test_cross_platform_pipeline_omits_visualize_prediction_task_when_disabled():
    """create_tasks() returns exactly 6 tasks when visualize=False (tasks.md 8.2).

    Including when prediction.enabled=True alone (prediction with no
    visualization).
    """
    config = load_cross_platform_config(
        HARNESS_DIR / "cross_platform_prediction_wiring.yaml"
    )
    assert config.prediction.enabled is True
    assert config.prediction.visualize is False
    pipeline = CrossPlatformPipeline(config=config)

    tasks = pipeline.create_tasks()

    assert len(tasks) == 6
    assert all(not t.name.startswith("07_") for t in tasks)
