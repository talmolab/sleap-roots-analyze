# Proposal: fix-heritability-config-contradiction

## Why

When `statistics.calculate_heritability=False` and `heritability.enabled=True` are set
together in a viz pipeline config, `FilterHeritabilityStep` receives an empty
`heritability_results` dict and silently removes **all** trait columns from the
DataFrame. The pipeline reports success while producing scientifically meaningless
results — no error, no warning, complete data loss.

This was a latent bug that becomes reachable now that PR #88 makes
`StatisticalAnalysisStep` correctly respect the `calculate_heritability` flag.

Both `StatisticalAnalysisStep` and `FilterHeritabilityStep` are used in both pipelines
(QC Step 8/9, Viz Step 2/6), but the contradiction is only possible in the viz pipeline
because `QCPipelineConfig` has no `statistics` section (always defaults to computing
heritability).

## What Changes

- **Config validation**: Add cross-field validation in `validate_viz_config()` to reject
  `heritability.enabled=True` when `statistics.calculate_heritability=False` at pipeline
  startup, before any steps run.
- **Defense-in-depth guard**: Add a guard in `FilterHeritabilityStep.execute()` that
  detects empty `heritability_results` when filtering is enabled, logs a warning, and
  passes all traits through instead of silently removing them.

## Impact

- **Spec affected**: `config-management` (ADDED requirement for cross-field validation)
- **Code affected**:
  - `src/sleap_roots_analyze/pipeline/config/utils.py` (`validate_viz_config`)
  - `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py` (`FilterHeritabilityStep.execute`)
- **Test files affected**:
  - `tests/test_viz_pipeline_config.py` (new validation tests)
  - `tests/test_step_filter_heritability.py` (new guard tests)
- **No breaking changes**: Invalid configs that previously caused silent data loss will
  now raise a clear `ValueError` at startup.
- **QC pipeline not affected**: `QCPipelineConfig` has no `statistics` section, so
  `StatisticalAnalysisStep` always defaults `calculate_heritability=True`.
