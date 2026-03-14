## Why

The `StatisticalAnalysisStep` unconditionally calculates heritability estimates
(`calculate_heritability_estimates()`) regardless of the
`config.statistics.calculate_heritability` flag. This wastes compute time (MixedLM
iterations cause ~24min CI timeouts) and produces heritability results even when
the user explicitly disabled them, confusing downstream steps like
`FilterHeritabilityStep`.

GitHub Issue: #70

## What Changes

- **Respect config flag**: Check `config.statistics.calculate_heritability` before
  calling `calculate_heritability_estimates()` in `StatisticalAnalysisStep.execute()`
- **Metadata key always present**: Always include `heritability_results` key in
  output metadata; set to `{}` when disabled, populated with results when enabled
- **Empty heritability path**: When heritability is disabled, set
  `heritability_results` to an empty dict `{}` and skip CSV generation for
  `08_heritability_results.csv`
- **Summary update**: Reflect skipped heritability in the statistical analysis
  summary JSON
- **Downstream compatibility**: `FilterHeritabilityStep` already handles missing
  heritability gracefully (skips when no results); verify this path works with
  empty dict

## Impact

- Affected specs: `visualization-pipeline` (statistical analysis step behavior)
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py` (primary fix)
  - `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py` (verify compatibility)
  - `tests/test_step_statistical_analysis.py` (new tests)
- No breaking changes to public API or config schema
- Performance improvement: Skips expensive MixedLM when heritability disabled
