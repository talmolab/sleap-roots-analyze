## Why

The `VisualizeCrossPlatformStep` crashes with `ValueError("correlation_df cannot be
empty")` when all cross-platform trait correlations are filtered out (e.g., insufficient
shared genotypes). The upstream `CalculateCrossPlatformCorrelationsStep` correctly creates
an empty DataFrame with proper schema, but the visualization step passes it directly to
`create_correlation_summary_plot()` without checking for emptiness first.

This crashes the entire cross-platform pipeline when experiments have no significant
correlations, which is a valid and expected scenario (e.g., incompatible experiments,
strict filtering thresholds).

GitHub Issue: #86

## What Changes

- **Guard empty DataFrame**: Add early check in `VisualizeCrossPlatformStep.execute()`
  for empty `correlation_df` before calling any plotting functions
- **Graceful skip**: When correlation_df is empty, skip all plot generation, log a
  warning, and return a successful StepResult with `plots_generated: 0`
- **Summary metadata**: Include `"empty_correlations": true` in output metadata when
  skipped
- **Upstream validation**: Ensure `create_correlation_summary_plot()` error message is
  still clear for direct API callers (keep the ValueError)

## Impact

- Affected specs: `cross-platform-analysis` (visualization step behavior)
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/visualize_cross_platform.py` (primary fix)
  - `tests/test_step_visualize_cross_platform.py` (new tests)
- No breaking changes to public API
- No changes to `cross_experiment_analysis.py` (the ValueError is correct for direct callers)
