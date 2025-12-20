## Why

The cross-platform correlation pipeline currently stores only the correlation method specified by `correlation_method` config (e.g., only Spearman). However, visualizations display BOTH Pearson and Spearman values in plot annotations. This creates an inconsistency where:

- Spearman ρ uses the pre-computed value from CSV (filtered data)
- Pearson r is recalculated fresh from plot data (potentially different filtering)

This violates the "single source of truth" principle and can cause subtle data inconsistencies between what's in the CSV and what's displayed in visualizations.

## What Changes

1. **Store both Pearson and Spearman in CSV** - The `CalculateCrossPlatformCorrelationsStep` will compute and store both correlation coefficients regardless of which method is primary (used for ranking/filtering).

2. **Rename CSV columns** - Change from generic `correlation`/`p_value` to explicit `spearman_r`/`spearman_p`/`pearson_r`/`pearson_p` for clarity.

3. **Update visualization functions** - `create_joint_plot` will accept both Pearson and Spearman pre-computed values instead of recalculating Pearson.

4. **Preserve `correlation_method` behavior** - The config option still determines which method is used for:
   - Sorting/ranking correlations (by absolute value)
   - Significance filtering
   - The "primary" metric shown prominently

## Impact

- Affected specs: `cross-platform-analysis`
- **CSV Schema Change** - Breaking change for existing CSV parsers expecting `correlation` column. New columns: `spearman_r`, `spearman_p`, `pearson_r`, `pearson_p`
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py`
  - `src/sleap_roots_analyze/pipeline/steps/visualize_cross_platform.py`
  - `src/sleap_roots_analyze/cross_experiment_analysis.py` (create_joint_plot, create_scatter_plot_grid)
  - `tests/test_step_calculate_cross_platform_correlations.py`
  - `tests/test_step_visualize_cross_platform.py`
