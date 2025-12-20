## Why

Cross-platform visualization functions (`create_joint_plot`, `create_scatter_plot_grid`) independently recalculate correlation statistics instead of using the pre-computed values from `CalculateCrossPlatformCorrelationsStep`. This violates DRY (Don't Repeat Yourself) and causes **data inconsistency** where:

- CSV output shows: correlation=-0.7399, p=0.000447, n=18 genotypes (correct)
- Image annotation shows: Spearman ρ=-0.556, p=0.0134, n=19 genotypes (incorrect)

The root cause is that `VisualizeCrossPlatformStep` recalculates genotype means without filtering to `common_genotypes` (which respects `min_samples_per_genotype`), causing the visualization to include an extra genotype that was excluded from the correlation calculation.

## What Changes

1. **Modify `create_joint_plot`** - Add optional parameters to accept pre-computed correlation values (correlation, p_value, n_genotypes). When provided, display these instead of recalculating.

2. **Modify `create_scatter_plot_grid`** - Add optional `correlation_df` parameter to look up pre-computed values for each trait pair.

3. **Update `VisualizeCrossPlatformStep`** - Pass pre-computed correlation values from `correlation_df` to visualization functions.

4. **Add regression test** - Test that verifies the correlation values displayed in images match the CSV output exactly.

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/cross_experiment_analysis.py` (create_joint_plot, create_scatter_plot_grid)
  - `src/sleap_roots_analyze/pipeline/steps/visualize_cross_platform.py`
  - `tests/test_step_visualize_cross_platform.py`
