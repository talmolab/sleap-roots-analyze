## Why

When computing thousands of pairwise correlations between traits across two experiments, approximately 5% will appear "significant" at p<0.05 purely by chance. The cross-platform pipeline currently outputs raw p-values without multiple testing correction, risking false discoveries and over-interpretation of spurious correlations.

**Related Issue**: #44 (Epic: Biostatistical Improvements for Cross-Platform Correlation Pipeline)

## What Changes

- Add configurable FDR (False Discovery Rate) correction to `CrossPlatformConfig`
- Support three correction methods:
  - `fdr_bh`: Benjamini-Hochberg (assumes test independence)
  - `fdr_by`: Benjamini-Yekutieli (valid under arbitrary dependence) - **DEFAULT**
  - `none`: Disable correction for exploratory analysis
- Add new columns to correlation CSV output:
  - `spearman_p_adjusted`: FDR-corrected Spearman p-value
  - `pearson_p_adjusted`: FDR-corrected Pearson p-value
  - `significant_fdr`: Boolean flag (adjusted p < significance_level)
- Update volcano plot annotation to show FDR-corrected significance count
- Update metadata to include correction method and significant correlation count

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/pipeline/config/components.py` (CrossPlatformConfig)
  - `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py`
  - `src/sleap_roots_analyze/cross_experiment_analysis.py` (create_correlation_summary_plot)
  - `src/sleap_roots_analyze/pipeline/steps/visualize_cross_platform.py`
  - `configs/cross_platform_*.yaml` (example configs)
  - `tests/test_step_calculate_cross_platform_correlations.py`
- Backward compatible: default `fdr_by` works with existing configs (new field has default)
- No new dependencies: `statsmodels.stats.multitest.multipletests` already used in codebase
