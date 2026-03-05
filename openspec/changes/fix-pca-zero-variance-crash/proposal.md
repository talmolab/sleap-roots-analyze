## Why

The PCA analysis pipeline step crashes with a shape mismatch error when trait columns
contain constant values (zero variance). The upstream `perform_pca_analysis()` function
silently filters out zero-variance columns, but `PCAAnalysisStep` uses the original
unfiltered `trait_cols` list as the DataFrame index for the loadings matrix, causing a
dimension mismatch (e.g., 263 actual features vs. 267 expected). This blocks all pipeline
executions when any trait exhibits zero variance—common in early-stage plant phenotyping
where measurements remain uniform across samples.

GitHub Issue: #74

## What Changes

- **Fix shape mismatch**: Use `pca_results["feature_names"]` (the post-filtering feature
  list already returned by `perform_pca_analysis()`) instead of `trait_cols` for:
  - Loadings DataFrame index (line 117)
  - `n_features_total` in `select_top_features_from_pca()` (line 91)
  - Top feature name lookup (line 96)
- **Log excluded traits**: Log which traits were excluded due to zero variance and how
  many remain, for traceability.
- **Warn on high exclusion rate**: Emit a Python `UserWarning` when >50% of traits are
  zero-variance, alerting users to potential data quality issues.
- **Metadata tracking**: Store `excluded_zero_variance_traits` and
  `n_traits_after_filtering` in the step's output metadata for downstream inspection.

## Impact

- Affected specs: `visualization-pipeline` (PCA analysis step behavior)
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/pca_analysis.py` (primary fix)
  - `tests/test_step_pca_analysis.py` (new tests)
- No breaking changes to public API or config schema
- No changes to `pca.py` (the filtering behavior is correct; the bug is in the step)
