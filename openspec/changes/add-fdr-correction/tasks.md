## 1. Configuration

- [x] 1.1 Add `fdr_correction_method` field to `CrossPlatformConfig` dataclass with default `"fdr_by"`
- [x] 1.2 Add docstring documentation for the new field
- [x] 1.3 Add validation in `__post_init__` for valid methods: `["fdr_bh", "fdr_by", "none"]`

## 2. Correlation Step (TDD - Tests First)

- [x] 2.1 Write `test_fdr_correction_bh_method` - verifies adjusted columns exist and p_adj >= p_raw
- [x] 2.2 Write `test_fdr_correction_by_method` - verifies BY is more conservative than BH
- [x] 2.3 Write `test_fdr_correction_none_method` - verifies p_adj == p_raw when disabled
- [x] 2.4 Write `test_fdr_correction_invalid_method` - verifies ValueError on invalid method
- [x] 2.5 Write `test_csv_output_contains_fdr_columns` - verifies CSV schema
- [x] 2.6 Write `test_metadata_includes_fdr_info` - verifies metadata fields
- [x] 2.7 Add `multipletests` import to `calculate_cross_platform_correlations.py`
- [x] 2.8 Implement FDR correction logic after correlation DataFrame creation
- [x] 2.9 Add `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr` columns
- [x] 2.10 Update metadata with `fdr_correction_method` and `significant_correlations`
- [x] 2.11 Run tests and verify all pass

## 3. Visualization Function (TDD - Tests First)

- [x] 3.1 Write test for `create_correlation_summary_plot` with `significant_col` parameter
- [x] 3.2 Add optional `significant_col: Optional[str] = None` parameter to `create_correlation_summary_plot()`
- [x] 3.3 Update significance count annotation to use `significant_col` if provided
- [x] 3.4 Update annotation label from "Significant (p<0.05)" to "Significant (FDR)" when using corrected values
- [x] 3.5 Run tests and verify visualization works correctly

## 4. Visualization Step

- [x] 4.1 Update `VisualizeCrossPlatformStep` to pass `significant_col="significant_fdr"` to `create_correlation_summary_plot()`
- [x] 4.2 Verify integration works end-to-end

## 5. Example Configs

- [x] 5.1 Add `fdr_correction_method` parameter with comments to `configs/cross_platform_turface19_vs_cylinder.yaml`
- [x] 5.2 Update other cross_platform config files in `configs/` and `configs/active/cross_platform/`

## 6. Cleanup

- [x] 6.1 Run full test suite: `uv run pytest tests/test_step_calculate_cross_platform_correlations.py -v`
- [x] 6.2 Run linting: `uv run ruff check --fix && uv run black .`
- [x] 6.3 Delete stray plan file: `C:\Users\Elizabeth\.claude\plans\enumerated-hatching-planet.md`

## 7. Documentation

- [x] 7.1 Create `docs/CROSS_PLATFORM_ANALYSIS.md` with comprehensive guide
- [x] 7.2 Add mathematical formulation of Benjamini-Hochberg (BH) procedure
- [x] 7.3 Add mathematical formulation of Benjamini-Yekutieli (BY) procedure
- [x] 7.4 Explain why BY often produces no significant results (expected behavior)
- [x] 7.5 Document output CSV columns and their meanings
- [x] 7.6 Add practical recommendations for improving statistical power
- [x] 7.7 Include references to original BH and BY papers

## 8. Pipeline Summary Updates

- [x] 8.1 Update `base_pipeline.py` to merge StepResult metadata into pipeline summary JSON
- [x] 8.2 Update `pipeline_runner.py` to use `spearman_r` column (new schema) for top correlation
- [x] 8.3 Update Methods section in SUMMARY.md to mention FDR correction instead of Bonferroni
- [x] 8.4 Run `run-all --cross-only` and verify SUMMARY.md shows correct top correlations
