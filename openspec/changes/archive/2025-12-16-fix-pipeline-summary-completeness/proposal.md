## Why

After a comprehensive review of pipeline outputs from the most recent pipeline run (`pipeline_runs/2025-12-15_222551/`), multiple issues were identified that compromise traceability, reproducibility, and scientific communication:

### Critical Issues Found

1. **Character Encoding Issues**: Unicode characters (H², mm²) display as garbled text (H�, mm�) in Markdown summaries due to improper encoding handling.

2. **Unfilled Placeholders in Methods Section**: The `SUMMARY.md` Methods section contains template placeholders (`{max_nan_fraction}%`, `{h2_threshold}`) that are never filled with actual values from the configs.

3. **Cross-Platform Alignment Metrics Missing**: The cross-platform summary table shows "N/A" for Common Genotypes, Exp1 Samples, Exp2 Samples columns despite `cross_platform_alignment_summary.csv` containing the data. The CSV parser expects different column names than what's produced.

4. **Viz Figure Count Incorrect**: The Viz summary counts figures from `figures/` directory but static figures are in `static_figures/` and interactive plots are in `pca/`, `umap/`, etc.

5. **Empty `data_source` Fields**: Viz and Cross-Platform `pipeline_summary.json` have empty `data_source` fields, violating the cli-pipeline spec requirement for data source tracking.

6. **Empty `files_generated` Arrays**: All JSON summaries have empty `files_generated` arrays despite producing many output files.

7. **Missing Heritability Values for Removed Traits**: The "Removed Traits by Dataset" section lists trait names but not their H² values, making it unclear why each trait was removed.

8. **Viz `n_traits_initial` Shows 0**: The Viz SUMMARY.md shows "Traits (initial): 0" because this value isn't populated from the input data.

9. **Cross-Platform Folder Names Have Spaces**: Output folders like `cross_platform_Root Core EDPIE (QC'd)_vs_Cylinder EDPIE_20251215_224620` have spaces from experiment names, causing shell escaping issues.

### Missing Features for Reproducibility

10. **No Package Dependencies Recorded**: The summaries don't record installed package versions beyond the main package.

11. **No Input Data Checksums**: No MD5/SHA256 checksums of input CSV files are recorded.

12. **No Cross-Reference Links**: Pipeline summaries don't link to related outputs (e.g., QC summary doesn't link to Viz summary using the same data).

### Missing Visual Aids

13. **No Summary Statistics Figure**: A bar chart or table showing sample/trait counts across all datasets would aid quick comparison.

14. **No Heritability Distribution Figure**: A visualization showing H² distribution for retained vs. removed traits per dataset.

## What Changes

### Bug Fixes

- **FIX** UTF-8 encoding when writing Markdown files to properly display Unicode characters
- **FIX** Methods section placeholder replacement using actual config values
- **FIX** Cross-platform CSV parsing to match actual column names (`genotype` instead of `common_genotype_id`)
- **FIX** Viz figure counting to check `static_figures/` and `interactive_figures/` directories
- **FIX** `data_source` population in Viz and CrossPlatform pipeline summary JSON
- **FIX** `n_traits_initial` calculation in Viz pipeline from input data shape
- **FIX** Cross-platform output folder naming to sanitize experiment names (replace spaces with underscores)

### Enhancements

- **ADD** `files_generated` population with list of all output files
- **ADD** Heritability values next to removed trait names
- **ADD** `dependencies` field to `code_snapshot` with key package versions (pandas, numpy, scipy, etc.)
- **ADD** `input_checksums` field with MD5 hash of input CSV files
- **ADD** Cross-reference links between related pipeline outputs
- **ADD** QC summary statistics bar chart showing samples/traits/genotypes across datasets
- **ADD** Heritability distribution visualization (retained vs. removed traits)

## Impact

### Affected Specs

- `pipeline-runner-skill` - Summary generation requirements (MODIFIED)
- `cli-pipeline` - Data source tracking requirements (MODIFIED)

### Affected Code

- `src/sleap_roots_analyze/pipeline_runner.py`:
  - `_format_qc_summary()` - Add H² values to removed traits, fix encoding
  - `_format_viz_summary()` - Fix figure counting
  - `_format_cross_platform_summary()` - Fix CSV parsing, fix folder naming
  - `_format_methods_section()` - Fill placeholders from config
  - `_generate_summary()` - Add encoding parameter, generate summary figures
  - New: `_generate_summary_figures()` - Create bar charts and H² visualizations
  - New: `_calculate_checksum()` - MD5 hash helper

- `src/sleap_roots_analyze/pipeline/steps/generate_summary.py`:
  - `run()` - Populate `files_generated`, add `dependencies` to code_snapshot
  - `_collect_output_files()` - New helper to list output files

- `src/sleap_roots_analyze/pipeline/steps/generate_summary_viz.py`:
  - `run()` - Fix `data_source` and `n_traits_initial` population
  - `_collect_output_files()` - New helper

- `src/sleap_roots_analyze/cross_platform/pipeline.py`:
  - `run()` - Fix `data_source` population
  - Sanitize experiment names in output folder creation

### New Tests (TDD)

- `tests/test_pipeline_runner_summary.py`:
  - `test_summary_encoding_unicode()` - Verify H², mm² display correctly
  - `test_methods_section_placeholders_filled()` - No `{placeholder}` in output
  - `test_cross_platform_alignment_parsing()` - Correct column name handling
  - `test_viz_figure_counting()` - Counts from correct directories
  - `test_removed_traits_include_heritability()` - H² values shown
  - `test_summary_figures_generated()` - Bar chart and H² plots created
  - `test_input_checksums_recorded()` - MD5 hashes present
  - `test_files_generated_populated()` - Non-empty list of outputs
  - `test_cross_reference_links()` - Links between related summaries

- `tests/test_generate_summary.py`:
  - `test_dependencies_in_code_snapshot()` - Package versions recorded
  - `test_data_source_viz_populated()` - Input path recorded
  - `test_data_source_cross_platform_populated()` - Both experiment paths recorded

### Risk Assessment

- **Low Risk**: All changes are to output formatting, not core pipeline logic
- **Backwards Compatible**: JSON schema additions don't break existing consumers
- **Easy Rollback**: Summary generation is independent of data processing
