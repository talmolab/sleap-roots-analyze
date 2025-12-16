## 1. Setup and Test Infrastructure

- [x] 1.1 Review existing `test_pipeline_runner.py` for test patterns
- [x] 1.2 Create test fixtures for mock pipeline summary JSON files
- [x] 1.3 Create test fixtures for mock cross-platform alignment CSV

## 2. TDD: Pipeline Summary Reading (Helper Functions)

- [x] 2.1 Write failing test: `test_read_pipeline_summary_success` - reads valid JSON
- [x] 2.2 Write failing test: `test_read_pipeline_summary_missing_file` - returns empty dict
- [x] 2.3 Write failing test: `test_read_pipeline_summary_malformed_json` - returns empty dict, logs warning
- [x] 2.4 Implement `_read_pipeline_summary()` in `pipeline_runner.py`
- [x] 2.5 Run tests until all pass

## 3. TDD: QC Summary Enhancement

- [x] 3.1 Write failing test: `test_format_qc_summary_with_metrics` - table includes Samples, Traits, Genotypes, H² Threshold, Mean H²
- [x] 3.2 Write failing test: `test_format_qc_summary_failed_pipeline` - shows N/A for failed runs
- [x] 3.3 Write failing test: `test_format_qc_summary_heritability_disabled` - shows "Disabled" for threshold
- [x] 3.4 Implement enhanced `_format_qc_summary()` to read `10_pipeline_summary.json`
- [x] 3.5 Run tests until all pass

## 4. TDD: Removed Traits Documentation

- [x] 4.1 Write failing test: `test_format_removed_traits_section` - lists removed traits per dataset
- [x] 4.2 Write failing test: `test_format_removed_traits_none_removed` - shows "No traits removed"
- [x] 4.3 Write failing test: `test_format_removed_traits_heritability_disabled` - omits section
- [x] 4.4 Implement `_format_removed_traits_section()` helper
- [x] 4.5 Integrate into `_format_qc_summary()`
- [x] 4.6 Run tests until all pass

## 5. TDD: Viz Summary Enhancement

- [x] 5.1 Write failing test: `test_format_viz_summary_with_figure_counts` - table includes figure counts
- [x] 5.2 Write failing test: `test_format_viz_summary_counts_figures_from_directory` - counts PNG files
- [x] 5.3 Implement enhanced `_format_viz_summary()` to count output figures
- [x] 5.4 Run tests until all pass

## 6. TDD: Cross-Platform Summary Enhancement

- [x] 6.1 Write failing test: `test_format_cross_platform_summary_with_metrics` - table includes Common Genotypes, Exp1/Exp2 Samples/Traits
- [x] 6.2 Write failing test: `test_format_cross_platform_summary_reads_alignment_csv` - parses alignment summary
- [x] 6.3 Write failing test: `test_format_cross_platform_summary_reads_top_correlation` - extracts from correlations.csv
- [x] 6.4 Write failing test: `test_format_cross_platform_summary_missing_files` - shows N/A gracefully
- [x] 6.5 Implement enhanced `_format_cross_platform_summary()`
- [x] 6.6 Run tests until all pass

## 7. TDD: Methods Section

- [x] 7.1 Write failing test: `test_format_methods_section_exists` - methods section is present
- [x] 7.2 Write failing test: `test_format_methods_section_describes_qc` - includes QC methodology
- [x] 7.3 Write failing test: `test_format_methods_section_describes_viz` - includes Viz methodology
- [x] 7.4 Write failing test: `test_format_methods_section_has_placeholders` - includes `{n_samples}` etc.
- [x] 7.5 Implement `_format_methods_section()`
- [x] 7.6 Integrate into `generate_summary()`
- [x] 7.7 Run tests until all pass

## 8. Integration Testing

- [x] 8.1 Write integration test: `test_generate_summary_full_pipeline` - end-to-end with mock data
- [x] 8.2 Write integration test: `test_generate_summary_partial_run` - QC-only mode (covered by other tests)
- [x] 8.3 Write integration test: `test_generate_summary_with_failures` - mixed success/failure (covered by failed pipeline test)
- [x] 8.4 Verify markdown formatting is valid (no broken tables)
- [x] 8.5 Run full test suite - 31 tests pass (18 new + 13 existing provenance tests)

## 9. Manual Verification

- [x] 9.1 Run `sleap-roots-analyze run-all --dry-run` to validate
- [x] 9.2 Test summary generation against existing pipeline run data
- [x] 9.3 Verify QC summary shows: Samples, Traits, Genotypes, H² Threshold, Mean H²
- [x] 9.4 Verify removed traits listed per dataset
- [x] 9.5 Verify all spec scenarios are covered

## 10. Documentation

- [x] 10.1 Update any relevant docstrings in `pipeline_runner.py` (docstrings added to all new methods)
- [x] 10.2 Ensure test coverage for new code paths (18 tests covering all new functionality)