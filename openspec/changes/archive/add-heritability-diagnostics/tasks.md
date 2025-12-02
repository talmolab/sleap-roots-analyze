# Implementation Tasks

## 1. Test Fixtures and Test Infrastructure
- [ ] 1.1 Create `heritability_diagnostic_zero_variance` fixture - Trait with no between-genotype variance
- [ ] 1.2 Create `heritability_diagnostic_high_within_variance` fixture - Trait with high replicate variation
- [ ] 1.3 Create `heritability_diagnostic_low_sample_size` fixture - Trait with minimal data
- [ ] 1.4 Create `heritability_diagnostic_mixed_quality` fixture - Dataset with mix of good and bad traits
- [ ] 1.5 Create helper function `assert_diagnostic_result_structure()` for validating diagnostic output format

## 2. Variance Analysis Function (TDD)
- [ ] 2.1 Write tests for `analyze_trait_variance()` - Success cases with known variance values
- [ ] 2.2 Write tests for `analyze_trait_variance()` - Edge cases (empty data, single genotype, all NaN)
- [ ] 2.3 Write tests for `analyze_trait_variance()` - Variance decomposition correctness (between + within = total)
- [ ] 2.4 Implement `analyze_trait_variance()` in statistics.py to pass all tests
- [ ] 2.5 Add docstring with Google format and example usage

## 3. Issue Diagnosis Function (TDD)
- [ ] 3.1 Write tests for `diagnose_heritability_issues()` - Correctly identifies zero variance issue
- [ ] 3.2 Write tests for `diagnose_heritability_issues()` - Correctly identifies high within-variance issue
- [ ] 3.3 Write tests for `diagnose_heritability_issues()` - Correctly identifies low sample size issue
- [ ] 3.4 Write tests for `diagnose_heritability_issues()` - Returns empty issues list for healthy traits
- [ ] 3.5 Write tests for `diagnose_heritability_issues()` - Handles missing heritability results gracefully
- [ ] 3.6 Implement `diagnose_heritability_issues()` in statistics.py to pass all tests
- [ ] 3.7 Add docstring with Google format and example usage

## 4. Trait Comparison Function (TDD)
- [ ] 4.1 Write tests for `compare_trait_heritabilities()` - Returns correctly structured DataFrame
- [ ] 4.2 Write tests for `compare_trait_heritabilities()` - Includes all expected columns (H², var_genetic, var_residual, etc.)
- [ ] 4.3 Write tests for `compare_trait_heritabilities()` - Handles traits with errors in heritability results
- [ ] 4.4 Write tests for `compare_trait_heritabilities()` - Handles empty trait list
- [ ] 4.5 Write tests for `compare_trait_heritabilities()` - Correctly calculates percentage variance between genotypes
- [ ] 4.6 Implement `compare_trait_heritabilities()` in statistics.py to pass all tests
- [ ] 4.7 Add docstring with Google format and example usage

## 5. Variance Decomposition Plot (TDD)
- [ ] 5.1 Write tests for `create_variance_decomposition_plot()` - Creates 4-panel figure
- [ ] 5.2 Write tests for `create_variance_decomposition_plot()` - Each subplot has correct data and labels
- [ ] 5.3 Write tests for `create_variance_decomposition_plot()` - Handles empty comparison DataFrame
- [ ] 5.4 Write tests for `create_variance_decomposition_plot()` - Returns matplotlib Figure object
- [ ] 5.5 Write tests for `create_variance_decomposition_plot()` - Optional save_path parameter works
- [ ] 5.6 Implement `create_variance_decomposition_plot()` in visualization.py to pass all tests
- [ ] 5.7 Add docstring with Google format and example usage

## 6. Trait Boxplot Visualization (TDD)
- [ ] 6.1 Write tests for `create_trait_by_genotype_boxplots()` - Creates multi-panel boxplot figure
- [ ] 6.2 Write tests for `create_trait_by_genotype_boxplots()` - Boxplots show correct trait distributions by genotype
- [ ] 6.3 Write tests for `create_trait_by_genotype_boxplots()` - H² annotations appear in subplot titles
- [ ] 6.4 Write tests for `create_trait_by_genotype_boxplots()` - Handles single trait case
- [ ] 6.5 Write tests for `create_trait_by_genotype_boxplots()` - Handles traits with missing data
- [ ] 6.6 Implement `create_trait_by_genotype_boxplots()` in visualization.py to pass all tests
- [ ] 6.7 Add docstring with Google format and example usage

## 7. Diagnostic Dashboard Visualization (TDD)
- [ ] 7.1 Write tests for `create_heritability_diagnostic_dashboard()` - Combines comparison data and boxplots
- [ ] 7.2 Write tests for `create_heritability_diagnostic_dashboard()` - Creates comprehensive 4-panel + boxplot figure
- [ ] 7.3 Write tests for `create_heritability_diagnostic_dashboard()` - Optional layout parameter (vertical vs horizontal)
- [ ] 7.4 Write tests for `create_heritability_diagnostic_dashboard()` - Returns Figure with correct subplot arrangement
- [ ] 7.5 Implement `create_heritability_diagnostic_dashboard()` in visualization.py to pass all tests
- [ ] 7.6 Add docstring with Google format and example usage

## 8. Pipeline Integration (TDD)
- [ ] 8.1 Write tests for FilterHeritabilityStep with `generate_diagnostics=True`
- [ ] 8.2 Write tests verifying diagnostic CSV files are exported
- [ ] 8.3 Write tests verifying diagnostic plots are saved
- [ ] 8.4 Write tests verifying diagnostic results stored in metadata
- [ ] 8.5 Write tests ensuring diagnostic mode doesn't break existing functionality
- [ ] 8.6 Modify FilterHeritabilityStep to add diagnostic mode
- [ ] 8.7 Update FilterHeritabilityStep docstring with diagnostic mode documentation

## 9. Integration Tests
- [ ] 9.1 Write end-to-end test: Load data → Calculate H² → Run diagnostics → Verify results
- [ ] 9.2 Write integration test: Multiple traits with different variance patterns
- [ ] 9.3 Write integration test: Pipeline with diagnostic mode enabled
- [ ] 9.4 Write integration test: Diagnostic results match manual variance calculations
- [ ] 9.5 Verify all integration tests pass

## 10. Documentation and Examples
- [ ] 10.1 Add example usage to statistics.py module docstring
- [ ] 10.2 Add example usage to visualization.py diagnostic functions
- [ ] 10.3 Create example notebook: `examples/heritability_diagnostics.ipynb`
- [ ] 10.4 Update CLAUDE.md with diagnostic function descriptions
- [ ] 10.5 Add diagnostic workflow to pipeline documentation

## 11. Code Quality and Coverage
- [ ] 11.1 Run `uv run pytest tests/test_statistics.py -v` - All new tests pass
- [ ] 11.2 Run `uv run pytest tests/test_visualization.py -v` - All new tests pass
- [ ] 11.3 Run `uv run pytest --cov=src/sleap_roots_analyze/statistics --cov-branch` - Maintain >95% coverage
- [ ] 11.4 Run `uv run pytest --cov=src/sleap_roots_analyze/visualization --cov-branch` - Maintain >90% coverage
- [ ] 11.5 Run `uv run black src/sleap_roots_analyze tests` - Format all code
- [ ] 11.6 Run `uv run ruff check src/sleap_roots_analyze tests` - Fix all linting issues
- [ ] 11.7 Verify all docstrings pass pydocstyle checks

## 12. Validation and Cleanup
- [ ] 12.1 Remove temporary diagnostic scripts (diagnose_heritability.py, diagnose_heritability_notebook.py)
- [ ] 12.2 Run full test suite: `uv run pytest` - All tests pass
- [ ] 12.3 Verify no regression in existing functionality
- [ ] 12.4 Update all task checkboxes to [x] in this file
- [ ] 12.5 Run `openspec validate add-heritability-diagnostics --strict` - Passes validation
