# Implementation Tasks

## 1. Preparation
- [ ] 1.1 Review existing test patterns in `test_step_load_data.py` and `test_step_cleanup.py`
- [ ] 1.2 Review existing visualization step tests for modern patterns (`test_step_generate_static_figures.py`, etc.)
- [ ] 1.3 Review centralized fixtures in `conftest.py` and `fixtures.py`
- [ ] 1.4 Identify reusable fixtures vs. step-specific fixtures needed

## 2. Test File: Step 3 - Validate Clean
- [ ] 2.1 Create `tests/test_step_validate_clean.py`
- [ ] 2.2 Implement tests for basic execution with valid cleaned data
- [ ] 2.3 Implement tests for validation failure cases (unexpected NaNs, invalid columns)
- [ ] 2.4 Implement tests for edge cases (empty data, single sample)
- [ ] 2.5 Implement tests for output file generation and manifest
- [ ] 2.6 Implement tests for metadata propagation
- [ ] 2.7 Run tests and verify coverage

## 3. Test File: Step 4 - Exploratory Analysis
- [ ] 3.1 Create `tests/test_step_exploratory_analysis.py`
- [ ] 3.2 Implement tests for basic execution (generates correlation matrix, trait statistics)
- [ ] 3.3 Implement tests for edge cases (single trait, highly correlated traits)
- [ ] 3.4 Implement tests for error handling (non-numeric data, insufficient samples)
- [ ] 3.5 Implement tests for CSV output validation
- [ ] 3.6 Implement tests for visualization generation (if applicable)
- [ ] 3.7 Run tests and verify coverage

## 4. Test File: Step 5 - Detect Outliers
- [ ] 4.1 Create `tests/test_step_detect_outliers.py`
- [ ] 4.2 Implement tests for Mahalanobis distance detection
- [ ] 4.3 Implement tests for Isolation Forest detection (if enabled)
- [ ] 4.4 Implement tests for combined outlier detection methods
- [ ] 4.5 Implement tests for edge cases (no outliers, all outliers, insufficient samples)
- [ ] 4.6 Implement tests for outlier indices output and manifest
- [ ] 4.7 Implement tests for configuration variations (thresholds, methods)
- [ ] 4.8 Run tests and verify coverage

## 5. Test File: Step 6 - Visualize Outliers
- [ ] 5.1 Create `tests/test_step_visualize_outliers.py`
- [ ] 5.2 Implement tests for PCA outlier plots generation
- [ ] 5.3 Implement tests for outlier distance plots
- [ ] 5.4 Implement tests for edge cases (no outliers detected, no PCA results)
- [ ] 5.5 Implement tests for multiple output formats (PNG, PDF, SVG)
- [ ] 5.6 Implement tests for figure file validation
- [ ] 5.7 Implement tests for manifest generation
- [ ] 5.8 Run tests and verify coverage

## 6. Test File: Step 8 - Statistical Analysis
- [ ] 6.1 Create `tests/test_step_statistical_analysis.py`
- [ ] 6.2 Implement tests for heritability calculation
- [ ] 6.3 Implement tests for ANOVA analysis
- [ ] 6.4 Implement tests for trait statistics computation
- [ ] 6.5 Implement tests for edge cases (single genotype, insufficient replicates)
- [ ] 6.6 Implement tests for CSV output validation
- [ ] 6.7 Implement tests for metadata propagation
- [ ] 6.8 Run tests and verify coverage

## 7. Test File: Step 9 - Filter Heritability
- [ ] 7.1 Create `tests/test_step_filter_heritability.py`
- [ ] 7.2 Implement tests for basic heritability filtering
- [ ] 7.3 Implement tests for threshold variations (0.0, 0.3, 0.5, 1.0)
- [ ] 7.4 Implement tests for edge cases (all traits pass, all traits fail, no H² data)
- [ ] 7.5 Implement tests for removed traits tracking
- [ ] 7.6 Implement tests for filtered data validation
- [ ] 7.7 Implement tests for manifest generation
- [ ] 7.8 Run tests and verify coverage

## 8. Test File: Step 10 - Generate Summary
- [ ] 8.1 Create `tests/test_step_generate_summary.py`
- [ ] 8.2 Implement tests for pipeline summary JSON generation
- [ ] 8.3 Implement tests for summary statistics aggregation
- [ ] 8.4 Implement tests for file list compilation
- [ ] 8.5 Implement tests for edge cases (minimal pipeline, full pipeline)
- [ ] 8.6 Implement tests for summary content validation
- [ ] 8.7 Implement tests for JSON schema validation
- [ ] 8.8 Run tests and verify coverage

## 9. Fixtures and Utilities
- [ ] 9.1 Add any new centralized fixtures to `conftest.py`
- [ ] 9.2 Add step-specific fixtures to individual test files
- [ ] 9.3 Ensure all fixtures follow established naming conventions
- [ ] 9.4 Document any complex fixtures with docstrings

## 10. Final Validation
- [ ] 10.1 Run full test suite: `uv run pytest`
- [ ] 10.2 Run coverage analysis: `uv run pytest --cov --cov-branch`
- [ ] 10.3 Verify coverage meets 90%+ target for new test files
- [ ] 10.4 Run linting: `uv run ruff check tests/test_step_*.py`
- [ ] 10.5 Run formatting: `uv run black tests/test_step_*.py`
- [ ] 10.6 Verify all tests pass in CI (if applicable)
- [ ] 10.7 Update test count in README.md if significantly changed
