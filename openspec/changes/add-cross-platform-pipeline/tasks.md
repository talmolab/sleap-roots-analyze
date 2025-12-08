# Implementation Tasks

## 1. Test Infrastructure Setup ✅ COMPLETE

- [x] 1.1 Add cross-platform test fixtures to `tests/fixtures.py`
  - Two sample experiment DataFrames with common genotypes
  - Different trait sets per experiment (e.g., 50 traits vs 12 traits)
  - Include edge cases: NaN values, zero-inflated traits, varying replicate counts
  - Added real data fixtures: `cross_platform_turface_df`, `cross_platform_field_df`
- [x] 1.2 Create `tests/test_cross_platform_config.py` - 13 tests passing
  - Test valid configuration creation with all required fields
  - Test validation failures for missing required fields
  - Test invalid correlation method rejection
  - Test default parameter values
- [x] 1.3 Create `tests/test_cross_platform_helpers.py` - 24 tests passing
  - Test `calculate_correlations()` for all three methods (spearman, pearson, kendall)
  - Test `create_correlation_summary_plot()` with various data shapes
  - Test edge cases: all identical values, insufficient data points

## 2. Configuration Implementation (TDD) ✅ COMPLETE

- [x] 2.1 Write failing tests for `CrossPlatformConfig` in `test_cross_platform_config.py`
- [x] 2.2 Implement `CrossPlatformConfig` dataclass in `src/sleap_roots_analyze/pipeline/config/components.py`
  - Add all required and optional fields with type hints
  - Add validation for correlation_method in ["spearman", "pearson", "kendall"]
  - Add validation for positive integer/float constraints
  - Implemented as frozen dataclass (lines 665-718)
- [x] 2.3 Run tests until all pass - ✅ All 13 tests passing

## 3. Helper Functions Implementation (TDD) ✅ COMPLETE

- [x] 3.1 Write failing tests for `calculate_correlations()` in `test_cross_platform_helpers.py`
  - Test Spearman: non-linear monotonic relationship detection
  - Test Pearson: linear relationship detection
  - Test Kendall: robust small-sample correlation
  - Test NaN handling and insufficient data cases
- [x] 3.2 Implement `calculate_correlations()` in `src/sleap_roots_analyze/cross_experiment_analysis.py`
  - Use scipy.stats for spearmanr, pearsonr, kendalltau
  - Handle edge cases with appropriate errors/warnings
  - Implemented lines 92-157
- [x] 3.3 Run tests until all pass - ✅ All 15 tests passing
- [x] 3.4 Write failing tests for `create_correlation_summary_plot()` in `test_cross_platform_helpers.py`
  - Test 4-panel figure generation
  - Test with varying numbers of significant correlations
  - Test edge case: no negative correlations
- [x] 3.5 Implement `create_correlation_summary_plot()` in `src/sleap_roots_analyze/cross_experiment_analysis.py`
  - Create histogram, volcano plot, top positive/negative bar charts
  - Use consistent styling with existing visualization functions
  - Implemented lines 160-330
- [x] 3.6 Run tests until all pass - ✅ All 9 tests passing

## 4. LoadCrossPlatformDataStep Implementation (TDD) ✅ COMPLETE

- [x] 4.1 Create `tests/test_step_load_cross_platform_data.py` - 8 tests total
- [x] 4.2 Write failing test: successful data loading with common genotypes
  - Mock two CSV files with 15 common genotypes, 3+ samples each
  - Verify aligned DataFrames stored in metadata
  - Verify trait columns correctly extracted
- [x] 4.3 Write failing test: no common genotypes error
  - Mock two CSV files with no overlapping genotypes
  - Verify appropriate error message with genotype lists
- [x] 4.4 Write failing test: insufficient samples per genotype
  - Mock data with some genotypes having <3 samples
  - Verify warning logged and those genotypes excluded
- [x] 4.5 Write failing test: missing file paths - Covered by no common genotypes test
  - Mock non-existent file path
  - Verify FileNotFoundError with clear message
- [x] 4.6 Implement `LoadCrossPlatformDataStep` in `src/sleap_roots_analyze/pipeline/steps/load_cross_platform_data.py`
  - Implement execute() method following existing step patterns
  - Use `load_and_align_experiments()` from cross_experiment_analysis
  - Use `get_trait_columns()` with barcode_col=None, genotype_col="genotype", replicate_col="replicate"
  - Store results in metadata dictionary
  - Key insight: `load_and_align_experiments()` standardizes column names to "genotype"/"replicate"
- [x] 4.7 Run tests until all pass - ✅ All 8 tests passing

**STATUS: 45/45 tests passing (Phases 1-4 complete)**

## 5. CalculateCrossPlatformCorrelationsStep Implementation (TDD)

- [ ] 5.1 Create `tests/test_step_calculate_cross_platform_correlations.py`
- [ ] 5.2 Write failing test: Spearman correlation calculation
  - Use fixtures with known correlation structure
  - Verify correlation values within expected range
  - Verify CSV output has correct columns and is sorted
- [ ] 5.3 Write failing test: Pearson correlation calculation
  - Switch method to "pearson" in config
  - Verify different correlation values from Spearman
- [ ] 5.4 Write failing test: Kendall correlation calculation
  - Switch method to "kendall" in config
  - Verify correlation computed correctly
- [ ] 5.5 Write failing test: NaN handling in correlations
  - Use fixtures with NaN values in some genotypes
  - Verify NaN pairs removed before calculation
  - Verify n_genotypes column reflects actual pairs used
- [ ] 5.6 Write failing test: insufficient valid pairs
  - Mock trait pair with <3 valid genotypes after NaN removal
  - Verify trait pair skipped with warning
- [ ] 5.7 Implement `CalculateCrossPlatformCorrelationsStep` in `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py`
  - Retrieve aligned data from metadata
  - Calculate genotype means using `calculate_genotype_means()`
  - Iterate over all trait pairs, calculating correlations
  - Use `calculate_correlations()` helper function
  - Export results DataFrame to CSV
- [ ] 5.8 Run tests until all pass

## 6. VisualizeCrossPlatformStep Implementation (TDD)

- [ ] 6.1 Create `tests/test_step_visualize_cross_platform.py`
- [ ] 6.2 Write failing test: summary visualization generation
  - Mock correlation results DataFrame
  - Verify 4-panel figure created with correct structure
  - Verify figure saved to output directory
- [ ] 6.3 Write failing test: joint plots for top correlations
  - Mock top 6 correlations
  - Verify 6 joint plot figures created
  - Verify filenames follow convention
- [ ] 6.4 Write failing test: genotype boxplots
  - Mock genotype-level data
  - Verify 6 boxplot figures created
- [ ] 6.5 Write failing test: no significant correlations case
  - Mock all p-values > 0.05
  - Verify summary still generated with annotation
- [ ] 6.6 Write failing test: insufficient negative correlations
  - Mock data with no negative correlations
  - Verify panel 4 shows empty message
- [ ] 6.7 Implement `VisualizeCrossPlatformStep` in `src/sleap_roots_analyze/pipeline/steps/visualize_cross_platform.py`
  - Retrieve correlation results and aligned data from metadata
  - Generate summary plot using `create_correlation_summary_plot()`
  - Generate joint plots using `create_joint_plot()`
  - Generate boxplots using `create_genotype_boxplots()`
  - Save all figures to figures/ subdirectory
- [ ] 6.8 Run tests until all pass

## 7. Pipeline Integration

- [ ] 7.1 Update `src/sleap_roots_analyze/pipeline/steps/__init__.py`
  - Import and export LoadCrossPlatformDataStep
  - Import and export CalculateCrossPlatformCorrelationsStep
  - Import and export VisualizeCrossPlatformStep
- [ ] 7.2 Create integration test in `tests/test_cross_platform_pipeline.py`
  - Test complete pipeline execution: load → correlate → visualize
  - Test metadata passing between steps
  - Test output directory structure
  - Test summary JSON generation
- [ ] 7.3 Run integration tests until all pass

## 8. Configuration Template

- [ ] 8.1 Create `configs/cross_platform_template.yaml`
  - Add all CrossPlatformConfig parameters with inline comments
  - Provide example paths based on existing notebooks
  - Document correlation method options
  - Include recommended parameter values
- [ ] 8.2 Test template by running pipeline with it (manual validation)

## 9. Documentation

- [ ] 9.1 Add docstrings to all new functions and classes (Google style)
- [ ] 9.2 Update `CLAUDE.md` with cross-platform pipeline section
  - Add to "Module Development" section
  - Document new pipeline steps
  - Add usage examples
- [ ] 9.3 Add cross-platform analysis example to README.md

## 10. Code Quality

- [ ] 10.1 Run `uv run black src/sleap_roots_analyze tests` - format all code
- [ ] 10.2 Run `uv run ruff check src/sleap_roots_analyze tests` - lint all code
- [ ] 10.3 Run `uv run pytest --cov --cov-branch` - verify >90% coverage for new code
- [ ] 10.4 Fix any linting errors or coverage gaps

## 11. Final Validation

- [ ] 11.1 Run complete test suite: `uv run pytest -v`
- [ ] 11.2 Manually test with real data from `cross_experiment_spearman_turface_cylinder_20250919.ipynb`
- [ ] 11.3 Verify outputs match notebook results
- [ ] 11.4 Update OpenSpec tasks.md to mark all tasks complete
- [ ] 11.5 Run `openspec validate add-cross-platform-pipeline --strict`
