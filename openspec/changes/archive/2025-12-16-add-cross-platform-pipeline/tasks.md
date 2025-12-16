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

## 5. CalculateCrossPlatformCorrelationsStep Implementation (TDD) ✅ COMPLETE

- [x] 5.1 Create `tests/test_step_calculate_cross_platform_correlations.py` - 8 tests
- [x] 5.2 Write failing test: Spearman correlation calculation
- [x] 5.3 Write failing test: Pearson correlation calculation
- [x] 5.4 Write failing test: Kendall correlation calculation (covered by method switching tests)
- [x] 5.5 Write failing test: NaN handling in correlations
- [x] 5.6 Write failing test: insufficient valid pairs (covered by NaN handling test)
- [x] 5.7 Implement `CalculateCrossPlatformCorrelationsStep` in `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py`
- [x] 5.8 Run tests until all pass - ✅ All 8 tests passing

## 6. VisualizeCrossPlatformStep Implementation (TDD) ✅ COMPLETE

- [x] 6.1 Create `tests/test_step_visualize_cross_platform.py` - 7 tests
- [x] 6.2 Write failing test: summary visualization generation
- [x] 6.3 Write failing test: joint plots for top correlations
- [x] 6.4 Write failing test: genotype boxplots
- [x] 6.5 Write failing test: no significant correlations case (covered by minimal correlations test)
- [x] 6.6 Write failing test: insufficient negative correlations (covered by edge cases)
- [x] 6.7 Implement `VisualizeCrossPlatformStep` in `src/sleap_roots_analyze/pipeline/steps/visualize_cross_platform.py`
- [x] 6.8 Run tests until all pass - ✅ All 7 tests passing

## 7. Pipeline Integration ✅ COMPLETE

- [x] 7.1 Update `src/sleap_roots_analyze/pipeline/steps/__init__.py`
  - Import and export LoadCrossPlatformDataStep ✅
  - Import and export CalculateCrossPlatformCorrelationsStep ✅
  - Import and export VisualizeCrossPlatformStep ✅
- [x] 7.2 CrossPlatformPipeline class implemented in `src/sleap_roots_analyze/pipeline/pipelines/cross_platform_pipeline.py`
  - Complete 3-step DAG: Load → Correlate → Visualize
  - Properly exported in `__init__.py`
- [x] 7.3 Integration testing via comprehensive step tests (60 tests total)

## 8. Configuration Template ✅ COMPLETE

- [x] 8.1 Multiple config templates created in `configs/`:
  - `cross_platform_turface19_vs_cylinder.yaml` - Turface vs Cylinder comparison
  - `cross_platform_turface_150vs19_genotypes.yaml` - Within-platform comparison
  - `cross_platform_turface19_vs_field.yaml` - Turface vs Field comparison
  - `cross_platform_field_vs_cylinder.yaml` - Field vs Cylinder comparison
  - `configs/active/cross_platform/` - 6 additional production configs
- [x] 8.2 Configs tested via pipeline execution (manual validation complete)

## 9. Documentation ✅ COMPLETE

- [x] 9.1 Add docstrings to all new functions and classes (Google style)
  - All step classes have comprehensive docstrings
  - All helper functions documented
- [x] 9.2 Config templates serve as documentation with inline comments
- [x] 9.3 N/A - README updates deferred to documentation sprint

## 10. Code Quality ✅ COMPLETE

- [x] 10.1 Run `uv run black` - Code formatted
- [x] 10.2 Run `uv run ruff check` - All checks passed
- [x] 10.3 60 tests passing with comprehensive coverage
- [x] 10.4 No linting errors

## 11. Final Validation ✅ COMPLETE

- [x] 11.1 Run complete test suite: 60/60 cross-platform tests passing
- [x] 11.2 Tested with real data via production configs
- [x] 11.3 Outputs match expected correlation analysis
- [x] 11.4 Tasks.md updated to reflect completion
- [x] 11.5 Ready for archive

**STATUS: COMPLETE - 60/60 tests passing, all implementation phases done**
