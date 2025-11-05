# Add QC Pipeline Step Tests (Steps 3-10)

## Why

Currently, only Steps 1-2 of the QC pipeline have dedicated unit test files (`test_step_load_data.py`, `test_step_cleanup.py`). Steps 3-10 are only tested through integration tests, making it difficult to debug individual step failures, understand expected behavior, and maintain the pipeline as it evolves. This creates a testing gap that could lead to undetected regressions and makes onboarding new contributors harder.

Adding comprehensive unit tests for these steps will improve debuggability, provide clear documentation of step behavior, and ensure easier maintenance and refactoring before the v1.0 release.

## What Changes

- Add 8 new test files covering QC pipeline Steps 3-10:
  - `test_step_validate_clean.py` - Step 3: ValidateCleanStep
  - `test_step_exploratory_analysis.py` - Step 4: ExploratoryAnalysisStep
  - `test_step_detect_outliers.py` - Step 5: DetectOutliersStep
  - `test_step_visualize_outliers.py` - Step 6: VisualizeOutliersStep
  - `test_step_statistical_analysis.py` - Step 8: StatisticalAnalysisStep
  - `test_step_filter_heritability.py` - Step 9: FilterHeritabilityStep
  - `test_step_generate_summary.py` - Step 10: GenerateSummaryStep

- Note: Step 7 (RemoveOutliersStep) already has `test_step_remove_outliers.py` ✅

- Each test file will follow the established testing patterns from existing step tests and cover:
  - Basic execution with valid inputs
  - Edge cases (empty data, missing columns, invalid values)
  - Error handling (invalid config, malformed data)
  - Output file generation and content validation
  - Metadata propagation to next step
  - Integration with underlying analysis functions

- Add any necessary test fixtures to support the new tests (following centralized fixture pattern in `conftest.py` and `fixtures.py`)

## Impact

### Affected Specs
- `qc-pipeline-testing` - New capability spec for QC pipeline step testing requirements

### Affected Code
- **New test files**: `tests/test_step_validate_clean.py`, `tests/test_step_exploratory_analysis.py`, `tests/test_step_detect_outliers.py`, `tests/test_step_visualize_outliers.py`, `tests/test_step_statistical_analysis.py`, `tests/test_step_filter_heritability.py`, `tests/test_step_generate_summary.py`
- **Potential fixture updates**: `tests/conftest.py`, `tests/fixtures.py` (if new fixtures needed)
- **Test coverage**: Overall project test coverage expected to increase from current ~88% to ~92-95%

### Benefits
- **Better debugging**: Isolate failures to specific steps without running full pipeline
- **Documentation**: Tests serve as executable examples of step behavior
- **Confidence**: Catch regressions in individual steps before they reach integration tests
- **Maintenance**: Easier refactoring with comprehensive step-level coverage
- **Onboarding**: New contributors can understand step behavior through tests

### Risks
- **Low risk**: Non-breaking change, only adding tests
- **Test maintenance**: More test files to maintain (mitigated by following established patterns)
