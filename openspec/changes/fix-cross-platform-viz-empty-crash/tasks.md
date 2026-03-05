# Tasks: fix-cross-platform-viz-empty-crash

## Task 1: Write failing tests -- empty correlation crash (TDD Red Phase)
- [x] 1.1 Add `test_visualize_empty_correlation_df_does_not_crash` -- pass empty correlation_df with correct schema; verify step completes without error and returns status "success"
- [x] 1.2 Add `test_visualize_empty_correlation_df_generates_no_files` -- verify `files_generated` is empty list and `plots_generated` is 0 in metadata
- [x] 1.3 Add `test_visualize_empty_correlation_df_logs_warning` -- verify a warning is logged when correlation_df is empty
- [x] 1.4 Run tests, confirm all FAIL (red) on current code (ValueError crash)

## Task 2: Write failing tests -- metadata and downstream (TDD Red Phase)
- [x] 2.1 Add `test_visualize_empty_metadata_includes_flag` -- verify `metadata["empty_correlations"]` is `True` when correlation_df is empty
- [x] 2.2 Add `test_visualize_nonempty_metadata_no_empty_flag` -- verify `metadata` does NOT contain `"empty_correlations"` when correlation_df has data (default behavior preserved)
- [x] 2.3 Run tests, confirm FAIL on current code

## Task 3: Write failing integration test -- pipeline with empty correlations (TDD Red Phase)
- [x] 3.1 Add `test_cross_platform_pipeline_completes_with_no_shared_genotypes` -- integration test: two experiments with zero shared genotypes; verify full pipeline returns status "success"
- [x] 3.2 Run test, confirm FAIL (crash in visualization step)

## Task 4: Implement the fix (TDD Green Phase)
- [x] 4.1 In `VisualizeCrossPlatformStep.execute()`, after extracting `correlation_df`, add early return when `correlation_df.empty`
- [x] 4.2 In the early return path: log warning, return StepResult with `data=data`, `metadata={"plots_generated": 0, "empty_correlations": True, ...}`, `files_generated=[]`
- [x] 4.3 Run all new tests, confirm all PASS (green)

## Task 5: Verify no regressions
- [x] 5.1 All existing `VisualizeCrossPlatformStep` tests pass unchanged
- [x] 5.2 All existing `CalculateCrossPlatformCorrelationsStep` tests pass unchanged
- [x] 5.3 Full test suite passes (`uv run pytest`)
- [x] 5.4 Linting and formatting pass (`uv run ruff check`, `uv run black --check`)
