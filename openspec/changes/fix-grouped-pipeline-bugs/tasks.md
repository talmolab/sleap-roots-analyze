# Tasks: Fix Critical Bugs in Grouped Pipeline Execution

## Phase 1: Write Failing Tests (TDD RED)

### Bug #1: Config Persistence Tests
- [x] 1.1 Create `tests/test_grouped_pipeline_config_persistence.py`
- [x] 1.2 Write `test_saved_config_csv_path_exists`
- [x] 1.3 Write `test_saved_config_is_reproducible`
- [x] 1.4 Write `test_input_csv_preserved_in_output_directory`
- [x] 1.5 Write `test_no_temporary_files_in_tmp_dir`
- [x] 1.6 Run tests - confirmed they FAIL (2/4 failing as expected)

### Bug #2: CLI Group-By Tests
- [x] 2.1 Create `tests/test_run_all_cli_group_by.py`
- [x] 2.2 Write `test_cli_group_by_triggers_viz_fanout_when_config_has_no_group_by`
- [x] 2.3 Write `test_cli_group_by_overrides_config_group_by_for_viz_fanout`
- [x] 2.4 Write `test_effective_group_by_logged_correctly`
- [x] 2.5 Write `test_viz_fanout_creates_per_group_directories`
- [x] 2.6 Run tests - confirmed they FAIL

### Bug #3: NaN Handling Tests
- [x] 3.1 Create `tests/test_grouped_pipeline_nan_handling.py`
- [x] 3.2 Write `test_nan_group_values_logged_and_dropped_by_default`
- [x] 3.3 Write `test_nan_handling_with_treat_as_group_option`
- [x] 3.4 Write `test_dropped_nan_samples_saved_to_csv`
- [x] 3.5 Write `test_dropped_samples_metadata_file_created`
- [x] 3.6 Write `test_dropped_samples_tracked_in_summary`
- [x] 3.7 Write `test_no_dropped_samples_when_no_nans`
- [x] 3.8 Run tests - confirmed they FAIL

### Bug #4: ANOVA Error Handling Tests (BLOCKING)
- [x] 4.1 Create `tests/test_statistical_analysis_error_handling.py`
- [x] 4.2 Write `test_anova_string_error_handled_gracefully`
- [x] 4.3 Write `test_anova_mixed_success_and_failure`
- [x] 4.4 Run tests - confirmed they FAIL

## Phase 2: Implement Fixes (TDD GREEN)

### Bug #4: Fix ANOVA Error Handling (PRIORITY - unblocks other tests)
- [x] 5.1 Add type check `isinstance(result, str)` before calling `.get()`
- [x] 5.2 Store error message in error column when ANOVA fails
- [x] 5.3 Run Bug #4 tests - confirm they PASS (3/3 passing)
- [x] 5.4 Re-run Bug #1 tests - now complete without crashes
- [x] 5.5 Commit Bug #4 fix (8616f43)

### Bug #1: Fix Config Persistence
- [x] 4.1 Modify `run_grouped_pipelines()` to write group CSV to output dir (not temp)
- [x] 4.2 Update filename to `00_input_data_{group_label}.csv`
- [x] 4.3 Remove `finally` block that deletes temp CSV
- [x] 4.4 Run Bug #1 tests - confirm they PASS (4/4 passing)
- [x] 4.5 Commit Bug #1 fix (386c0b2)

### Bug #2: Fix CLI Group-By Detection
- [x] 5.1 Add effective group_by tracking to `_run_qc_pipelines()`
- [x] 5.2 Use `self.group_by or config_group_by` for detection logic
- [x] 5.3 Update grouped output detection to use effective group_by
- [x] 5.4 Add logging for CLI/config/effective group_by values
- [x] 5.5 Run Bug #2 tests - confirm they PASS (4/4 passing)
- [x] 5.6 Commit Bug #2 fix (3db0560)

### Bug #3: Fix NaN Handling
- [x] 6.1 Add `handle_na` parameter to `split_data_by_group()`
- [x] 6.2 Check for NaN values before grouping
- [x] 6.3 Log warning when NaN values are dropped
- [x] 6.4 Use `df.groupby(..., dropna=True/False)` instead of manual filtering
- [x] 6.5 Save dropped samples to CSV with metadata
- [x] 6.6 Track dropped_samples in result metadata
- [x] 6.7 Run Bug #3 tests - confirm they PASS (6/6 passing)
- [x] 6.8 Commit Bug #3 fix (cb49cb7)

## Phase 3: Integration Testing

- [x] 7.1 Run full test suite: All 17 bug fix tests passing (5 min)
- [x] 7.2 Run grouped integration tests: 67 tests passed (20 min)
- [x] 7.3 Verify no regressions in existing tests
- [x] 7.4 All tests passing with no failures

## Phase 4: Manual Verification

- [ ] 8.1 Run `run-all --group-by plant_age_days` with real data
- [ ] 8.2 Inspect output directory structure
- [ ] 8.3 Verify each group has `00_input_data_{group}.csv` in its output dir
- [ ] 8.4 Load saved `config.yaml` from one group
- [ ] 8.5 Verify csv_path points to existing file in output dir
- [ ] 8.6 Attempt to re-run using saved config: `sleap-roots-analyze qc <saved_config>`
- [ ] 8.7 Verify re-run succeeds (reproducibility test)
- [ ] 8.8 Verify viz fan-out occurred for all groups
- [ ] 8.9 Check viz results match number of QC groups
- [ ] 8.10 Verify dropped NaN samples are saved with metadata

## Phase 5: Commit and Archive

- [x] 9.1 Commit Bug #4 fix with passing tests (8616f43)
- [x] 9.2 Commit Bug #1 fix with passing tests (386c0b2)
- [x] 9.3 Commit Bug #2 fix with passing tests (3db0560)
- [x] 9.4 Commit Bug #3 fix with passing tests (cb49cb7)
- [x] 9.5 Update this tasks.md with completion status
- [ ] 9.6 Archive OpenSpec change
- [ ] 9.7 Push commits to PR
