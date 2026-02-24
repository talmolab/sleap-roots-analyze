# Tasks: Fix Critical Bugs in Grouped Pipeline Execution

## Phase 1: Write Failing Tests (TDD RED)

### Bug #1: Config Persistence Tests
- [ ] 1.1 Create `tests/test_grouped_pipeline_config_persistence.py`
- [ ] 1.2 Write `test_saved_config_csv_path_exists`
- [ ] 1.3 Write `test_saved_config_is_reproducible`
- [ ] 1.4 Write `test_input_csv_preserved_in_output_directory`
- [ ] 1.5 Run tests - confirm they FAIL

### Bug #2: CLI Group-By Tests
- [ ] 2.1 Create `tests/test_run_all_cli_group_by.py`
- [ ] 2.2 Write `test_cli_group_by_triggers_viz_fanout_when_config_has_no_group_by`
- [ ] 2.3 Write `test_cli_group_by_overrides_config_group_by_for_viz_fanout`
- [ ] 2.4 Write `test_effective_group_by_tracked_correctly`
- [ ] 2.5 Run tests - confirm they FAIL

### Bug #3: NaN Handling Tests
- [ ] 3.1 Create `tests/test_grouped_pipeline_nan_handling.py`
- [ ] 3.2 Write `test_nan_group_values_logged_and_dropped_by_default`
- [ ] 3.3 Write `test_nan_group_values_can_be_treated_as_group`
- [ ] 3.4 Write `test_grouped_pipeline_with_nan_values`
- [ ] 3.5 Run tests - confirm they FAIL

## Phase 2: Implement Fixes (TDD GREEN)

### Bug #1: Fix Config Persistence
- [ ] 4.1 Modify `run_grouped_pipelines()` to write group CSV to output dir (not temp)
- [ ] 4.2 Update filename to `00_input_data_{group_label}.csv`
- [ ] 4.3 Remove `finally` block that deletes temp CSV
- [ ] 4.4 Run Bug #1 tests - confirm they PASS

### Bug #2: Fix CLI Group-By Detection
- [ ] 5.1 Add effective group_by tracking to `_run_qc_pipelines()`
- [ ] 5.2 Use `self.group_by or config_group_by` for detection logic
- [ ] 5.3 Update grouped output detection to use effective group_by
- [ ] 5.4 Add logging for CLI/config/effective group_by values
- [ ] 5.5 Run Bug #2 tests - confirm they PASS

### Bug #3: Fix NaN Handling
- [ ] 6.1 Add `handle_na` parameter to `split_data_by_group()`
- [ ] 6.2 Check for NaN values before grouping
- [ ] 6.3 Log warning when NaN values are dropped
- [ ] 6.4 Use `df.groupby(..., dropna=True/False)` instead of manual filtering
- [ ] 6.5 Run Bug #3 tests - confirm they PASS

## Phase 3: Integration Testing

- [ ] 7.1 Run full test suite: `uv run pytest tests/ -v`
- [ ] 7.2 Verify no regressions in existing tests
- [ ] 7.3 Run grouped integration tests specifically
- [ ] 7.4 Verify all 1,856+ tests still pass

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

## Phase 5: Commit and Archive

- [ ] 9.1 Commit Bug #1 fix with passing tests
- [ ] 9.2 Commit Bug #2 fix with passing tests
- [ ] 9.3 Commit Bug #3 fix with passing tests
- [ ] 9.4 Update this tasks.md with completion status
- [ ] 9.5 Archive OpenSpec change
- [ ] 9.6 Push commits to PR
