# Tasks: Polish Grouped Pipeline Robustness

## Phase 1: TDD RED - Write Failing Tests

### Test File Setup
- [ ] 1.1 Create `tests/test_grouped_pipeline_robustness.py`
- [ ] 1.2 Add imports and shared fixtures (MockPipeline, test data generators)

### HIGH Priority Tests
- [ ] 2.1 Write `test_pipeline_failure_logs_group_context`
  - Mock pipeline that raises exception
  - Verify error log includes group context
  - Verify exception re-raised (fail fast)

### MEDIUM Priority Tests
- [ ] 3.1 Write `test_empty_groups_after_filtering_warns`
  - Create sparse data (all groups < min_samples)
  - Verify WARNING log about no valid groups
  - Verify empty result returned

- [ ] 3.2 Write `test_mixed_type_group_values_sorted_gracefully`
  - Create data with mixed string/numeric group IDs
  - Verify no TypeError during sorting
  - Verify all groups processed

- [ ] 3.3 Write `test_missing_csv_error_message_includes_context`
  - Use nonexistent CSV path
  - Verify error message mentions group_by

- [ ] 3.4 Write `test_empty_csv_error_message_includes_context`
  - Use empty CSV file
  - Verify error message mentions group_by context

- [ ] 3.5 Write `test_missing_group_column_error_message_clear`
  - Use CSV without group_by column
  - Verify KeyError mentions missing column

### Run Tests to Confirm RED
- [ ] 4.1 Run: `pytest tests/test_grouped_pipeline_robustness.py -v`
- [ ] 4.2 Verify all 7 tests FAIL (expected behavior)

## Phase 2: TDD GREEN - Implement Fixes

### HIGH Priority Fixes
- [ ] 5.1 Add exception logging in `run_grouped_pipelines()` at line 671
  - Add try-except around `pipeline.run()`
  - Log error with `logger.exception()` including group context
  - Re-raise exception to fail fast
- [ ] 5.2 Run: `pytest tests/test_grouped_pipeline_robustness.py::test_pipeline_failure_logs_group_context -v`
- [ ] 5.3 Verify test PASSES

### MEDIUM Priority Fixes
- [ ] 6.1 Add empty groups warning at line 607
  - Check `if not valid_groups:` after filtering
  - Log WARNING with min_samples context
  - Log skipped groups if available
  - Return empty dict
- [ ] 6.2 Run: `pytest tests/test_grouped_pipeline_robustness.py::test_empty_groups_after_filtering_warns -v`
- [ ] 6.3 Verify test PASSES

- [ ] 7.1 Add mixed-type sorting fallback at line 609
  - Wrap `sorted()` in try-except
  - Catch TypeError
  - Log WARNING about mixed types
  - Fallback to `sorted(keys, key=lambda v: str(v))`
- [ ] 7.2 Run: `pytest tests/test_grouped_pipeline_robustness.py::test_mixed_type_group_values_sorted_gracefully -v`
  - 7.3 Verify test PASSES

- [ ] 8.1 Add contextualized error handling at line 595
  - Wrap CSV read in try-except FileNotFoundError
  - Wrap grouping in try-except (catch all except KeyError)
  - Add group_by context to all error messages
  - Preserve original KeyError from split_data_by_group
- [ ] 8.2 Run: `pytest tests/test_grouped_pipeline_robustness.py -k error_message -v`
- [ ] 8.3 Verify all error message tests PASS

### LOW Priority Fixes (Code Cleanup)
- [ ] 9.1 Remove `import tempfile` from line ~20 (no longer needed after Bug #1 fix)
- [ ] 9.2 Remove `from copy import deepcopy` from line ~20 (no longer needed)
- [ ] 9.3 Remove `original_pipeline_name = config.pipeline_name` from line ~630
- [ ] 9.4 Update `.github/workflows/ci.yml` comment: "issue #70" → "issue #69"

### Run All Tests to Confirm GREEN
- [ ] 10.1 Run: `pytest tests/test_grouped_pipeline_robustness.py -v`
- [ ] 10.2 Verify all 7 tests PASS

## Phase 3: Integration Testing

### Regression Testing
- [ ] 11.1 Run: `pytest tests/test_grouped_pipeline*.py -v`
- [ ] 11.2 Verify all existing grouped pipeline tests still pass
- [ ] 11.3 Run: `pytest tests/test_data_grouping.py -v`
- [ ] 11.4 Verify all data grouping tests still pass

### Code Quality Checks
- [ ] 12.1 Run: `uv run black --check src/sleap_roots_analyze tests`
- [ ] 12.2 Run: `uv run ruff check src/sleap_roots_analyze`
- [ ] 12.3 Verify no lint issues

### Coverage Check
- [ ] 13.1 Run: `pytest --cov=src/sleap_roots_analyze/pipeline/utils --cov-report=term-missing tests/`
- [ ] 13.2 Verify coverage for utils.py >90%
- [ ] 13.3 Verify new error handling branches covered

## Phase 4: Manual Verification

### Edge Case Testing
- [ ] 14.1 Create test config with min_samples_per_trait that filters all groups
- [ ] 14.2 Run: `sleap-roots-analyze qc <config>`
- [ ] 14.3 Verify WARNING log about no valid groups appears
- [ ] 14.4 Verify pipeline exits gracefully

- [ ] 14.5 Create test data with mixed string/numeric group IDs
- [ ] 14.6 Run grouped pipeline
- [ ] 14.7 Verify WARNING about mixed types appears
- [ ] 14.8 Verify all groups processed successfully

### Failure Scenario Testing
- [ ] 15.1 Intentionally create pipeline that fails mid-execution
- [ ] 15.2 Run grouped pipeline
- [ ] 15.3 Verify ERROR log shows which group failed
- [ ] 15.4 Verify pipeline stops (doesn't continue with other groups)

## Phase 5: Commit and Documentation

### Git Commit
- [ ] 16.1 Stage all changes: `git add src/sleap_roots_analyze/pipeline/utils.py .github/workflows/ci.yml tests/test_grouped_pipeline_robustness.py`
- [ ] 16.2 Commit with detailed message (see proposal for full commit message template)
- [ ] 16.3 Verify commit includes all 7 fixes + 7 tests

### Update OpenSpec
- [ ] 17.1 Mark all tasks complete in this tasks.md
- [ ] 17.2 Update proposal.md status: "Proposed" → "Implemented"

### PR Description Update
- [ ] 18.1 Add note to PR #63 description about polish improvements
- [ ] 18.2 Link to closed Copilot review comments
- [ ] 18.3 Mention all 7 fixes addressed

## Summary

**Total Tasks**: 45
**Estimated Time**: 1.5-2 hours
**Test Count**: 7 new tests
**Files Modified**: 2 (utils.py, ci.yml)
**Files Created**: 1 (test_grouped_pipeline_robustness.py)
