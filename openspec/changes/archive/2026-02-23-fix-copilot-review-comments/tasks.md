# Tasks: Address Copilot Review Comments

## Phase 1: Code Cleanup (No Tests)

- [x] 1.1 Remove unused `original_pipeline_name` variable in `src/sleap_roots_analyze/pipeline/utils.py` (already fixed in codebase)
- [x] 1.2 Move `import tempfile` to top of file (already fixed in codebase)
- [x] 1.3 Move `import copy` to top of file (already fixed in codebase)
- [x] 1.4 Fix issue reference in `.github/workflows/ci.yml` (#70 → #69)
- [x] 1.5 Commit: "chore: remove unused variable and fix imports in pipeline utils" (commit: 1f66e2b)

## Phase 2: Error Handling (TDD)

### 2.1 Write Tests First
- [x] 2.1.1 Add `TestErrorHandling` class to `tests/test_grouped_pipeline_execution_error_handling.py`
- [x] 2.1.2 Write `test_file_not_found_error_includes_context`
- [x] 2.1.3 Write `test_empty_data_error_includes_context`
- [x] 2.1.4 Write `test_key_error_from_split_preserved`
- [x] 2.1.5 Write `test_empty_groups_warning_logged`
- [x] 2.1.6 Write `test_mixed_type_groups_sorted_safely`
- [x] 2.1.7 Write `test_group_failure_logged_and_continues`
- [x] 2.1.8 Run tests - confirmed they FAIL (TDD Red phase)

### 2.2 Implement Fixes
- [x] 2.2.1 Add try-except block for CSV read/group split with contextual errors
- [x] 2.2.2 Add empty groups warning after `filter_valid_groups` (already existed)
- [x] 2.2.3 Add mixed-type group handling with try-except around `sorted()`
- [x] 2.2.4 Add exception logging in group pipeline execution except block
- [x] 2.2.5 Run tests - confirmed they PASS (TDD Green phase)
- [x] 2.2.6 Commit: "feat: improve error handling in grouped pipeline execution" (commit: 43b5b8b)

## Phase 3: Early Validation (TDD)

### 3.1 Write Tests First
- [x] 3.1.1 Add tests to `tests/test_pipeline_config_group_by_validation.py` (new file)
- [x] 3.1.2 Write `test_validate_qc_config_group_by_column_missing`
- [x] 3.1.3 Write `test_validate_qc_config_group_by_validation_skipped_when_check_files_false`
- [x] 3.1.4 Write `test_validate_qc_config_group_by_validation_skipped_when_csv_missing`
- [x] 3.1.5 Write `test_validate_qc_config_group_by_null_skips_validation`
- [x] 3.1.6 Write `test_validate_qc_config_group_by_exists_passes`
- [x] 3.1.7 Write `test_validate_qc_config_group_by_with_empty_csv_skips`
- [x] 3.1.8 Run tests - confirmed they FAIL (TDD Red phase)

### 3.2 Implement Fix
- [x] 3.2.1 Add group_by validation to `validate_qc_config()` in `src/sleap_roots_analyze/pipeline/config/utils.py`
- [x] 3.2.2 Add pandas import to config/utils.py
- [x] 3.2.3 Run tests - confirmed they PASS (TDD Green phase)
- [x] 3.2.4 Commit: "feat: validate group_by column exists in config validation" (commit: 5ab6f4f)

## Phase 4: Integration Testing

- [x] 4.1 Run full test suite: `uv run pytest tests/ -v` (1,856 tests passed, 0 failed)
- [x] 4.2 Run grouped integration tests: included in full suite (733s, 412s)
- [x] 4.3 Verify no regressions - confirmed clean

## Phase 5: Archive

- [x] 5.1 Update this tasks.md with completion status
- [ ] 5.2 Mark OpenSpec change as complete (archive the change)
- [ ] 5.3 Push commits to PR

---

## Completion Summary

**Completed:** 2026-02-23

**All Copilot review comments addressed:**
- Issue 1 (unused variable): Already fixed
- Issue 2 (inline imports): Already fixed
- Issue 3 (wrong issue reference): Fixed in commit 1f66e2b
- Issue 4 (contextual error messages): Fixed in commit 43b5b8b
- Issue 5 (empty groups warning): Already existed
- Issue 6 (mixed-type groups): Fixed in commit 43b5b8b
- Issue 7 (group failure logging): Fixed in commit 43b5b8b
- Issue 8 (early validation): Fixed in commit 5ab6f4f

**Test Results:**
- Phase 2: 6/6 tests passing
- Phase 3: 6/6 tests passing
- Integration: 1,856/1,856 tests passing, 0 failures

**Commits:**
1. `1f66e2b` - chore: fix issue reference in CI workflow
2. `43b5b8b` - feat: improve error handling in grouped pipeline execution
3. `5ab6f4f` - feat: validate group_by column exists in config validation
