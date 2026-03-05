# Tasks: fix-trait-column-id-pattern-match

## Task 1: Write failing unit tests (TDD Red Phase)
- [x] Add `test_width_columns_not_excluded_by_id_pattern` to `TestGetTraitColumns`
- [x] Add `test_solidity_columns_not_excluded_by_id_pattern` to `TestGetTraitColumns`
- [x] Add `test_actual_id_columns_still_excluded` to `TestGetTraitColumns`
- [x] Add `test_mixed_id_and_width_columns` to `TestGetTraitColumns`
- [x] Run tests, confirm all 4 new tests FAIL (red) — confirmed 3 FAILED, 1 PASSED

## Task 2: Write failing integration tests (TDD Red Phase)
- [x] Add `test_pipeline_trait_classification_11dag` — SLEAP Roots data (880 cols, width/solidity/ID columns)
- [x] Add `test_pipeline_trait_classification_traits_summary` — summary data (924 cols, different column order)
- [x] Add `test_pipeline_trait_classification_traits_summary_lateral` — lateral root data (different trait prefixes)
- [x] Add `test_pipeline_trait_classification_turface` — agronomic data (41 cols, different naming conventions)
- [x] Add `test_pipeline_trait_classification_features` — RhizoVision output (dotted PascalCase column names)
- [x] Add `test_metadata_columns_are_complement_of_traits` — regression guard across all datasets
- [x] Run tests, confirm all 6 integration tests FAIL (red) — confirmed 4 FAILED, 2 PASSED

## Task 3: Implement the fix (TDD Green Phase)
- [x] Split `common_metadata` into `metadata_substring_patterns` and `metadata_suffix_patterns`
- [x] Remove bare `"id"` and redundant `"plant_id"` from patterns
- [x] Add `"_id"` to suffix patterns list
- [x] Update matching loop to use `endswith()` for suffix patterns
- [x] Run all 10 new tests, confirm all PASS (green) — all 15 tests passed

## Task 4: Verify no regressions
- [x] All 5 existing `TestGetTraitColumns` tests still pass
- [x] Full test suite passes (`uv run pytest`) — 1889 passed, 0 failed
- [x] No other `common_metadata` patterns produce false positives on fixture data

## Task 5: Address Copilot review (Round 2)
- [x] Replace tautological assertions in `test_metadata_columns_are_complement_of_traits` with meaningful checks
- [x] Fix stale print message in `test_qc_pipeline_turface_integration`
- [x] Run tests, confirm all pass — 71 data_cleanup tests passed, turface pipeline test passed
- [x] Full test suite passes — 1889 passed, 0 failed
