# Tasks: fix-trait-column-id-pattern-match

## Task 1: Write failing unit tests (TDD Red Phase)
- [ ] Add `test_width_columns_not_excluded_by_id_pattern` to `TestGetTraitColumns`
- [ ] Add `test_solidity_columns_not_excluded_by_id_pattern` to `TestGetTraitColumns`
- [ ] Add `test_actual_id_columns_still_excluded` to `TestGetTraitColumns`
- [ ] Add `test_mixed_id_and_width_columns` to `TestGetTraitColumns`
- [ ] Run tests, confirm all 4 new tests FAIL (red)

## Task 2: Write failing integration tests (TDD Red Phase)
- [ ] Add `test_pipeline_trait_classification_11dag` — SLEAP Roots data (880 cols, width/solidity/ID columns)
- [ ] Add `test_pipeline_trait_classification_traits_summary` — summary data (924 cols, different column order)
- [ ] Add `test_pipeline_trait_classification_traits_summary_lateral` — lateral root data (different trait prefixes)
- [ ] Add `test_pipeline_trait_classification_turface` — agronomic data (41 cols, different naming conventions)
- [ ] Add `test_pipeline_trait_classification_features` — RhizoVision output (dotted PascalCase column names)
- [ ] Add `test_metadata_columns_are_complement_of_traits` — regression guard across all datasets
- [ ] Run tests, confirm all 6 integration tests FAIL (red)

## Task 3: Implement the fix (TDD Green Phase)
- [ ] Split `common_metadata` into `metadata_substring_patterns` and `metadata_suffix_patterns`
- [ ] Remove bare `"id"` and redundant `"plant_id"` from patterns
- [ ] Add `"_id"` to suffix patterns list
- [ ] Update matching loop to use `endswith()` for suffix patterns
- [ ] Run all 10 new tests, confirm all PASS (green)

## Task 4: Verify no regressions
- [ ] All 5 existing `TestGetTraitColumns` tests still pass
- [ ] Full test suite passes (`uv run pytest`)
- [ ] No other `common_metadata` patterns produce false positives on fixture data
