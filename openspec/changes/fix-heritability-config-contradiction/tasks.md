# Tasks: fix-heritability-config-contradiction

## Task 1: Write failing tests — config validation (TDD Red Phase)
- [x] 1.1 Add `test_validate_viz_config_rejects_heritability_without_calculation` — set `statistics.calculate_heritability=False` and `heritability.enabled=True`; verify `validate_viz_config()` raises `ValueError` with message mentioning both config fields
- [x] 1.2 Add `test_validate_viz_config_accepts_both_enabled` — set both to `True`; verify no error
- [x] 1.3 Add `test_validate_viz_config_accepts_calculate_without_filter` — set `calculate_heritability=True`, `heritability.enabled=False`; verify no error
- [x] 1.4 Add `test_validate_viz_config_accepts_both_disabled` — set both to `False`; verify no error
- [x] 1.5 Add `test_validate_viz_config_error_message_suggests_fix` — verify error message includes both solution options (enable calculation OR disable filtering)
- [x] 1.6 Add `test_validate_viz_config_default_values_pass` — construct `VizPipelineConfig` with defaults; verify no error
- [x] 1.7 Run tests, confirm all FAIL (red) on current code (1.1 and 1.5 fail; 1.2-1.4, 1.6 pass)

## Task 2: Write failing tests — FilterHeritabilityStep guard (TDD Red Phase)
- [x] 2.1 Add `test_guard_prevents_silent_trait_removal` — pass `heritability_results={}` with `heritability.enabled=True`; verify all traits retained in output DataFrame
- [x] 2.2 Add `test_guard_logs_warning` — verify warning is logged when guard activates
- [x] 2.3 Add `test_guard_metadata_includes_flag` — verify `guard_activated: True` and `guard_reason` (string) in result metadata when guard triggers
- [x] 2.4 Add `test_guard_preserves_metadata` — verify previous step metadata (trait_names, valid_trait_names, heritability_results, etc.) is preserved when guard activates
- [x] 2.5 Add `test_guard_generates_correct_files` — verify 09_data_high_heritability.csv contains all traits, 09_removed_traits.json is `[]`, and summary JSON shows `traits_removed: 0` and `guard_activated: True`
- [x] 2.6 Add `test_guard_does_not_activate_with_populated_results` — pass real heritability results with `enabled=True`; verify normal filtering occurs and no `guard_activated` in metadata
- [x] 2.7 Add `test_guard_does_not_activate_when_filtering_disabled` — pass `heritability_results={}` with `heritability.enabled=False`; verify existing disabled path and no `guard_activated` in metadata
- [x] 2.8 Run tests, confirm all FAIL (red) on current code (2.1-2.5 fail; 2.6-2.7 pass)

## Task 3: Implement config validation (TDD Green Phase)
- [x] 3.1 In `validate_viz_config()`, add cross-field check after existing validation: if `heritability.enabled=True` and `statistics.calculate_heritability=False`, raise `ValueError` with message naming both fields and suggesting both fix options
- [x] 3.2 Run Task 1 tests, confirm all PASS (green)
- [x] 3.3 Run full test suite to confirm no regressions from validation change

## Task 4: Implement FilterHeritabilityStep guard (TDD Green Phase)
- [x] 4.1 In `FilterHeritabilityStep.execute()`, after extracting `heritability_results` (line 86) and before the `if not config.heritability.enabled:` check (line 90), add guard: if `config.heritability.enabled` and `not heritability_results`, log warning and return early with all traits preserved
- [x] 4.2 Include `guard_activated: True` and `guard_reason` (string) in both result metadata and summary JSON
- [x] 4.3 Preserve all previous step metadata via `**prev_result.metadata` spread
- [x] 4.4 Generate same output files as the disabled path (09_data_high_heritability.csv, 09_removed_traits.json as `[]`, 09_heritability_filter_summary.json) so downstream steps see consistent file structure
- [x] 4.5 Run Task 2 tests, confirm all PASS (green)

## Task 5: Verify no regressions
- [x] 5.1 All existing `FilterHeritabilityStep` tests pass unchanged
- [x] 5.2 All existing `validate_viz_config` tests pass unchanged
- [x] 5.3 All existing `StatisticalAnalysisStep` tests pass unchanged
- [ ] 5.4 Full test suite passes (`uv run pytest`)
- [x] 5.5 Linting and formatting pass (`uv run ruff check`, `uv run black --check`)

## Additional fix discovered during implementation
- [x] Fix `viz_config_minimal` and `viz_config_with_stats` fixtures in `tests/fixtures.py`: they used nonexistent `heritability.filter_enabled` instead of `heritability.enabled`, masking the exact contradiction this change prevents
