# Tasks: fix-heritability-flag-ignored

## Task 1: Write failing tests -- heritability flag respected (TDD Red Phase)
- [ ] 1.1 Add `test_heritability_skipped_when_disabled` -- set `config.statistics.calculate_heritability = False`; verify `calculate_heritability_estimates` is NOT called and `heritability_results` is empty dict in metadata
- [ ] 1.2 Add `test_heritability_calculated_when_enabled` -- set `config.statistics.calculate_heritability = True` (default); verify `calculate_heritability_estimates` IS called and `heritability_results` is populated
- [ ] 1.3 Add `test_heritability_csv_not_generated_when_disabled` -- verify `08_heritability_results.csv` is NOT created when heritability is disabled
- [ ] 1.4 Run tests, confirm all FAIL (red) on current code

## Task 2: Write failing tests -- metadata and summary (TDD Red Phase)
- [ ] 2.1 Add `test_summary_reflects_skipped_heritability` -- verify summary JSON contains `"heritability_summary": {"skipped": true}` when disabled
- [ ] 2.2 Add `test_metadata_heritability_results_empty_when_disabled` -- verify `metadata["heritability_results"]` is `{}` when disabled
- [ ] 2.3 Run tests, confirm all FAIL (red) on current code

## Task 3: Write failing tests -- downstream compatibility (TDD Red Phase)
- [ ] 3.1 Add `test_filter_heritability_handles_empty_results` -- verify `FilterHeritabilityStep` completes without error when `heritability_results` is `{}`
- [ ] 3.2 Add `test_pipeline_completes_with_heritability_disabled` -- integration test with full pipeline, `calculate_heritability = False`, verify pipeline status is "success"
- [ ] 3.3 Run tests, confirm FAIL or verify existing behavior

## Task 4: Implement the fix (TDD Green Phase)
- [ ] 4.1 In `StatisticalAnalysisStep.execute()`, read `config.statistics.calculate_heritability` (default `True`)
- [ ] 4.2 Wrap heritability calculation block (lines 144-186) in `if calculate_heritability:` guard
- [ ] 4.3 In the `else` branch, set `heritability_results = {}`, `heritability_df = pd.DataFrame()`, and skip CSV save
- [ ] 4.4 Update summary dict to include `"skipped": True` in `heritability_summary` when disabled
- [ ] 4.5 Run all new tests, confirm all PASS (green)

## Task 5: Verify no regressions
- [ ] 5.1 All existing `StatisticalAnalysisStep` tests pass unchanged
- [ ] 5.2 All existing `FilterHeritabilityStep` tests pass unchanged
- [ ] 5.3 Full test suite passes (`uv run pytest`)
- [ ] 5.4 Linting and formatting pass (`uv run ruff check`, `uv run black --check`)
