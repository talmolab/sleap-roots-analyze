# Tasks: fix-nan-removal-tracking

## Status: APPLIED

### 1. Write failing regression test (red — must fail before fix)
- [x] Add `test_apply_data_cleanup_filters_propagates_removal_details` to
  `tests/test_data_cleanup.py`
  - GIVEN a DataFrame where ≥1 sample exceeds `max_nan_fraction`
  - WHEN `apply_data_cleanup_filters()` is called
  - THEN `cleanup_log["removed_samples_detail"]` is non-empty
  - AND its length equals the number of samples removed
  - AND each entry contains keys: `sample_index`, `barcode`, `genotype`, `rep`,
    `nan_count`, `nan_fraction`, `nan_traits`, `removal_reason`
  - Confirmed **failed** against the buggy code before fix was applied
- [x] Add `test_removed_samples_csv_contains_correct_rows` to
  `tests/test_step_cleanup.py`
  - GIVEN a dataset with samples that have NaN values above the threshold
  - WHEN `CleanupTraitsStep.execute()` runs
  - THEN `02_removed_samples_detail.csv` has N data rows (not 0, not header-only)
  - AND columns include: `sample_index`, `barcode`, `genotype`, `rep`,
    `nan_count`, `nan_fraction`, `nan_traits`, `removal_reason`
  - Confirmed **failed** against the buggy code before fix was applied

### 2. Apply the fixes (green — tests must pass after)
- [x] Fix primary key mismatch in `apply_data_cleanup_filters()`:
  - `data_cleanup.py:704`: changed `"removed_samples_detail"` to `"removal_details"`
    in `removal_stats.get()` call
- [x] Fix secondary key mismatch in `CleanupTraitsStep`:
  - `cleanup_traits.py:118`: changed `sample_info["index"]` to
    `sample_info["sample_index"]`
  - `cleanup_traits.py:149`: changed column name `"index"` to `"sample_index"` in
    empty-DataFrame fallback; expanded to all 8 required columns
- [x] Update `apply_data_cleanup_filters()` docstring to document `cleanup_log`
  key schema, explicitly naming `"removed_samples_detail"` and its source
- [x] Confirmed both new tests pass after fixes

### 3. Full validation
- [ ] Run full test suite: `uv run pytest --cov=src/sleap_roots_analyze --cov-branch -q`
  (running — awaiting results)

### 4. Documentation
- [x] Added `### Fixed` entry to `[Unreleased]` section of `docs/CHANGELOG.md`

### Post-merge manual validation (not a PR gate)
After merging, re-run the QC pipeline on Turface 19 genotypes and confirm
`02_removed_samples_detail.csv` contains 29 rows with correct barcode, genotype,
rep, and NaN trait information.
