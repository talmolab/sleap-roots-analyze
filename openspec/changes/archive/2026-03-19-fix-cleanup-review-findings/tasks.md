# Tasks: fix-cleanup-review-findings

## Status: DONE

### 1. Fix baseline import errors and write failing tests (red)

Fix the broken imports first so CI produces meaningful failures, then add assertions
that are red before the implementation fix.

- [x] Fix all `from src.sleap_roots_analyze` imports in `tests/test_data_cleanup.py`:
  - Module-level at lines 11, 19: `from src.sleap_roots_analyze.data_cleanup import ...`
    → `from sleap_roots_analyze.data_cleanup import ...`
  - Inline imports at lines 1435, 1467, 1498, 1534, 1562, 1588, 1624, 1660
    (run `grep -n "from src\." tests/test_data_cleanup.py` to enumerate all sites)
- [x] Fix `from src.sleap_roots_analyze.statistics import` at `tests/test_statistics.py:8`
- [x] Add assertions to `test_removed_samples_csv_contains_correct_rows`
  (`tests/test_step_cleanup.py`) for genotype and rep values:
  - `assert (removed_samples["genotype"] == "B").all()`
  - `assert removed_samples["rep"].ne("").all()`
    _(rows 15/16/17 have rep [2, 1, 2] — not all equal, so assert non-empty not a specific value)_
  - `assert (removed_samples["nan_count"] == 2).all()`
  - `assert (removed_samples["nan_fraction"] == 1.0).all()`
- [x] Add unit test `test_apply_data_cleanup_filters_uses_genotype_col_and_replicate_col`
  to `TestApplyDataCleanupFilters` in `tests/test_data_cleanup.py`:
  - Build a DataFrame with columns `"Genotype"` and `"Replicate"` (not `"geno"`/`"rep"`)
  - Remove at least one sample by exceeding `max_nans_per_sample`
  - Call `apply_data_cleanup_filters(..., genotype_col="Genotype", replicate_col="Replicate")`
  - Assert `cleanup_log["removed_samples_detail"][0]["genotype"]` is not `""`
  - Assert `cleanup_log["removed_samples_detail"][0]["rep"]` is not `""`
- [x] Add unit test `test_apply_data_cleanup_filters_removed_samples_is_independent_copy`
  to `tests/test_data_cleanup.py`:
  - Run `apply_data_cleanup_filters` producing at least one removed sample
  - Assert `cleanup_log["removed_samples"] is not cleanup_log["removed_samples_detail"]`
  - Mutate `cleanup_log["removed_samples"]` (e.g., `append({"sentinel": True})`)
  - Assert `cleanup_log["removed_samples_detail"]` is unchanged (length unchanged, no sentinel)
  - Also assert mutating a field in `cleanup_log["removed_samples"][0]` does not affect
    `cleanup_log["removed_samples_detail"][0]` (dict-level independence)
- [x] Add unit test `test_apply_data_cleanup_filters_empty_detail_when_no_samples_removed`
  to `tests/test_data_cleanup.py`:
  - Build a DataFrame with no NaN values
  - Assert `cleanup_log["removed_samples_detail"] == []`
  - Assert `cleanup_log["removed_samples"] == []`
- [x] Add unit test `test_remove_nan_samples_max_nan_fraction_zero`
  to `tests/test_data_cleanup.py`:
  - One sample with exactly 1 NaN out of N traits, `max_nan_fraction=0.0`
  - Assert that sample appears in removal details
  - Assert `nan_count == 1` and `nan_fraction > 0.0`
- [x] Add unit test `test_remove_nan_samples_max_nan_fraction_one_keeps_partial_nan`
  to `tests/test_data_cleanup.py`:
  - Samples with partial NaN (< 100%), `max_nan_fraction=1.0`
  - Assert `cleanup_log["removed_samples_detail"] == []`
- [x] Add unit test `test_remove_nan_samples_missing_column_fallback`
  to `tests/test_data_cleanup.py`:
  - DataFrame missing the `replicate_col` column
  - Assert no `KeyError`; `rep == ""` in the removal entry
- [x] Run `uv run pytest tests/test_step_cleanup.py::test_removed_samples_csv_contains_correct_rows -v`
  and confirm failure on genotype/rep assertions (not import error)

### 2. Apply fixes — commit test + fix together (green)

**Note:** The failing tests from Task 1 and the implementation fixes below MUST be
committed in a single atomic commit to keep CI green. Do not commit the failing tests
without their fix.

- [x] Add `barcode_col`, `genotype_col`, `replicate_col` keyword params to
  `apply_data_cleanup_filters()` in `data_cleanup.py`:
  - Defaults: `barcode_col="Barcode"`, `genotype_col="geno"`, `replicate_col="rep"`
    _(these match `remove_nan_samples()` defaults, preserving backward compat for direct callers)_
  - Forward all three to `remove_nan_samples()` call
  - Update docstring to document new params
- [x] Update `CleanupTraitsStep.execute()` in `cleanup_traits.py` to pass sanitized
  column names when calling `apply_data_cleanup_filters()`:
  - `barcode_col="Barcode"`, `genotype_col="Genotype"`, `replicate_col="Replicate"`
  - These are set at lines 87-89 after `sanitize_trait_names()` returns
- [x] Fix mutable alias at `data_cleanup.py` (the `removed_samples = removed_samples_detail` line):
  - Change to `cleanup_log["removed_samples"] = [dict(e) for e in cleanup_log["removed_samples_detail"]]`
    _(deep copy — dict entries must be independent, not just the list)_
- [x] Remove dead code block `cleanup_traits.py:112-118` (the orphaned comment AND the
  `removed_sample_indices` loop — both lines 112 through 118 inclusive)
- [x] Add `warnings.warn` / `logging.warning` in `remove_nan_samples()` when `barcode_col`,
  `genotype_col`, or `replicate_col` is not found in the DataFrame (before falling back to `""`)

### 3. Verify tests pass (green)

- [x] Run `uv run pytest tests/test_step_cleanup.py::test_removed_samples_csv_contains_correct_rows -v`
  — confirm passes with correct genotype/rep values
- [x] Run `uv run pytest tests/test_data_cleanup.py tests/test_step_cleanup.py -v`
  — confirm all pass, no regressions

### 4. Full validation

- [x] Run full test suite: `uv run pytest --cov=src/sleap_roots_analyze -q`
- [x] Run lint: `uv run black --check src/sleap_roots_analyze tests && uv run ruff check src/sleap_roots_analyze`

### 5. Update CHANGELOG

- [x] Add entry under `[Unreleased] ### Fixed` in `docs/CHANGELOG.md`
