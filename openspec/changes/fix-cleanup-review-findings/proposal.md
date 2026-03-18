# Fix Cleanup Review Findings

## Status: PROPOSED

## Why

Code review of PR #99 identified four issues in the NaN removal tracking fix: genotype/replicate fields are silently empty in `02_removed_samples_detail.csv` due to post-sanitization column names not being forwarded to `remove_nan_samples()`, a shared mutable alias creates silent mutation risk in the cleanup log, dead code exists in `cleanup_traits.py`, and multiple test files use an incorrect `from src.` import path.

## What Changes

- Add `barcode_col`, `genotype_col`, `replicate_col` keyword parameters to
  `apply_data_cleanup_filters()` with defaults matching `remove_nan_samples()` defaults
  (`"Barcode"`, `"geno"`, `"rep"`); forward them to `remove_nan_samples()`
- Update `CleanupTraitsStep.execute()` to pass the post-sanitization column names
  (`"Barcode"`, `"Genotype"`, `"Replicate"`) when calling `apply_data_cleanup_filters()`
- Change `cleanup_log["removed_samples"] = cleanup_log["removed_samples_detail"]` to a
  deep copy: `[dict(e) for e in cleanup_log["removed_samples_detail"]]`
- Delete the unused `removed_sample_indices` block and its orphaned comment
  (`cleanup_traits.py:112-118`)
- Fix all `from src.sleap_roots_analyze...` imports in `tests/test_data_cleanup.py`
  (module-level lines 11, 19 and inline imports at lines 1435, 1467, 1498, 1534, 1562,
  1588, 1624, 1660) and `tests/test_statistics.py` (line 8)

## Impact

**Affected specs:** `data-sanitization` (ADDED requirement: NaN Sample Removal Detail
Tracking — extends the requirement added by `fix-nan-removal-tracking`; that proposal
must archive first for the canonical spec to be consistent)

**Affected code files:**
- `src/sleap_roots_analyze/data_cleanup.py` (new params + deep copy)
- `src/sleap_roots_analyze/pipeline/steps/cleanup_traits.py` (pass sanitized cols,
  remove dead code)
- `tests/test_data_cleanup.py` (fix all `from src.` imports, add new unit tests)
- `tests/test_step_cleanup.py` (add genotype/rep value assertions)
- `tests/test_statistics.py` (fix `from src.` import)

**Unaffected callers:** `src/sleap_roots_analyze/visualization.py` calls
`apply_data_cleanup_filters()` without column kwargs and always receives pre-sanitization
DataFrames (`"geno"`, `"rep"` columns present), so the new defaults leave it unaffected.

**Breaking changes:** None — new params have defaults matching current behavior for any
caller that does not use `CleanupTraitsStep`

**Risk:** Low — additive signature change; existing callers unaffected by defaults
