# Fix NaN Removal Tracking

## Status: PROPOSED

## Why

The `apply_data_cleanup_filters()` function silently drops per-sample NaN removal
details due to a dictionary key mismatch between producer (`remove_nan_samples()`
returns `"removal_details"`) and consumer (the caller looks up
`"removed_samples_detail"`). In the Turface 19 dataset, 29 samples (15.5%) are
removed for missing `Root_Biomass_mg` values but their identities are never
recorded in `02_removed_samples_detail.csv`, breaking the traceability requirement
for publication-grade analysis.

A secondary key mismatch in `CleanupTraitsStep` (`sample_info["index"]` instead of
`sample_info["sample_index"]`) would raise `KeyError` once the primary fix is
applied, and must be fixed in the same change.

## What Changes

- Fix `removal_stats.get("removed_samples_detail", [])` →
  `removal_stats.get("removal_details", [])` at `data_cleanup.py:704`
- Fix `sample_info["index"]` → `sample_info["sample_index"]` at
  `cleanup_traits.py:118`
- Fix column name `"index"` → `"sample_index"` in empty-DataFrame fallback at
  `cleanup_traits.py:149`
- Add regression test asserting `cleanup_log["removed_samples_detail"]` is
  non-empty when samples are removed (must fail before fix is applied)
- Add CSV content test asserting `02_removed_samples_detail.csv` has correct rows
  and columns (not just existence)
- Add CHANGELOG `### Fixed` entry
- Update `apply_data_cleanup_filters()` docstring to document `cleanup_log` key
  schema

## Impact

**Affected specs:** `data-sanitization` (new ADDED requirement)

**Affected code files:**
- `src/sleap_roots_analyze/data_cleanup.py` (primary key mismatch fix)
- `src/sleap_roots_analyze/pipeline/steps/cleanup_traits.py` (secondary key
  mismatch fix)
- `tests/test_data_cleanup.py` (new regression test)
- `tests/test_step_cleanup.py` (new CSV content test)
- `docs/CHANGELOG.md` (Fixed entry)

**Breaking changes:** None

**Risk:** Minimal — restores intended data flow; no logic changes; 3 one-line fixes

## Affected Datasets

| Dataset | Samples Removed for NaN | Currently Tracked? |
|---------|------------------------|--------------------|
| Root Core EDPIE | 0 | N/A |
| Cylinder EDPIE | 0 | N/A |
| Turface 150 | 1 | NO (bug) |
| Turface 19 | 29 | NO (bug) |
