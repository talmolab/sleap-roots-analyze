## Why

The FDR correction implementation (PR #45) correctly uses `statsmodels.stats.multitest.multipletests`, but edge case handling needs improvement. If any correlation produces NaN p-values (e.g., from constant-valued traits), `multipletests` returns **all NaN** for the entire adjusted p-value array, silently corrupting all FDR corrections. Additionally, the current test suite lacks coverage for edge cases like NaN propagation and single-test scenarios.

## What Changes

- Add explicit NaN p-value handling before calling `multipletests`:
  - Filter out NaN p-values before correction
  - Merge corrected values back with NaN for invalid correlations
  - Log warning when NaN p-values are encountered
- Add comprehensive edge case tests:
  - Test NaN p-value handling
  - Test single correlation (m=1) behavior
  - Test behavior with constant-valued traits
- Document edge case behavior in `docs/CROSS_PLATFORM_ANALYSIS.md`

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py`
  - `tests/test_step_calculate_cross_platform_correlations.py`
  - `docs/CROSS_PLATFORM_ANALYSIS.md`
- Backward compatible: No API changes, only defensive improvements
- No new dependencies
