## Why

PR #46 review feedback identified missing input validation in `calculate_correlation_ci()`. As a public API function, it must defensively validate inputs to prevent silent incorrect results from invalid correlation coefficients or confidence levels. Additionally, the docstring needs clarity on Spearman approximation accuracy for small samples.

## What Changes

- Add input validation for `r` parameter: must be in [-1, 1] or NaN
- Add input validation for `confidence_level` parameter: must be in (0, 1)
- Enhance docstring with Spearman n < 10 accuracy warning
- Add cross-reference to DataFrame-based `calculate_correlation_confidence_intervals`
- Refactor old `calculate_correlation_confidence_intervals` to use new `calculate_correlation_ci` for consistency
- Remove unused test variable `expected_ratio`

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/cross_experiment_analysis.py`
  - `tests/test_cross_experiment_analysis.py`
