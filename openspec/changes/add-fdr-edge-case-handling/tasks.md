## 1. Edge Case Tests (TDD - Tests First)

- [x] 1.1 Write `test_fdr_correction_with_nan_pvalues` - verify NaN doesn't corrupt other results
- [x] 1.2 Write `test_fdr_correction_single_correlation` - verify m=1 case works correctly
- [x] 1.3 Write `test_fdr_correction_with_constant_trait` - verify constant-valued trait produces NaN and doesn't break FDR
- [x] 1.4 Write `test_significant_fdr_false_for_nan` - verify NaN adjusted p-values result in False significance
- [x] 1.5 Run tests to confirm they fail (TDD red phase)

## 2. NaN P-value Handling Implementation

- [x] 2.1 Add NaN detection before calling `multipletests` in `calculate_cross_platform_correlations.py`
- [x] 2.2 Filter out NaN p-values, apply FDR correction to valid p-values only
- [x] 2.3 Merge corrected p-values back, preserving NaN for invalid correlations
- [x] 2.4 Add logging when NaN p-values are encountered (count and warning)
- [x] 2.5 Ensure `significant_fdr` is False for rows with NaN adjusted p-values
- [x] 2.6 Run tests to confirm they pass (TDD green phase)

## 3. Documentation

- [x] 3.1 Add "Edge Cases" section to `docs/CROSS_PLATFORM_ANALYSIS.md`
- [x] 3.2 Document behavior when traits have constant values (zero variance)
- [x] 3.3 Document behavior when fewer than 3 genotypes have valid data for a trait pair

## 4. Validation

- [x] 4.1 Run full test suite: `uv run pytest tests/test_step_calculate_cross_platform_correlations.py -v`
- [x] 4.2 Run linting: `uv run ruff check --fix && uv run black .`
