## 1. TDD: Write Failing Tests First

- [x] 1.1 Write test for `r` validation: values outside [-1, 1] should raise ValueError
- [x] 1.2 Write test for `confidence_level` validation: values outside (0, 1) should raise ValueError
- [x] 1.3 Write test verifying NaN r still returns (NaN, NaN) without raising
- [x] 1.4 Run tests to confirm they fail (red phase)

## 2. Implementation

- [x] 2.1 Add `r` validation in `calculate_correlation_ci()` after NaN check
- [x] 2.2 Add `confidence_level` validation in `calculate_correlation_ci()`
- [x] 2.3 Update docstring with Spearman n < 10 accuracy warning
- [x] 2.4 Add cross-reference to DataFrame-based function in docstring
- [x] 2.5 Update docstring Raises section
- [x] 2.6 Run tests to confirm they pass (green phase)

## 3. Refactor

- [x] 3.1 Refactor `calculate_correlation_confidence_intervals` to use `calculate_correlation_ci` internally
- [x] 3.2 Ensure existing tests still pass after refactor

## 4. Cleanup

- [x] 4.1 Remove unused `expected_ratio` variable in test
- [x] 4.2 Run full test suite
- [x] 4.3 Run linting (ruff, black)

## 5. Validation

- [x] 5.1 Run `openspec validate harden-correlation-ci-validation --strict`
- [x] 5.2 Push changes and verify CI passes
