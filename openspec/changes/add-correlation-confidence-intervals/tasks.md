## 1. TDD: Write Tests First

- [x] 1.1 Write `test_calculate_correlation_ci_perfect_correlation` - verify CI for r=1.0 edge case
- [x] 1.2 Write `test_calculate_correlation_ci_zero_correlation` - verify CI contains 0 for uncorrelated data
- [x] 1.3 Write `test_calculate_correlation_ci_small_n` - verify NaN CI for n < 4
- [x] 1.4 Write `test_calculate_correlation_ci_bounds` - verify -1 ≤ ci_low ≤ r ≤ ci_high ≤ 1
- [x] 1.5 Write `test_calculate_correlation_ci_confidence_levels` - verify 99% CI wider than 95% CI
- [x] 1.6 Write `test_correlation_step_outputs_ci_columns` - verify CSV contains CI columns
- [x] 1.7 Write `test_confidence_level_config_validation` - verify config accepts valid levels, rejects invalid
- [x] 1.8 Write `test_metadata_includes_confidence_level` - verify metadata includes confidence_level
- [x] 1.9 Run tests to confirm they fail (TDD red phase)

## 2. Implement CI Calculation Function

- [x] 2.1 Add `calculate_correlation_ci(r, n, confidence_level=0.95)` to `cross_experiment_analysis.py`
- [x] 2.2 Implement Fisher z-transformation: z = arctanh(r)
- [x] 2.3 Implement CI on z-scale: z ± z_{α/2} / √(n-3)
- [x] 2.4 Implement back-transformation: r = tanh(z)
- [x] 2.5 Handle edge case: r = ±1.0 returns (r, r) as CI (point mass at boundary)
- [x] 2.6 Handle edge case: n < 4 returns (NaN, NaN)
- [x] 2.7 Clamp CI bounds to [-1, 1] to handle numerical precision issues
- [x] 2.8 Add comprehensive docstring with mathematical formulation and references

## 3. Add Configuration Parameter

- [x] 3.1 Add `confidence_level: float = 0.95` to `CrossPlatformConfig` in `components.py`
- [x] 3.2 Add validation: confidence_level must be in (0, 1) exclusive
- [x] 3.3 Update config docstring with parameter description
- [ ] 3.4 Update template YAML configs with `confidence_level` parameter

## 4. Integrate CI into Pipeline Step

- [x] 4.1 Import `calculate_correlation_ci` in `calculate_cross_platform_correlations.py`
- [x] 4.2 Call `calculate_correlation_ci` for both Spearman and Pearson correlations
- [x] 4.3 Add columns to correlation_results dict: `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r_ci_low`, `pearson_r_ci_high`
- [x] 4.4 Pass `config.confidence_level` to CI function
- [x] 4.5 Add `confidence_level` to step metadata
- [x] 4.6 Run tests to confirm they pass (TDD green phase)

## 5. Documentation

- [x] 5.1 Add "Confidence Intervals" section to `docs/CROSS_PLATFORM_ANALYSIS.md`
- [x] 5.2 Document Fisher z-transformation formula with mathematical notation
- [x] 5.3 Explain why n ≥ 4 is required (variance formula has n-3 in denominator)
- [x] 5.4 Document Spearman CI as asymptotic approximation (exact for Pearson)
- [x] 5.5 Add CI column descriptions to CSV schema documentation
- [x] 5.6 Add academic reference: Fisher, R.A. (1921) for z-transformation

## 6. Validation

- [x] 6.1 Run full test suite: `uv run pytest tests/test_step_calculate_cross_platform_correlations.py -v`
- [x] 6.2 Run CI function unit tests: `uv run pytest tests/test_cross_experiment_analysis.py -v -k ci`
- [x] 6.3 Run linting: `uv run ruff check --fix && uv run black .`
- [ ] 6.4 Verify CI output in sample pipeline run
