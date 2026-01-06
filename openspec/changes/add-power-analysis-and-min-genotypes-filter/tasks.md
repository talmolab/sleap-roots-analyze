## 1. TDD: Write Failing Tests First (Power Analysis)

- [x] 1.1 Write `test_minimum_detectable_correlation` - verify formula with known values
- [x] 1.2 Write `test_achieved_power` - verify power calculation matches expected values
- [x] 1.3 Write `test_achieved_power_edge_cases` - test n < 4 returns NaN, r = 0 returns α
- [x] 1.4 Write `test_power_functions_input_validation` - test invalid inputs raise errors
- [x] 1.5 Run tests to confirm they fail (TDD red phase)

## 2. TDD: Write Failing Tests First (Min Genotypes Filter)

- [x] 2.1 Write `test_min_genotypes_filter_excludes_low_n` - verify pairs below threshold excluded
- [x] 2.2 Write `test_min_genotypes_filter_keeps_high_n` - verify pairs above threshold kept
- [x] 2.3 Write `test_min_genotypes_filter_metadata` - verify filtered count in metadata
- [x] 2.4 Write `test_min_genotypes_config_validation` - verify config accepts valid, rejects invalid
- [x] 2.5 Run tests to confirm they fail (TDD red phase)

## 3. TDD: Write Failing Integration Tests

- [x] 3.1 Write `test_correlation_step_outputs_power_column` - verify CSV has achieved_power
- [x] 3.2 Write `test_correlation_step_power_metadata` - verify metadata has power summary
- [x] 3.3 Write `test_correlation_step_filters_low_n` - verify rows filtered from CSV
- [x] 3.4 Write `test_power_config_parameters_in_metadata` - verify α, power in metadata
- [x] 3.5 Run tests to confirm they fail (TDD red phase)

## 4. Implement Power Analysis Functions

- [x] 4.1 Add `minimum_detectable_correlation(n, alpha, power)` to `cross_experiment_analysis.py`
- [x] 4.2 Add `achieved_power(r, n, alpha)` to `cross_experiment_analysis.py`
- [x] 4.3 Add input validation for n > 0, alpha in (0, 1), power in (0, 1), r in [-1, 1]
- [x] 4.4 Handle edge cases: n < 4 returns NaN, r = 0 returns alpha (no effect to detect)
- [x] 4.5 Add comprehensive docstrings with formulas and references
- [x] 4.6 Run unit tests to confirm they pass (TDD green phase)

## 5. Add Configuration Parameters

- [x] 5.1 Add `min_genotypes_for_correlation: int = 10` to `CrossPlatformConfig`
- [x] 5.2 Add `power_analysis_alpha: float = 0.05` to `CrossPlatformConfig`
- [x] 5.3 Add `power_analysis_power: float = 0.80` to `CrossPlatformConfig`
- [x] 5.4 Add validation: min_genotypes >= 3, alpha in (0, 1), power in (0, 1)
- [x] 5.5 Add docstrings explaining each parameter's purpose and defaults

## 6. Integrate into Pipeline Step

- [x] 6.1 Import power functions in `calculate_cross_platform_correlations.py`
- [x] 6.2 Add hard filter: skip trait pairs where n_genotypes < min_genotypes_for_correlation
- [x] 6.3 Track filtered count and reason for metadata
- [x] 6.4 Calculate `achieved_power` for each correlation in loop
- [x] 6.5 Add `achieved_power` column to correlation_results dict
- [x] 6.6 Calculate `minimum_detectable_r` using modal n_genotypes
- [x] 6.7 Add power analysis summary to step metadata
- [x] 6.8 Run integration tests to confirm they pass (TDD green phase)

## 7. Documentation

- [x] 7.1 Add "Power Analysis" section to `docs/CROSS_PLATFORM_ANALYSIS.md`
- [x] 7.2 Document power formulas with mathematical notation
- [x] 7.3 Explain minimum detectable r and interpretation
- [x] 7.4 Add "Minimum Genotypes Filter" section explaining filtering behavior
- [x] 7.5 Add interpretation guidance: what power thresholds mean
- [x] 7.6 Add academic references (Cohen 1988, Fisher 1921)
- [x] 7.7 Update CSV schema documentation with `achieved_power` column
- [x] 7.8 Update metadata JSON example

## 8. Update YAML Configs

- [x] 8.1 Add `min_genotypes_for_correlation: 10` to all cross_platform_*.yaml
- [x] 8.2 Add `power_analysis_alpha: 0.05` to all cross_platform_*.yaml
- [x] 8.3 Add `power_analysis_power: 0.80` to all cross_platform_*.yaml
- [x] 8.4 Add comments explaining parameter meaning and defaults

## 9. Validation

- [x] 9.1 Run full test suite: `uv run pytest -v`
- [x] 9.2 Run linting: `uv run ruff check --fix && uv run black .`
- [ ] 9.3 Run `openspec validate add-power-analysis-and-min-genotypes-filter --strict`
- [ ] 9.4 Verify pipeline output with sample data
- [x] 9.5 Verify metadata includes all new fields
- [x] 9.6 Verify filtered correlations are logged clearly
