## Why

With n≈18 genotypes, many correlations in the cross-platform pipeline are statistically underpowered. Users need to understand which correlations have sufficient statistical power to be meaningful, and trait pairs with very few valid genotypes (due to NaN removal) should be excluded to prevent spurious results.

This proposal addresses Issue #44 P1 priorities:
1. **Power Analysis Integration**: Report achieved power per correlation and minimum detectable effect size
2. **Minimum Genotypes Filter**: Hard filter to exclude trait pairs with insufficient sample size

## Design Decisions

### Power Analysis Method

**Decision**: Use Fisher z-transformation for power calculation (same method used for Pearson correlation).

**Rationale**: Research confirms that power analysis for Spearman's rank correlation coefficient is computationally identical to Pearson's product-moment coefficient, as both use the Fisher z-transformation. G*Power and other standard tools use this same approach for both correlation types.

**References**:
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum.
- Bonett, D.G. & Wright, T.A. (2000). Sample size requirements for estimating Pearson, Kendall and Spearman correlations. Psychometrika, 65(1), 23-28.
- [Sample Size Calculators](https://sample-size.net/correlation-sample-size/)

### Default Parameters

**Decision**: Use α=0.05 and power=0.80 as defaults, but make them configurable.

**Rationale**: These are near-universal conventions in biostatistics. α=0.05 (5% false positive rate) and power=0.80 (80% true positive rate) represent the standard trade-off between Type I and Type II errors. Making them configurable allows users with different requirements (e.g., power=0.90 for high-stakes decisions) to adjust.

### Minimum Genotypes Filter Behavior

**Decision**: Implement as a hard filter that excludes trait pairs from the CSV, with clear documentation and metadata logging of filtered counts.

**Rationale**:
1. A hard filter prevents meaningless correlations from polluting results
2. Correlations with very low n (e.g., n=5) have undefined or extremely wide confidence intervals
3. The Fisher z approximation is only accurate for n ≥ 10 for Spearman correlations
4. Users need to know correlations were filtered, so metadata includes the count and reason

**Default value**: `min_genotypes_for_correlation: 10` (recommended for accurate Fisher z approximation)

### Output Strategy

**Decision**: Include `achieved_power` as a per-correlation CSV column AND summary statistics in pipeline metadata.

**Rationale**:
1. Per-correlation power varies by both effect size (|r|) and sample size (n_genotypes)
2. Users need row-level power to interpret individual correlations
3. Summary statistics (minimum detectable r, count above threshold) provide quick assessment

## What Changes

- Add `min_genotypes_for_correlation` config parameter (default: 10) to filter trait pairs
- Add `power_analysis_alpha` config parameter (default: 0.05)
- Add `power_analysis_power` config parameter (default: 0.80)
- Add `achieved_power` column to CSV output (per-correlation)
- Add power analysis metadata: `minimum_detectable_r`, `n_correlations_above_mdr`, `n_correlations_filtered_low_n`, `filtered_reason`
- Document statistical methodology and interpretation guidance

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/cross_experiment_analysis.py` (new power functions)
  - `src/sleap_roots_analyze/pipeline/config/components.py` (new config parameters)
  - `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py` (filtering + power column)
  - `docs/CROSS_PLATFORM_ANALYSIS.md` (documentation)
  - `configs/cross_platform_*.yaml` (explicit parameters)
  - `tests/test_cross_experiment_analysis.py` (unit tests)
  - `tests/test_step_calculate_cross_platform_correlations.py` (integration tests)
