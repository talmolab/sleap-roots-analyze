## Why

Cross-platform correlation analysis currently reports point estimates (r or ρ) without uncertainty quantification. A correlation of r=0.7 with n=10 genotypes has much wider confidence intervals than r=0.7 with n=100 genotypes, but this distinction is invisible in the current output. Confidence intervals are essential for:

1. **Reproducibility**: Communicating precision of estimates in publications
2. **Decision-making**: Distinguishing reliably-estimated correlations from unstable ones
3. **Statistical rigor**: Standard practice in biostatistics to report effect sizes with CIs

## What Changes

### Core Implementation
- Extend `calculate_correlations()` function to optionally return confidence interval bounds
- Add new function `calculate_correlation_ci()` using Fisher z-transformation
- Add 4 new columns to CSV output: `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r_ci_low`, `pearson_r_ci_high`
- Add `confidence_level` config parameter (default 0.95)

### Statistical Method
- **Fisher z-transformation**: Transform r → z = arctanh(r), compute CI on z-scale, back-transform
- Formula: z ± z_{α/2} / √(n-3), then r = tanh(z)
- Valid for both Pearson (exact) and Spearman (asymptotic approximation)
- Handle edge cases: r = ±1.0 (undefined z), n < 4 (undefined variance)

### Metadata & Traceability
- Include `confidence_level` in pipeline metadata
- Document CI method in output summary
- Add CI columns to CSV schema documentation

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/cross_experiment_analysis.py` (new CI function)
  - `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py` (CI computation)
  - `src/sleap_roots_analyze/pipeline/config/components.py` (new config field)
  - `docs/CROSS_PLATFORM_ANALYSIS.md` (documentation)
- No breaking changes: new columns are additive, existing columns unchanged
