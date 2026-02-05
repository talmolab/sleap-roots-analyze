# Improve Cross-Platform Summary Clarity and Accuracy

## Problem Statement

The current cross-platform summary has several issues that could mislead scientists or make interpretation difficult:

1. **Factual error**: Methods section hardcodes "Benjamini-Yekutieli" but configs use "fdr_bh" (Benjamini-Hochberg)
2. **Broken image paths**: Images use relative paths that don't resolve from SUMMARY.md location
3. **Missing variable definitions**: r, p, q, Power, n, H², FDR are used without definition
4. **No interpretation guidance**: Empty "Sig?" column appears like a bug rather than a meaningful result
5. **Missing confidence intervals**: CSV has CIs but they're not displayed
6. **No power analysis interpretation**: Scientists don't know what low power means for their data
7. **No guidance when FDR=0**: No explanation of what to do when no correlations survive correction

## Proposed Solution

### 1. Fix FDR Method in Methods Section
- Read actual `fdr_correction_method` from cross-platform configs
- Map "fdr_bh" → "Benjamini-Hochberg", "fdr_by" → "Benjamini-Yekutieli"

### 2. Embed Images as Base64
- Convert PNG images to base64 data URIs
- Self-contained markdown that renders anywhere

### 3. Inline Variable Definitions in Table Headers
Change from:
```
| Exp1 Trait | Exp2 Trait | r | p | q | Power | n | Sig? |
```
To:
```
| Exp1 Trait | Exp2 Trait | ρ (Spearman) | p | q (FDR-adj) | Power | n | FDR Sig |
```

Add a brief legend below the table:
```
*ρ = Spearman correlation coefficient, p = raw p-value, q = FDR-adjusted p-value, n = genotypes*
```

### 4. Add Confidence Intervals
Include 95% CI for correlation coefficients:
```
| Exp1 Trait | Exp2 Trait | ρ (95% CI) | p | q | ... |
| Max Num Roots | Root Shoot Ratio | -0.64 [-0.85, -0.26] | 0.003 | 0.30 | ... |
```

### 5. Power Analysis with Calculations and Warnings
Show the formula and parameters used:
```markdown
### Power Analysis

**Parameters**: α = 0.05, two-tailed test, n = 19 genotypes

| Metric | Value |
|--------|-------|
| Minimum Detectable |r| (at 80% power) | 0.58 |
| Achieved Power Range | 0.05 - 0.85 |
| Median Achieved Power | 0.12 |
| Correlations with Power ≥ 80% | 1/88 (1.1%) |

⚠️ **Underpowered Analysis**: With n=19 genotypes, only correlations with |r| ≥ 0.58
can be detected at 80% power. Consider increasing sample size or focusing on
correlations with achieved power > 0.5.

To achieve 80% power for detecting |r| = 0.40:
- Required n ≈ 46 genotypes
```

### 6. FDR=0 Interpretation Section
When no correlations survive FDR correction:
```markdown
### Statistical Significance

**FDR-Corrected Results**: 0/88 correlations significant at q < 0.05

ℹ️ **Interpretation**: No correlations remained significant after FDR correction.
This is expected when:
- Sample size is small relative to the number of tests
- True effect sizes are modest
- Multiple testing burden is high (88 tests)

**Recommendations**:
1. Examine raw p-values (p < 0.05): 5 correlations show nominal significance
2. Focus on correlations with highest power (>0.5) and lowest p-values
3. For definitive conclusions, increase sample size to ≥30 genotypes

*Note: Absence of FDR significance does not mean absence of biological relationships.*
```

## Acceptance Criteria

1. [ ] FDR method dynamically extracted from config and correctly named
2. [ ] Images embedded as base64 data URIs in markdown
3. [ ] Table headers include inline variable definitions
4. [ ] Correlation table includes 95% confidence intervals
5. [ ] Power analysis shows calculation parameters and sample size recommendations
6. [ ] Warning displayed when >50% of correlations have power < 80%
7. [ ] Interpretation section added when FDR significant count = 0
8. [ ] All changes covered by unit tests
9. [ ] Existing tests continue to pass

## Files to Modify

- `src/sleap_roots_analyze/summary/cross_platform_summary.py` - Main changes
- `src/sleap_roots_analyze/pipeline_runner.py` - Fix Methods section FDR method
- `tests/test_cross_platform_summary.py` - New tests

## Out of Scope

- Changes to correlation calculation itself
- Changes to power analysis calculation
- Interactive visualizations
