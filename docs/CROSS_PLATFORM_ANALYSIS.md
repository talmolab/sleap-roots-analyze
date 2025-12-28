# Cross-Platform Analysis Guide

This guide covers the cross-platform correlation analysis pipeline, which compares trait measurements between two experimental platforms (e.g., greenhouse vs. field, RhizoVision vs. SLEAP Roots).

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Multiple Testing Correction](#multiple-testing-correction)
- [Output Files](#output-files)
- [Interpreting Results](#interpreting-results)
- [Examples](#examples)
- [References](#references)

---

## Overview

The cross-platform pipeline:

1. Loads trait data from two experiments
2. Identifies common genotypes between experiments
3. Calculates genotype means for each trait
4. Computes pairwise correlations between all trait combinations
5. Applies multiple testing correction (FDR)
6. Generates visualizations and exports results

---

## Configuration

### Required Parameters

```yaml
# Experiment 1
exp1_data_path: "path/to/experiment1.csv"
exp1_name: "Experiment 1 Name"
exp1_genotype_col: "Genotype"

# Experiment 2
exp2_data_path: "path/to/experiment2.csv"
exp2_name: "Experiment 2 Name"
exp2_genotype_col: "Genotype"
```

### Optional Parameters

```yaml
# Correlation settings
correlation_method: "spearman"      # Options: "spearman", "pearson"
min_samples_per_genotype: 3         # Minimum replicates per genotype
significance_level: 0.05            # Alpha level for significance

# Multiple Testing Correction
fdr_correction_method: "fdr_by"     # Options: "fdr_bh", "fdr_by", "none"

# Visualization
top_n_correlations: 30
top_n_joint_plots: 10
top_n_boxplots: 10
```

---

## Multiple Testing Correction

### The Multiple Testing Problem

When testing thousands of correlations simultaneously, false positives accumulate. At significance level α = 0.05, approximately 5% of tests will appear "significant" purely by chance.

**Example:** Testing 10,000 trait pairs:
- Expected false positives: 10,000 × 0.05 = 500 spurious "significant" results

### The Solution: False Discovery Rate (FDR) Control

FDR correction controls the **expected proportion of false discoveries** among all discoveries, rather than controlling the probability of any false positive (family-wise error rate).

**Definition:** If we call *R* tests significant and *V* of those are false positives, then:

```
FDR = E[V/R]  (expected proportion of false discoveries)
```

### Benjamini-Hochberg (BH) Procedure

The original FDR procedure (Benjamini & Hochberg, 1995), valid when tests are independent or positively correlated.

#### Mathematical Formulation

Given *m* hypothesis tests with p-values, let p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎ denote the ordered p-values.

**Step 1:** For each test *i* (in rank order), calculate the critical value:

```
α_i = (i / m) × α
```

where α is the desired FDR level (e.g., 0.05).

**Step 2:** Find the largest rank *k* such that:

```
p_(k) ≤ (k / m) × α
```

**Step 3:** Reject all hypotheses H₍₁₎, H₍₂₎, ..., H₍ₖ₎ (i.e., those with the *k* smallest p-values).

#### Adjusted P-Value Formula

The BH-adjusted p-value (q-value) for the test with rank *i* is:

```
p_adj(i) = min(1, min_{j≥i}(m/j × p_(j)))
```

This is computed by working backwards from the largest p-value:
1. Start with p_adj(m) = p_(m)
2. For i = m-1, m-2, ..., 1: p_adj(i) = min(p_adj(i+1), m/i × p_(i))

#### When to Use BH (`fdr_bh`)

- Tests are independent or positively correlated
- Exploratory analysis where some false positives are acceptable
- Larger sample sizes with sufficient statistical power
- Screening for candidates to validate in follow-up studies

### Benjamini-Yekutieli (BY) Procedure

A conservative modification (Benjamini & Yekutieli, 2001) that controls FDR under **arbitrary dependence** between tests.

#### Mathematical Formulation

The BY procedure modifies the critical values by adding a correction factor *c(m)*:

```
α_i = (i / (m × c(m))) × α
```

where *c(m)* is the *m*-th harmonic number:

```
c(m) = Σ_{i=1}^{m} (1/i) = 1 + 1/2 + 1/3 + ... + 1/m
```

This sum grows logarithmically: c(m) ≈ ln(m) + γ, where γ ≈ 0.5772 (Euler-Mascheroni constant).

#### Adjusted P-Value Formula

The BY-adjusted p-value is:

```
p_adj(i) = min(1, min_{j≥i}((m × c(m))/j × p_(j)))
```

#### Correction Factor Examples

| Number of tests (m) | c(m) | Effective multiplier (m × c(m)) |
|--------------------:|-----:|--------------------------------:|
| 100                 | 5.19 | 519 |
| 1,000               | 7.49 | 7,490 |
| 10,000              | 9.79 | 97,900 |
| 16,380              | 10.30| 168,700 |
| 100,000             | 12.09| 1,209,000 |

#### When to Use BY (`fdr_by`) — Default

- Trait correlations are often correlated (not independent)
- Conservative inference is preferred
- Publication-quality results requiring defensible statistics
- Small sample sizes (n < 30 genotypes) where power is limited anyway

### No Correction (`none`)

Raw p-values are used without adjustment. The `_adjusted` columns equal the raw values.

#### When to Use No Correction

- Purely exploratory/hypothesis-generating analysis
- When follow-up validation is already planned
- When presenting raw associations alongside adjusted values for transparency
- Sensitivity analysis comparing corrected vs. uncorrected results

### Why BY Often Yields No Significant Results

With conservative BY correction on thousands of correlated tests, it is common and **expected** to find zero significant correlations. This occurs because:

1. **Large correction factor:** c(m) grows logarithmically with *m*
2. **Many tests:** Cross-platform analysis often involves thousands of trait pairs
3. **Small sample size:** Limited genotypes (e.g., n=18) constrain achievable p-values
4. **Arbitrary dependence penalty:** BY assumes worst-case correlation structure

#### Worked Example

Consider the Turface 19 vs. Cylinder analysis:
- **m = 16,380 tests** (8 Turface traits × 2,048 Cylinder traits after filtering)
- **n = 18 genotypes** with valid data
- **Strongest correlation:** ρ = -0.742, raw p = 0.000423

Calculating the BY-adjusted p-value:

```
c(m) = c(16380) ≈ ln(16380) + 0.5772 ≈ 10.30

For the top-ranked test (rank 1):
p_adj = min(1, (m × c(m)) / 1 × p_(1))
p_adj = min(1, (16380 × 10.30) / 1 × 0.000423)
p_adj = min(1, 168,714 × 0.000423)
p_adj = min(1, 71.4)
p_adj = 1.0  (capped)
```

Even the strongest correlation cannot survive correction because 16,380 × 10.30 × 0.000423 ≈ 71 >> 1.

**This is correct behavior**, not a bug. It indicates that with the current sample size and number of tests, no correlations can be confidently distinguished from chance after proper multiple testing correction.

### Recommendations for Improving Statistical Power

If BY correction yields no significant results, consider:

1. **Increase sample size:** More genotypes provide smaller achievable p-values
   - n=30 genotypes can achieve p ≈ 10⁻⁵ for ρ=0.8
   - n=50 genotypes can achieve p ≈ 10⁻⁸ for ρ=0.8

2. **Reduce number of tests:** Pre-filter traits before correlation
   - Filter by heritability (H² > 0.5)
   - Focus on biologically motivated trait pairs
   - Use trait categories to reduce dimensionality

3. **Use BH for exploration:** If tests have positive dependence
   - BH is less conservative and may identify candidates
   - Follow up with validation experiments

4. **Report both raw and adjusted:** For transparency
   - Show top correlations by raw p-value
   - Note they did not survive FDR correction
   - Treat as hypotheses for future studies

---

## Output Files

### CSV Output: `cross_platform_correlations.csv`

| Column | Type | Description |
|--------|------|-------------|
| `exp1_trait` | string | Trait name from experiment 1 |
| `exp2_trait` | string | Trait name from experiment 2 |
| `spearman_r` | float | Spearman correlation coefficient (ρ), range [-1, 1] |
| `spearman_p` | float | Raw Spearman p-value (two-tailed) |
| `pearson_r` | float | Pearson correlation coefficient (r), range [-1, 1] |
| `pearson_p` | float | Raw Pearson p-value (two-tailed) |
| `n_genotypes` | int | Number of genotypes used (after removing NaN pairs) |
| `spearman_p_adjusted` | float | FDR-corrected Spearman p-value (q-value) |
| `pearson_p_adjusted` | float | FDR-corrected Pearson p-value (q-value) |
| `significant_fdr` | bool | True if primary adjusted p < significance_level |

**Notes:**
- Results are sorted by absolute value of the primary correlation (descending)
- `significant_fdr` uses the primary method's adjusted p-value (Spearman if `correlation_method: "spearman"`)
- Adjusted p-values are capped at 1.0

### Metadata: `pipeline_summary.json`

Key fields related to FDR correction:

```json
{
  "config": {
    "fdr_correction_method": "fdr_by",
    "significance_level": 0.05,
    "correlation_method": "spearman"
  },
  "steps": [
    {
      "name": "02_calculate_correlations",
      "metadata": {
        "fdr_correction_method": "fdr_by",
        "significance_level": 0.05,
        "significant_correlations": 0,
        "total_correlations": 16380
      }
    }
  ]
}
```

### Visualizations

**Summary plot** (`cross_platform_correlation_summary.png`):
- Panel 1: Histogram with "Significant (FDR): N" annotation
- Panel 2: Volcano plot using raw p-values (for visual interpretation)
- Panel 3-4: Top positive/negative correlation bar charts

**Joint plots**: Display both Pearson and Spearman values from the CSV (single source of truth)

**Boxplots**: Genotype distributions for top trait pairs

---

## Interpreting Results

### Correlation Strength Guidelines

| |r| or |ρ| | Interpretation |
|------------|----------------|
| 0.0 - 0.1  | Negligible |
| 0.1 - 0.3  | Weak |
| 0.3 - 0.5  | Moderate |
| 0.5 - 0.7  | Strong |
| 0.7 - 1.0  | Very strong |

### When Many Correlations Are Significant

If many correlations survive FDR correction:
1. Focus on those with biological plausibility
2. Check effect sizes (|r| > 0.5 for strong effects)
3. Verify sample sizes are adequate (n_genotypes > 10)
4. Consider trait redundancy (correlated traits in same experiment)

### When No Correlations Are Significant

This is common with BY correction on large test sets. Actions:
1. Examine top correlations by raw p-value as hypotheses
2. Consider using BH for exploratory analysis
3. Plan validation with increased sample size
4. Report honestly: "No correlations survived FDR correction at α=0.05"

---

## Examples

### Conservative Analysis (BY, default)

```yaml
fdr_correction_method: "fdr_by"
significance_level: 0.05
```

Best for: Publications, final analyses, small sample sizes

### Exploratory Analysis (BH)

```yaml
fdr_correction_method: "fdr_bh"
significance_level: 0.05
```

Best for: Hypothesis generation, moderate sample sizes, independent traits

### No Correction (Raw)

```yaml
fdr_correction_method: "none"
```

Best for: Transparency, sensitivity analysis, when validation is planned

### CLI Usage

```bash
# Run cross-platform analysis
sleap-roots-analyze cross-platform configs/cross_platform_example.yaml

# Run with custom output directory
sleap-roots-analyze cross-platform configs/cross_platform_example.yaml -o results/
```

---

## References

1. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

2. Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188. https://doi.org/10.1214/aos/1013699998

3. Storey, J. D., & Tibshirani, R. (2003). Statistical significance for genomewide studies. *Proceedings of the National Academy of Sciences*, 100(16), 9440-9445. https://doi.org/10.1073/pnas.1530509100
