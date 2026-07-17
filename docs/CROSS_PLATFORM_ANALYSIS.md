# Cross-Platform Analysis Guide

This guide covers the cross-platform correlation analysis pipeline, which compares trait measurements between two experimental platforms (e.g., greenhouse vs. field, RhizoVision vs. SLEAP Roots).

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Multiple Testing Correction](#multiple-testing-correction)
- [Edge Cases](#edge-cases)
- [Confidence Intervals](#confidence-intervals)
- [Power Analysis](#power-analysis)
- [Minimum Genotypes Filter](#minimum-genotypes-filter)
- [Output Files](#output-files)
- [Interpreting Results](#interpreting-results)
- [Examples](#examples)
- [Public PC-Correlation and Trait-Enrichment Workflows](#public-pc-correlation-and-trait-enrichment-workflows)
- [Cross-Platform Genotype-Effect Prediction](#cross-platform-genotype-effect-prediction)
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
confidence_level: 0.95              # Confidence level for CI (0.95 = 95% CI)

# Multiple Testing Correction
fdr_correction_method: "fdr_by"     # Options: "fdr_bh", "fdr_by", "none"

# Minimum Genotypes Filter
min_genotypes_for_correlation: 10   # Min valid genotypes for correlation (excludes low-n pairs)

# Power Analysis
power_analysis_alpha: 0.05          # Significance level for power calculations
power_analysis_power: 0.80          # Target power for minimum detectable effect

# Visualization
top_n_correlations: 30
top_n_joint_plots: 10
top_n_boxplots: 10

# Input-contract validation (requires the optional `contracts` extra)
validate_input: "warn"              # Options: "off", "warn", "strict"
```

### Input-contract validation (`validate_input`)

When the optional [`sleap-roots-contracts`](https://github.com/talmolab/sleap-roots-contracts)
extra is installed (`pip install "sleap-roots-analyze[contracts]"`), each loaded and
aligned experiment frame is validated on a **discarded copy** at the load boundary —
it never alters the data fed to the analysis, so the mode never changes results. When
the extra is absent, validation degrades to a logged no-op.

- `off` — skip validation entirely.
- `warn` (default) — log non-fatal issues; raise only on universal structural errors
  (missing/blank `genotype`, no numeric trait, bad role dtype). Rows with a blank
  genotype are dropped during alignment (they cannot participate in any
  genotype-aligned comparison), so they never trigger a failure.
- `strict` — escalate recommended-column issues to errors. Aligned frames carry no
  per-sample id (the source barcode is dropped during alignment), so `strict` injects a
  synthetic positional `sample_id` into the discarded copy rather than failing on the
  structurally-absent role; it otherwise enforces the full contract. For routine runs,
  `warn` is recommended.

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

## Edge Cases

### Constant-Valued Traits (Zero Variance)

When a trait has constant values across all genotypes (zero variance), correlation cannot be computed.

**Behavior:**
- Correlation coefficient (r or ρ) = NaN
- Raw p-value = NaN
- Adjusted p-value = NaN (excluded from FDR correction)
- `significant_fdr` = False

**Why this happens:**
Correlation measures the relationship between two variables as they vary together. If one variable doesn't vary (all genotypes have the same trait value), there's no covariation to measure. Mathematically, the correlation formula divides by the standard deviation, which is zero for constant data.

**Example:** If all 20 genotypes have `root_count = 5.0`, then `root_count` vs any other trait will have NaN correlation.

### Fewer Than 3 Valid Samples

When fewer than 3 genotypes have valid (non-NaN) data for a trait pair, correlation is not computed.

**Behavior:**
- Correlation coefficient = NaN
- Raw p-value = NaN
- Adjusted p-value = NaN
- `significant_fdr` = False

**Why 3 is the minimum:**
1. **Degrees of freedom:** Correlation tests have n-2 degrees of freedom. With n=2, df=0, making p-value calculation undefined.
2. **Statistical meaning:** With only 2 points, you always get r=±1.0 (perfect correlation), which is meaningless.
3. **Convention:** Most statistical software requires n≥3 for correlation.

### NaN P-value Handling in FDR Correction

The `statsmodels.multipletests` function returns **all NaN** if any input p-value is NaN. The pipeline handles this by:

1. **Filtering:** NaN p-values are excluded before calling `multipletests`
2. **Correcting:** FDR correction is applied only to valid p-values
3. **Merging:** Results are merged back, preserving NaN for invalid correlations
4. **Logging:** A warning is logged when NaN p-values are encountered

**Example log output:**
```
WARNING: Found 5 NaN p-values out of 1000 total. These will be excluded from FDR correction and remain NaN.
```

### Single Correlation (m=1)

When only one trait pair exists (e.g., one trait in each experiment), there's no multiple testing to correct.

**Behavior:**
- Adjusted p-value = Raw p-value (no correction applied)
- `significant_fdr` based on raw p-value vs significance level

This is correct because with m=1 test, the expected false discoveries under the null hypothesis is α × 1 = α, which is what we're already controlling.

---

## Confidence Intervals

### Fisher z-Transformation Method

Confidence intervals for correlation coefficients are computed using Fisher's z-transformation, which normalizes the sampling distribution of r:

**Step 1: Transform to z-scale**
```
z = arctanh(r) = 0.5 × ln((1+r)/(1-r))
```

**Step 2: Compute standard error**
```
SE_z = 1 / √(n-3)
```

**Step 3: Compute CI on z-scale**
```
z_low = z - z_{α/2} × SE_z
z_high = z + z_{α/2} × SE_z
```
where z_{α/2} is the critical value (1.96 for 95% CI, 2.576 for 99% CI).

**Step 4: Back-transform to r-scale**
```
ci_low = tanh(z_low)
ci_high = tanh(z_high)
```

### Why n ≥ 4 is Required

The standard error formula has (n-3) in the denominator. When n < 4:
- n = 3: SE = 1/√0 = undefined (division by zero)
- n = 2: Would give negative variance

For trait pairs with n < 4 genotypes, CI bounds are set to NaN.

### Accuracy Notes

- **Pearson r**: Fisher z-transformation is exact under bivariate normality
- **Spearman ρ**: Fisher z provides a good asymptotic approximation (accurate for n ≥ 10)

### Example Calculation

For r = 0.6, n = 25, 95% CI:

1. z = arctanh(0.6) = 0.693
2. SE_z = 1/√(25-3) = 1/√22 = 0.213
3. z_low = 0.693 - 1.96 × 0.213 = 0.275
4. z_high = 0.693 + 1.96 × 0.213 = 1.111
5. ci_low = tanh(0.275) = 0.269
6. ci_high = tanh(1.111) = 0.804

Result: 95% CI = (0.27, 0.80)

### Interpreting Confidence Intervals

- **Narrower CI**: More precise estimate (larger sample size)
- **Wider CI**: Less precise estimate (smaller sample size)
- **CI contains 0**: Correlation may not be significantly different from zero
- **Non-overlapping CIs**: Correlations are likely significantly different

### Reference

Fisher, R.A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. Metron, 1, 3-32.

---

## Power Analysis

Statistical power analysis helps assess whether your sample size is sufficient to detect meaningful correlations. The pipeline provides two key metrics:

### Achieved Power

For each correlation, the pipeline calculates the **achieved power** — the probability that a correlation of the observed magnitude would be detected as statistically significant given the sample size.

**Formula (using Fisher z-transformation):**

1. Transform correlation: z_r = arctanh(|r|)
2. Effect size: λ = z_r × √(n-3)
3. Critical z-value: z_α = Φ⁻¹(1 - α/2)
4. Power: Φ(λ - z_α) + Φ(-λ - z_α)

**Interpretation:**
- Power ≥ 0.80: Adequate power (standard convention)
- Power 0.50-0.79: Moderate power, interpret with caution
- Power < 0.50: Low power, underpowered to detect this effect

### Minimum Detectable Correlation

The pipeline calculates the **minimum detectable correlation** (MDR) — the smallest correlation that can be detected with the specified power at the given sample size.

**Formula:**

1. Critical z-values: z_α = Φ⁻¹(1 - α/2), z_β = Φ⁻¹(power)
2. Minimum detectable z: z_r = (z_α + z_β) / √(n-3)
3. Back-transform: r = tanh(z_r)

**Example values (α=0.05, power=0.80):**

| n (genotypes) | Minimum Detectable |r| |
|--------------:|--------------------:|
| 10            | 0.76 |
| 15            | 0.62 |
| 20            | 0.58 |
| 30            | 0.49 |
| 50            | 0.38 |
| 100           | 0.28 |

### Configuration

```yaml
# Power analysis parameters
power_analysis_alpha: 0.05   # Significance level (Type I error rate)
power_analysis_power: 0.80   # Target power (1 - Type II error rate)
```

### When Power is NaN

Power is NaN when:
- n < 4 (Fisher z variance undefined with n-3 denominator)
- Correlation is NaN (constant trait or insufficient data)

### Why Power Matters

With n ≈ 18 genotypes (typical in greenhouse experiments), many correlations may be statistically underpowered. The power analysis helps:

1. **Interpret non-significant results**: Low power means "absence of evidence is not evidence of absence"
2. **Focus on robust findings**: High-power correlations are more reliable
3. **Plan future experiments**: MDR tells you what effect size you can detect
4. **Prioritize follow-up studies**: Target high-power, high-effect correlations

### Reference

Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum Associates, 2nd edition.

---

## Minimum Genotypes Filter

The pipeline applies a hard filter to exclude trait pairs with too few valid genotypes for reliable correlation estimation.

### Why This Filter Exists

1. **Statistical reliability**: Correlations with very few observations (n < 10) have extremely wide confidence intervals and unreliable p-values
2. **Fisher z accuracy**: The Fisher z-transformation approximation for Spearman correlations is most accurate for n ≥ 10
3. **Spurious results prevention**: With low n, random fluctuations can produce misleadingly strong correlations

### Configuration

```yaml
# Minimum genotypes for correlation calculation
min_genotypes_for_correlation: 10   # Default: 10 (recommended for Fisher z accuracy)
```

Valid range: 3 to unlimited. The default of 10 is recommended for reliable Spearman correlation CIs.

### Behavior

When a trait pair has fewer valid genotypes than `min_genotypes_for_correlation` (after NaN removal):

1. **Excluded from CSV**: The trait pair is not included in the output
2. **Counted in metadata**: `n_correlations_filtered_low_n` tracks how many pairs were filtered
3. **Logged**: A summary message indicates how many pairs were filtered and why

**Example log output:**
```
INFO: Filtered 25/1000 trait pairs with n_genotypes < 10
```

**If all pairs are filtered:**
```
WARNING: All 500 trait pairs were filtered out. Consider lowering min_genotypes_for_correlation (current: 10).
```

### Relationship to min_samples_per_genotype

These two parameters serve different purposes:

| Parameter | What it filters | When applied |
|-----------|-----------------|--------------|
| `min_samples_per_genotype` | Genotypes with too few replicates | During data loading |
| `min_genotypes_for_correlation` | Trait pairs with too few valid genotypes | During correlation calculation |

A genotype may pass `min_samples_per_genotype` but still be excluded from a correlation if it has NaN values for those traits, reducing n below `min_genotypes_for_correlation`.

### Recommendations

- **n ≥ 10**: Recommended for publication-quality results
- **n ≥ 5**: Acceptable for exploratory analysis
- **n = 3**: Minimum valid (use only for small pilot studies)

---

## Output Files

### CSV Output: `cross_platform_correlations.csv`

| Column | Type | Description |
|--------|------|-------------|
| `exp1_trait` | string | Trait name from experiment 1 |
| `exp2_trait` | string | Trait name from experiment 2 |
| `spearman_r` | float | Spearman correlation coefficient (ρ), range [-1, 1] |
| `spearman_p` | float | Raw Spearman p-value (two-tailed) |
| `spearman_r_ci_low` | float | Lower bound of Spearman ρ confidence interval |
| `spearman_r_ci_high` | float | Upper bound of Spearman ρ confidence interval |
| `pearson_r` | float | Pearson correlation coefficient (r), range [-1, 1] |
| `pearson_p` | float | Raw Pearson p-value (two-tailed) |
| `pearson_r_ci_low` | float | Lower bound of Pearson r confidence interval |
| `pearson_r_ci_high` | float | Upper bound of Pearson r confidence interval |
| `n_genotypes` | int | Number of genotypes used (after removing NaN pairs) |
| `achieved_power` | float | Statistical power for detecting the observed correlation |
| `spearman_p_adjusted` | float | FDR-corrected Spearman p-value (q-value) |
| `pearson_p_adjusted` | float | FDR-corrected Pearson p-value (q-value) |
| `significant_fdr` | bool | True if primary adjusted p < significance_level |

**Notes:**
- Results are sorted by absolute value of the primary correlation (descending)
- `significant_fdr` uses the primary method's adjusted p-value (Spearman if `correlation_method: "spearman"`)
- Adjusted p-values are capped at 1.0
- CI columns are NaN when n_genotypes < 4 or correlation is NaN

### Metadata: `pipeline_summary.json`

Key fields related to statistical analysis:

```json
{
  "config": {
    "fdr_correction_method": "fdr_by",
    "significance_level": 0.05,
    "confidence_level": 0.95,
    "correlation_method": "spearman",
    "min_genotypes_for_correlation": 10,
    "power_analysis_alpha": 0.05,
    "power_analysis_power": 0.80
  },
  "steps": [
    {
      "name": "02_calculate_correlations",
      "metadata": {
        "fdr_correction_method": "fdr_by",
        "significance_level": 0.05,
        "confidence_level": 0.95,
        "significant_correlations": 0,
        "total_correlations": 16380,
        "min_genotypes_for_correlation": 10,
        "n_correlations_filtered_low_n": 25,
        "power_analysis_alpha": 0.05,
        "power_analysis_power": 0.80,
        "minimum_detectable_r": 0.62,
        "modal_n_genotypes": 15
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

## Summary Generation

After running cross-platform analyses, you can generate a detailed summary document that aggregates results from all comparisons. This is especially useful when running multiple cross-platform pipelines in a single run.

### Automatic Summary Generation

When using `sleap-roots-analyze run-all`, a comprehensive `SUMMARY.md` file is automatically generated in the pipeline run directory. This includes:

- Trait reduction statistics (when clustering is enabled)
- Correlation statistics (total, nominal significant, FDR significant)
- Power analysis with sample size recommendations
- Links to visualizations (dendrograms, heatmaps, joint plots)
- Interpretation guidance for common scenarios (e.g., FDR=0)

**Note:** The summary uses relative file paths for images, keeping the file small (~50KB) and viewable in VS Code markdown preview. To view in a browser, use the HTML conversion script (see below).

### Claude Command: `/cross-platform-summary`

For interactive summary generation with validation, use the Claude command:

```
/cross-platform-summary pipeline_runs/2026-02-03_092935
```

This command:
1. Reads all cross-platform results in the specified directory
2. Validates reported statistics against source CSVs
3. Generates a detailed markdown summary with embedded images
4. Provides interpretation guidance based on results

### Summary Sections

The generated summary includes:

**Configuration:**
- Correlation method (Spearman/Pearson)
- Trait reduction method and target
- Clustering threshold (when applicable)
- FDR correction method

**Trait Reduction (when clustering enabled):**
- Original trait counts per experiment
- Number of clusters formed
- Representative traits selected
- Reduction percentage achieved
- Dendrogram and heatmap visualizations

**Correlation Statistics:**
- Total correlations calculated
- Nominally significant (p < 0.05)
- FDR-significant (q < 0.05)
- Top correlations table with:
  - ρ (Spearman correlation)
  - 95% confidence intervals
  - Raw and FDR-adjusted p-values
  - Achieved power
  - Sample size

**Power Analysis:**
- Analysis parameters (α, n, minimum detectable |r|)
- Power distribution (min, median, max)
- Percentage of tests with adequate power (≥80%)
- Sample size recommendations for future studies
- Warnings when study is underpowered

**Interpretation Guidance:**
- When FDR=0: Explanation of why no correlations survived correction
- Sample size recommendations for detecting specific effect sizes
- Suggestions for follow-up studies

### Validation Guardrails

The summary generator includes validation checks to ensure accuracy:

| Check | Purpose |
|-------|---------|
| Correlation counts | Verify total/nominal/FDR counts match source CSV |
| Top correlations | Verify top N values match sorted CSV |
| Power statistics | Verify power range and percentiles match CSV |
| Trait reduction | Verify cluster counts match cluster membership CSV |
| Missing files | Gracefully handle missing optional files |

If validation fails, warnings are included in the summary with details about the discrepancy.

### Output Format Options

The summary generator supports multiple output formats:

| Mode | Description | Use Case |
|------|-------------|----------|
| `file_path` | Relative image paths (default) | VS Code markdown preview |
| `embed` | Base64-embedded images | Portable single-file sharing |
| `auto` | Smart selection based on size | Intelligent default |

**Memory limits:** Embedded images are limited to 10MB total. If exceeded, the generator falls back to file paths with a warning.

### HTML Conversion for Browser Viewing

Since browsers don't render markdown files directly, use the conversion script:

```bash
# Convert latest run to HTML
uv run python scripts/convert_summary_to_html.py

# Convert specific run
uv run python scripts/convert_summary_to_html.py pipeline_runs/2026-02-04_120723

# Open in browser (Windows)
start pipeline_runs/2026-02-04_120723/SUMMARY.html
```

The HTML output includes:
- Styled tables with alternating row colors
- Properly rendered headers and formatting
- Image references that work when opened from the run directory

### Example Output

```markdown
## Turface 19 Genotypes vs Cylinder 23 Genotypes

### Configuration

- **Correlation Method**: spearman
- **Trait Reduction**: clustering
- **Reduction Target**: both
- **Clustering Threshold**: 0.7

### Trait Reduction

**Turface 19 Genotypes**:
- 82 original traits → 15 clusters → 15 representatives (81.7% reduction)

**Cylinder 23 Genotypes**:
- 2048 original traits → 245 clusters → 245 representatives (88.0% reduction)

### Correlation Statistics

| Metric | Value |
| --- | --- |
| Total Correlations | 3675 |
| Nominal Significant (p < 0.05) | 184 |
| FDR Significant | 0 |

#### Interpretation: No FDR-Significant Correlations

**Note:** No correlations survived FDR correction. This is common when:
- Sample sizes are small
- Effect sizes are modest
- Testing many correlations simultaneously

**Nominal Significant (p < 0.05):** 184 correlations reached nominal significance
before FDR correction. These may warrant further investigation with larger sample sizes.

### Power Analysis

**Analysis Parameters:**
- **Significance level (α):** 0.05
- **Modal sample size (n):** 19
- **Minimum detectable |r| at 80% power:** 0.58
- **Required n for |r|=0.40 at 80% power:** 46

**Power Distribution:**
- **Min Power:** 0.05
- **Median Power:** 0.12
- **Max Power:** 0.89
- **% Above 80%:** 2.3%

**⚠️ Warning: Study may be underpowered.** Only 2.3% of correlations have ≥80% power.
Consider increasing sample size for future studies.
```

---

## Interpreting Results

### Correlation Strength Guidelines

| \|r\| or \|ρ\| | Interpretation |
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

3. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum Associates, 2nd edition.

4. Fisher, R.A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. *Metron*, 1, 3-32.

5. Bonett, D.G. & Wright, T.A. (2000). Sample size requirements for estimating Pearson, Kendall and Spearman correlations. *Psychometrika*, 65(1), 23-28.

6. Storey, J. D., & Tibshirani, R. (2003). Statistical significance for genomewide studies. *Proceedings of the National Academy of Sciences*, 100(16), 9440-9445. https://doi.org/10.1073/pnas.1530509100

## Public PC-Correlation and Trait-Enrichment Workflows

Two reusable, DAG-structured workflows expose the wheat-EDPIE cross-platform
analyses as single public calls (issue #119). They are **independent**: the
trait-enrichment workflow is *not* downstream of the PC-correlation workflow.

### PC-level cross-platform correlations

`cross_platform_pc_correlations` runs the full PC-level DAG from a pipeline-run
directory: load per-platform sample PC scores (`pc_scores.csv`) and QC genotype
labels → aggregate sample PC scores to **genotype means** (sample-level PCA is
done upstream, so this preserves the sample-PCA → genotype-mean → correlate
ordering) → align on common genotypes → correlate every PC against every PC per
platform pair → FDR-correct at `combined` and/or `per_pair` scope → write
`correlations.csv`, `significant_*.csv`, `genotype_means_*.csv`, figures, and
`metadata.json`.

```python
from sleap_roots_analyze import cross_platform_pc_correlations
from sleap_roots_analyze.pc_correlations.aggregate import WHEAT_EDPIE_PLATFORMS

result = cross_platform_pc_correlations(
    pipeline_run="/path/to/pipeline_run",
    platform_config=WHEAT_EDPIE_PLATFORMS,   # example config; supply your own
    output_dir="./outputs/pc_correlations",
    primary_fdr_method="fdr_by",
    fdr_scope="both",
)
result.summary  # {"n_tests": 47, "n_genotypes": 19, "n_significant_combined": 0, ...}
```

CLI reproduction: `uv run scripts/run_pc_correlations.py --pipeline-run <dir> --output-dir <dir>`.

Confidence intervals, achieved power, and the minimum detectable correlation are
reused from `cross_experiment_analysis` (`calculate_correlation_ci`,
`achieved_power`, `minimum_detectable_correlation`) — a single source of truth.

### Trait-level correlation enrichment

`trait_correlation_enrichment` tests whether the number of nominally significant
(`p < alpha`) trait correlations in existing `cross_platform_correlations.csv`
files deviates from chance, using an exact binomial test per pair and pooled
(`Combined`). It writes `enrichment_results.csv`, an enrichment figure, and
`metadata.json`, and returns typed `EnrichmentResult` records.

```python
from sleap_roots_analyze import trait_correlation_enrichment

result = trait_correlation_enrichment(
    correlation_files={
        "Turface vs Cylinder": "/path/.../cross_platform_correlations.csv",
        "Field vs Cylinder": "/path/.../cross_platform_correlations.csv",
    },
    output_dir="./outputs/trait_enrichment",
)
result["results"][0]  # Combined EnrichmentResult (fold_enrichment, interpretation, ...)
```

CLI reproduction: `uv run scripts/run_trait_enrichment.py --pair "Label=<csv>" --output-dir <dir>`.

## Cross-Platform Genotype-Effect Prediction

Tier 3 of the wheat EDPIE cross-platform genotype-prediction program (issue #194)
reframes the cross-platform result from *correlation* to *predictability*: given
genotype BLUPs (Tier 1, `extract_blup_table`) estimated within one platform, test
whether they predict genotype effects in another platform via ridge regression /
Partial Least Squares (PLS) with leave-one-genotype-out (LOGO) cross-validation.

`logo_cv_predict` implements the CV-hygiene contract underlying every LOGO-CV
oracle in this program: a fresh `sklearn.pipeline.Pipeline` is instantiated and
fit **inside** each fold, so no step ever sees the held-out genotype during fit.
Three `reduction_method` values are supported: `pls_latent` (default —
`PLSRegression(n_components=1)`, fixed rather than searched via an inner CV loop,
both for statistical reasons at n≈18 training genotypes and to keep a future
1000-permutation null tractable) and `representatives` (variance-based cluster
representatives, unsupervised and selected once before the fold loop) fit a
`Ridge()`/`PLSRegression` model directly; `pc1` reduces each fold's predictors to
a single principal-component score via `fit_pca_on_fold` — a per-fold PCA utility
deliberately distinct from the pipeline-level `PCA` step, since fitting PCA on all
genotypes before the fold loop would leak the held-out genotype's position into
the component loadings.

```python
from sleap_roots_analyze import logo_cv_predict, CrossPlatformPredictionResult

result = logo_cv_predict(
    X=source_platform_blup_table,          # (n_genotypes, n_traits) DataFrame
    y=target_platform_blup_table["target_trait"].values,
    genotypes=source_platform_blup_table.index.tolist(),
    reduction_method="pls_latent",
)
print(result.r2, result.rmse, result.spearman_rho)

# Bundle multiple prediction targets (representative traits + PC1) into one
# JSON-serializable result:
prediction = CrossPlatformPredictionResult.from_logo_cv_results(
    source_platform="Turface19",
    target_platform="Cylinder",
    predictor_source="blup",
    reduction_method="pls_latent",
    logo_cv_results={"target_trait": result, "PC1": pc1_result},
)
```

Trait-set continuity with the paper's own published Section 3.4 result (cluster
each platform's traits independently at |ρ|≥0.80 → correlate every
representative pair → filter to |ρ|≥0.55 → count distinct traits per side) is
verified against the real `cluster_correlated_traits`/
`select_cluster_representatives` functions in `cross_experiment_analysis`, not a
hardcoded lookup — see `tests/test_cross_platform_prediction.py`'s
`TestTraitSetIdentityOracle`.

Tier 3.5 (issue #196) wires this machinery into `CrossPlatformPipeline` itself, as
an optional 6th step — `PredictCrossPlatformStep` — on the same per-pair
`cross-platform` command already used for correlation. The permutation null and
its figures (Tier 4) remain a separate, later change.

### Configuration

Prediction is disabled by default (`enabled: false`), so no existing
`CrossPlatformConfig` YAML is affected. To enable it for a directed pair, add a
`prediction:` block to that pair's existing YAML and rerun the same
`cross-platform` command — one run directory then contains both correlation and
predictability numbers for the same pair:

```yaml
exp1_data_path: "path/to/turface19_qc.csv"
exp1_name: "Turface19"
exp1_genotype_col: "Genotype"
exp2_data_path: "path/to/cylinder_qc.csv"
exp2_name: "Cylinder"
exp2_genotype_col: "Genotype"

prediction:
  enabled: true
  predictor_source: "blup"          # "blup" (default) or "genotype_means"
  reduction_method: "pls_latent"    # "pls_latent" (default), "representatives", "pc1"
  comparison_methods: ["representatives"]  # additional methods reported alongside
  representative_selection_metric: "variance"  # only "variance" is supported
  platform_pairs:
    - source: "Turface19"           # must equal exp1_name or exp2_name
      target: "Cylinder"            # the other one
  source_blup_path: "path/to/turface19_08_blup_adjusted_means.csv"
  target_blup_path: "path/to/cylinder_08_blup_adjusted_means.csv"
```

`source_blup_path`/`target_blup_path` (Tier 1's `08_blup_adjusted_means.csv`
output, one row per genotype) are only required when `predictor_source="blup"`;
`predictor_source="genotype_means"` instead aggregates the same raw per-sample
data `exp1_data_path`/`exp2_data_path` already load for correlation, via a plain
per-genotype mean — a "full raw-data ablation" against the BLUP-based predictor.
`platform_pairs` takes exactly one entry, naming which of `exp1_name`/`exp2_name`
is the predictor (`source`) and which is predicted (`target`) — prediction is
directional (Turface19→Cylinder is a different model from Cylinder→Turface19),
unlike correlation.

Output: one `06_prediction_<method>.json` file per method (`reduction_method`
plus each of `comparison_methods`), holding a `CrossPlatformPredictionResult`
with one entry per prediction target — the target platform's cluster-
representative traits, plus a `PC1` entry.

### Current Limitations

- The `PC1` target's computation (`sklearn.decomposition.PCA` via
  `pca.fit_pca()`, with `StandardScaler` applied first and `random_state=42`
  fixed) is **not user-configurable** in this tier — no `PCAConfig` exists on
  this pipeline.
- `representative_selection_metric` only supports `"variance"` in this tier;
  `"heritability"`-based representative selection is not yet implemented
  (`select_cluster_representatives` has no metric parameter to select by it).
- `blup_refit_per_fold` is present in the config schema (for forward
  compatibility with a future heritability-based
  `representative_selection_metric`) but is currently inert — no valid
  `representative_selection_metric` value triggers any behavior from it.
- Only genotypes common to **both** the source and target BLUP/genotype-mean
  tables are used for prediction. Genotypes present in only one side are
  silently excluded from the result (not merely from an error path) — check
  `06_prediction_<method>.json`'s genotype count against each platform's own
  genotype count if this matters for your analysis.
- `CrossPlatformSummaryGenerator`/`/cross-platform-summary` does not yet
  surface prediction results — tracked as follow-up
  [#197](https://github.com/talmolab/sleap-roots-analyze/issues/197).
