## MODIFIED Requirements

### Requirement: Cross-Platform Configuration

The system SHALL provide configuration options for cross-platform trait correlation analysis through the `CrossPlatformConfig` dataclass with the following required parameters:

- `exp1_data_path`: Path to experiment 1 cleaned traits CSV
- `exp1_name`: Display name for experiment 1 (e.g., "Cylinder")
- `exp1_genotype_col`: Column name containing genotype identifiers in experiment 1
- `exp2_data_path`: Path to experiment 2 cleaned traits CSV
- `exp2_name`: Display name for experiment 2 (e.g., "Turface")
- `exp2_genotype_col`: Column name containing genotype identifiers in experiment 2

And the following optional parameters with defaults:

- `correlation_method`: Statistical method ("spearman", "pearson", "kendall"), default "spearman"
- `min_samples_per_genotype`: Minimum samples required per genotype, default 3
- `significance_level`: P-value threshold for significance, default 0.05
- `top_n_correlations`: Number of top correlations to display in summary, default 20
- `top_n_joint_plots`: Number of joint plots to generate, default 6
- `top_n_boxplots`: Number of boxplots to generate, default 6
- `figsize_summary`: Summary figure size tuple, default (14, 12)
- `figsize_joint`: Joint plot figure size tuple, default (10, 10)
- `figsize_boxplot`: Boxplot figure size tuple, default (14, 6)
- `exp1_exclude_cols`: List of column names to exclude from experiment 1 trait analysis, default None
- `exp2_exclude_cols`: List of column names to exclude from experiment 2 trait analysis, default None
- `fdr_correction_method`: Method for multiple testing correction ("fdr_bh", "fdr_by", "none"), default "fdr_by"
- `confidence_level`: Confidence level for correlation coefficient intervals, default 0.95

#### Scenario: Valid configuration with required fields

- **WHEN** user provides valid paths and column names for both experiments
- **THEN** configuration object is created successfully with default optional parameters

#### Scenario: Missing required fields

- **WHEN** user provides configuration missing required fields (data paths or genotype columns)
- **THEN** configuration validation fails with clear error message indicating missing fields

#### Scenario: Invalid correlation method

- **WHEN** user specifies correlation method not in ["spearman", "pearson", "kendall"]
- **THEN** configuration validation fails with error listing valid options

#### Scenario: Invalid FDR correction method

- **WHEN** user specifies fdr_correction_method not in ["fdr_bh", "fdr_by", "none"]
- **THEN** configuration validation fails with error listing valid options

#### Scenario: Invalid confidence level

- **WHEN** user specifies confidence_level outside (0, 1) exclusive range
- **THEN** configuration validation fails with error indicating valid range

#### Scenario: Custom confidence level

- **WHEN** user specifies confidence_level as 0.99
- **THEN** 99% confidence intervals are computed for all correlations
- **AND** intervals are wider than default 95% intervals

#### Scenario: Exclude metadata columns from experiment 1

- **WHEN** user specifies exp1_exclude_cols with metadata column names like ["Ent", "Sub", "Cid"]
- **THEN** those columns are excluded from experiment 1 trait analysis
- **AND** they do not appear in correlation results

#### Scenario: Exclude metadata columns from experiment 2

- **WHEN** user specifies exp2_exclude_cols with metadata column names like ["File.me", "scanner"]
- **THEN** those columns are excluded from experiment 2 trait analysis
- **AND** they do not appear in correlation results

#### Scenario: Different exclusion lists per experiment

- **WHEN** exp1 has field metadata columns and exp2 has imaging metadata columns
- **THEN** each experiment's exclusion list is applied independently
- **AND** correlations only include biological trait columns

### Requirement: Calculate Cross-Platform Correlations

The system SHALL calculate pairwise trait correlations between experiments through `CalculateCrossPlatformCorrelationsStep` with the following behavior:

- Calculate genotype means for each trait in both experiments using `calculate_genotype_means()`
- Compute **both Pearson and Spearman** correlations for all trait pairs, regardless of which method is configured as primary
- Remove NaN pairs before correlation calculation
- Calculate p-values for each correlation (both Pearson and Spearman)
- **Calculate confidence intervals** for each correlation coefficient using Fisher z-transformation
- Apply FDR correction to p-values based on `fdr_correction_method` config:
  - `fdr_bh`: Benjamini-Hochberg correction (assumes test independence)
  - `fdr_by`: Benjamini-Yekutieli correction (valid under arbitrary dependence)
  - `none`: No correction applied (adjusted p-values equal raw p-values)
- **Handle NaN p-values gracefully**: If any correlation produces NaN p-values (e.g., from constant-valued traits or insufficient data), the FDR correction SHALL:
  - Filter out NaN p-values before applying correction
  - Apply correction only to valid p-values
  - Preserve NaN in adjusted p-value columns for invalid correlations
  - Set `significant_fdr` to False for rows with NaN adjusted p-values
  - Log a warning indicating the count of NaN p-values encountered
- Store results with columns: `exp1_trait`, `exp2_trait`, `spearman_r`, `spearman_p`, `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r`, `pearson_p`, `pearson_r_ci_low`, `pearson_r_ci_high`, `n_genotypes`, `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
- Sort results by absolute value of the **primary** correlation (determined by `correlation_method` config), descending
- Export results to `cross_platform_correlations.csv` in output directory

#### Scenario: Confidence intervals computed for all correlations

- **WHEN** correlation step executes with 50 trait pairs and n=20 genotypes
- **THEN** CSV output contains `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r_ci_low`, `pearson_r_ci_high` columns
- **AND** all CI bounds satisfy: -1 ≤ ci_low ≤ r ≤ ci_high ≤ 1
- **AND** CI width is inversely related to n_genotypes

#### Scenario: Confidence interval for perfect correlation

- **WHEN** two traits have r = 1.0 (perfect positive correlation)
- **THEN** confidence interval is (1.0, 1.0) as a point mass at the boundary
- **AND** no mathematical error occurs from arctanh(1.0)

#### Scenario: Confidence interval with small sample size

- **WHEN** n_genotypes < 4 for a trait pair
- **THEN** CI bounds are NaN (undefined variance with n-3 in denominator)
- **AND** correlation coefficient r is still computed if n >= 3

#### Scenario: Higher confidence level produces wider intervals

- **WHEN** confidence_level is 0.99 instead of 0.95
- **THEN** all confidence intervals are wider
- **AND** width difference follows z_{0.005}/z_{0.025} ratio ≈ 1.32

#### Scenario: NaN p-values from constant trait do not corrupt FDR correction

- **WHEN** one trait pair produces NaN p-values (e.g., constant values in one trait)
- **AND** other trait pairs have valid p-values
- **THEN** FDR correction is applied only to valid p-values
- **AND** NaN p-values remain NaN in the adjusted columns
- **AND** `significant_fdr` is False for rows with NaN adjusted p-values
- **AND** valid correlations receive correct FDR-adjusted p-values

#### Scenario: Single correlation (m=1)

- **WHEN** only one trait pair is tested (m=1)
- **THEN** no FDR correction is applied (single test, no multiple testing)
- **AND** adjusted p-values equal raw p-values
- **AND** `significant_fdr` is based on raw p-value comparison

#### Scenario: Fewer than 3 genotypes for a trait pair

- **WHEN** after removing NaN pairs, fewer than 3 valid genotype pairs remain
- **THEN** step sets all correlation values to NaN for that trait pair
- **AND** adjusted p-values are NaN for that trait pair
- **AND** confidence intervals are NaN for that trait pair
- **AND** `significant_fdr` is False for that trait pair
- **AND** logs warning about insufficient data

#### Scenario: Dual correlation calculation with Spearman primary

- **WHEN** correlation method is "spearman" with 18 valid genotypes and 50 trait pairs
- **THEN** step calculates BOTH Spearman and Pearson correlations for all 50 pairs
- **AND** exports CSV with columns including CI bounds for both methods
- **AND** sorts results by absolute Spearman correlation (descending)

#### Scenario: FDR correction with Benjamini-Hochberg

- **WHEN** fdr_correction_method is "fdr_bh" and 1000 trait pairs are tested
- **THEN** spearman_p_adjusted and pearson_p_adjusted contain BH-corrected p-values
- **AND** adjusted p-values are >= raw p-values
- **AND** significant_fdr is True when primary adjusted p < significance_level

#### Scenario: FDR correction with Benjamini-Yekutieli (default)

- **WHEN** fdr_correction_method is "fdr_by" (default) and traits are correlated
- **THEN** BY correction is applied (valid under arbitrary dependence)
- **AND** BY produces more conservative (larger) adjusted p-values than BH

#### Scenario: No FDR correction

- **WHEN** fdr_correction_method is "none"
- **THEN** spearman_p_adjusted equals spearman_p
- **AND** pearson_p_adjusted equals pearson_p
- **AND** significant_fdr uses raw p-values for threshold comparison

### Requirement: Reproducible Output Structure

The system SHALL generate reproducible outputs with consistent directory structure:

```
<output_base_dir>/
├── cross_platform_correlations.csv     # All correlation results (Pearson + Spearman + FDR + CI)
│   Columns: exp1_trait, exp2_trait, spearman_r, spearman_p, spearman_r_ci_low, spearman_r_ci_high,
│            pearson_r, pearson_p, pearson_r_ci_low, pearson_r_ci_high, n_genotypes,
│            spearman_p_adjusted, pearson_p_adjusted, significant_fdr
├── summary.json                         # Analysis metadata and summary statistics
├── figures/
│   ├── correlation_summary.png          # 4-panel summary visualization
│   ├── joint_01_trait1_vs_trait2.png   # Top correlation joint plots
│   ├── joint_02_trait1_vs_trait2.png
│   └── ...
│   ├── boxplot_01.png                   # Genotype distribution comparisons
│   ├── boxplot_02.png
│   └── ...
└── pipeline.log                         # Execution log
```

#### Scenario: CSV schema includes confidence interval columns

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains CI columns: `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r_ci_low`, `pearson_r_ci_high`
- **AND** column order places CI columns adjacent to their corresponding r values

#### Scenario: CSV schema includes FDR correction columns

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains columns for both raw and adjusted p-values
- **AND** column names include: spearman_p_adjusted, pearson_p_adjusted, significant_fdr

#### Scenario: Metadata includes confidence level

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes `confidence_level` parameter value used

#### Scenario: Metadata includes FDR correction information

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes fdr_correction_method and significant_correlations count

#### Scenario: Consistent output structure across runs

- **WHEN** user runs cross-platform pipeline multiple times with different configs
- **THEN** each run creates timestamped directory with identical internal structure for easy comparison

## ADDED Requirements

### Requirement: Correlation Confidence Interval Calculation

The system SHALL provide a function `calculate_correlation_ci(r, n, confidence_level=0.95)` in `cross_experiment_analysis.py` that computes confidence intervals for correlation coefficients using the Fisher z-transformation method:

1. **Transform to z-scale**: z = arctanh(r) = 0.5 × ln((1+r)/(1-r))
2. **Compute standard error**: SE_z = 1 / √(n-3)
3. **Compute z-scale CI**: z ± z_{α/2} × SE_z where α = 1 - confidence_level
4. **Back-transform to r-scale**: r = tanh(z)
5. **Clamp bounds**: Ensure -1 ≤ ci_low ≤ ci_high ≤ 1

Edge case handling:
- **r = ±1.0**: Return (r, r) as point mass (arctanh undefined at boundaries)
- **n < 4**: Return (NaN, NaN) as variance undefined (n-3 in denominator)
- **r = NaN**: Return (NaN, NaN)

#### Scenario: CI for moderate correlation with adequate sample size

- **WHEN** r = 0.5 and n = 20 with confidence_level = 0.95
- **THEN** CI bounds are approximately (0.06, 0.78)
- **AND** interval is symmetric on z-scale but asymmetric on r-scale

#### Scenario: CI for zero correlation

- **WHEN** r = 0.0 and n = 30 with confidence_level = 0.95
- **THEN** CI bounds are approximately (-0.36, 0.36)
- **AND** interval contains zero (expected for null correlation)

#### Scenario: CI for perfect correlation

- **WHEN** r = 1.0 and n = 50
- **THEN** CI is (1.0, 1.0) as point mass
- **AND** no mathematical error from arctanh(1.0) = infinity

#### Scenario: CI for negative perfect correlation

- **WHEN** r = -1.0 and n = 50
- **THEN** CI is (-1.0, -1.0) as point mass
- **AND** no mathematical error from arctanh(-1.0) = -infinity

#### Scenario: CI undefined for very small n

- **WHEN** r = 0.5 and n = 3
- **THEN** CI is (NaN, NaN)
- **AND** warning is logged about insufficient sample size for CI

#### Scenario: CI for near-boundary correlation

- **WHEN** r = 0.99 and n = 100
- **THEN** CI bounds are valid: ci_low < 0.99 < ci_high ≤ 1.0
- **AND** bounds are clamped to [-1, 1] if numerical precision causes overshoot

#### Scenario: Higher confidence level widens interval

- **WHEN** same r and n but confidence_level changes from 0.95 to 0.99
- **THEN** CI width increases by factor of approximately z_{0.005}/z_{0.025} ≈ 2.576/1.96 ≈ 1.31

### Requirement: Confidence Interval Documentation

The system SHALL document confidence interval methodology in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Mathematical formulation of Fisher z-transformation with:
  - Forward transformation: z = arctanh(r)
  - Standard error formula: SE_z = 1/√(n-3)
  - CI formula on z-scale: z ± z_{α/2} × SE_z
  - Back-transformation: r = tanh(z)
- Explanation of why n ≥ 4 is required (variance formula denominator)
- Note that Fisher CI is exact for Pearson, asymptotic approximation for Spearman
- Example calculation with r = 0.6, n = 25, 95% CI
- Interpretation guidance: narrower CI = more precise estimate
- CSV column descriptions for CI bounds
- Academic reference: Fisher, R.A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. Metron, 1, 3-32.

#### Scenario: User understands CI calculation

- **WHEN** user reads the Confidence Intervals section
- **THEN** they can manually calculate CI for a given r and n
- **AND** they understand why CI width depends on sample size

#### Scenario: User interprets CI in results

- **WHEN** user examines cross_platform_correlations.csv with CI columns
- **THEN** documentation explains that overlapping CIs suggest non-significant difference
- **AND** documentation clarifies CI is for the correlation coefficient, not for predictions
