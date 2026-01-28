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
- `min_genotypes_for_correlation`: Minimum number of valid genotypes required for a trait pair correlation, default 10. Trait pairs with fewer valid genotypes after NaN removal are excluded from output.
- `power_analysis_alpha`: Significance level (α) for power analysis, default 0.05. Used to calculate minimum detectable effect size and achieved power.
- `power_analysis_power`: Target power (1-β) for minimum detectable effect size calculation, default 0.80. Standard convention is 80% power.

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

#### Scenario: Invalid min_genotypes_for_correlation

- **WHEN** user specifies min_genotypes_for_correlation < 3
- **THEN** configuration validation fails with error indicating minimum value is 3

#### Scenario: Invalid power_analysis_alpha

- **WHEN** user specifies power_analysis_alpha outside (0, 1) exclusive range
- **THEN** configuration validation fails with error indicating valid range

#### Scenario: Invalid power_analysis_power

- **WHEN** user specifies power_analysis_power outside (0, 1) exclusive range
- **THEN** configuration validation fails with error indicating valid range

#### Scenario: Custom power analysis parameters

- **WHEN** user specifies power_analysis_alpha as 0.01 and power_analysis_power as 0.90
- **THEN** minimum detectable r is computed using α=0.01 and power=0.90
- **AND** achieved power is computed using α=0.01

### Requirement: Calculate Cross-Platform Correlations

The system SHALL calculate pairwise trait correlations between experiments through `CalculateCrossPlatformCorrelationsStep` with the following behavior:

- Calculate genotype means for each trait in both experiments using `calculate_genotype_means()`
- **Filter trait pairs**: Exclude trait pairs where n_genotypes < min_genotypes_for_correlation after NaN removal
- Compute **both Pearson and Spearman** correlations for all remaining trait pairs, regardless of which method is configured as primary
- Remove NaN pairs before correlation calculation
- Calculate p-values for each correlation (both Pearson and Spearman)
- **Calculate confidence intervals** for each correlation coefficient using Fisher z-transformation
- **Calculate achieved power** for each correlation using Fisher z-transformation power formula
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
- Store results with columns: `exp1_trait`, `exp2_trait`, `spearman_r`, `spearman_p`, `spearman_r_ci_low`, `spearman_r_ci_high`, `pearson_r`, `pearson_p`, `pearson_r_ci_low`, `pearson_r_ci_high`, `n_genotypes`, `achieved_power`, `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
- Sort results by absolute value of the **primary** correlation (determined by `correlation_method` config), descending
- Export results to `cross_platform_correlations.csv` in output directory
- **Log filtering information**: Log the count and reason for filtered trait pairs

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

#### Scenario: Trait pairs filtered by min_genotypes_for_correlation

- **WHEN** min_genotypes_for_correlation is 10 and a trait pair has n_genotypes = 7
- **THEN** that trait pair is excluded from the CSV output
- **AND** metadata includes count of filtered trait pairs
- **AND** metadata includes reason: "n_genotypes below threshold"
- **AND** log message indicates how many trait pairs were filtered

#### Scenario: Achieved power computed for all correlations

- **WHEN** correlation step executes with varying r values and n_genotypes
- **THEN** CSV output contains `achieved_power` column
- **AND** achieved_power is higher for larger |r| values (same n)
- **AND** achieved_power is higher for larger n_genotypes (same r)
- **AND** achieved_power is in range [0, 1]

#### Scenario: Achieved power for zero correlation

- **WHEN** r = 0.0 for a trait pair
- **THEN** achieved_power equals power_analysis_alpha (false positive rate, no effect to detect)

#### Scenario: Achieved power for very small n

- **WHEN** n_genotypes < 4 for a trait pair
- **THEN** achieved_power is NaN (power undefined for undefined variance)

### Requirement: Reproducible Output Structure

The system SHALL generate reproducible outputs with consistent directory structure:

```
<output_base_dir>/
├── cross_platform_correlations.csv     # All correlation results (Pearson + Spearman + FDR + CI + Power)
│   Columns: exp1_trait, exp2_trait, spearman_r, spearman_p, spearman_r_ci_low, spearman_r_ci_high,
│            pearson_r, pearson_p, pearson_r_ci_low, pearson_r_ci_high, n_genotypes, achieved_power,
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

#### Scenario: CSV schema includes achieved_power column

- **WHEN** cross-platform pipeline completes successfully
- **THEN** cross_platform_correlations.csv contains `achieved_power` column
- **AND** column is placed after n_genotypes

#### Scenario: Metadata includes confidence level

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes `confidence_level` parameter value used

#### Scenario: Metadata includes FDR correction information

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes fdr_correction_method and significant_correlations count

#### Scenario: Metadata includes power analysis parameters

- **WHEN** cross-platform pipeline completes successfully
- **THEN** metadata includes `power_analysis_alpha`, `power_analysis_power`, `minimum_detectable_r`
- **AND** metadata includes `modal_n_genotypes` (most common sample size used for MDR calculation)
- **AND** metadata includes `n_correlations_filtered_low_n` (count of trait pairs excluded due to low n)

#### Scenario: Consistent output structure across runs

- **WHEN** user runs cross-platform pipeline multiple times with different configs
- **THEN** each run creates timestamped directory with identical internal structure for easy comparison

## ADDED Requirements

### Requirement: Power Analysis Functions

The system SHALL provide power analysis functions in `cross_experiment_analysis.py` for computing statistical power of correlation tests:

**Function: `minimum_detectable_correlation(n, alpha=0.05, power=0.80)`**

Calculates the minimum absolute correlation coefficient detectable at the specified power level using Fisher z-transformation:

1. **Validate inputs**: Raise ValueError if n <= 0, alpha not in (0, 1), or power not in (0, 1)
2. **Compute critical z-values**: z_α = Φ⁻¹(1 - α/2), z_β = Φ⁻¹(power)
3. **Compute z_r threshold**: z_r = (z_α + z_β) / √(n - 3)
4. **Transform to correlation scale**: r = tanh(z_r)
5. **Handle edge cases**: n < 4 returns NaN (undefined variance)

**Function: `achieved_power(r, n, alpha=0.05)`**

Calculates the statistical power achieved for a given correlation coefficient using Fisher z-transformation:

1. **Validate inputs**: Raise ValueError if r not in [-1, 1] (and not NaN), alpha not in (0, 1), n <= 0
2. **Compute critical z-value**: z_α = Φ⁻¹(1 - α/2)
3. **Transform r to z-scale**: z_r = arctanh(|r|)
4. **Compute non-centrality parameter**: ncp = z_r × √(n - 3)
5. **Compute power**: power = Φ(ncp - z_α) + Φ(-ncp - z_α)
6. **Handle edge cases**:
   - r = 0: Return alpha (no effect to detect, power equals false positive rate)
   - r = NaN: Return NaN
   - n < 4: Return NaN (undefined variance)
   - r = ±1.0: Return 1.0 (perfect correlation, maximum power)

**References**:
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum Associates.
- Fisher, R.A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. Metron, 1, 3-32.

#### Scenario: Minimum detectable r with standard parameters

- **WHEN** n = 20, alpha = 0.05, power = 0.80
- **THEN** minimum_detectable_correlation returns approximately 0.58
- **AND** correlations with |r| < 0.58 have power < 80% at this sample size

#### Scenario: Minimum detectable r with large sample

- **WHEN** n = 100, alpha = 0.05, power = 0.80
- **THEN** minimum_detectable_correlation returns approximately 0.28
- **AND** larger samples can detect smaller effect sizes at 80% power

#### Scenario: Minimum detectable r with small sample

- **WHEN** n = 10, alpha = 0.05, power = 0.80
- **THEN** minimum_detectable_correlation returns approximately 0.76
- **AND** value is close to 1.0, indicating severe power limitations

#### Scenario: Achieved power for moderate correlation

- **WHEN** r = 0.5, n = 20, alpha = 0.05
- **THEN** achieved_power returns approximately 0.65
- **AND** this is below the 80% threshold, indicating underpowered

#### Scenario: Achieved power for strong correlation

- **WHEN** r = 0.7, n = 20, alpha = 0.05
- **THEN** achieved_power returns approximately 0.92
- **AND** this is above the 80% threshold, indicating adequate power

#### Scenario: Achieved power for zero correlation

- **WHEN** r = 0.0, n = 20, alpha = 0.05
- **THEN** achieved_power returns 0.05 (equals alpha)
- **AND** this represents the false positive rate (no effect to detect)

#### Scenario: Achieved power for perfect correlation

- **WHEN** r = 1.0, n = 20, alpha = 0.05
- **THEN** achieved_power returns 1.0 (maximum power)

#### Scenario: Achieved power with very small n

- **WHEN** r = 0.5, n = 3, alpha = 0.05
- **THEN** achieved_power returns NaN (variance undefined with n-3 denominator)

#### Scenario: Invalid inputs raise errors

- **WHEN** minimum_detectable_correlation is called with n = 0
- **THEN** ValueError is raised with message indicating n must be positive

- **WHEN** achieved_power is called with r = 1.5
- **THEN** ValueError is raised with message indicating r must be in [-1, 1]

- **WHEN** either function is called with alpha = 0 or alpha = 1
- **THEN** ValueError is raised with message indicating alpha must be in (0, 1)

### Requirement: Power Analysis Documentation

The system SHALL document power analysis methodology in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Explanation of statistical power and why it matters for correlation analysis
- Mathematical formulation of power calculation using Fisher z-transformation:
  - Minimum detectable r formula: r = tanh((z_α + z_β) / √(n-3))
  - Achieved power formula: power = Φ(z_r × √(n-3) - z_α) + Φ(-z_r × √(n-3) - z_α)
- Table of minimum detectable r values for common sample sizes (n = 10, 15, 20, 25, 30, 50, 100)
- Interpretation guidance:
  - power < 0.50: severely underpowered, results unreliable
  - power 0.50-0.80: moderately underpowered, interpret with caution
  - power >= 0.80: adequately powered, standard threshold
  - power >= 0.90: well-powered, high confidence
- Explanation that same formula applies to both Pearson and Spearman correlations (asymptotically)
- Note about n >= 10 recommendation for Spearman (Fisher z approximation accuracy)
- Academic references: Cohen (1988), Fisher (1921)

#### Scenario: User understands power interpretation

- **WHEN** user examines achieved_power column in CSV
- **THEN** documentation explains that power < 0.80 indicates underpowered correlation
- **AND** documentation provides context for interpreting specific power values

#### Scenario: User understands minimum detectable r

- **WHEN** user sees minimum_detectable_r in metadata
- **THEN** documentation explains this is the smallest effect size detectable at 80% power
- **AND** documentation clarifies correlations below this threshold are statistically unreliable

### Requirement: Minimum Genotypes Filter Documentation

The system SHALL document the minimum genotypes filter in `docs/CROSS_PLATFORM_ANALYSIS.md` with the following content:

- Explanation of why filtering is necessary:
  - NaN values in specific traits can reduce effective sample size below threshold
  - Very low n leads to unreliable correlations and undefined confidence intervals
  - Fisher z approximation is accurate for n >= 10 for Spearman correlations
- Description of the filtering behavior:
  - Hard filter: trait pairs below threshold are excluded from CSV
  - Filtering occurs after NaN removal, using actual n_genotypes
  - Metadata logs count and reason for filtered pairs
- Distinction from `min_samples_per_genotype`:
  - `min_samples_per_genotype`: Filters genotypes with too few biological replicates
  - `min_genotypes_for_correlation`: Filters trait pairs with too few valid genotypes after NaN removal
- Guidance on choosing the threshold:
  - n >= 4: Minimum for CI calculation (variance undefined below this)
  - n >= 10: Recommended for accurate Fisher z approximation (Spearman)
  - n >= 20: Preferred for reliable correlation estimates
- Example showing how trait-specific NaN patterns can reduce effective n

#### Scenario: User understands filtering reason

- **WHEN** user sees n_correlations_filtered_low_n in metadata
- **THEN** documentation explains these trait pairs had insufficient valid genotypes
- **AND** documentation clarifies this is protective, not a bug

#### Scenario: User chooses appropriate threshold

- **WHEN** user decides on min_genotypes_for_correlation value
- **THEN** documentation provides guidance based on statistical requirements
- **AND** documentation recommends n >= 10 as default for Spearman accuracy
